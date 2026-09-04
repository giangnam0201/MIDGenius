"""Velocity and expression: recovering the dynamics a plain note grid loses.

Naive converters emit every note at velocity 100. That is the biggest single
reason machine transcriptions sound robotic even when the pitches are right.

Here velocity comes from the actual attack energy in the stem audio, measured
in a band around each note's fundamental so that a quiet inner voice does not
inherit the loudness of the bass note underneath it.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np

from midgenius.notes import Note

log = logging.getLogger("midgenius.dynamics")


def _midi_to_hz(m: float) -> float:
    return 440.0 * 2.0 ** ((m - 69.0) / 12.0)


class BandEnergy:
    """Cached CQT of a stem, for per-note attack energy lookups."""

    def __init__(self, y: np.ndarray, sr: int, fmin_midi: int = 21,
                 n_octaves: int = 8, bins_per_octave: int = 12,
                 hop_length: int = 256):
        import librosa

        self.sr = sr
        self.hop = hop_length
        self.fmin_midi = fmin_midi
        self.bins_per_octave = bins_per_octave
        n_bins = n_octaves * bins_per_octave

        # Keep the CQT inside Nyquist.
        while n_bins > bins_per_octave:
            top = _midi_to_hz(fmin_midi + n_bins / (bins_per_octave / 12.0))
            if top < sr / 2.0 * 0.95:
                break
            n_bins -= bins_per_octave

        try:
            C = np.abs(librosa.cqt(
                y=y.astype(np.float32), sr=sr, hop_length=hop_length,
                fmin=_midi_to_hz(fmin_midi), n_bins=n_bins,
                bins_per_octave=bins_per_octave,
            ))
        except Exception as e:  # pragma: no cover - very short signals
            log.debug("CQT failed (%r); falling back to broadband RMS", e)
            C = None

        self.C = C
        self.n_bins = n_bins
        if C is not None:
            self.times = librosa.frames_to_time(np.arange(C.shape[1]), sr=sr,
                                                hop_length=hop_length)
        else:
            self.times = np.zeros(0)

        frame = hop_length * 4
        rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop_length)[0]
        self.rms = rms
        self.rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr,
                                                hop_length=hop_length)

    def _frame(self, t: float) -> int:
        return int(np.clip(round(t * self.sr / self.hop), 0, max(len(self.times) - 1, 0)))

    def attack_energy(self, note: Note, window_ms: float = 45.0,
                      semitone_span: int = 1) -> float:
        """Peak energy near the note's fundamental over its first few frames."""
        if self.C is None or self.C.size == 0:
            return self.broadband(note.start, window_ms)
        bin_idx = int(round(note.pitch - self.fmin_midi))
        if bin_idx < 0 or bin_idx >= self.n_bins:
            return self.broadband(note.start, window_ms)
        lo_b = max(0, bin_idx - semitone_span)
        hi_b = min(self.n_bins, bin_idx + semitone_span + 1)

        f0 = self._frame(note.start)
        span = max(1, int(round(window_ms / 1000.0 * self.sr / self.hop)))
        f1 = min(self.C.shape[1], f0 + span)
        if f1 <= f0:
            f1 = min(self.C.shape[1], f0 + 1)
        if f1 <= f0:
            return 0.0
        return float(np.max(self.C[lo_b:hi_b, f0:f1]))

    def broadband(self, t: float, window_ms: float = 45.0) -> float:
        if len(self.rms) == 0:
            return 0.0
        f0 = int(np.clip(round(t * self.sr / self.hop), 0, len(self.rms) - 1))
        span = max(1, int(round(window_ms / 1000.0 * self.sr / self.hop)))
        f1 = min(len(self.rms), f0 + span)
        return float(np.max(self.rms[f0:f1]))

    def envelope(self, t0: float, t1: float, n: int = 16) -> List[Tuple[float, float]]:
        """Normalised loudness curve over [t0, t1], for CC11 expression."""
        if len(self.rms) == 0 or t1 <= t0:
            return []
        peak = float(self.rms.max()) or 1.0
        ts = np.linspace(t0, t1, max(2, n))
        out = []
        for t in ts:
            f = int(np.clip(round(t * self.sr / self.hop), 0, len(self.rms) - 1))
            out.append((float(t), float(np.clip(self.rms[f] / peak, 0.0, 1.0))))
        return out


def assign_velocities(
    notes: Sequence[Note],
    band: Optional[BandEnergy],
    vel_min: int = 28,
    vel_max: int = 127,
    curve: float = 0.62,
    use_confidence_fallback: bool = True,
) -> None:
    """Set ``note.velocity`` from measured attack energy, in place.

    Energies are mapped through a log scale (loudness is perceptual, not
    linear) and then percentile-normalised across the track, so a quiet
    recording still uses the full velocity range instead of clumping at 20.
    """
    notes = list(notes)
    if not notes:
        return

    if band is None:
        if use_confidence_fallback:
            for n in notes:
                v = vel_min + (vel_max - vel_min) * float(np.clip(n.confidence, 0, 1))
                n.velocity = int(np.clip(round(v), 1, 127))
        return

    energies = np.array([band.attack_energy(n) for n in notes], dtype=np.float64)
    if not np.any(energies > 0):
        for n in notes:
            n.velocity = int(np.clip(round(vel_min + (vel_max - vel_min) * n.confidence), 1, 127))
        return

    db = 20.0 * np.log10(energies + 1e-9)
    # Robust range: ignore the loudest/quietest 5% so one outlier does not
    # compress everything else into a couple of velocity steps.
    lo = np.percentile(db, 5)
    hi = np.percentile(db, 95)
    if hi - lo < 6.0:                 # nearly flat dynamics
        hi = lo + 6.0
    norm = np.clip((db - lo) / (hi - lo), 0.0, 1.0)
    norm = norm ** curve              # curve < 1 lifts the quiet end

    for n, v in zip(notes, norm):
        n.velocity = int(np.clip(round(vel_min + (vel_max - vel_min) * v), 1, 127))


def attach_expression(notes: Sequence[Note], band: Optional[BandEnergy],
                      points: int = 12) -> None:
    """Attach a CC11 loudness curve to each note (for sustained instruments)."""
    if band is None:
        return
    for n in notes:
        if n.duration >= 0.25:
            n.expression = band.envelope(n.start, n.end, points)


def detect_sustain(y: np.ndarray, sr: int, notes: Sequence[Note],
                   hop: int = 512, min_hold: float = 0.35) -> List[Tuple[float, bool]]:
    """Infer sustain-pedal spans from note overlap density.

    When a pedalled passage is transcribed, many notes ring simultaneously and
    decay slowly. Writing an explicit CC64 span reproduces that on playback and
    lets the note-offs stay where the attacks actually stopped, which keeps the
    piano roll readable.
    """
    if not notes:
        return []

    ordered = sorted(notes, key=lambda n: n.start)
    end_t = max(n.end for n in ordered)
    grid_hop = 0.05
    grid = np.arange(0.0, end_t + grid_hop, grid_hop)
    density = np.zeros(len(grid), dtype=np.int32)
    for n in ordered:
        a = int(n.start / grid_hop)
        b = min(len(grid) - 1, int(n.end / grid_hop))
        if b > a:
            density[a:b] += 1

    pedal = density >= 3
    events: List[Tuple[float, bool]] = []
    state = False
    run_start = 0.0
    for i, p in enumerate(pedal):
        t = float(grid[i])
        if p and not state:
            state, run_start = True, t
            events.append((t, True))
        elif not p and state:
            if t - run_start < min_hold:
                events.pop()          # too short to be a real pedal press
            else:
                events.append((t, False))
            state = False
    if state:
        events.append((float(grid[-1]), False))
    return events
