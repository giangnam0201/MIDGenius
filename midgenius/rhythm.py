"""Tempo, beats, downbeats, key, and quantisation.

A transcription that is right about every pitch but has no tempo map is still
painful to use: imported into a DAW its bar lines drift away from the music
within a few seconds, and nothing can be edited on a grid.

So we estimate a beat grid, express every note position in *beats* rather than
seconds, and write a real tempo map. Live playing speeds up and slows down, and
a single average BPM cannot represent that - hence the variable tempo map,
which tracks the beat-to-beat interval and keeps bar one, beat one where the
music actually puts it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from midgenius.notes import Note

log = logging.getLogger("midgenius.rhythm")

# Quantisation grids, in quarter notes.
GRIDS: Dict[str, float] = {
    "1/1": 4.0, "1/2": 2.0, "1/4": 1.0, "1/8": 0.5,
    "1/16": 0.25, "1/32": 0.125,
    "1/4t": 2.0 / 3.0, "1/8t": 1.0 / 3.0, "1/16t": 1.0 / 6.0,
}


@dataclass
class TempoMap:
    """Maps seconds to musical position and back.

    ``beat_times[i]`` is the wall-clock time of beat ``i`` (a quarter note).
    Everything in between is linearly interpolated, which is exactly the
    piecewise-constant-tempo model MIDI itself uses.
    """

    beat_times: np.ndarray
    bpm: float
    downbeat_index: int = 0
    beats_per_bar: int = 4

    @property
    def is_variable(self) -> bool:
        return len(self.beat_times) > 2

    def time_to_beat(self, t: float | np.ndarray) -> np.ndarray:
        """Seconds -> position in quarter notes."""
        bt = self.beat_times
        t = np.asarray(t, dtype=np.float64)
        if len(bt) < 2:
            return t * (self.bpm / 60.0)
        idx = np.arange(len(bt), dtype=np.float64)
        out = np.interp(t, bt, idx)
        # Extrapolate outside the tracked range at the edge tempo.
        first_dt = max(bt[1] - bt[0], 1e-6)
        last_dt = max(bt[-1] - bt[-2], 1e-6)
        before = t < bt[0]
        after = t > bt[-1]
        if np.any(before):
            out = np.where(before, (t - bt[0]) / first_dt, out)
        if np.any(after):
            out = np.where(after, (len(bt) - 1) + (t - bt[-1]) / last_dt, out)
        return out

    def beat_to_time(self, b: float | np.ndarray) -> np.ndarray:
        bt = self.beat_times
        b = np.asarray(b, dtype=np.float64)
        if len(bt) < 2:
            return b * (60.0 / self.bpm)
        idx = np.arange(len(bt), dtype=np.float64)
        out = np.interp(b, idx, bt)
        first_dt = max(bt[1] - bt[0], 1e-6)
        last_dt = max(bt[-1] - bt[-2], 1e-6)
        out = np.where(b < 0, bt[0] + b * first_dt, out)
        out = np.where(b > len(bt) - 1, bt[-1] + (b - (len(bt) - 1)) * last_dt, out)
        return out

    def segment_tempi(self) -> List[Tuple[float, float]]:
        """(time_seconds, bpm) tempo events, one per beat, run-length compressed."""
        bt = self.beat_times
        if len(bt) < 2:
            return [(0.0, self.bpm)]
        events: List[Tuple[float, float]] = []
        last = None
        clamped = 0
        for i in range(len(bt) - 1):
            dt = max(bt[i + 1] - bt[i], 1e-6)
            raw = 60.0 / dt
            # A clamp here is not cosmetic: the tempo map *is* the file's time
            # axis, so any interval we cannot express exactly shifts every note
            # after it. The range is wide enough that clamping means something
            # upstream produced a degenerate beat interval - so say so.
            bpm = float(np.clip(raw, 5.0, 900.0))
            if abs(bpm - raw) > 1e-6:
                clamped += 1
            # Only emit when the tempo actually moved - a tempo event per beat
            # for a machine-steady track is noise in the file.
            if last is None or abs(bpm - last) > 0.05:
                events.append((float(bt[i]), bpm))
                last = bpm
        if clamped:
            log.warning("%d beat interval(s) needed an out-of-range tempo and "
                        "were clamped; output timing may drift", clamped)
        if not events:
            events = [(float(bt[0]), self.bpm)]
        return events


def analyze_rhythm(y: np.ndarray, sr: int, fixed_tempo: Optional[float] = None,
                   beats_per_bar: int = 4,
                   percussive_hint: Optional[np.ndarray] = None) -> TempoMap:
    """Estimate tempo, beat positions and the downbeat phase.

    ``percussive_hint`` (the drum stem, if we have one) makes beat tracking far
    more reliable than the full mix: no sustained pads or vocal onsets to
    confuse the onset envelope.
    """
    import librosa

    src = percussive_hint if percussive_hint is not None and percussive_hint.size else y
    src = np.asarray(src, dtype=np.float32).reshape(-1)
    if src.size < sr:
        return TempoMap(np.zeros(0), float(fixed_tempo or 120.0), 0, beats_per_bar)

    hop = 512
    onset_env = librosa.onset.onset_strength(y=src, sr=sr, hop_length=hop,
                                             aggregate=np.median)

    kwargs = dict(onset_envelope=onset_env, sr=sr, hop_length=hop, units="time",
                  trim=False)
    if fixed_tempo:
        kwargs["bpm"] = float(fixed_tempo)
    else:
        kwargs["start_bpm"] = 120.0

    tempo, beats = librosa.beat.beat_track(**kwargs)
    beats = np.asarray(beats, dtype=np.float64)
    tempo = float(np.atleast_1d(tempo)[0])

    if len(beats) < 2:
        return TempoMap(beats, float(fixed_tempo or tempo or 120.0), 0, beats_per_bar)

    downbeat = _estimate_downbeat(onset_env, beats, sr, hop, beats_per_bar)
    beats = _anchor_grid(beats, downbeat, beats_per_bar)
    log.info("tempo %.1f BPM, %d beats, downbeat phase %d",
             tempo, len(beats), downbeat)
    return TempoMap(beats, tempo, 0, beats_per_bar)


def _anchor_grid(beats: np.ndarray, downbeat: int, beats_per_bar: int) -> np.ndarray:
    """Fill the beat grid in from t=0, so musical position 0 is audio time 0.

    Beat tracking starts at the first detected beat, not at the start of the
    file. That leaves two problems, and the fix for one must not create the
    other:

    * Anything before the first beat maps to a *negative* musical position and
      gets clamped onto tick 0 - a whole intro piled into one chord.
    * Anchoring beat 0 anywhere before t=0 is worse still: MIDI tick 0 is the
      start of the file by definition, so a grid starting at -0.5 s silently
      shifts every note in the output later by half a second.

    So beat 0 is pinned to exactly t=0, and the gap up to the first detected
    beat is filled with evenly spaced pickup beats. The *number* of pickup
    beats is free, which lets us also satisfy the downbeat: choosing a count
    congruent to the detected phase puts bar one where a musician would count
    it. If that would require an implausible pickup tempo, timing wins and the
    bar phase is left alone.
    """
    if len(beats) < 2:
        return beats

    ibi = float(np.median(np.diff(beats[:min(len(beats), 16)])))
    if not np.isfinite(ibi) or ibi <= 0:
        return beats

    def pull_to_zero() -> np.ndarray:
        out = beats.copy()
        out[0] = 0.0
        return out

    # Too small a gap to hold even one pickup beat. Inserting one anyway would
    # demand an absurd tempo for that interval, which then gets clamped - and a
    # clamped tempo shifts every following note in the file.
    if beats[0] < 0.45 * ibi:
        return pull_to_zero()

    n_min = max(1, int(round(beats[0] / ibi)))
    n_lead = n_min
    want = (-downbeat) % beats_per_bar
    while n_lead % beats_per_bar != want:
        n_lead += 1

    if not (0.45 * ibi <= beats[0] / n_lead <= 2.2 * ibi):
        n_lead = n_min          # bar alignment not reachable at a sane tempo
    if beats[0] / n_lead < 0.45 * ibi:
        return pull_to_zero()

    lead = np.linspace(0.0, float(beats[0]), n_lead + 1)[:-1]
    return np.concatenate([lead, beats])


def _estimate_downbeat(onset_env: np.ndarray, beats: np.ndarray, sr: int,
                       hop: int, beats_per_bar: int) -> int:
    """Pick which beat of the bar carries the most accent.

    Bar one is where the strongest accents recur every ``beats_per_bar`` beats.
    Getting this right is what puts the transcription's bar lines where a
    musician would put them.
    """
    if len(beats) < beats_per_bar * 2:
        return 0
    frames = np.clip((beats * sr / hop).astype(int), 0, len(onset_env) - 1)
    strengths = onset_env[frames]
    scores = [float(strengths[phase::beats_per_bar].mean())
              for phase in range(beats_per_bar)]
    return int(np.argmax(scores))


# --------------------------------------------------------------------------
# quantisation
# --------------------------------------------------------------------------

def quantize_notes(notes: Sequence[Note], tempo_map: TempoMap, grid: str,
                   strength: float = 0.75, quantize_ends: bool = True,
                   min_duration_beats: float = 0.0625) -> List[Note]:
    """Pull note positions toward a rhythmic grid.

    ``strength`` is a blend, not a snap: 1.0 puts every note exactly on the
    grid (right for programmed music, wrong for anything played by a human),
    while 0.7-0.8 tightens the timing but keeps the push-and-pull that makes a
    performance sound alive. That trade-off is why this is a dial rather than a
    checkbox.
    """
    if grid in ("off", "", None) or not notes:
        return list(notes)
    step = GRIDS.get(grid)
    if step is None:
        log.warning("unknown quantise grid %r - skipping", grid)
        return list(notes)

    strength = float(np.clip(strength, 0.0, 1.0))
    out: List[Note] = []
    for n in notes:
        b0 = float(tempo_map.time_to_beat(n.start))
        b1 = float(tempo_map.time_to_beat(n.end))

        g0 = round(b0 / step) * step
        nb0 = b0 + (g0 - b0) * strength

        if quantize_ends:
            g1 = round(b1 / step) * step
            nb1 = b1 + (g1 - b1) * strength
            # A note quantised to zero length is worse than an unquantised one.
            if nb1 - nb0 < min_duration_beats:
                nb1 = nb0 + max(min_duration_beats, (b1 - b0) * 0.5)
        else:
            nb1 = nb0 + (b1 - b0)

        t0 = float(tempo_map.beat_to_time(nb0))
        t1 = float(tempo_map.beat_to_time(nb1))
        shift = t0 - n.start
        m = Note(start=t0, end=max(t1, t0 + 0.01), pitch=n.pitch,
                 velocity=n.velocity, confidence=n.confidence)
        if n.bends:
            m.bends = [(t + shift, v) for t, v in n.bends]
        if n.expression:
            m.expression = [(t + shift, v) for t, v in n.expression]
        out.append(m)

    out.sort(key=lambda n: (n.start, n.pitch))
    return out


def snap_drums(notes: Sequence[Note], tempo_map: TempoMap, grid: str = "1/16",
               strength: float = 0.9, max_shift_ms: float = 60.0) -> List[Note]:
    """Quantise percussion harder than pitched material, but never far.

    Drum hits carry the groove, so they benefit most from a tight grid. The
    distance cap stops a genuinely syncopated or flammed hit from being dragged
    onto the wrong subdivision.
    """
    if grid in ("off", "", None) or not notes:
        return list(notes)
    step = GRIDS.get(grid)
    if step is None:
        return list(notes)

    cap = max_shift_ms / 1000.0
    out: List[Note] = []
    for n in notes:
        b = float(tempo_map.time_to_beat(n.start))
        g = round(b / step) * step
        nb = b + (g - b) * strength
        t = float(tempo_map.beat_to_time(nb))
        if abs(t - n.start) > cap:
            t = n.start + np.sign(t - n.start) * cap
        dur = n.end - n.start
        out.append(Note(start=float(t), end=float(t + dur), pitch=n.pitch,
                        velocity=n.velocity, confidence=n.confidence))
    out.sort(key=lambda n: (n.start, n.pitch))
    return out


# --------------------------------------------------------------------------
# key detection
# --------------------------------------------------------------------------

# Krumhansl-Schmuckler key profiles.
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52,
                   5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54,
                   4.75, 3.98, 2.69, 3.34, 3.17])
PITCH_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def detect_key(notes: Sequence[Note]) -> Tuple[int, bool, str]:
    """Estimate the key from transcribed notes.

    Returns (tonic_pitch_class, is_major, label). Weighting each pitch class by
    total sounding duration rather than note count is what keeps a flurry of
    passing sixteenths from outvoting the held chord tones.
    """
    if not notes:
        return 0, True, "C"

    profile = np.zeros(12)
    for n in notes:
        profile[n.pitch % 12] += max(n.duration, 0.02) * (n.velocity / 127.0)
    if profile.sum() <= 0:
        return 0, True, "C"
    profile = profile / profile.sum()

    best_score, best = -np.inf, (0, True)
    for tonic in range(12):
        for is_major, template in ((True, _MAJOR), (False, _MINOR)):
            rotated = np.roll(template, tonic)
            rotated = rotated / rotated.sum()
            score = float(np.corrcoef(profile, rotated)[0, 1])
            if np.isfinite(score) and score > best_score:
                best_score, best = score, (tonic, is_major)

    tonic, is_major = best
    label = PITCH_NAMES[tonic] + ("" if is_major else "m")
    return tonic, is_major, label
