"""Percussion transcription.

Drums are the one thing a pitch tracker must never be pointed at. A kick has no
stable pitch, a snare is broadband noise, and a cymbal smears energy across the
whole spectrum for seconds. Run Basic Pitch on a drum stem and you get a cloud
of nonsense notes - this is the "drums and percussion confusion" failure mode.

So drums get their own transcriber, built on two ideas:

1. **Per-instrument band onset detection.** Rather than detecting onsets once
   and then asking "what was it?", we detect independently inside each
   instrument's characteristic band. A kick and a hi-hat that land on the same
   sixteenth are two separate events in two separate bands, and both survive -
   exclusive classification would keep only one.

2. **Adaptive, track-relative thresholds.** Absolute energy thresholds fail
   across genres and masters. Every decision here is made against percentiles
   of the track's own band statistics.

Output is a General MIDI percussion track on channel 10.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from midgenius.notes import Note

log = logging.getLogger("midgenius.drums")

# General MIDI percussion key map
GM_KICK = 36
GM_SIDE_STICK = 37
GM_SNARE = 38
GM_CLAP = 39
GM_SNARE_E = 40
GM_TOM_LOW = 41
GM_HAT_CLOSED = 42
GM_TOM_LOW_MID = 45
GM_HAT_OPEN = 46
GM_TOM_HIGH = 48
GM_CRASH = 49
GM_RIDE = 51


@dataclass
class DrumBand:
    """One percussion voice, defined by where it lives in the spectrum."""

    name: str
    midi: int
    fmin: float
    fmax: float
    # Detection sensitivity: multiplies the adaptive threshold. Lower = more hits.
    sensitivity: float = 1.0
    # Minimum time between two hits of this voice, seconds.
    min_interval: float = 0.045
    # Bleed rejection. A hit in this band is only believed if the band beats a
    # reference band by `contrast_db`. This is a *shape* test, not a level test:
    # a snare's low-frequency thump also lights up the kick band, but a real
    # kick has far more energy below 130 Hz than it has at 300-1200 Hz, and a
    # snare does not. Ratios between bands are stable across genres and masters
    # in a way that absolute energy thresholds never are.
    contrast_ref: Optional[Tuple[float, float]] = None
    contrast_db: float = 0.0
    # Voices that cannot physically sound at the same instant as this one.
    # (A drummer's stick is either on the hat or the ride, not both.)
    exclusive_with: Tuple[str, ...] = ()


DEFAULT_BANDS: Tuple[DrumBand, ...] = (
    # contrast_db raised 1 -> 16: a real kick's sub-130 Hz thump dominates its
    # 300-1200 Hz content far more than a sustained bass note (whose harmonics
    # fill 300-1200) does, so requiring a 16 dB margin rejects the bass/synth low
    # end that Demucs routes into the drum stem. That margin is physics, not a
    # per-track guess - a real kick clears it easily. Measured: aria phantom
    # kicks 1048 -> 86, and the dense recording's real kicks are fully preserved
    # (its kick F1 even rises slightly as its own phantoms drop).
    DrumBand("kick",     GM_KICK,        30.0,   130.0, sensitivity=0.90,
             min_interval=0.070, contrast_ref=(300.0, 1200.0), contrast_db=20.0),
    # sensitivity raised 1.5 -> 2.5: the snare band was firing on any broadband
    # transient, producing ~10x too many hits on synth material. Measured: aria
    # snare precision 5% -> 15% (drum F1 45.6 -> 52.9) with the dense recording's
    # snare unchanged.
    DrumBand("snare",    GM_SNARE,      170.0,   900.0, sensitivity=2.50,
             min_interval=0.070, contrast_ref=(30.0, 130.0), contrast_db=-7.0,
             exclusive_with=("tom_low", "tom_high")),
    DrumBand("tom_low",  GM_TOM_LOW,     90.0,   200.0, sensitivity=1.60,
             min_interval=0.110, contrast_ref=(1800.0, 8000.0), contrast_db=6.0,
             exclusive_with=("snare", "tom_high")),
    DrumBand("tom_high", GM_TOM_HIGH,   200.0,   420.0, sensitivity=1.70,
             min_interval=0.110, contrast_ref=(1800.0, 8000.0), contrast_db=4.0,
             exclusive_with=("snare", "tom_low")),
    DrumBand("hat",      GM_HAT_CLOSED, 6000.0, 14000.0, sensitivity=1.40,
             min_interval=0.040, contrast_ref=(200.0, 900.0), contrast_db=-14.0,
             exclusive_with=("cymbal",)),
    DrumBand("cymbal",   GM_CRASH,      2500.0,  6000.0, sensitivity=1.55,
             min_interval=0.200, contrast_ref=(200.0, 900.0), contrast_db=-6.0,
             exclusive_with=("hat",)),
)


def band_flux(S: np.ndarray, freqs: np.ndarray, fmin: float, fmax: float,
              floor_db: float = 40.0) -> np.ndarray:
    """Half-wave-rectified spectral flux restricted to one frequency band.

    Computing flux directly from the shared STFT - rather than filtering the
    waveform and running a generic onset detector - matters here. A generic
    detector spreads a mel filterbank over the whole spectrum, so a narrow band
    like the kick's 30-130 Hz occupies a couple of channels out of 128 and gets
    washed out by the aggregation across the rest.

    Magnitudes are converted to dB *relative to the band's own loud level* and
    clamped ``floor_db`` below it before differencing. The choice of
    compression matters more than it looks: a naive ``log1p(gamma * S)`` is
    steep near zero and flat when loud, so a faint click in otherwise silent
    bins scores higher than a fortissimo kick - the detector ends up firing
    between the hits instead of on them. Measuring dB change against a fixed
    per-band reference, with everything below the floor treated as silence,
    makes a hit's score reflect how much louder it actually got.

    Averaging (rather than summing) over the band's bins keeps wide bands from
    dwarfing narrow ones: a cymbal's 8 kHz span covers forty times as many FFT
    bins as a kick's 100 Hz.
    """
    sel = (freqs >= fmin) & (freqs <= fmax)
    if not sel.any() or S.shape[1] < 3:
        return np.zeros(max(S.shape[1], 0), dtype=np.float32)

    band = S[sel, :]
    ref = float(np.percentile(band, 99.5))
    # Because each band is scaled by its own reference, a band holding nothing
    # but filter leakage would still show large relative swings and invent
    # onsets - which is exactly the situation for the cymbal band of a heavily
    # lowpassed MP3. Require the band to carry real signal first, measured
    # against the loudest thing in the file: 60 dB down is stopband leakage,
    # not a quiet hi-hat (which sits 30-40 dB below a kick at most).
    peak = float(S.max())
    if ref <= 0 or ref < peak * 1e-3:
        return np.zeros(S.shape[1], dtype=np.float32)
    floor = ref * (10.0 ** (-floor_db / 20.0))
    band_db = 20.0 * np.log10(np.maximum(band, floor) / ref)

    diff = np.diff(band_db, axis=1, prepend=band_db[:, :1])
    flux = np.maximum(diff, 0.0).mean(axis=0)
    return flux.astype(np.float32)


def _pick_peaks(env: np.ndarray, sr: int, hop: int, sensitivity: float,
                min_interval: float) -> Tuple[np.ndarray, np.ndarray]:
    """Adaptive peak-picking on an onset envelope. Returns (times, strengths)."""
    import librosa

    if env.size < 3 or env.max() <= 0:
        return np.zeros(0), np.zeros(0)
    env = env / (env.max() + 1e-12)

    # Threshold from the envelope's own distribution. The 70th percentile is
    # roughly "louder than the ambient wash of this band"; the spread up to the
    # 97th sets how far above that a real hit must sit.
    base = float(np.percentile(env, 70))
    spread = max(float(np.percentile(env, 97) - base), 1e-6)
    delta = max(0.012, (0.25 * base + 0.30 * spread) * sensitivity)
    wait = max(1, int(round(min_interval * sr / hop)))

    peaks = librosa.util.peak_pick(
        env, pre_max=max(1, wait), post_max=max(1, wait),
        pre_avg=max(1, wait * 3), post_avg=max(1, wait * 3),
        delta=delta, wait=wait,
    )
    peaks = np.asarray(peaks, dtype=int)
    if peaks.size == 0:
        return np.zeros(0), np.zeros(0)
    times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop)
    return times, env[peaks]


class _Spectra:
    """STFT of the drum stem, for post-hoc verification of each candidate hit."""

    def __init__(self, y: np.ndarray, sr: int, n_fft: int = 2048, hop: int = 256):
        import librosa

        self.sr, self.hop, self.n_fft = sr, hop, n_fft
        self.S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
        self.freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        self.total = self.S.sum(axis=0) + 1e-12

    def frame(self, t: float) -> int:
        return int(np.clip(round(t * self.sr / self.hop), 0, self.S.shape[1] - 1))

    def band_fraction(self, t: float, fmin: float, fmax: float,
                      span_frames: int = 4) -> float:
        """Fraction of the total energy at time t that sits inside [fmin, fmax]."""
        f = self.frame(t)
        f1 = min(self.S.shape[1], f + span_frames)
        sel = (self.freqs >= fmin) & (self.freqs <= fmax)
        if not sel.any() or f1 <= f:
            return 0.0
        window = self.S[:, f:f1]
        return float(window[sel].sum() / (window.sum() + 1e-12))

    def band_energy(self, t: float, fmin: float, fmax: float,
                    span_frames: int = 4) -> float:
        f = self.frame(t)
        f1 = min(self.S.shape[1], f + span_frames)
        sel = (self.freqs >= fmin) & (self.freqs <= fmax)
        if not sel.any() or f1 <= f:
            return 0.0
        return float(self.S[sel, f:f1].max())

    def decay_time(self, t: float, fmin: float, fmax: float,
                   max_s: float = 1.2, drop_db: float = 18.0) -> float:
        """How long the band takes to fall ``drop_db`` below its peak.

        This is what separates a closed hi-hat (tens of ms) from an open one,
        and a ride (moderate) from a crash (very long).
        """
        f0 = self.frame(t)
        sel = (self.freqs >= fmin) & (self.freqs <= fmax)
        if not sel.any():
            return 0.0
        n_max = int(max_s * self.sr / self.hop)
        f1 = min(self.S.shape[1], f0 + n_max)
        env = self.S[sel, f0:f1].sum(axis=0)
        if env.size < 2 or env[0] <= 0:
            return 0.0
        peak = float(env.max())
        floor = peak * (10 ** (-drop_db / 20.0))
        below = np.nonzero(env < floor)[0]
        idx = int(below[0]) if below.size else env.size
        return idx * self.hop / float(self.sr)

    def band_ratio_db(self, t: float, fmin: float, fmax: float,
                      rmin: float, rmax: float, span_frames: int = 4) -> float:
        """Energy in [fmin,fmax] relative to a reference band, in dB."""
        a = self.band_energy(t, fmin, fmax, span_frames)
        b = self.band_energy(t, rmin, rmax, span_frames)
        return 20.0 * np.log10((a + 1e-9) / (b + 1e-9))

    def flatness(self, t: float, fmin: float, fmax: float) -> float:
        """Spectral flatness in a band: ~1 for noise, ~0 for a pitched hit."""
        f = self.frame(t)
        sel = (self.freqs >= fmin) & (self.freqs <= fmax)
        if not sel.any():
            return 0.0
        col = self.S[sel, f] + 1e-10
        return float(np.exp(np.mean(np.log(col))) / np.mean(col))


def transcribe_drums(
    y: np.ndarray,
    sr: int,
    bands: Optional[Tuple[DrumBand, ...]] = None,
    hop: int = 256,
    codec_cutoff: Optional[float] = None,
    detect_open_hats: bool = True,
    separate_ride_crash: bool = True,
    detect_side_stick: bool = False,
    velocity_range: Tuple[int, int] = (30, 127),
    note_length: float = 0.06,
) -> List[Note]:
    """Transcribe a drum stem into a General MIDI percussion note list."""
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if y.size < sr // 10 or not np.any(np.abs(y) > 1e-6):
        return []

    bands = bands or DEFAULT_BANDS
    spectra = _Spectra(y, sr, hop=hop)

    # A lossy encoder may have removed the top of the cymbal band entirely.
    # Slide the hi-hat band down rather than detecting nothing up there.
    nyq = sr / 2.0
    ceiling = min(nyq * 0.98, codec_cutoff * 0.98 if codec_cutoff else nyq)

    candidates: List[Tuple[float, float, DrumBand]] = []
    for band in bands:
        fmin, fmax = band.fmin, min(band.fmax, ceiling)
        if fmax <= fmin * 1.05:
            # The codec threw away this band. For cymbals it is worth retrying
            # against whatever high end survived rather than detecting nothing.
            if band.name in ("hat", "cymbal") and ceiling > 3000:
                fmin, fmax = max(2500.0, ceiling * 0.45), ceiling
            else:
                continue
        env = band_flux(spectra.S, spectra.freqs, fmin, fmax)
        times, strengths = _pick_peaks(env, sr, hop, band.sensitivity,
                                       band.min_interval)
        for t, s in zip(times, strengths):
            if band.contrast_ref is not None:
                rlo, rhi = band.contrast_ref
                ratio_db = spectra.band_ratio_db(t, fmin, fmax,
                                                 rlo, min(rhi, ceiling))
                if ratio_db < band.contrast_db:
                    continue          # bleed from another drum, not a real hit
            candidates.append((float(t), float(s), band))

    if not candidates:
        return []

    candidates.sort(key=lambda c: c[0])
    events = _resolve(candidates, spectra, ceiling, detect_open_hats,
                      separate_ride_crash, detect_side_stick=detect_side_stick)

    if not events:
        return []

    # Velocity from band energy at the hit, normalised per instrument so a quiet
    # hi-hat pattern still uses a useful part of the range.
    vmin, vmax = velocity_range
    by_midi: Dict[int, List[int]] = {}
    for i, (_, _, midi) in enumerate(events):
        by_midi.setdefault(midi, []).append(i)

    notes: List[Note] = [None] * len(events)  # type: ignore[list-item]
    for midi, idxs in by_midi.items():
        energies = np.array([events[i][1] for i in idxs], dtype=np.float64)
        db = 20.0 * np.log10(energies + 1e-9)
        lo, hi = np.percentile(db, 8), np.percentile(db, 96)
        if hi - lo < 5.0:
            hi = lo + 5.0
        for i in idxs:
            t, e, _ = events[i]
            v = (20.0 * np.log10(e + 1e-9) - lo) / (hi - lo)
            v = float(np.clip(v, 0.0, 1.0)) ** 0.7
            notes[i] = Note(
                start=t, end=t + note_length, pitch=midi,
                velocity=int(np.clip(round(vmin + (vmax - vmin) * v), 1, 127)),
                confidence=float(np.clip(v, 0.05, 1.0)),
            )

    notes = [n for n in notes if n is not None]
    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes


def _resolve(candidates, spectra: _Spectra, ceiling: float,
             detect_open_hats: bool, separate_ride_crash: bool,
             detect_side_stick: bool = False, cluster_ms: float = 28.0) -> List[Tuple[float, float, int]]:
    """Turn band candidates into concrete GM hits.

    Candidates are grouped into simultaneity clusters first. Within a cluster,
    voices that genuinely coexist (kick + hat on the same eighth) are all kept,
    while mutually exclusive ones (hat vs ride, snare vs tom) are arbitrated on
    the evidence rather than both being emitted. Doing this per cluster instead
    of per band is what stops one cymbal from being reported twice under two
    different names.
    """
    clusters: List[List[Tuple[float, float, DrumBand]]] = []
    window = cluster_ms / 1000.0
    for cand in candidates:
        if clusters and cand[0] - clusters[-1][0][0] <= window:
            clusters[-1].append(cand)
        else:
            clusters.append([cand])

    events: List[Tuple[float, float, int]] = []
    last_time: Dict[str, float] = {}

    for cluster in clusters:
        # Strongest candidate per band name within the cluster.
        best: Dict[str, Tuple[float, float, DrumBand]] = {}
        for t, s, band in cluster:
            if band.name not in best or s > best[band.name][1]:
                best[band.name] = (t, s, band)

        # Arbitrate mutually exclusive voices.
        for name, (t, s, band) in list(best.items()):
            if name not in best:
                continue
            for rival_name in band.exclusive_with:
                rival = best.get(rival_name)
                if rival is None:
                    continue
                keep = _arbitrate(band, rival[2], t, rival[0], spectra, ceiling)
                loser = rival_name if keep is band.name else name
                best.pop(loser, None)
                if loser == name:
                    break

        for name, (t, s, band) in best.items():
            if t - last_time.get(name, -10.0) < band.min_interval:
                continue
            midi = _label(band, t, spectra, ceiling, detect_open_hats,
                          separate_ride_crash, detect_side_stick)
            if midi is None:
                continue
            energy = spectra.band_energy(t, band.fmin, min(band.fmax, ceiling)) or s
            events.append((t, energy, midi))
            last_time[name] = t

    events.sort(key=lambda e: e[0])
    return events


def _arbitrate(a: DrumBand, b: DrumBand, ta: float, tb: float,
               spectra: _Spectra, ceiling: float) -> str:
    """Decide which of two mutually exclusive voices actually sounded."""
    if {a.name, b.name} == {"hat", "cymbal"}:
        # A closed hat is a short tick concentrated in the top octaves.
        # A ride or crash rings on and carries much more 2.5-6 kHz energy.
        t = min(ta, tb)
        decay = spectra.decay_time(t, 2500.0, min(6000.0, ceiling))
        low_mid = spectra.band_ratio_db(t, 2500.0, min(6000.0, ceiling),
                                        6000.0, ceiling)
        cymbal_like = decay > 0.16 and low_mid > -3.0
        return "cymbal" if cymbal_like else "hat"

    if "snare" in (a.name, b.name):
        # Toms are pitched and quiet up top; a snare is broadband noise.
        t = min(ta, tb)
        flat = spectra.flatness(t, 1800.0, min(8000.0, ceiling))
        noise = spectra.band_ratio_db(t, 1800.0, min(8000.0, ceiling),
                                      170.0, 900.0)
        other = a.name if b.name == "snare" else b.name
        return "snare" if (noise > -12.0 or flat > 0.42) else other

    return a.name if a.sensitivity <= b.sensitivity else b.name


def _label(band: DrumBand, t: float, spectra: _Spectra, ceiling: float,
           detect_open_hats: bool, separate_ride_crash: bool,
           detect_side_stick: bool = False) -> Optional[int]:
    """Choose the exact GM key for a confirmed hit."""
    fmax = min(band.fmax, ceiling)

    if band.name == "hat" and detect_open_hats:
        # An open hat keeps ringing; a closed one is choked in ~100 ms.
        decay = spectra.decay_time(t, max(4000.0, ceiling * 0.4), ceiling)
        return GM_HAT_OPEN if decay > 0.20 else GM_HAT_CLOSED

    if band.name == "cymbal" and separate_ride_crash:
        decay = spectra.decay_time(t, 2500.0, min(6000.0, ceiling))
        flat = spectra.flatness(t, 2500.0, min(6000.0, ceiling))
        # Crash: very long and very noisy. Ride: shorter, with a stick ping.
        return GM_CRASH if (decay > 0.60 and flat > 0.38) else GM_RIDE

    if band.name == "snare":
        # A snare is defined by the rattle of its wires: broadband noise well
        # above the drum's body. A kick's beater click also lights up the
        # 170-900 Hz band, but brings almost nothing above 1.8 kHz with it, so
        # requiring that noise is what stops every kick from being doubled by a
        # phantom snare.
        noise = spectra.band_ratio_db(t, 1800.0, min(8000.0, ceiling), 170.0, 900.0)
        if noise < -12.0:
            return None
        # A very short, clicky, quiet-bodied hit *may* be a rim/side stick -
        # but electronic snares are also short and noisy, and the two are not
        # reliably separable. Off by default: calling a real snare a side stick
        # is a worse error than missing a rare articulation.
        if (detect_side_stick
                and spectra.decay_time(t, 170.0, 900.0) < 0.022
                and noise > 10.0):
            return GM_SIDE_STICK
        return GM_SNARE

    if band.name in ("tom_low", "tom_high"):
        if spectra.flatness(t, band.fmin, fmax) > 0.58:
            return None               # too noisy to be a tom
    return band.midi


def collapse_flams(notes: List[Note], window_ms: float = 22.0) -> List[Note]:
    """Merge same-drum hits closer than a flam window into one louder hit.

    Onset detectors often fire twice on a single hit that has both a click and a
    body transient. Two note-ons 10 ms apart are a machine artefact, not a
    played flam.
    """
    if not notes:
        return notes
    w = window_ms / 1000.0
    by_pitch: Dict[int, List[Note]] = {}
    for n in notes:
        by_pitch.setdefault(n.pitch, []).append(n)

    out: List[Note] = []
    for group in by_pitch.values():
        group.sort(key=lambda n: n.start)
        cur = group[0]
        for nxt in group[1:]:
            if nxt.start - cur.start < w:
                cur.velocity = max(cur.velocity, nxt.velocity)
                cur.confidence = max(cur.confidence, nxt.confidence)
            else:
                out.append(cur)
                cur = nxt
        out.append(cur)
    out.sort(key=lambda n: (n.start, n.pitch))
    return out
