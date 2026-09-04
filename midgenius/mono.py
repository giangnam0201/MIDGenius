"""Monophonic transcription for bass lines and lead vocals.

For a genuinely monophonic source, a dedicated f0 tracker beats a polyphonic
model by a wide margin. Basic Pitch has to hedge across 88 possible pitches per
frame; pYIN commits to one, with a proper voicing decision, and its octave
errors are far rarer on bass than any spectral-peak method.

Just as importantly, a continuous f0 contour is where the *expression* lives.
Vibrato, scoops into a note, portamento between notes and blue-note bends are
all visible in the contour and would be flattened away by rounding each frame
to the nearest semitone.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from midgenius.notes import Note

log = logging.getLogger("midgenius.mono")

HOP = 256
PYIN_SR = 22050


def track_f0(y: np.ndarray, sr: int, fmin: float = 55.0, fmax: float = 1200.0,
             frame_length: int = 2048) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run pYIN. Returns (midi_float, voiced_prob, times); unvoiced frames NaN."""
    import librosa
    from midgenius.audio import resample

    if sr != PYIN_SR:
        y = resample(np.asarray(y, np.float32), sr, PYIN_SR)
        sr = PYIN_SR
    y = np.asarray(y, dtype=np.float32)

    fmax = min(fmax, sr / 2.0 * 0.95)
    fmin = max(fmin, 20.0)
    if fmax <= fmin * 1.5 or len(y) < frame_length * 2:
        empty = np.zeros(0)
        return empty, empty, empty

    # pYIN needs a few periods of the lowest pitch inside the analysis frame.
    min_frame = int(2 ** np.ceil(np.log2(4.0 * sr / fmin)))
    frame_length = max(frame_length, min_frame)

    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, fmin=fmin, fmax=fmax, sr=sr,
        frame_length=frame_length, hop_length=HOP,
        fill_na=np.nan, center=True,
    )
    midi = np.full(len(f0), np.nan)
    ok = np.isfinite(f0) & (f0 > 0)
    midi[ok] = 69.0 + 12.0 * np.log2(f0[ok] / 440.0)
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=HOP)
    voiced_prob = np.nan_to_num(voiced_prob, nan=0.0)
    return midi, voiced_prob, times


def _onset_frames(y: np.ndarray, sr: int, times: np.ndarray) -> np.ndarray:
    """Frame indices (on the pYIN grid) where a new attack was detected.

    Needed to split repeated notes at the same pitch, which the contour alone
    cannot distinguish from one long held note.
    """
    import librosa
    from midgenius.audio import resample

    if sr != PYIN_SR:
        y = resample(np.asarray(y, np.float32), sr, PYIN_SR)
        sr = PYIN_SR
    if len(y) < 2048 or len(times) == 0:
        return np.zeros(0, dtype=int)
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    peaks = librosa.util.peak_pick(
        env, pre_max=6, post_max=6, pre_avg=12, post_avg=12,
        delta=float(np.percentile(env, 65)) * 0.35, wait=6,
    )
    return np.asarray(peaks, dtype=int)


def transcribe_mono(
    y: np.ndarray,
    sr: int,
    fmin: float = 55.0,
    fmax: float = 1200.0,
    min_voiced_prob: float = 0.55,
    min_note_ms: float = 70.0,
    min_midi: int = 21,
    max_midi: int = 108,
    pitch_change_semitones: float = 0.62,
    smooth_ms: float = 45.0,
    keep_bends: bool = True,
    vibrato_preserve: bool = True,
    split_on_onsets: bool = True,
) -> List[Note]:
    """Transcribe a monophonic stem into notes with pitch-bend curves."""
    midi, vprob, times = track_f0(y, sr, fmin, fmax)
    if len(midi) == 0:
        return []

    voiced = np.isfinite(midi) & (vprob >= min_voiced_prob)
    if not voiced.any():
        return []

    # Median-smooth the contour to kill single-frame octave slips without
    # blurring genuine vibrato (median preserves edges; a mean would not).
    span = max(3, int(round(smooth_ms / 1000.0 * PYIN_SR / HOP)) | 1)
    smoothed = _median_filter_nan(midi, span)

    onsets = set(_onset_frames(y, sr, times).tolist()) if split_on_onsets else set()

    segments = _segment(smoothed, voiced, onsets, pitch_change_semitones)

    frame_dt = HOP / float(PYIN_SR)
    min_frames = max(1, int(round(min_note_ms / 1000.0 * PYIN_SR / HOP)))

    notes: List[Note] = []
    for s, e in segments:
        if e - s < min_frames:
            continue
        seg = smoothed[s:e]
        seg = seg[np.isfinite(seg)]
        if seg.size == 0:
            continue

        # Robust centre. Using the median rather than the mean keeps a scoop
        # into the note from dragging the reported pitch flat.
        centre = float(np.median(seg))
        pitch = int(round(centre))
        if not (min_midi <= pitch <= max_midi):
            continue

        start = float(times[s])
        end = float(times[min(e, len(times) - 1)])
        if end - start <= 0:
            continue
        conf = float(np.mean(vprob[s:e])) if e > s else 0.0

        note = Note(start=start, end=end, pitch=pitch, confidence=conf)

        if keep_bends:
            dev = smoothed[s:e] - pitch
            dev = np.nan_to_num(dev, nan=0.0)
            if not vibrato_preserve:
                dev = _median_filter_nan(dev, span)
                dev = np.nan_to_num(dev, nan=0.0)
            # Only bother writing a bend if it is musically visible (>12 cents).
            if np.max(np.abs(dev)) > 0.12:
                t = times[s:e]
                note.bends = [(float(a), float(np.clip(b, -2.0, 2.0)))
                              for a, b in zip(t, dev)]
        notes.append(note)

    notes.sort(key=lambda n: n.start)
    return notes


def _median_filter_nan(x: np.ndarray, span: int) -> np.ndarray:
    """Median filter that ignores NaNs (unvoiced frames)."""
    if span <= 1 or len(x) == 0:
        return x.copy()
    half = span // 2
    padded = np.pad(x.astype(np.float64), (half, half), constant_values=np.nan)
    out = np.empty_like(x, dtype=np.float64)
    for i in range(len(x)):
        w = padded[i:i + span]
        w = w[np.isfinite(w)]
        out[i] = np.median(w) if w.size else np.nan
    return out


def _segment(midi: np.ndarray, voiced: np.ndarray, onsets: set,
             pitch_change: float) -> List[Tuple[int, int]]:
    """Cut the contour into note segments.

    A new note begins when the source goes voiced, when the pitch steps by more
    than ``pitch_change`` semitones away from the current note's running centre,
    or when a percussive attack was detected at the same pitch.
    """
    segments: List[Tuple[int, int]] = []
    n = len(midi)
    i = 0
    while i < n:
        if not voiced[i] or not np.isfinite(midi[i]):
            i += 1
            continue
        start = i
        ref = midi[i]
        acc = [midi[i]]
        i += 1
        while i < n and voiced[i] and np.isfinite(midi[i]):
            if abs(midi[i] - ref) > pitch_change:
                break
            if i in onsets and i - start > 3:
                break
            acc.append(midi[i])
            # Track the running centre so a slow glide eventually splits.
            ref = float(np.median(acc[-24:]))
            i += 1
        segments.append((start, i))
    return segments


def fill_short_gaps(notes: List[Note], max_gap_ms: float = 45.0) -> List[Note]:
    """Close sub-perceptual gaps between consecutive same-pitch notes.

    pYIN briefly loses voicing on consonants and on note transitions; without
    this, a sung phrase turns into a machine-gun of 80 ms fragments.
    """
    if not notes:
        return notes
    gap = max_gap_ms / 1000.0
    out = [notes[0]]
    for n in notes[1:]:
        prev = out[-1]
        if n.pitch == prev.pitch and (n.start - prev.end) <= gap:
            prev.end = n.end
            prev.confidence = max(prev.confidence, n.confidence)
            if prev.bends and n.bends:
                prev.bends = prev.bends + n.bends
            continue
        out.append(n)
    return out


def fix_octave_jumps(notes: List[Note], max_jump: int = 12) -> List[Note]:
    """Correct isolated one-octave outliers in an otherwise smooth line.

    A single short note an octave from both its neighbours is an f0 halving or
    doubling error, not a melodic leap.
    """
    if len(notes) < 3:
        return notes
    for i in range(1, len(notes) - 1):
        prev, cur, nxt = notes[i - 1], notes[i], notes[i + 1]
        if cur.duration > 0.35:
            continue
        d_prev = cur.pitch - prev.pitch
        d_next = cur.pitch - nxt.pitch
        if abs(d_prev) == max_jump and abs(d_next) == max_jump and d_prev == d_next:
            cur.pitch -= int(np.sign(d_prev)) * max_jump
    return notes
