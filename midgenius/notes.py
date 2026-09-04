"""Note representation, polyphonic decoding, and phantom-note suppression."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from midgenius.basicpitch import (
    ANNOTATIONS_FPS,
    CONTOURS_BINS_PER_SEMITONE,
    MAX_FREQ_IDX,
    MIDI_OFFSET,
    N_FREQ_BINS_CONTOURS,
    Posteriorgram,
    midi_to_contour_bin,
)

log = logging.getLogger("midgenius.notes")


@dataclass
class Note:
    """One transcribed note event."""

    start: float                 # seconds
    end: float                   # seconds
    pitch: int                   # MIDI note number
    velocity: int = 80           # 1..127
    confidence: float = 1.0      # mean model activation, 0..1
    # Pitch bend curve as (seconds, semitone_offset) pairs, relative to `pitch`.
    bends: Optional[List[Tuple[float, float]]] = None
    # Expression (CC11) curve as (seconds, 0..1) pairs.
    expression: Optional[List[Tuple[float, float]]] = None

    @property
    def duration(self) -> float:
        return self.end - self.start

    def shifted(self, dt: float) -> "Note":
        n = Note(self.start + dt, self.end + dt, self.pitch, self.velocity,
                 self.confidence, None, None)
        if self.bends:
            n.bends = [(t + dt, v) for t, v in self.bends]
        if self.expression:
            n.expression = [(t + dt, v) for t, v in self.expression]
        return n


@dataclass
class Track:
    """A named collection of notes destined for one MIDI track."""

    name: str
    notes: List[Note] = field(default_factory=list)
    program: int = 0
    channel: Optional[int] = None
    is_drum: bool = False
    # Sustain pedal as (seconds, bool) pairs.
    sustain: List[Tuple[float, bool]] = field(default_factory=list)

    def sort(self) -> None:
        self.notes.sort(key=lambda n: (n.start, n.pitch))


# --------------------------------------------------------------------------
# polyphonic decoding
# --------------------------------------------------------------------------

def infer_onsets(onset: np.ndarray, note: np.ndarray, n_diff: int = 2) -> np.ndarray:
    """Add onsets implied by sharp rises in the frame activations.

    The onset head misses soft attacks (legato piano, bowed strings). A positive
    jump in the note head is independent evidence of a new note, so we take the
    elementwise max of the two.
    """
    diffs = []
    for n in range(1, n_diff + 1):
        padded = np.concatenate([np.zeros((n, note.shape[1]), note.dtype), note])
        diffs.append(padded[n:, :] - padded[:-n, :])
    frame_diff = np.min(diffs, axis=0)
    frame_diff[frame_diff < 0] = 0
    frame_diff[:n_diff, :] = 0
    peak = frame_diff.max()
    if peak > 0:
        frame_diff = onset.max() * frame_diff / peak
    return np.maximum(onset, frame_diff)


def adaptive_thresholds(post: Posteriorgram, scale: float = 1.22,
                        frame_ratio: float = 0.5,
                        lo: float = 0.15, hi: float = 0.75
                        ) -> Tuple[float, float]:
    """Derive onset/frame thresholds from the model's own confidence spread.

    A fixed threshold cannot serve different material. Basic Pitch's onset head
    reports how *confident* it is, and that confidence scales with how sharp the
    attacks are: a chiptune with hard square-wave attacks peaks near 1.0, while
    soft synth pads on the same passage peak near 0.3. Applying one number to
    both means either drowning in phantoms or, as happened here, transcribing a
    sixth of the notes and calling the rest silence.

    So the threshold is set relative to the distribution of onset activation
    peaks on *this* material. Measured against reference transcriptions of two
    deliberately dissimilar tracks, the best threshold sat at essentially the
    same multiple of the 99th percentile of those peaks:

        soft synth pads   best onset 0.40 = 1.21x p99
        chiptune          best onset 0.60 = 1.23x p99

    Two tracks is thin evidence for the exact constant, but the *principle* -
    normalise to the model's own scale, rather than guessing an absolute - is
    the same one the percussion stage already relies on. ``lo``/``hi`` keep a
    degenerate stem from producing an absurd threshold.
    """
    if post.n_frames == 0 or post.onset.size == 0:
        return lo, lo * frame_ratio
    t, f = _argrelmax_time(post.onset)
    peaks = post.onset[t, f] if len(t) else post.onset.reshape(-1)
    if peaks.size == 0:
        return lo, lo * frame_ratio
    onset = float(np.clip(scale * float(np.percentile(peaks, 99)), lo, hi))
    return onset, max(0.10, onset * frame_ratio)


def decode_polyphonic(
    post: Posteriorgram,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    min_note_ms: float = 58.0,
    min_midi: int = 21,
    max_midi: int = 108,
    infer_onsets_flag: bool = True,
    melodia_trick: bool = True,
    energy_tolerance: int = 11,
) -> List[Note]:
    """Turn onset/frame posteriorgrams into discrete note events.

    Greedy peak-picking on the onset head, then each note is extended forward
    through the frame head until its energy stays below threshold for
    ``energy_tolerance`` frames. Claimed energy is erased (including the two
    neighbouring semitone bins, which absorbs the model's frequency smearing)
    so it cannot be claimed twice.

    The optional "melodia trick" then sweeps up whatever loud frame energy is
    left over with no onset attached to it, which recovers notes whose attacks
    were masked in a dense mix.
    """
    if post.n_frames == 0:
        return []

    note_act = post.note.copy()
    onset_act = post.onset.copy()
    n_frames, n_bins = note_act.shape

    # Hard pitch gate: outside an instrument's range every activation is a ghost.
    lo = max(0, int(min_midi) - MIDI_OFFSET)
    hi = min(n_bins, int(max_midi) - MIDI_OFFSET + 1)
    if lo > 0:
        onset_act[:, :lo] = 0
        note_act[:, :lo] = 0
    if hi < n_bins:
        onset_act[:, hi:] = 0
        note_act[:, hi:] = 0

    if infer_onsets_flag:
        onset_act = infer_onsets(onset_act, note_act)

    min_note_len = max(1, int(round(min_note_ms / 1000.0 * ANNOTATIONS_FPS)))

    # Local maxima in time, above threshold.
    peaks = _argrelmax_time(onset_act)
    peak_mat = np.zeros_like(onset_act)
    peak_mat[peaks] = onset_act[peaks]
    onset_idx = np.where(peak_mat >= onset_threshold)
    # Process backwards in time so that a note is bounded by the next onset.
    onset_t = onset_idx[0][::-1]
    onset_f = onset_idx[1][::-1]

    remaining = note_act.copy()
    events: List[Tuple[int, int, int, float]] = []

    for start_idx, freq_idx in zip(onset_t, onset_f):
        if start_idx >= n_frames - 1:
            continue
        i = start_idx + 1
        k = 0
        while i < n_frames - 1 and k < energy_tolerance:
            k = k + 1 if remaining[i, freq_idx] < frame_threshold else 0
            i += 1
        i -= k
        if i - start_idx <= min_note_len:
            continue

        remaining[start_idx:i, freq_idx] = 0
        if freq_idx < MAX_FREQ_IDX:
            remaining[start_idx:i, freq_idx + 1] = 0
        if freq_idx > 0:
            remaining[start_idx:i, freq_idx - 1] = 0

        amplitude = float(np.mean(note_act[start_idx:i, freq_idx]))
        events.append((int(start_idx), int(i), int(freq_idx + MIDI_OFFSET), amplitude))

    if melodia_trick:
        events.extend(_melodia_sweep(remaining, note_act, frame_threshold,
                                     min_note_len, energy_tolerance))

    times = post.times
    last_t = float(times[-1]) if len(times) else 0.0
    notes: List[Note] = []
    for s, e, pitch, amp in events:
        if not (min_midi <= pitch <= max_midi):
            continue
        t0 = float(times[s]) if s < len(times) else last_t
        t1 = float(times[e]) if e < len(times) else last_t
        if t1 <= t0:
            continue
        notes.append(Note(start=t0, end=t1, pitch=int(pitch), confidence=float(amp)))

    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes


def _argrelmax_time(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Indices of strict local maxima along the time axis."""
    if x.shape[0] < 3:
        return (np.array([], int), np.array([], int))
    mask = (x[1:-1] > x[:-2]) & (x[1:-1] > x[2:])
    t, f = np.nonzero(mask)
    return (t + 1, f)


def _melodia_sweep(remaining: np.ndarray, note_act: np.ndarray,
                   frame_threshold: float, min_note_len: int,
                   energy_tolerance: int) -> List[Tuple[int, int, int, float]]:
    """Recover notes from leftover frame energy that had no detected onset."""
    n_frames = remaining.shape[0]
    events = []
    guard = 0
    max_iters = 20000
    while remaining.max() > frame_threshold and guard < max_iters:
        guard += 1
        i_mid, freq_idx = np.unravel_index(int(np.argmax(remaining)), remaining.shape)
        remaining[i_mid, freq_idx] = 0

        def erase(i):
            remaining[i, freq_idx] = 0
            if freq_idx < MAX_FREQ_IDX:
                remaining[i, freq_idx + 1] = 0
            if freq_idx > 0:
                remaining[i, freq_idx - 1] = 0

        i, k = i_mid + 1, 0
        while i < n_frames - 1 and k < energy_tolerance:
            k = k + 1 if remaining[i, freq_idx] < frame_threshold else 0
            erase(i)
            i += 1
        i_end = i - 1 - k

        i, k = i_mid - 1, 0
        while i > 0 and k < energy_tolerance:
            k = k + 1 if remaining[i, freq_idx] < frame_threshold else 0
            erase(i)
            i -= 1
        i_start = max(0, i + 1 + k)

        if i_end - i_start <= min_note_len:
            continue
        amplitude = float(np.mean(note_act[i_start:i_end, freq_idx]))
        events.append((int(i_start), int(i_end), int(freq_idx + MIDI_OFFSET), amplitude))
    return events


# --------------------------------------------------------------------------
# pitch bends
# --------------------------------------------------------------------------

def estimate_pitch_bends(post: Posteriorgram, notes: Sequence[Note],
                         n_bins_tolerance: int = 25,
                         drop_when_overlapping: bool = True) -> None:
    """Attach a pitch-bend curve to each note, in place.

    The contour head has 3 bins per semitone, so tracking the argmax within a
    window around the note's nominal pitch recovers vibrato, slides and
    intonation drift - the expressive detail that plain note-on/note-off
    quantisation throws away.

    Overlapping notes share one MIDI pitch-bend controller per channel, so a
    bend written for one note would detune its neighbours. Those bends are
    dropped rather than written wrongly.
    """
    if post.n_frames == 0 or not notes:
        return

    import scipy.signal

    contours = post.contour
    times = post.times
    window_length = n_bins_tolerance * 2 + 1
    gaussian = scipy.signal.windows.gaussian(window_length, std=5)

    for note in notes:
        # Look the frame up in the model's own time axis. That axis is *not* a
        # uniform grid - each analysis window carries a small offset correction
        # - so computing an index as round(t * frame_rate) drifts steadily and
        # is over a second out by the end of a three minute track, which would
        # read the bend curve off entirely the wrong notes.
        s = int(np.clip(np.searchsorted(times, note.start), 0, post.n_frames - 1))
        e = int(np.clip(np.searchsorted(times, note.end), s + 1, post.n_frames))
        freq_idx = int(round(midi_to_contour_bin(note.pitch)))
        lo = max(freq_idx - n_bins_tolerance, 0)
        hi = min(N_FREQ_BINS_CONTOURS, freq_idx + n_bins_tolerance + 1)
        if hi <= lo:
            continue

        w_lo = max(0, n_bins_tolerance - freq_idx)
        w_hi = window_length - max(0, freq_idx - (N_FREQ_BINS_CONTOURS - n_bins_tolerance - 1))
        weights = gaussian[w_lo:w_hi]
        sub = contours[s:e, lo:hi]
        if sub.size == 0 or weights.shape[0] != sub.shape[1]:
            continue

        weighted = sub * weights
        shift = n_bins_tolerance - w_lo
        bins = (np.argmax(weighted, axis=1) - shift).astype(np.float64)
        semitones = bins / float(CONTOURS_BINS_PER_SEMITONE)

        # Frames with no pitch evidence must not vote. argmax over an all-zero
        # row returns bin 0, which would read as a bend of minus eight
        # semitones - a loud, audible defect invented out of silence.
        strength = weighted.max(axis=1)
        quiet = strength < max(1e-3, 0.08 * float(strength.max()))
        if quiet.all():
            note.bends = None
            continue
        semitones[quiet] = np.nan
        semitones = _fill_nan(semitones)

        # A jump larger than a whole tone between adjacent frames is the tracker
        # latching onto a neighbouring partial, not real expression.
        semitones = _despike(semitones, max_step=1.0)
        if np.allclose(semitones, 0.0):
            note.bends = None
            continue
        t = times[s:e] if e <= len(times) else np.linspace(note.start, note.end, len(semitones))
        note.bends = [(float(a), float(b)) for a, b in zip(t, semitones)]

    if drop_when_overlapping:
        _drop_overlapping_bends(notes)


def _fill_nan(x: np.ndarray) -> np.ndarray:
    """Carry the last good value across gaps; back-fill a leading gap."""
    out = x.astype(np.float64).copy()
    good = np.isfinite(out)
    if not good.any():
        return np.zeros_like(out)
    idx = np.where(good, np.arange(len(out)), 0)
    np.maximum.accumulate(idx, out=idx)
    out = out[idx]
    first = int(np.argmax(good))
    out[:first] = out[first]
    return out


def _despike(x: np.ndarray, max_step: float = 1.0) -> np.ndarray:
    if len(x) < 2:
        return x
    out = x.astype(np.float64).copy()
    for i in range(1, len(out)):
        if abs(out[i] - out[i - 1]) > max_step:
            out[i] = out[i - 1]
    return out


def _drop_overlapping_bends(notes: Sequence[Note]) -> None:
    """Remove bends from any note that overlaps another in time."""
    ordered = sorted(notes, key=lambda n: n.start)
    for i in range(len(ordered) - 1):
        a = ordered[i]
        for j in range(i + 1, len(ordered)):
            b = ordered[j]
            if b.start >= a.end:
                break
            a.bends = None
            b.bends = None


# --------------------------------------------------------------------------
# phantom note suppression
# --------------------------------------------------------------------------

def suppress_harmonic_ghosts(notes: List[Note], ratio: float = 0.28,
                             overlap_frac: float = 0.55) -> List[Note]:
    """Drop notes that are just harmonics or subharmonics of a stronger note.

    A single sounding note produces energy at +12, +19 and +24 semitones (and,
    for low-pitched sources with a missing fundamental, the model sometimes
    reports the octave *below*). Those "phantom" notes are time-aligned with
    their parent and much weaker. A note is removed when it

      * overlaps a stronger note by more than ``overlap_frac`` of its length,
      * sits at a harmonic interval from it, and
      * carries less than ``ratio`` of its confidence.
    """
    if not notes:
        return notes

    HARMONIC_INTERVALS = (12, 19, 24, 28, 31, 36, -12)
    order = sorted(range(len(notes)), key=lambda i: -notes[i].confidence)
    alive = [True] * len(notes)
    by_start = sorted(range(len(notes)), key=lambda i: notes[i].start)
    starts = np.array([notes[i].start for i in by_start])

    for idx in order:
        if not alive[idx]:
            continue
        parent = notes[idx]
        # Candidate children start before the parent ends.
        j0 = int(np.searchsorted(starts, parent.start - 0.12, side="left"))
        j1 = int(np.searchsorted(starts, parent.end, side="right"))
        for j in range(j0, j1):
            k = by_start[j]
            if k == idx or not alive[k]:
                continue
            child = notes[k]
            interval = child.pitch - parent.pitch
            if interval not in HARMONIC_INTERVALS:
                continue
            if child.confidence > parent.confidence * ratio:
                continue
            ov = min(parent.end, child.end) - max(parent.start, child.start)
            if child.duration <= 0 or ov / child.duration < overlap_frac:
                continue
            # Attacks must line up; a real note at the octave rarely starts
            # within 40 ms of its "parent" AND is this much quieter.
            if abs(child.start - parent.start) > 0.12:
                continue
            alive[k] = False

    kept = [n for n, a in zip(notes, alive) if a]
    if len(kept) != len(notes):
        log.debug("harmonic ghost filter removed %d notes", len(notes) - len(kept))
    return kept


def correct_octaves(post: Posteriorgram, notes: List[Note],
                    sub_ratio: float = 0.80, onset_ratio: float = 0.50,
                    min_pitch: int = 40) -> List[Note]:
    """Pull octave-too-high detections down to their true fundamental.

    Basic Pitch's characteristic error is reporting a note an octave above where
    it sounds: when the fundamental is weak (a missing-fundamental timbre, or a
    note masked in a chord) but its second harmonic is strong, the peak-picker
    latches onto ``p + 12``. Left alone, each such note is scored twice wrong -
    a false positive at ``p + 12`` and a false negative at ``p`` - so it is the
    most expensive single error class on soft, sustained material.

    For every detected note at pitch ``p`` this compares the note-head evidence
    at ``p`` with the evidence an octave below over the same frames. The note is
    moved down a octave when

      * the lower octave carries at least ``sub_ratio`` of the note-head energy
        at ``p`` (a real, not incidental, presence),
      * an onset actually fires there near the attack (``onset_ratio`` of the
        original), so we are moving to a genuine note start rather than into the
        middle of a held sub-harmonic, and
      * no already-detected note occupies ``p - 12`` at that time (otherwise the
        octave pair is real polyphony, not an error).

    Restricted to ``p >= min_pitch``: below that the "octave below" is often out
    of an instrument's range and the test misfires.
    """
    if post.n_frames == 0 or not notes:
        return notes

    note_act = post.note
    onset_act = post.onset
    times = post.times
    n_bins = note_act.shape[1]

    # Occupancy: for a quick "is p-12 already taken here" test, index note
    # spans by pitch.
    by_pitch: dict = {}
    for i, n in enumerate(notes):
        by_pitch.setdefault(n.pitch, []).append((n.start, n.end))
    for v in by_pitch.values():
        v.sort()

    def occupied(pitch: int, s: float, e: float) -> bool:
        for a, b in by_pitch.get(pitch, ()):  # small lists in practice
            if a < e and s < b:
                return True
        return False

    moved = 0
    for n in notes:
        if n.pitch < min_pitch:
            continue
        p_bin = n.pitch - MIDI_OFFSET
        if p_bin >= n_bins:
            continue
        s = int(np.clip(np.searchsorted(times, n.start), 0, post.n_frames - 1))
        e = int(np.clip(np.searchsorted(times, n.end), s + 1, post.n_frames))
        e_p = float(note_act[s:e, p_bin].mean())
        if e_p <= 1e-6:
            continue
        a0 = max(0, s - 2)
        a1 = min(post.n_frames, s + 3)
        on_p = float(onset_act[a0:a1, p_bin].max())

        # Consider one and two octaves below. Basic Pitch's harmonic false
        # notes land at +12 and +24 above the fundamental. Prefer the deepest
        # candidate that clears the tests, so a note two octaves high is pulled
        # all the way down rather than one step at a time.
        best = None
        for down in (24, 12):
            sub_bin = p_bin - down
            if sub_bin < 0:
                continue
            e_sub = float(note_act[s:e, sub_bin].mean())
            if e_sub < sub_ratio * e_p:
                continue
            if onset_ratio > 0.0:
                on_sub = float(onset_act[a0:a1, sub_bin].max())
                if on_sub < onset_ratio * max(on_p, 1e-6):
                    continue
            if occupied(n.pitch - down, n.start, n.end):
                continue
            best = down
            break
        if best is None:
            continue
        # Keep occupancy current so we do not stack two moved notes onto one slot.
        by_pitch.setdefault(n.pitch - best, []).append((n.start, n.end))
        n.pitch -= best
        moved += 1

    if moved:
        log.debug("octave correction pulled %d notes down an octave", moved)
    return notes


def deghost_octaves(post: Posteriorgram, notes: List[Note],
                    onset_ratio: float = 0.5, overlap_frac: float = 0.5,
                    intervals: Tuple[int, ...] = (12, 24),
                    max_attack_gap: float = 0.05) -> List[Note]:
    """Drop octave notes that are harmonics, judged by onset independence.

    Basic Pitch's note head fires at a note's octave harmonics as well as its
    fundamental, so a single sounded note at pitch ``p`` often produces a
    spurious extra note at ``p + 12`` (and ``p + 24``). The hard part is telling
    that ghost apart from a genuine octave doubling the music actually plays.

    Note *confidence* (mean frame activation) does not separate them - a real
    octave note and a strong harmonic ghost both light up the frame head - which
    is why the confidence-ratio filter loses real notes when pushed hard. The
    **onset** head does separate them: a genuinely played octave note has its own
    attack, so an independent onset activation fires at ``p + 12``; a harmonic
    ghost has only the bleed of the fundamental's onset there.

    So a note at ``p + interval`` is removed only when a stronger, near-
    simultaneous, overlapping note sits at ``p`` and the upper note's own onset
    activation is less than ``onset_ratio`` of the lower note's. That keeps
    deliberate octaves while cutting the harmonic doublings.
    """
    if post.n_frames == 0 or not notes:
        return notes

    times = post.times
    onset = post.onset
    n_bins = onset.shape[1]

    def onset_strength(n: Note) -> float:
        s = int(np.clip(np.searchsorted(times, n.start), 0, post.n_frames - 1))
        b = n.pitch - MIDI_OFFSET
        if not (0 <= b < n_bins):
            return 0.0
        lo = max(0, s - 1)
        hi = min(post.n_frames, s + 2)
        return float(onset[lo:hi, b].max())

    strength = {id(n): onset_strength(n) for n in notes}

    by_pitch: dict = {}
    for n in notes:
        by_pitch.setdefault(n.pitch, []).append(n)
    for v in by_pitch.values():
        v.sort(key=lambda n: n.start)

    alive = {id(n): True for n in notes}
    # Weakest-onset notes are the likeliest ghosts; test them first.
    for child in sorted(notes, key=lambda n: strength[id(n)]):
        if not alive[id(child)]:
            continue
        killed = False
        for iv in intervals:
            for parent in by_pitch.get(child.pitch - iv, ()):
                if not alive[id(parent)] or parent is child:
                    continue
                if abs(parent.start - child.start) > max_attack_gap:
                    continue
                ov = min(parent.end, child.end) - max(parent.start, child.start)
                if child.duration <= 0 or ov / child.duration < overlap_frac:
                    continue
                if strength[id(parent)] <= strength[id(child)]:
                    continue
                if strength[id(child)] < onset_ratio * strength[id(parent)]:
                    alive[id(child)] = False
                    killed = True
                    break
            if killed:
                break

    kept = [n for n in notes if alive[id(n)]]
    if len(kept) != len(notes):
        log.debug("octave deghost removed %d notes", len(notes) - len(kept))
    return kept


def drop_low_confidence(notes: List[Note], min_confidence: float) -> List[Note]:
    if min_confidence <= 0:
        return notes
    return [n for n in notes if n.confidence >= min_confidence]


def limit_polyphony(notes: List[Note], max_voices: int) -> List[Note]:
    """Cap simultaneous voices, keeping the most confident ones.

    Dense mixes make the model hallucinate a wash of weak notes. Real
    instruments have a voice limit; enforcing it is a cheap, musically
    motivated way to cut that wash.
    """
    if max_voices <= 0 or not notes:
        return notes

    events = []
    for i, n in enumerate(notes):
        events.append((n.start, 1, i))
        events.append((n.end, 0, i))
    events.sort(key=lambda e: (e[0], e[1]))

    active: set = set()
    drop: set = set()
    for _, kind, i in events:
        if kind == 0:
            active.discard(i)
            continue
        active.add(i)
        if len(active) > max_voices:
            ranked = sorted(active, key=lambda k: notes[k].confidence)
            for victim in ranked[: len(active) - max_voices]:
                drop.add(victim)
                active.discard(victim)
    return [n for i, n in enumerate(notes) if i not in drop]


def merge_repeats(notes: List[Note], gap_ms: float = 30.0,
                  max_total_ms: float = 6000.0) -> List[Note]:
    """Join same-pitch notes separated by a sub-perceptual gap.

    Frame-threshold dropouts split one held note into a stutter of short
    repeats. Anything under ~30 ms apart was never two separate attacks.
    """
    if not notes:
        return notes
    gap = gap_ms / 1000.0
    max_total = max_total_ms / 1000.0
    by_pitch = {}
    for n in notes:
        by_pitch.setdefault(n.pitch, []).append(n)

    out: List[Note] = []
    for pitch, group in by_pitch.items():
        group.sort(key=lambda n: n.start)
        cur = group[0]
        for nxt in group[1:]:
            if (nxt.start - cur.end) <= gap and (nxt.end - cur.start) <= max_total:
                w1, w2 = cur.duration, nxt.duration
                total = max(w1 + w2, 1e-9)
                cur = Note(
                    start=cur.start, end=max(cur.end, nxt.end), pitch=pitch,
                    velocity=max(cur.velocity, nxt.velocity),
                    confidence=(cur.confidence * w1 + nxt.confidence * w2) / total,
                    bends=cur.bends, expression=cur.expression,
                )
            else:
                out.append(cur)
                cur = nxt
        out.append(cur)
    out.sort(key=lambda n: (n.start, n.pitch))
    return out


def remove_duplicates(notes: List[Note], tol: float = 0.02) -> List[Note]:
    """Drop notes that duplicate another at the same pitch and near-same time."""
    out: List[Note] = []
    seen: List[Note] = []
    for n in sorted(notes, key=lambda x: (x.pitch, x.start, -x.confidence)):
        dup = False
        for m in reversed(seen):
            if m.pitch != n.pitch or n.start - m.start > tol:
                break
            if abs(n.start - m.start) <= tol:
                dup = True
                break
        if not dup:
            out.append(n)
            seen.append(n)
    out.sort(key=lambda n: (n.start, n.pitch))
    return out


def enforce_min_duration(notes: List[Note], min_ms: float) -> List[Note]:
    m = min_ms / 1000.0
    return [n for n in notes if n.duration >= m]


def trim_overlaps(notes: List[Note], min_gap_ms: float = 6.0) -> List[Note]:
    """Stop a note before the next attack at the same pitch.

    Overlapping same-pitch notes are ambiguous in MIDI - most synths kill the
    first voice when the second arrives, producing a shortened, wrong-sounding
    note. Trimming explicitly makes the file mean what it plays.
    """
    gap = min_gap_ms / 1000.0
    by_pitch = {}
    for n in notes:
        by_pitch.setdefault(n.pitch, []).append(n)
    for group in by_pitch.values():
        group.sort(key=lambda n: n.start)
        for a, b in zip(group, group[1:]):
            if a.end > b.start - gap:
                a.end = max(a.start + 0.01, b.start - gap)
    return [n for n in notes if n.duration > 0]
