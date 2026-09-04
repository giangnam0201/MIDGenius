"""Ground-truth round-trip benchmark for MIDGenius.

Transcription cannot be judged by listening to the output and nodding. This
builds a score whose every note is known, renders it to audio, encodes it as a
real MP3, converts it back, and scores the result with `mir_eval` - the same
metrics used in the MIR transcription literature.

    python tools/benchmark.py                # run every case
    python tools/benchmark.py --case bass    # one case
    python tools/benchmark.py --keep out/    # keep the rendered audio

Metrics, per instrument:

    P / R / F1 (onset)      note is correct if its onset is within 50 ms and
                            its pitch is within 50 cents          [mir_eval]
    P / R / F1 (on+off)     as above, and the offset must also match
    onset error             median |predicted - true| onset time
    octave errors           right pitch class, wrong octave - the classic
                            f0-tracker failure, counted separately
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

SR = 44100


# --------------------------------------------------------------------------
# ground truth score
# --------------------------------------------------------------------------

@dataclass
class GTNote:
    start: float
    end: float
    pitch: int
    velocity: int = 100


@dataclass
class Case:
    """A synthetic piece with known content."""

    name: str
    bpm: float = 120.0
    bass: List[GTNote] = field(default_factory=list)
    keys: List[GTNote] = field(default_factory=list)
    lead: List[GTNote] = field(default_factory=list)
    drums: List[Tuple[float, str]] = field(default_factory=list)

    def pitched(self) -> Dict[str, List[GTNote]]:
        out = {}
        if self.bass:
            out["bass"] = self.bass
        if self.keys:
            out["keys"] = self.keys
        if self.lead:
            out["lead"] = self.lead
        return out


def _beats(bpm: float, n: int) -> np.ndarray:
    return np.arange(n) * (60.0 / bpm)


def case_bass() -> Case:
    """A monophonic bass line - tests the pYIN path and octave stability."""
    bpm = 100.0
    b = 60.0 / bpm
    pitches = [40, 40, 43, 45, 40, 38, 36, 43]
    notes = [GTNote(i * b, i * b + b * 0.85, p) for i, p in enumerate(pitches * 2)]
    return Case("bass", bpm=bpm, bass=notes)


def case_chords() -> Case:
    """Block chords - tests polyphony and harmonic ghost rejection."""
    bpm = 90.0
    b = 60.0 / bpm
    chords = [(60, 64, 67), (57, 60, 64), (62, 65, 69), (59, 62, 67)]
    notes = []
    for i, ch in enumerate(chords * 3):
        t = i * 2 * b
        for p in ch:
            notes.append(GTNote(t, t + 2 * b * 0.9, p))
    return Case("chords", bpm=bpm, keys=notes)


def case_drums() -> Case:
    """A backbeat - tests the percussion transcriber."""
    bpm = 120.0
    b = 60.0 / bpm
    hits = []
    for bar in range(8):
        t0 = bar * 4 * b
        hits += [(t0, "kick"), (t0 + 2 * b, "kick")]
        hits += [(t0 + b, "snare"), (t0 + 3 * b, "snare")]
        for e in range(8):
            hits.append((t0 + e * b / 2, "hat"))
    return Case("drums", bpm=bpm, drums=hits)


def case_mix() -> Case:
    """Everything at once - the real test: dense, polyphonic, with drums."""
    bpm = 110.0
    b = 60.0 / bpm
    bass = [GTNote(i * b, i * b + b * 0.9, p)
            for i, p in enumerate([36, 36, 43, 41, 38, 38, 45, 43] * 2)]
    chords = [(60, 63, 67), (58, 62, 65), (57, 60, 64), (55, 59, 62)]
    keys = []
    for i, ch in enumerate(chords * 4):
        t = i * b
        for p in ch:
            keys.append(GTNote(t, t + b * 0.85, p))
    lead = [GTNote(i * b * 0.5, i * b * 0.5 + b * 0.4, p)
            for i, p in enumerate([72, 74, 75, 79, 77, 75, 74, 72] * 4)]
    hits = []
    for bar in range(4):
        t0 = bar * 4 * b
        hits += [(t0, "kick"), (t0 + 2 * b, "kick"),
                 (t0 + b, "snare"), (t0 + 3 * b, "snare")]
        for e in range(8):
            hits.append((t0 + e * b / 2, "hat"))
    return Case("mix", bpm=bpm, bass=bass, keys=keys, lead=lead, drums=hits)


CASES = {c.name: c for c in (case_bass(), case_chords(), case_drums(), case_mix())}


# --------------------------------------------------------------------------
# synthesis
# --------------------------------------------------------------------------

def _adsr(n: int, sr: int, attack: float, decay: float, sustain: float,
          release: float) -> np.ndarray:
    a = max(1, int(attack * sr))
    d = max(1, int(decay * sr))
    r = max(1, int(release * sr))
    s = max(1, n - a - d - r)
    env = np.concatenate([
        np.linspace(0, 1, a),
        np.linspace(1, sustain, d),
        np.full(s, sustain),
        np.linspace(sustain, 0, r),
    ])
    return env[:n] if len(env) >= n else np.pad(env, (0, n - len(env)))


def synth_note(note: GTNote, sr: int, timbre: str) -> np.ndarray:
    n = max(1, int((note.end - note.start) * sr))
    t = np.arange(n) / sr
    f0 = 440.0 * 2 ** ((note.pitch - 69) / 12.0)
    amp = note.velocity / 127.0

    if timbre == "bass":
        # Strong fundamental, few harmonics - like a picked electric bass.
        partials = [(1, 1.0), (2, 0.45), (3, 0.20), (4, 0.08)]
        env = _adsr(n, sr, 0.006, 0.10, 0.55, 0.08)
    elif timbre == "keys":
        partials = [(1, 1.0), (2, 0.5), (3, 0.32), (4, 0.18), (5, 0.10), (6, 0.06)]
        env = _adsr(n, sr, 0.004, 0.15, 0.42, 0.10)
    else:  # lead
        partials = [(1, 1.0), (2, 0.6), (3, 0.4), (4, 0.25), (5, 0.15)]
        env = _adsr(n, sr, 0.010, 0.06, 0.70, 0.06)

    y = np.zeros(n)
    for h, a in partials:
        if f0 * h < sr / 2 * 0.95:
            y += a * np.sin(2 * np.pi * f0 * h * t)
    y /= max(sum(a for _, a in partials), 1e-9)
    return (y * env * amp).astype(np.float32)


def synth_drum(kind: str, sr: int) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(kind)) % 2**31)
    if kind == "kick":
        n = int(0.28 * sr)
        t = np.arange(n) / sr
        # Pitch sweep 110 -> 45 Hz, the shape of a real kick.
        f = 45 + 65 * np.exp(-t * 28)
        y = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t * 11)
        y += rng.standard_normal(n) * np.exp(-t * 220) * 0.10
        return (y * 0.95).astype(np.float32)
    if kind == "snare":
        n = int(0.20 * sr)
        t = np.arange(n) / sr
        body = (np.sin(2 * np.pi * 190 * t) + 0.6 * np.sin(2 * np.pi * 330 * t))
        noise = rng.standard_normal(n)
        import scipy.signal as sps
        sos = sps.butter(3, 1600, btype="high", fs=sr, output="sos")
        noise = sps.sosfilt(sos, noise)
        y = 0.35 * body * np.exp(-t * 26) + 0.85 * noise * np.exp(-t * 19)
        return (y * 0.7).astype(np.float32)
    # hat
    n = int(0.06 * sr)
    t = np.arange(n) / sr
    import scipy.signal as sps
    noise = rng.standard_normal(n)
    sos = sps.butter(4, 7000, btype="high", fs=sr, output="sos")
    y = sps.sosfilt(sos, noise) * np.exp(-t * 90)
    return (y * 0.42).astype(np.float32)


def render(case: Case, sr: int = SR) -> np.ndarray:
    end = 1.0
    for group in case.pitched().values():
        end = max(end, max(n.end for n in group))
    if case.drums:
        end = max(end, max(t for t, _ in case.drums) + 0.5)
    total = int((end + 1.0) * sr)
    out = np.zeros(total, np.float32)

    def place(sig, t, gain=1.0):
        i = int(t * sr)
        j = min(total, i + len(sig))
        if j > i:
            out[i:j] += sig[: j - i] * gain

    gains = {"bass": 0.85, "keys": 0.5, "lead": 0.42}
    for timbre, group in case.pitched().items():
        for note in group:
            place(synth_note(note, sr, timbre), note.start, gains[timbre])
    for t, kind in case.drums:
        place(synth_drum(kind, sr), t, 0.8)

    peak = np.abs(out).max()
    if peak > 0:
        out *= 0.89 / peak
    return out


def write_mp3(path: str, y: np.ndarray, sr: int = SR, bitrate: int = 192) -> None:
    """Encode to a genuine MP3, so the lossy path is exercised for real."""
    import lameenc

    enc = lameenc.Encoder()
    enc.set_bit_rate(bitrate)
    enc.set_in_sample_rate(sr)
    enc.set_channels(2)
    enc.set_quality(2)
    stereo = np.stack([y, y], axis=1)
    pcm = np.clip(stereo * 32767.0, -32768, 32767).astype(np.int16)
    data = enc.encode(pcm.tobytes()) + enc.flush()
    with open(path, "wb") as f:
        f.write(bytes(data))


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

@dataclass
class PredNote:
    start: float
    end: float
    pitch: int
    velocity: int


def read_midi_notes(path: str) -> Tuple[List[PredNote], List[PredNote]]:
    """Read a MIDI file back into absolute-time notes.

    Uses the file's own tempo map, exactly as a DAW or player would, so what is
    scored is what a user would actually hear.
    """
    import mido

    mid = mido.MidiFile(path)
    tpb = mid.ticks_per_beat

    tempo_changes: List[Tuple[int, int]] = []
    tick = 0
    for m in mid.tracks[0]:
        tick += m.time
        if m.type == "set_tempo":
            tempo_changes.append((tick, m.tempo))
    if not tempo_changes or tempo_changes[0][0] != 0:
        tempo_changes.insert(0, (0, 500000))

    def to_sec(target: int) -> float:
        sec, prev, tempo = 0.0, 0, tempo_changes[0][1]
        for ctick, ctempo in tempo_changes:
            if ctick >= target:
                break
            sec += mido.tick2second(ctick - prev, tpb, tempo)
            prev, tempo = ctick, ctempo
        return sec + mido.tick2second(target - prev, tpb, tempo)

    pitched: List[PredNote] = []
    drums: List[PredNote] = []
    for tr in mid.tracks[1:]:
        tick = 0
        open_notes: Dict[Tuple[int, int], Tuple[int, int]] = {}
        collected: List[PredNote] = []
        is_drum = False
        for m in tr:
            tick += m.time
            if getattr(m, "channel", None) == 9:
                is_drum = True
            if m.type == "note_on" and m.velocity > 0:
                open_notes[(m.channel, m.note)] = (tick, m.velocity)
            elif m.type == "note_off" or (m.type == "note_on" and m.velocity == 0):
                key = (m.channel, m.note)
                if key in open_notes:
                    on_tick, vel = open_notes.pop(key)
                    collected.append(PredNote(to_sec(on_tick), to_sec(tick),
                                              m.note, vel))
        for (ch, note), (on_tick, vel) in open_notes.items():   # stuck notes
            collected.append(PredNote(to_sec(on_tick), to_sec(tick), note, vel))
        (drums if is_drum else pitched).extend(collected)

    pitched.sort(key=lambda n: n.start)
    drums.sort(key=lambda n: n.start)
    return pitched, drums


def to_arrays(notes: Sequence) -> Tuple[np.ndarray, np.ndarray]:
    if not notes:
        return np.zeros((0, 2)), np.zeros(0)
    intervals = np.array([[n.start, max(n.end, n.start + 1e-3)] for n in notes])
    pitches = np.array([440.0 * 2 ** ((n.pitch - 69) / 12.0) for n in notes])
    return intervals, pitches


def score_pitched(truth: Sequence, pred: Sequence, onset_tol: float = 0.05) -> Dict:
    import mir_eval

    ti, tp = to_arrays(truth)
    pi, pp = to_arrays(pred)
    if len(ti) == 0:
        return {}
    if len(pi) == 0:
        return dict(p=0.0, r=0.0, f=0.0, p_off=0.0, r_off=0.0, f_off=0.0,
                    onset_err=float("nan"), octave=0, n_true=len(ti), n_pred=0)

    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ti, tp, pi, pp, onset_tolerance=onset_tol, offset_ratio=None)
    p2, r2, f2, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ti, tp, pi, pp, onset_tolerance=onset_tol, offset_ratio=0.2)

    matches = mir_eval.transcription.match_notes(
        ti, tp, pi, pp, onset_tolerance=onset_tol, offset_ratio=None)
    errs = [abs(pi[j][0] - ti[i][0]) for i, j in matches]

    # Octave errors: an unmatched prediction whose pitch class is right but
    # whose octave is not, aligned in time with a true note.
    matched_pred = {j for _, j in matches}
    octave = 0
    t_mid = ti.mean(axis=1)
    t_pitch = np.array([n.pitch for n in truth])
    for j, n in enumerate(pred):
        if j in matched_pred:
            continue
        mid = (pi[j][0] + pi[j][1]) / 2
        near = np.where(np.abs(t_mid - mid) < 0.15)[0]
        for i in near:
            d = n.pitch - t_pitch[i]
            if d != 0 and d % 12 == 0:
                octave += 1
                break

    return dict(p=p, r=r, f=f, p_off=p2, r_off=r2, f_off=f2,
                onset_err=float(np.median(errs)) if errs else float("nan"),
                octave=octave, n_true=len(ti), n_pred=len(pi))


DRUM_MAP = {"kick": {36, 35}, "snare": {38, 40}, "hat": {42, 44, 46}}


def score_drums(truth: Sequence[Tuple[float, str]], pred: Sequence,
                tol: float = 0.05) -> Dict[str, Dict]:
    out = {}
    for kind, keys in DRUM_MAP.items():
        t_times = np.array(sorted(t for t, k in truth if k == kind))
        p_times = np.array(sorted(n.start for n in pred if n.pitch in keys))
        if len(t_times) == 0:
            continue
        if len(p_times) == 0:
            out[kind] = dict(p=0.0, r=0.0, f=0.0, n_true=len(t_times), n_pred=0,
                             onset_err=float("nan"))
            continue
        # Greedy one-to-one matching within the tolerance.
        used = set()
        errs = []
        for t in t_times:
            best, bestd = None, tol
            for j, pt in enumerate(p_times):
                if j in used:
                    continue
                d = abs(pt - t)
                if d <= bestd:
                    best, bestd = j, d
            if best is not None:
                used.add(best)
                errs.append(bestd)
        tp = len(used)
        prec = tp / len(p_times)
        rec = tp / len(t_times)
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[kind] = dict(p=prec, r=rec, f=f1, n_true=len(t_times),
                         n_pred=len(p_times),
                         onset_err=float(np.median(errs)) if errs else float("nan"))
    return out


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------

def run_case(case: Case, workdir: str, separate: bool = True,
             quality: str = "good") -> Dict:
    from midgenius.config import Config
    from midgenius.pipeline import convert

    mp3 = os.path.join(workdir, case.name + ".mp3")
    write_mp3(mp3, render(case))

    cfg = Config()
    cfg.separate = separate
    cfg.verbose = False
    if not separate:
        cfg.mixdown_stem.max_polyphony = 0

    out = os.path.join(workdir, case.name + ".mid")
    result = convert(mp3, out, cfg)

    # Score the file that was actually written, not the in-memory notes.
    # Everything between the two - the tempo map, the beat grid, tick
    # rounding - can shift timing, and scoring the objects would hide it.
    pitched_pred, drum_pred = read_midi_notes(out)

    report: Dict = {"case": case.name, "result": result, "pitched": {}, "drums": {}}

    # Score all pitched truth against all pitched predictions together: which
    # stem a note landed in is a routing question, not a transcription error.
    all_truth = [n for g in case.pitched().values() for n in g]
    if all_truth:
        report["pitched"]["all"] = score_pitched(all_truth, pitched_pred)
        for name, group in case.pitched().items():
            report["pitched"][name] = score_pitched(group, pitched_pred)
    if case.drums:
        report["drums"] = score_drums(case.drums, drum_pred)
    return report


def print_report(rep: Dict) -> None:
    print()
    print("=" * 74)
    print("CASE: %s" % rep["case"])
    print("=" * 74)
    res = rep["result"]
    print("  tempo %.1f BPM   key %s   %d notes   separation %s"
          % (res.tempo_map.bpm if res.tempo_map else 0, res.key, res.n_notes,
             res.backend))
    for t in res.tracks:
        print("     %-8s %4d notes" % (t.name, len(t.notes)))

    if rep["pitched"]:
        print()
        print("  PITCHED                 P      R      F1  |  F1+off  onset  oct  true/pred")
        for name, m in rep["pitched"].items():
            if not m:
                continue
            print("    %-14s %6.1f%% %6.1f%% %6.1f%%  |  %5.1f%%  %5s  %3d  %d/%d"
                  % (name, 100 * m["p"], 100 * m["r"], 100 * m["f"],
                     100 * m["f_off"],
                     "%.0fms" % (1000 * m["onset_err"]) if m["onset_err"] == m["onset_err"] else "-",
                     m["octave"], m["n_true"], m["n_pred"]))
    if rep["drums"]:
        print()
        print("  DRUMS                   P      R      F1  |  onset  true/pred")
        for name, m in rep["drums"].items():
            print("    %-14s %6.1f%% %6.1f%% %6.1f%%  |  %5s  %d/%d"
                  % (name, 100 * m["p"], 100 * m["r"], 100 * m["f"],
                     "%.0fms" % (1000 * m["onset_err"]) if m["onset_err"] == m["onset_err"] else "-",
                     m["n_true"], m["n_pred"]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", help="run only this case")
    ap.add_argument("--keep", help="directory to keep rendered audio and midi in")
    ap.add_argument("--no-separate", action="store_true")
    ap.add_argument("--quality", default="good")
    args = ap.parse_args()

    cases = [CASES[args.case]] if args.case else list(CASES.values())

    import logging
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    ctx = (tempfile.TemporaryDirectory() if not args.keep else None)
    workdir = args.keep or ctx.name
    os.makedirs(workdir, exist_ok=True)
    try:
        reports = []
        for case in cases:
            rep = run_case(case, workdir, separate=not args.no_separate,
                           quality=args.quality)
            print_report(rep)
            reports.append(rep)

        print()
        print("=" * 74)
        print("SUMMARY")
        print("=" * 74)
        for rep in reports:
            bits = []
            if rep["pitched"].get("all"):
                bits.append("pitched F1 %.1f%%" % (100 * rep["pitched"]["all"]["f"]))
            if rep["drums"]:
                mean_f = np.mean([m["f"] for m in rep["drums"].values()])
                bits.append("drums F1 %.1f%%" % (100 * mean_f))
            print("  %-8s %s" % (rep["case"], "   ".join(bits)))
    finally:
        if ctx:
            ctx.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
