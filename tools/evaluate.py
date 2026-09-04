"""Score a produced MIDI against a human-made reference transcription.

Synthetic benchmarks are too kind: their notes have clean attacks, no
reverb, no sidechain ducking and no two instruments sharing a partial. This
scores against a real arrangement of the real track instead.

    python tools/evaluate.py music.mp3 out/music.mid --reference reference.mid

Alignment is solved rather than assumed. The reference is usually a different
length from the recording (a loop, a shorter arrangement, a pickup bar), so the
tool finds where it sits in the audio by chroma cross-correlation, refines the
offset on onset envelopes, and scores only the overlapping span. A looping track
matches in several places; each pass is scored separately.

Reported per pass:

  pitched P/R/F1   mir_eval note scoring: onset within 50 ms and pitch within
                   50 cents. "+off" additionally requires the offset to match.
  octave errors    right pitch class, wrong octave, counted separately because
                   it is the characteristic f0-tracker failure and is much less
                   damaging musically than a wrong pitch class.
  drums P/R/F1     by percussion *class* (kick / snare-clap / hat / tom /
                   cymbal), not raw GM key: a clap where the reference wrote a
                   snare is a naming difference, not a transcription error.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

SR = 22050
HOP = 2048

# Percussion classes. Arrangers pick different GM keys for the same musical
# role, so agreement is judged on the role.
DRUM_CLASSES: Dict[str, set] = {
    "kick": {35, 36},
    "snare/clap": {37, 38, 39, 40},
    "hat": {42, 44, 46},
    "tom": {41, 43, 45, 47, 48, 50},
    "cymbal": {49, 51, 52, 55, 57, 59},
}


@dataclass
class RefNote:
    start: float
    end: float
    pitch: int
    velocity: int
    is_drum: bool


def read_reference(path: str) -> List[RefNote]:
    """Read any MIDI file into absolute-time notes.

    ``clip=True`` because hand-made and exported files in the wild routinely
    contain out-of-range data bytes that strict parsing rejects outright.
    """
    import mido

    mid = mido.MidiFile(path, clip=True)
    notes: List[RefNote] = []
    open_n: Dict[Tuple[int, int], List[Tuple[float, int]]] = {}
    t = 0.0
    for msg in mid:                      # mido applies the tempo map for us
        t += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            open_n.setdefault((msg.channel, msg.note), []).append((t, msg.velocity))
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            key = (msg.channel, msg.note)
            if open_n.get(key):
                st, vel = open_n[key].pop(0)
                notes.append(RefNote(st, max(t, st + 1e-3), msg.note, vel,
                                     msg.channel == 9))
    notes.sort(key=lambda n: n.start)
    return notes


def synth_pitched(notes: Sequence[RefNote], sr: int = SR,
                  duration: Optional[float] = None) -> np.ndarray:
    """Cheap render of the pitched notes, for chroma alignment only."""
    pitched = [n for n in notes if not n.is_drum]
    if not pitched:
        return np.zeros(sr, np.float32)
    end = duration or (max(n.end for n in pitched) + 1.0)
    y = np.zeros(int(end * sr) + sr, np.float32)
    for n in pitched:
        dur = min(n.end - n.start, 2.0)
        ln = int(dur * sr)
        if ln < 16:
            continue
        t = np.arange(ln) / sr
        f0 = 440.0 * 2 ** ((n.pitch - 69) / 12.0)
        if f0 > sr / 2 * 0.9:
            continue
        sig = np.sin(2 * np.pi * f0 * t) + 0.4 * np.sin(4 * np.pi * f0 * t)
        env = np.exp(-1.5 * t)
        env[:64] *= np.linspace(0, 1, min(64, ln))[:min(64, ln)]
        i = int(n.start * sr)
        j = min(len(y), i + ln)
        if j > i:
            y[i:j] += (sig * env)[: j - i] * (n.velocity / 127.0)
    peak = np.abs(y).max()
    return (y / peak).astype(np.float32) if peak > 0 else y


def _chroma(y: np.ndarray, sr: int) -> np.ndarray:
    import librosa
    c = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP)
    return c / (np.linalg.norm(c, axis=0, keepdims=True) + 1e-9)


def find_alignments(ref_audio: np.ndarray, src_audio: np.ndarray, sr: int,
                    max_passes: int = 4, min_separation_s: float = 20.0
                    ) -> List[Tuple[float, float]]:
    """Where does the reference sit inside the recording?

    Returns (offset_seconds, score) for each distinct match, best first. A
    looping track legitimately matches in several places; each is a separate
    pass to score.
    """
    cr, ca = _chroma(ref_audio, sr), _chroma(src_audio, sr)
    n_ref, n_src = cr.shape[1], ca.shape[1]
    if n_ref >= n_src:
        return [(0.0, float((ca * cr[:, :n_src]).sum() / max(n_src, 1)))]

    scores = np.array([float((ca[:, lag:lag + n_ref] * cr).sum() / n_ref)
                       for lag in range(n_src - n_ref + 1)])
    out: List[Tuple[float, float]] = []
    taken: List[int] = []
    sep = int(min_separation_s * sr / HOP)
    for lag in np.argsort(scores)[::-1]:
        if any(abs(lag - t) < sep for t in taken):
            continue
        taken.append(int(lag))
        out.append((float(lag) * HOP / sr, float(scores[lag])))
        if len(out) >= max_passes:
            break
    return out


def refine_offset(ref_audio: np.ndarray, src_audio: np.ndarray, sr: int,
                  coarse: float, search_s: float = 2.0, hop: int = 512) -> float:
    """Refine a coarse offset using chroma at a finer hop.

    Refining on onset envelopes is tempting but wrong here: the reference
    render is pitched-only while the recording is a full mix whose onset
    envelope is dominated by drums, so the two envelopes correlate on the wrong
    features and the "refinement" can move the offset by over a second.
    Harmony is what the two signals genuinely share.
    """
    import librosa

    a0 = int(max(0, (coarse - search_s) * sr))
    a1 = int(min(len(src_audio),
                 (coarse + len(ref_audio) / sr + search_s) * sr))
    seg = src_audio[a0:a1]
    if len(seg) < hop * 8 or len(ref_audio) < hop * 8:
        return coarse

    cr = librosa.feature.chroma_cqt(y=ref_audio, sr=sr, hop_length=hop)
    ca = librosa.feature.chroma_cqt(y=seg, sr=sr, hop_length=hop)
    cr = cr / (np.linalg.norm(cr, axis=0, keepdims=True) + 1e-9)
    ca = ca / (np.linalg.norm(ca, axis=0, keepdims=True) + 1e-9)
    n_ref = cr.shape[1]
    if ca.shape[1] <= n_ref:
        return coarse
    scores = [float((ca[:, lag:lag + n_ref] * cr).sum())
              for lag in range(ca.shape[1] - n_ref + 1)]
    return a0 / sr + int(np.argmax(scores)) * hop / sr


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def _arrays(notes: Sequence) -> Tuple[np.ndarray, np.ndarray]:
    if not notes:
        return np.zeros((0, 2)), np.zeros(0)
    iv = np.array([[n.start, max(n.end, n.start + 1e-3)] for n in notes])
    hz = np.array([440.0 * 2 ** ((n.pitch - 69) / 12.0) for n in notes])
    return iv, hz


def score_pitched(truth: Sequence, pred: Sequence, tol: float = 0.05) -> Dict:
    import mir_eval

    ti, tp = _arrays(truth)
    pi, pp = _arrays(pred)
    if len(ti) == 0:
        return {}
    if len(pi) == 0:
        return dict(p=0.0, r=0.0, f=0.0, f_off=0.0, onset_err=float("nan"),
                    octave=0, n_true=len(ti), n_pred=0)

    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ti, tp, pi, pp, onset_tolerance=tol, offset_ratio=None)
    _, _, f_off, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ti, tp, pi, pp, onset_tolerance=tol, offset_ratio=0.2)
    matches = mir_eval.transcription.match_notes(
        ti, tp, pi, pp, onset_tolerance=tol, offset_ratio=None)
    errs = [pi[j][0] - ti[i][0] for i, j in matches]

    matched = {j for _, j in matches}
    t_mid = ti.mean(axis=1)
    t_pitch = np.array([n.pitch for n in truth])
    octave = 0
    for j, n in enumerate(pred):
        if j in matched:
            continue
        mid = (pi[j][0] + pi[j][1]) / 2
        near = np.where(np.abs(t_mid - mid) < 0.15)[0]
        if any((n.pitch - t_pitch[i]) != 0 and (n.pitch - t_pitch[i]) % 12 == 0
               for i in near):
            octave += 1

    return dict(p=p, r=r, f=f, f_off=f_off,
                onset_err=float(np.median(errs)) if errs else float("nan"),
                octave=octave, n_true=len(ti), n_pred=len(pi))


def score_drums(truth: Sequence, pred: Sequence, tol: float = 0.05) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for cls, keys in DRUM_CLASSES.items():
        tt = np.array(sorted(n.start for n in truth if n.pitch in keys))
        pt = np.array(sorted(n.start for n in pred if n.pitch in keys))
        if len(tt) == 0 and len(pt) == 0:
            continue
        if len(tt) == 0 or len(pt) == 0:
            out[cls] = dict(p=0.0, r=0.0, f=0.0, n_true=len(tt), n_pred=len(pt))
            continue
        used = set()
        for t in tt:
            cand = [(abs(t - x), j) for j, x in enumerate(pt)
                    if abs(t - x) <= tol and j not in used]
            if cand:
                used.add(min(cand)[1])
        tp = len(used)
        p = tp / len(pt)
        r = tp / len(tt)
        out[cls] = dict(p=p, r=r, f=2 * p * r / (p + r) if p + r else 0.0,
                        n_true=len(tt), n_pred=len(pt))
    return out


def window(notes: Sequence, t0: float, t1: float, shift: float = 0.0) -> List:
    """Notes starting inside [t0, t1), shifted onto the reference timeline."""
    out = []
    for n in notes:
        if t0 <= n.start < t1:
            m = type(n)(**{**n.__dict__})
            m.start = n.start - shift
            m.end = n.end - shift
            out.append(m)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio")
    ap.add_argument("midi", help="the MIDI produced by MIDGenius")
    ap.add_argument("--reference", required=True, help="ground-truth MIDI")
    ap.add_argument("--offset", type=float,
                    help="skip alignment search and use this offset (seconds)")
    ap.add_argument("--passes", type=int, default=2,
                    help="how many matching passes to score (looping tracks)")
    args = ap.parse_args()

    from midgenius.audio import load
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from benchmark import read_midi_notes, PredNote

    ref = read_reference(args.reference)
    ref_pitched = [n for n in ref if not n.is_drum]
    ref_drums = [n for n in ref if n.is_drum]
    ref_end = max(n.end for n in ref)

    pred_pitched_raw, pred_drums_raw = read_midi_notes(args.midi)

    print("=" * 74)
    print("EVALUATION vs reference: %s" % os.path.basename(args.reference))
    print("=" * 74)
    print("  reference   %d notes (%d pitched, %d drums), %.1f s"
          % (len(ref), len(ref_pitched), len(ref_drums), ref_end))
    print("  produced    %d notes (%d pitched, %d drums)"
          % (len(pred_pitched_raw) + len(pred_drums_raw),
             len(pred_pitched_raw), len(pred_drums_raw)))

    src = load(args.audio, sr=SR, mono=True)
    if args.offset is not None:
        passes = [(args.offset, float("nan"))]
    else:
        ref_audio = synth_pitched(ref, SR)
        passes = find_alignments(ref_audio, src.mono(), SR,
                                 max_passes=args.passes)
        passes = [(refine_offset(ref_audio, src.mono(), SR, off), sc)
                  for off, sc in passes]
        print("\n  alignment   the reference matches the recording at:")
        for off, sc in passes:
            print("                %7.2f s   (chroma score %.3f)" % (off, sc))

    for idx, (offset, _) in enumerate(passes[: args.passes], 1):
        t0, t1 = offset, offset + ref_end
        pp = window(pred_pitched_raw, t0, t1, offset)
        pd = window(pred_drums_raw, t0, t1, offset)

        print()
        print("-" * 74)
        print("PASS %d   audio %.1f s - %.1f s   (offset %+.2f s)"
              % (idx, t0, t1, offset))
        print("-" * 74)

        m = score_pitched(ref_pitched, pp)
        if m:
            print("  PITCHED   P %5.1f%%  R %5.1f%%  F1 %5.1f%%   (+offsets F1 %5.1f%%)"
                  % (100 * m["p"], 100 * m["r"], 100 * m["f"], 100 * m["f_off"]))
            print("            reference %d notes vs produced %d   octave errors %d"
                  % (m["n_true"], m["n_pred"], m["octave"]))
            if m["onset_err"] == m["onset_err"]:
                print("            median onset error %+.0f ms" % (1000 * m["onset_err"]))

        d = score_drums(ref_drums, pd)
        if d:
            print("  DRUMS                P        R       F1     ref/produced")
            for cls, s in d.items():
                print("    %-12s %6.1f%%  %6.1f%%  %6.1f%%    %d/%d"
                      % (cls, 100 * s["p"], 100 * s["r"], 100 * s["f"],
                         s["n_true"], s["n_pred"]))
            mean_f = np.mean([s["f"] for s in d.values() if s["n_true"]])
            print("    %-12s %22.1f%%" % ("mean F1", 100 * mean_f))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
