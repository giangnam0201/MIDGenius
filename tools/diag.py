"""Error anatomy for one produced MIDI vs a reference, at a known offset.

Where mir_eval gives one F1 number, this says *what kind* of errors make it up:
exact matches, octave errors (and their direction), other pitch-class confusions,
plain misses, and plain phantoms. Run at offset 0 on the 1:1 aria pair so every
discrepancy is the transcriber's.

    python tools/diag.py aria.mid out/aria.mid --offset 0
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from collections import Counter

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import mir_eval  # noqa: E402

from evaluate import read_reference, dedupe_unisons, _arrays, window  # noqa: E402
from benchmark import read_midi_notes  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("midi")
    ap.add_argument("--offset", type=float, default=0.0)
    ap.add_argument("--tol", type=float, default=0.05)
    args = ap.parse_args()

    ref = dedupe_unisons(read_reference(args.reference))
    ref_p = [n for n in ref if not n.is_drum]
    ref_end = max(n.end for n in ref)

    pred_p, _ = read_midi_notes(args.midi)
    pred_p = window(pred_p, args.offset, args.offset + ref_end, args.offset)

    ti, tf = _arrays(ref_p)
    pi, pf = _arrays(pred_p)
    matches = mir_eval.transcription.match_notes(
        ti, tf, pi, pf, onset_tolerance=args.tol, offset_ratio=None)
    matched_t = {i for i, _ in matches}
    matched_p = {j for _, j in matches}

    t_pitch = np.array([n.pitch for n in ref_p])
    t_start = np.array([n.start for n in ref_p])
    p_pitch = np.array([n.pitch for n in pred_p])
    p_start = np.array([n.start for n in pred_p])

    n_t, n_p = len(ref_p), len(pred_p)
    n_match = len(matches)
    p = n_match / n_p if n_p else 0.0
    r = n_match / n_t if n_t else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0

    print("=" * 66)
    print("ERROR ANATOMY  ref=%s  pred=%s  offset=%.2f"
          % (os.path.basename(args.reference), os.path.basename(args.midi), args.offset))
    print("=" * 66)
    print("  reference %d   produced %d   matched %d" % (n_t, n_p, n_match))
    print("  P %5.1f%%  R %5.1f%%  F1 %5.1f%%" % (100*p, 100*r, 100*f))

    # --- misses: for each unmatched ref note, is there a pred note nearby at
    #     an octave / other interval? ---
    miss_octave_hi = miss_octave_lo = miss_pc = miss_blank = 0
    for i in range(n_t):
        if i in matched_t:
            continue
        near = np.where(np.abs(p_start - t_start[i]) <= args.tol)[0]
        if len(near) == 0:
            miss_blank += 1
            continue
        diffs = p_pitch[near] - t_pitch[i]
        if np.any(diffs == 12):
            miss_octave_hi += 1
        elif np.any(diffs == -12):
            miss_octave_lo += 1
        elif np.any(diffs % 12 == 0):
            miss_octave_hi += 1  # ±24 etc
        elif np.any(np.abs(diffs) <= 12):
            miss_pc += 1
        else:
            miss_blank += 1

    print("\n  MISSES (%d ref notes unmatched):" % (n_t - n_match))
    print("    covered by a note +12 (pred an octave HIGH)  %d" % miss_octave_hi)
    print("    covered by a note -12 (pred an octave LOW)    %d" % miss_octave_lo)
    print("    covered by other wrong pitch nearby           %d" % miss_pc)
    print("    nothing produced near that time               %d" % miss_blank)

    # --- phantoms: unmatched pred notes ---
    ph_octave = ph_pc = ph_blank = 0
    dir_counter = Counter()
    for j in range(n_p):
        if j in matched_p:
            continue
        near = np.where(np.abs(t_start - p_start[j]) <= args.tol)[0]
        if len(near) == 0:
            ph_blank += 1
            continue
        diffs = p_pitch[j] - t_pitch[near]
        octd = [d for d in diffs if d != 0 and d % 12 == 0]
        if octd:
            ph_octave += 1
            dir_counter[int(min(octd, key=abs))] += 1
        elif np.any(np.abs(diffs) <= 12):
            ph_pc += 1
        else:
            ph_blank += 1

    print("\n  PHANTOMS (%d produced notes unmatched):" % (n_p - n_match))
    print("    at an octave of a real note   %d   %s"
          % (ph_octave, dict(sorted(dir_counter.items()))))
    print("    other wrong pitch near a real note   %d" % ph_pc)
    print("    nothing real near that time          %d" % ph_blank)

    # --- ceiling if octaves were forgiven ---
    tf12 = tf.copy()
    # match on pitch class by folding both into one octave is not valid for
    # mir_eval hz; instead count how many misses have an octave partner.
    recoverable = miss_octave_hi + miss_octave_lo
    r2 = (n_match + recoverable) / n_t if n_t else 0.0
    print("\n  If every octave-off note were placed right:")
    print("    recall ceiling ~ %.1f%% (+%.1f pts)" % (100*r2, 100*(r2-r)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
