"""Tune transcription parameters against a real reference transcription.

Thresholds tuned on synthesised test tones do not transfer: real recordings
have reverb tails, overlapping partials, sidechain ducking and instruments that
share a register. This sweeps the decoder against a human-made reference for
the actual track.

    python tools/sweep.py --stems out --reference reference.mid \\
        --audio music.mp3 --offset 0.07

Separation is the expensive stage and its output does not depend on any of the
parameters being swept, so it is done once (or reused from cached stems), and
the Basic Pitch posteriorgram is computed once too. Only note decoding is
repeated, which makes a full sweep seconds rather than hours.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from midgenius import basicpitch, drums as D, mono  # noqa: E402
from midgenius import notes as N  # noqa: E402
from midgenius.audio import load  # noqa: E402
from evaluate import (DRUM_CLASSES, dedupe_unisons, read_reference,  # noqa: E402
                      score_drums, score_pitched)


def in_window(notes: Sequence, t0: float, t1: float, shift: float) -> List:
    out = []
    for n in notes:
        if t0 <= n.start < t1:
            m = type(n)(**{**n.__dict__})
            m.start = n.start - shift
            m.end = n.end - shift
            out.append(m)
    return out


def sweep_poly(post, ref_pitched, bass_notes, t0, t1, offset) -> None:
    print("=" * 78)
    print("POLYPHONIC DECODER  (stem: other)")
    print("=" * 78)
    print("  onset frame melodia conf poly |     P      R     F1   oct  produced")

    grid = itertools.product(
        (0.3, 0.4, 0.5, 0.6),        # onset threshold
        (0.20, 0.30, 0.40),          # frame threshold
        (False, True),               # melodia trick
        (0.0, 0.12),                 # min confidence
    )
    rows = []
    for onset, frame, melodia, conf in grid:
        ns = N.decode_polyphonic(
            post, onset_threshold=onset, frame_threshold=frame,
            min_note_ms=58, min_midi=28, max_midi=104,
            melodia_trick=melodia)
        ns = N.drop_low_confidence(ns, conf)
        ns = N.suppress_harmonic_ghosts(ns)
        ns = N.merge_repeats(ns)
        ns = N.remove_duplicates(ns)
        ns = N.enforce_min_duration(ns, 58)
        ns = N.trim_overlaps(ns)

        combined = in_window(ns, t0, t1, offset) + bass_notes
        m = score_pitched(ref_pitched, combined)
        rows.append((m["f"], onset, frame, melodia, conf, m))
        print("  %5.2f %5.2f %7s %4.2f  n/a | %5.1f%% %5.1f%% %5.1f%%  %4d  %d"
              % (onset, frame, melodia, conf,
                 100 * m["p"], 100 * m["r"], 100 * m["f"], m["octave"],
                 m["n_pred"]))

    rows.sort(key=lambda r: -r[0])
    best = rows[0]
    print()
    print("  BEST F1 %.1f%%  at onset=%.2f frame=%.2f melodia=%s conf=%.2f"
          % (100 * best[0], best[1], best[2], best[3], best[4]))


def sweep_mono(y, sr, ref_pitched, other_notes, t0, t1, offset) -> None:
    print()
    print("=" * 78)
    print("MONOPHONIC DECODER  (stem: bass)")
    print("=" * 78)
    print("  vprob  minms  fmax |     P      R     F1   produced")

    rows = []
    for vprob, min_ms, fmax in itertools.product(
            (0.35, 0.50, 0.65), (50.0, 70.0, 100.0), (440.0,)):
        ns = mono.transcribe_mono(
            y, sr, fmin=30.0, fmax=fmax, min_voiced_prob=vprob,
            min_note_ms=min_ms, min_midi=24, max_midi=67)
        ns = mono.fill_short_gaps(ns)
        ns = mono.fix_octave_jumps(ns)
        ns = N.enforce_min_duration(ns, min_ms)
        ns = N.trim_overlaps(ns)

        combined = other_notes + in_window(ns, t0, t1, offset)
        m = score_pitched(ref_pitched, combined)
        rows.append((m["f"], vprob, min_ms, m))
        print("  %5.2f %6.0f %5.0f | %5.1f%% %5.1f%% %5.1f%%   %d"
              % (vprob, min_ms, fmax,
                 100 * m["p"], 100 * m["r"], 100 * m["f"], m["n_pred"]))

    rows.sort(key=lambda r: -r[0])
    print()
    print("  BEST F1 %.1f%%  at voiced_prob=%.2f min_note_ms=%.0f"
          % (100 * rows[0][0], rows[0][1], rows[0][2]))


def sweep_drums(y, sr, ref_drums, t0, t1, offset, cutoff) -> None:
    print()
    print("=" * 78)
    print("PERCUSSION")
    print("=" * 78)
    print("  kick_sens snare_sens hat_sens |  kick F1  snare F1  hat F1  mean")

    for ks, ss, hs in itertools.product((0.9, 1.3, 1.8), (1.0, 1.4), (1.05,)):
        bands = []
        for b in D.DEFAULT_BANDS:
            s = {"kick": ks, "snare": ss, "hat": hs}.get(b.name, b.sensitivity)
            bands.append(D.DrumBand(**{**b.__dict__, "sensitivity": s}))
        ns = D.collapse_flams(
            D.transcribe_drums(y, sr, bands=tuple(bands), codec_cutoff=cutoff))
        got = in_window(ns, t0, t1, offset)
        sc = score_drums(ref_drums, got)
        k = sc.get("kick", {}).get("f", 0.0)
        s_ = sc.get("snare/clap", {}).get("f", 0.0)
        h = sc.get("hat", {}).get("f", 0.0)
        mean = np.mean([v["f"] for v in sc.values() if v["n_true"]])
        print("  %9.2f %10.2f %8.2f |  %6.1f%%  %7.1f%%  %5.1f%%  %5.1f%%"
              % (ks, ss, hs, 100 * k, 100 * s_, 100 * h, 100 * mean))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stems", required=True,
                    help="directory of separated stems (*_other.wav or other.wav)")
    ap.add_argument("--reference", required=True)
    ap.add_argument("--offset", type=float, required=True,
                    help="where the reference starts in the recording, seconds")
    ap.add_argument("--cutoff", type=float, default=16645.0)
    ap.add_argument("--only", help="poly | mono | drums")
    args = ap.parse_args()

    def stem(name: str) -> str:
        for cand in (os.path.join(args.stems, name + ".wav"),
                     os.path.join(args.stems, "music_" + name + ".wav")):
            if os.path.exists(cand):
                return cand
        raise SystemExit("no %s stem in %s" % (name, args.stems))

    ref = dedupe_unisons(read_reference(args.reference))
    ref_pitched = [n for n in ref if not n.is_drum]
    ref_drums = [n for n in ref if n.is_drum]
    ref_end = max(n.end for n in ref)
    t0, t1 = args.offset, args.offset + ref_end
    print("reference: %d pitched, %d drums over %.1f s   window %.2f-%.2f s"
          % (len(ref_pitched), len(ref_drums), ref_end, t0, t1))

    which = args.only

    # Baseline parts, so each sweep is scored in the context of the whole mix.
    bass_a = load(stem("bass"))
    bass_notes_full = mono.fill_short_gaps(mono.transcribe_mono(
        bass_a.mono(), bass_a.sr, fmin=30.0, fmax=440.0, min_voiced_prob=0.50,
        min_note_ms=70.0, min_midi=24, max_midi=67))
    bass_notes = in_window(bass_notes_full, t0, t1, args.offset)

    other_a = load(stem("other"))
    post = basicpitch.predict(other_a.mono(), other_a.sr)

    other_base = N.trim_overlaps(N.enforce_min_duration(N.remove_duplicates(
        N.merge_repeats(N.suppress_harmonic_ghosts(N.decode_polyphonic(
            post, onset_threshold=0.6, frame_threshold=0.45, min_note_ms=58,
            min_midi=28, max_midi=104, melodia_trick=False)))), 58))
    other_notes = in_window(other_base, t0, t1, args.offset)

    if which in (None, "poly"):
        sweep_poly(post, ref_pitched, bass_notes, t0, t1, args.offset)
    if which in (None, "mono"):
        sweep_mono(bass_a.mono(), bass_a.sr, ref_pitched, other_notes,
                   t0, t1, args.offset)
    if which in (None, "drums"):
        drum_a = load(stem("drums"))
        sweep_drums(drum_a.mono(), drum_a.sr, ref_drums, t0, t1, args.offset,
                    args.cutoff)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
