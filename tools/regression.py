"""Run the full pipeline over every reference pair and print one table.

Tuning against a single track is how the decoder ended up transcribing a sixth
of the notes of a soft one: a threshold that suited a punchy chiptune was
carried over as if it were universal. This runs every pair we have, so a change
that helps one kind of material and wrecks another is visible immediately
rather than after someone listens.

    python tools/regression.py                     # every pair
    python tools/regression.py --pair aria         # just one
    python tools/regression.py --out artifacts/    # keep the MIDI

Each pair is (audio, reference MIDI, offset). An offset of 0 means the audio
was rendered from the reference, so alignment is exact and every discrepancy is
ours; ``None`` means the reference is a separate arrangement and alignment has
to be searched for.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import warnings
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

# name -> (audio, reference, offset or None to search)
PAIRS: Dict[str, Tuple[str, str, Optional[float]]] = {
    # Rendered directly from its own MIDI: alignment exact, no arrangement
    # differences, so every error is the transcriber's.
    "aria": ("aria.mp3", "aria.mid", 0.0),
    # A separate human arrangement of a real recording: harder and less
    # forgiving, and the reference does not always agree with the audio.
    "graze": ("music.mp3", "reference.mid", None),
}


def run_pair(name: str, audio: str, reference: str, offset: Optional[float],
             out_dir: str, extra_args: Optional[List[str]] = None) -> Dict:
    from midgenius.cli import config_from_args, build_parser
    from midgenius.pipeline import convert
    from evaluate import (dedupe_unisons, find_alignments, read_reference,
                          refine_offset, score_drums, score_pitched,
                          synth_pitched, window)
    from benchmark import read_midi_notes
    from midgenius.audio import load

    midi_path = os.path.join(out_dir, name + ".mid")
    args = build_parser().parse_args([audio, "-o", midi_path] + (extra_args or []))
    cfg = config_from_args(args)
    cfg.verbose = False
    result = convert(audio, midi_path, cfg)

    ref = dedupe_unisons(read_reference(reference))
    ref_pitched = [n for n in ref if not n.is_drum]
    ref_drums = [n for n in ref if n.is_drum]
    ref_end = max(n.end for n in ref)

    if offset is None:
        src = load(audio, sr=22050, mono=True)
        ref_audio = synth_pitched(ref, 22050)
        found = find_alignments(ref_audio, src.mono(), 22050, max_passes=1)
        offset = refine_offset(ref_audio, src.mono(), 22050, found[0][0])

    pred_pitched, pred_drums = read_midi_notes(midi_path)
    pp = window(pred_pitched, offset, offset + ref_end, offset)
    pd = window(pred_drums, offset, offset + ref_end, offset)

    m = score_pitched(ref_pitched, pp)
    d = score_drums(ref_drums, pd)
    drum_f1 = (float(np.mean([v["f"] for v in d.values() if v["n_true"]]))
               if d else float("nan"))
    return dict(name=name, offset=offset, pitched=m, drums=d, drum_f1=drum_f1,
                notes=result.n_notes, midi=midi_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", help="run only this pair")
    ap.add_argument("--out", help="directory to keep the produced MIDI in")
    ap.add_argument("--arg", action="append", default=[],
                    help="extra CLI argument to pass through (repeatable)")
    args = ap.parse_args()

    import logging
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    names = [args.pair] if args.pair else list(PAIRS)
    ctx = None if args.out else tempfile.TemporaryDirectory()
    out_dir = args.out or ctx.name
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    try:
        for name in names:
            audio, reference, offset = PAIRS[name]
            if not (os.path.exists(audio) and os.path.exists(reference)):
                print("skipping %s: missing %s or %s" % (name, audio, reference))
                continue
            rows.append(run_pair(name, audio, reference, offset, out_dir, args.arg))
    finally:
        if ctx:
            ctx.cleanup()

    print()
    print("=" * 78)
    print("REGRESSION SUMMARY")
    print("=" * 78)
    print("  %-8s %-8s | %7s %7s %7s | %8s | %6s" %
          ("pair", "offset", "P", "R", "F1", "drums F1", "notes"))
    print("  " + "-" * 74)
    for r in rows:
        m = r["pitched"]
        print("  %-8s %7.2fs | %6.1f%% %6.1f%% %6.1f%% | %7.1f%% | %6d"
              % (r["name"], r["offset"], 100 * m["p"], 100 * m["r"], 100 * m["f"],
                 100 * r["drum_f1"], r["notes"]))
    if rows:
        print("  " + "-" * 74)
        print("  %-17s | %6s %6s %6.1f%% | %7.1f%% |"
              % ("mean", "", "",
                 100 * float(np.mean([r["pitched"]["f"] for r in rows])),
                 100 * float(np.nanmean([r["drum_f1"] for r in rows]))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
