"""Fast drum-band tuning: score percussion per class from the cached drum stem.

Drum transcription is cheap (STFT flux, no pYIN / Basic Pitch), so this reloads
the cached separation, runs only the drum transcriber with overridden band
parameters, and scores per class against the reference - a couple of seconds a
config instead of a full pipeline run.

    python tools/drumeval.py aria.mp3 aria.mid 0
    python tools/drumeval.py aria.mp3 aria.mid 0 kick_sens=1.4 kick_contrast=5
    python tools/drumeval.py music.mp3 reference.mid 76.09 kick_contrast=5
"""

from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402


def main() -> int:
    audio, reference, offset = sys.argv[1], sys.argv[2], float(sys.argv[3])
    ov = {}
    for a in sys.argv[4:]:
        if "=" in a:
            k, v = a.split("=", 1)
            ov[k] = float(v)

    from midgenius.audio import load
    from midgenius import separation, drums as D
    from evaluate import read_reference, dedupe_unisons, score_drums, window

    src = load(audio)
    stems = separation.separate(src)
    drum = stems.get("drums")
    if drum is None:
        print("no drum stem"); return 1

    # Build bands from the defaults, overriding the kick (and optionally snare).
    bands = []
    for b in D.DEFAULT_BANDS:
        nb = D.DrumBand(**{**b.__dict__})
        if b.name == "kick":
            if "kick_sens" in ov:
                nb.sensitivity = ov["kick_sens"]
            if "kick_contrast" in ov:
                nb.contrast_db = ov["kick_contrast"]
        if b.name == "snare":
            if "snare_sens" in ov:
                nb.sensitivity = ov["snare_sens"]
            if "snare_contrast" in ov:
                nb.contrast_db = ov["snare_contrast"]
        bands.append(nb)

    notes = D.transcribe_drums(drum.mono(), drum.sr, bands=tuple(bands))
    notes = D.collapse_flams(notes)

    ref = dedupe_unisons(read_reference(reference))
    ref_d = [n for n in ref if n.is_drum]
    ref_end = max(n.end for n in ref)
    pd = window(notes, offset, offset + ref_end, offset)

    d = score_drums(ref_d, pd)
    tag = " ".join("%s=%g" % kv for kv in ov.items()) or "(default bands)"
    print("== %s  offset %.2f  :: %s ==" % (os.path.basename(audio), offset, tag))
    for cls, s in d.items():
        print("   %-11s P%5.1f R%5.1f F%5.1f   ref/prod %d/%d"
              % (cls, 100*s["p"], 100*s["r"], 100*s["f"], s["n_true"], s["n_pred"]))
    mean_f = np.mean([s["f"] for s in d.values() if s["n_true"]])
    print("   mean F1 (classes with ref) = %.1f%%" % (100*mean_f))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
