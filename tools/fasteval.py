"""Fast A/B evaluation on the aria 1:1 pair using cached stems.

Separation is cached (MIDGENIUS_STEM_CACHE), so this re-runs only transcription
and scoring - a couple of minutes instead of half an hour. Config overrides are
given as key=value and applied to the top-level Config and, for keys a StemConfig
also has, to every poly stem plus the mixdown stem.

    MIDGENIUS_STEM_CACHE=1 python tools/fasteval.py octave_correction=1
    MIDGENIUS_STEM_CACHE=1 python tools/fasteval.py octave_correction=1 octave_sub_ratio=0.7
"""

from __future__ import annotations

import os
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402


def _coerce(v: str):
    lo = v.lower()
    if lo in ("1", "true", "yes", "on"):
        return True
    if lo in ("0", "false", "no", "off"):
        return False
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def main() -> int:
    audio = os.environ.get("FASTEVAL_AUDIO", "aria.mp3")
    reference = os.environ.get("FASTEVAL_REF", "aria.mid")
    # Excerpt mode: the audio is a slice [start, start+dur] of the reference's
    # timeline (offset 0 material), so score against the reference windowed to
    # that span. Lets pYIN and basic-pitch run on 120 s instead of 8.5 min for
    # fast tuning, then the winner is confirmed on the full track.
    ex_start = float(os.environ.get("FASTEVAL_START", "0") or 0)
    ex_dur = float(os.environ.get("FASTEVAL_DUR", "0") or 0)
    overrides = {}
    for a in sys.argv[1:]:
        if "=" in a:
            k, v = a.split("=", 1)
            overrides[k] = _coerce(v)

    from midgenius.config import Config, StemConfig
    from midgenius.pipeline import convert
    from evaluate import (read_reference, dedupe_unisons, score_pitched,
                          score_drums, window)
    from benchmark import read_midi_notes

    cfg = Config()
    stem_keys = set(StemConfig("x").__dict__.keys())
    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
        if k in stem_keys:
            for sc in cfg.stems.values():
                setattr(sc, k, v)
            setattr(cfg.mixdown_stem, k, v)
    cfg.verbose = False

    tmp = tempfile.mkdtemp()
    midi_path = os.path.join(tmp, "aria.mid")
    result = convert(audio, midi_path, cfg)

    ref = dedupe_unisons(read_reference(reference))
    if ex_dur > 0:
        # Keep reference notes that start inside the excerpt, shifted to 0.
        ref = [type(n)(n.start - ex_start, n.end - ex_start, n.pitch,
                       n.velocity, n.is_drum)
               for n in ref if ex_start <= n.start < ex_start + ex_dur]
    ref_p = [n for n in ref if not n.is_drum]
    ref_d = [n for n in ref if n.is_drum]
    ref_end = max((n.end for n in ref), default=ex_dur or 1.0)

    pp, pd = read_midi_notes(midi_path)
    pp = window(pp, 0.0, ref_end, 0.0)
    pd = window(pd, 0.0, ref_end, 0.0)

    m = score_pitched(ref_p, pp)
    d = score_drums(ref_d, pd)
    drum_f1 = (float(np.mean([v["f"] for v in d.values() if v["n_true"]]))
               if d else float("nan"))

    tag = " ".join("%s=%s" % kv for kv in overrides.items()) or "(baseline)"
    print("OVERRIDES: %s" % tag)
    print("  notes=%d  P=%.1f%%  R=%.1f%%  F1=%.1f%%  +off=%.1f%%  octave_fp=%d  drums=%.1f%%"
          % (result.n_notes, 100*m["p"], 100*m["r"], 100*m["f"], 100*m["f_off"],
             m["octave"], 100*drum_f1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
