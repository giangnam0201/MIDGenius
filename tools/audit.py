"""Audit a produced MIDI file against the audio it came from.

Note-level metrics need ground truth, which real music does not come with. This
tool instead asks the question a listener asks: *does the MIDI sound like the
track?* It renders the MIDI back to audio with a simple synth and compares the
render to the original.

    python tools/audit.py song.mp3 song.mid
    python tools/audit.py song.mp3 song.mid --render out/preview.wav

What it reports:

  global offset     cross-correlation of the two onset envelopes. A non-zero
                    value means every note is early or late by a fixed amount -
                    the single most audible defect, and invisible to note-level
                    metrics computed on in-memory objects.
  onset F1          do notes start where the audio has attacks
  chroma similarity per-frame cosine similarity of pitch-class profiles: is the
                    right harmony sounding at the right time
  coverage          does the MIDI span the whole track
  density           notes per second, and simultaneous voices
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

SR = 22050

# General MIDI percussion -> (centre frequency, noisiness, decay)
DRUM_VOICES = {
    35: (50, 0.1, 12), 36: (50, 0.1, 12), 37: (400, 0.7, 45),
    38: (200, 0.8, 22), 40: (220, 0.85, 24), 39: (250, 0.9, 20),
    41: (100, 0.2, 14), 43: (130, 0.2, 14), 45: (160, 0.2, 14),
    47: (200, 0.2, 14), 48: (250, 0.2, 14), 50: (300, 0.2, 14),
    42: (9000, 1.0, 90), 44: (8000, 1.0, 80), 46: (8000, 1.0, 14),
    49: (5000, 1.0, 4), 51: (6000, 0.95, 6), 52: (5000, 1.0, 4),
    53: (6000, 0.9, 7), 55: (5000, 1.0, 4), 57: (5000, 1.0, 4),
    59: (6000, 0.95, 6),
}


def render_midi(path: str, sr: int = SR, duration: Optional[float] = None
                ) -> np.ndarray:
    """Render a MIDI file to audio with a simple, honest synth."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from benchmark import read_midi_notes

    pitched, drums = read_midi_notes(path)
    end = duration or (max([n.end for n in pitched + drums], default=1.0) + 0.5)
    n_total = int(end * sr) + sr
    out = np.zeros(n_total, np.float32)
    rng = np.random.default_rng(0)

    def place(sig, t, gain=1.0):
        i = int(t * sr)
        j = min(n_total, i + len(sig))
        if j > i:
            out[i:j] += sig[: j - i] * gain

    for n in pitched:
        dur = max(0.05, min(n.end - n.start, 4.0))
        ln = int(dur * sr)
        if ln < 8:
            continue
        t = np.arange(ln) / sr
        f0 = 440.0 * 2 ** ((n.pitch - 69) / 12.0)
        if f0 > sr / 2 * 0.9:
            continue
        y = np.zeros(ln)
        for h, a in ((1, 1.0), (2, 0.45), (3, 0.22), (4, 0.10)):
            if f0 * h < sr / 2 * 0.95:
                y += a * np.sin(2 * np.pi * f0 * h * t)
        atk = max(2, int(0.005 * sr))
        rel = max(2, int(0.03 * sr))
        env = np.ones(ln)
        env[:atk] = np.linspace(0, 1, atk)
        env[-rel:] = np.linspace(1, 0, rel)
        env *= np.exp(-1.2 * t)
        place((y * env).astype(np.float32), n.start, 0.28 * (n.velocity / 127.0))

    for n in drums:
        freq, noisiness, decay = DRUM_VOICES.get(n.pitch, (300, 0.8, 25))
        ln = int(min(0.5, 4.0 / decay + 0.05) * sr)
        t = np.arange(ln) / sr
        tone = np.sin(2 * np.pi * freq * np.exp(-t * 6) * t) if freq < 1000 else 0.0
        noise = rng.standard_normal(ln)
        if freq >= 1000:
            noise = np.diff(noise, prepend=0.0)      # crude highpass
        y = (1 - noisiness) * np.asarray(tone) + noisiness * noise
        place((y * np.exp(-t * decay)).astype(np.float32), n.start,
              0.5 * (n.velocity / 127.0))

    peak = np.abs(out).max()
    if peak > 0:
        out *= 0.9 / peak
    return out


def onset_envelope(y: np.ndarray, sr: int, hop: int = 256) -> np.ndarray:
    import librosa
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    return env / (env.max() + 1e-12)


def global_offset(a: np.ndarray, b: np.ndarray, sr: int, hop: int = 256,
                  max_shift_s: float = 1.0) -> Tuple[float, float]:
    """Best constant time shift aligning b to a, and the correlation there.

    A single number that catches the whole class of "everything is late"
    bugs - the kind that note-level metrics on in-memory objects cannot see,
    because the shift is introduced when the file is written.
    """
    ea, eb = onset_envelope(a, sr, hop), onset_envelope(b, sr, hop)
    n = min(len(ea), len(eb))
    ea, eb = ea[:n] - ea[:n].mean(), eb[:n] - eb[:n].mean()
    max_lag = int(max_shift_s * sr / hop)
    corr = np.correlate(ea, eb, mode="full")
    mid = len(corr) // 2
    lo, hi = mid - max_lag, mid + max_lag + 1
    window = corr[lo:hi]
    k = int(np.argmax(window))
    lag = (lo + k) - mid
    denom = np.sqrt((ea ** 2).sum() * (eb ** 2).sum()) + 1e-12
    return -lag * hop / sr, float(window[k] / denom)


def onset_f1(a: np.ndarray, b: np.ndarray, sr: int, tol: float = 0.05) -> Dict:
    import librosa
    oa = librosa.onset.onset_detect(y=a, sr=sr, hop_length=256, units="time",
                                    backtrack=False)
    ob = librosa.onset.onset_detect(y=b, sr=sr, hop_length=256, units="time",
                                    backtrack=False)
    if len(oa) == 0 or len(ob) == 0:
        return dict(p=0.0, r=0.0, f=0.0, n_a=len(oa), n_b=len(ob))
    used = set()
    for t in oa:
        cand = [(abs(t - x), j) for j, x in enumerate(ob)
                if abs(t - x) <= tol and j not in used]
        if cand:
            used.add(min(cand)[1])
    tp = len(used)
    p = tp / len(ob)
    r = tp / len(oa)
    return dict(p=p, r=r, f=2 * p * r / (p + r) if p + r else 0.0,
                n_a=len(oa), n_b=len(ob))


def chroma_similarity(a: np.ndarray, b: np.ndarray, sr: int) -> Dict:
    """Per-frame cosine similarity of pitch-class profiles."""
    import librosa
    ca = librosa.feature.chroma_cqt(y=a, sr=sr, hop_length=1024)
    cb = librosa.feature.chroma_cqt(y=b, sr=sr, hop_length=1024)
    n = min(ca.shape[1], cb.shape[1])
    ca, cb = ca[:, :n], cb[:, :n]
    na = np.linalg.norm(ca, axis=0) + 1e-9
    nb = np.linalg.norm(cb, axis=0) + 1e-9
    sims = (ca * cb).sum(axis=0) / (na * nb)
    # Only judge frames where the original actually has pitched content.
    active = na > np.percentile(na, 25)
    return dict(mean=float(sims[active].mean()) if active.any() else 0.0,
                median=float(np.median(sims[active])) if active.any() else 0.0,
                frames=int(active.sum()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio")
    ap.add_argument("midi")
    ap.add_argument("--render", help="write the MIDI render to this .wav")
    ap.add_argument("--sr", type=int, default=SR)
    args = ap.parse_args()

    from midgenius.audio import load, save_wav, Audio
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from benchmark import read_midi_notes

    src = load(args.audio, sr=args.sr, mono=True)
    y_src = src.mono()
    y_mid = render_midi(args.midi, args.sr, duration=src.duration)
    n = min(len(y_src), len(y_mid))
    y_src, y_mid = y_src[:n], y_mid[:n]

    if args.render:
        save_wav(args.render, Audio(y_mid[None, :], args.sr))

    pitched, drums = read_midi_notes(args.midi)
    alln = pitched + drums

    print("=" * 70)
    print("AUDIT  %s  vs  %s" % (os.path.basename(args.midi),
                                 os.path.basename(args.audio)))
    print("=" * 70)

    print("\n  COVERAGE")
    print("    audio duration      %.1f s" % src.duration)
    if alln:
        first = min(x.start for x in alln)
        last = max(x.end for x in alln)
        print("    midi spans          %.1f s -> %.1f s  (%.0f%% of track)"
              % (first, last, 100 * (last - first) / max(src.duration, 1e-9)))
        print("    notes               %d pitched + %d drums = %d"
              % (len(pitched), len(drums), len(alln)))
        print("    density             %.1f notes/s" % (len(alln) / max(last, 1e-9)))
        if pitched:
            ev = []
            for x in pitched:
                ev += [(x.start, 1), (x.end, -1)]
            ev.sort()
            cur, mx, series = 0, 0, []
            for _, d in ev:
                cur += d
                mx = max(mx, cur)
                series.append(cur)
            print("    polyphony           median %d, max %d"
                  % (int(np.median(series)), mx))

    off, corr = global_offset(y_src, y_mid, args.sr)
    print("\n  TIMING")
    print("    global offset       %+.0f ms   (midi relative to audio)" % (1000 * off))
    print("    onset-envelope corr %.3f" % corr)
    verdict = ("good" if abs(off) < 0.015 else
               "AUDIBLE - notes are consistently " + ("late" if off > 0 else "early"))
    print("    verdict             %s" % verdict)

    f = onset_f1(y_src, y_mid, args.sr)
    print("\n  ONSETS (rhythmic agreement)")
    print("    precision %.1f%%  recall %.1f%%  F1 %.1f%%   (audio %d / midi %d)"
          % (100 * f["p"], 100 * f["r"], 100 * f["f"], f["n_a"], f["n_b"]))

    ch = chroma_similarity(y_src, y_mid, args.sr)
    print("\n  HARMONY (pitch-class agreement)")
    print("    chroma cosine       mean %.3f   median %.3f   over %d frames"
          % (ch["mean"], ch["median"], ch["frames"]))
    print("    scale               1.00 identical | ~0.85 good | ~0.6 loose | <0.4 wrong")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
