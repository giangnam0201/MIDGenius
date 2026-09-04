"""Command line interface for MIDGenius."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from typing import List, Optional

from midgenius import __version__
from midgenius.config import Config, default_stems

EPILOG = """
examples:
  midgenius song.mp3
  midgenius song.mp3 -o out.mid --quantize 1/16
  midgenius song.mp3 --only bass,drums --write-stems
  midgenius song.mp3 --no-separate            # fast, lower quality
  midgenius song.mp3 --quality best           # shift trick, slower

quality presets:
  fast    no shift trick, higher thresholds     ~0.5x realtime
  good    the default                           ~1.5x realtime
  best    4x shift trick, lower thresholds      ~5x realtime
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="midgenius",
        description="Convert audio (MP3, WAV, FLAC, ...) to a multi-track MIDI file.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", nargs="+", help="audio file(s) to convert")
    p.add_argument("-o", "--output", help="output .mid path (single input only)")
    p.add_argument("-d", "--outdir", help="directory for outputs")
    p.add_argument("--version", action="version", version="MIDGenius " + __version__)

    g = p.add_argument_group("quality")
    g.add_argument("--quality", choices=("fast", "good", "best"), default="good",
                   help="speed/accuracy preset (default: good)")
    g.add_argument("--no-separate", action="store_true",
                   help="skip stem separation and transcribe the whole mix")
    g.add_argument("--pitched-from-mix-only", action="store_true",
                   help="take pitched notes from the whole mix, using stems only "
                        "for drums (cleaner on synth/electronic material)")
    g.add_argument("--no-mix-primary", action="store_true",
                   help="disable mix-primary merging (use the older stem-union)")
    g.add_argument("--min-stem-confidence", type=float, default=None,
                   help="confidence floor for stem notes added under mix-primary")
    g.add_argument("--no-mix-pass", action="store_true",
                   help="do not also transcribe the untouched mix; separation "
                        "can strip attacks, so this usually loses notes")
    g.add_argument("--model", default="htdemucs",
                   help="Demucs model name (default: htdemucs)")
    g.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    g.add_argument("--shifts", type=int, default=None,
                   help="separation shift-trick passes; higher is cleaner and slower")
    g.add_argument("--no-lossy-repair", action="store_true",
                   help="do not pre-condition MP3/AAC artefacts")

    g = p.add_argument_group("transcription")
    g.add_argument("--only", help="comma separated stems to keep "
                                  "(drums,bass,vocals,other)")
    g.add_argument("--skip", help="comma separated stems to drop")
    g.add_argument("--onset-threshold", type=float,
                   help="polyphonic onset sensitivity, 0-1 (lower = more notes)")
    g.add_argument("--frame-threshold", type=float,
                   help="polyphonic sustain sensitivity, 0-1")
    g.add_argument("--min-note-ms", type=float,
                   help="discard notes shorter than this")
    g.add_argument("--max-polyphony", type=int,
                   help="cap simultaneous notes per pitched stem (0 = unlimited)")
    g.add_argument("--no-bends", action="store_true",
                   help="do not write pitch-bend expression")
    g.add_argument("--harmonic-ratio", type=float, default=None,
                   help="octave/harmonic ghost is cut if this much weaker than "
                        "its parent (higher = more aggressive; default 0.28)")
    g.add_argument("--no-ghost-filter", action="store_true",
                   help="keep harmonic/octave phantom notes")
    g.add_argument("--fixed-threshold", action="store_true",
                   help="use the configured thresholds instead of deriving "
                        "them from the model's confidence on this material")
    g.add_argument("--melodia", action="store_true",
                   help="recover notes with no detected onset; finds quiet "
                        "notes in sparse material, adds phantoms in dense mixes")

    g = p.add_argument_group("rhythm")
    g.add_argument("-q", "--quantize", default="off",
                   choices=("off", "1/4", "1/8", "1/16", "1/32", "1/4t", "1/8t", "1/16t"),
                   help="snap notes to a grid (default: off)")
    g.add_argument("--quantize-strength", type=float, default=0.75,
                   help="0-1 blend toward the grid (default: 0.75)")
    g.add_argument("--quantize-drums-only", action="store_true")
    g.add_argument("--tempo", type=float, help="force a fixed BPM")
    g.add_argument("--no-tempo-map", action="store_true",
                   help="write one average tempo instead of a variable map")
    g.add_argument("--time-signature", default="4/4")

    g = p.add_argument_group("output")
    g.add_argument("--write-stems", action="store_true",
                   help="also save the separated audio stems as .wav")
    g.add_argument("--per-stem-midi", action="store_true",
                   help="also save one .mid per stem")
    g.add_argument("--single-track", action="store_true",
                   help="merge pitched instruments into one track")
    g.add_argument("--ticks-per-beat", type=int, default=480)
    g.add_argument("-v", "--verbose", action="store_true")
    g.add_argument("--quiet", action="store_true")
    return p


def apply_quality(cfg: Config, quality: str) -> None:
    if quality == "fast":
        cfg.separation_shifts = 1
        cfg.separation_overlap = 0.10
        for s in cfg.stems.values():
            s.onset_threshold = min(0.95, s.onset_threshold + 0.08)
            s.frame_threshold = min(0.95, s.frame_threshold + 0.06)
    elif quality == "best":
        cfg.separation_shifts = 4
        cfg.separation_overlap = 0.35
        for s in cfg.stems.values():
            s.onset_threshold = max(0.05, s.onset_threshold - 0.06)
            s.frame_threshold = max(0.05, s.frame_threshold - 0.05)
            s.min_confidence = max(0.0, s.min_confidence - 0.03)


def config_from_args(args: argparse.Namespace) -> Config:
    cfg = Config()
    cfg.stems = default_stems()
    apply_quality(cfg, args.quality)

    cfg.separate = not args.no_separate
    cfg.separation_model = args.model
    cfg.device = args.device
    if args.shifts is not None:
        cfg.separation_shifts = max(1, args.shifts)
    cfg.lossy_repair = not args.no_lossy_repair
    cfg.transcribe_mix = not args.no_mix_pass
    cfg.pitched_from_mix_only = args.pitched_from_mix_only
    if args.no_mix_primary:
        cfg.mix_primary = False
    if args.min_stem_confidence is not None:
        cfg.min_stem_confidence = args.min_stem_confidence

    if args.only:
        keep = {s.strip().lower() for s in args.only.split(",") if s.strip()}
        for name, s in cfg.stems.items():
            s.enabled = name in keep
    if args.skip:
        drop = {s.strip().lower() for s in args.skip.split(",") if s.strip()}
        for name, s in cfg.stems.items():
            if name in drop:
                s.enabled = False

    for s in list(cfg.stems.values()) + [cfg.mixdown_stem]:
        if args.onset_threshold is not None:
            s.onset_threshold = args.onset_threshold
            s.adaptive_threshold = False
        if args.frame_threshold is not None:
            s.frame_threshold = args.frame_threshold
            s.adaptive_threshold = False
        if args.fixed_threshold:
            s.adaptive_threshold = False
        if args.min_note_ms is not None:
            s.min_note_ms = args.min_note_ms
        if args.max_polyphony is not None:
            s.max_polyphony = args.max_polyphony
        if args.no_bends:
            s.pitch_bend = False
        if args.no_ghost_filter:
            s.harmonic_suppression = False
        if args.harmonic_ratio is not None:
            s.harmonic_ratio = args.harmonic_ratio
        if args.melodia:
            s.melodia_trick = True

    cfg.quantize = args.quantize
    cfg.quantize_strength = float(min(max(args.quantize_strength, 0.0), 1.0))
    cfg.quantize_drums_only = args.quantize_drums_only
    cfg.fixed_tempo = args.tempo
    cfg.detect_tempo = args.tempo is None
    cfg.variable_tempo = not args.no_tempo_map

    try:
        num, den = args.time_signature.split("/")
        cfg.time_signature = (int(num), int(den))
    except Exception:
        raise SystemExit("bad --time-signature %r (expected e.g. 4/4)"
                         % args.time_signature)

    cfg.write_stems = args.write_stems
    cfg.write_per_stem_midi = args.per_stem_midi
    cfg.merge_to_single_track = args.single_track
    cfg.ticks_per_beat = args.ticks_per_beat
    cfg.verbose = args.verbose and not args.quiet
    return cfg


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    level = logging.DEBUG if args.verbose else (
        logging.WARNING if args.quiet else logging.INFO)
    logging.basicConfig(level=level, format="%(message)s", stream=sys.stderr)
    if not args.verbose:
        warnings.filterwarnings("ignore")
        for noisy in ("numba", "matplotlib", "huggingface_hub", "httpx", "urllib3"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.output and len(args.input) > 1:
        raise SystemExit("--output takes a single input file; use --outdir instead")

    from midgenius.pipeline import convert

    failures = 0
    for path in args.input:
        if not os.path.exists(path):
            print("error: no such file: %s" % path, file=sys.stderr)
            failures += 1
            continue

        out = args.output
        if out is None:
            stem = os.path.splitext(os.path.basename(path))[0] + ".mid"
            out = os.path.join(args.outdir, stem) if args.outdir else \
                os.path.join(os.path.dirname(os.path.abspath(path)), stem)

        try:
            result = convert(path, out, config_from_args(args))
        except KeyboardInterrupt:
            print("\ninterrupted", file=sys.stderr)
            return 130
        except Exception as e:
            print("error: %s: %s" % (os.path.basename(path), e), file=sys.stderr)
            if args.verbose:
                import traceback
                traceback.print_exc()
            failures += 1
            continue

        if not args.quiet:
            print()
            print(result.report())
            print()

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
