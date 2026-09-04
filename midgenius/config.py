"""Configuration objects for the MIDGenius pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple


# General MIDI program numbers used for the default stem -> instrument mapping.
GM_ACOUSTIC_GRAND = 0
GM_ELECTRIC_PIANO = 4
GM_FINGERED_BASS = 33
GM_SYNTH_STRINGS = 51
GM_SAW_LEAD = 81
GM_VOICE_OOHS = 53


@dataclass
class StemConfig:
    """Per-stem transcription settings.

    Every stem gets its own thresholds because a bass line, a lead vocal and a
    dense pad have wildly different statistics. Using one global threshold is
    the single biggest cause of phantom notes in naive converters.
    """

    name: str
    # "poly"  -> Basic Pitch (ICASSP 2022) polyphonic transcription
    # "mono"  -> pYIN monophonic pitch tracking (bass, lead vocal)
    # "drums" -> spectral onset / percussion classifier
    method: str = "poly"

    enabled: bool = True

    # MIDI program + channel
    program: int = GM_ACOUSTIC_GRAND
    channel: Optional[int] = None  # None -> auto-assign

    # Pitch range gate (MIDI note numbers). Anything outside is impossible for
    # this instrument and is therefore a phantom note by definition.
    min_midi: int = 21
    max_midi: int = 108

    # Basic Pitch decoding thresholds
    onset_threshold: float = 0.6
    frame_threshold: float = 0.45
    min_note_ms: float = 58.0
    infer_onsets: bool = True
    # The "melodia trick" sweeps up leftover frame energy that has no onset
    # behind it. That leftover energy is mostly harmonics of notes already
    # transcribed, so on real material it is a phantom-note generator: on the
    # ground-truth chord benchmark it cost 36 points of precision and produced
    # every one of the octave errors. Off by default; worth enabling only for
    # sparse, quiet solo material where missed notes matter more.
    melodia_trick: bool = False
    energy_tolerance: int = 11

    # Phantom-note suppression
    harmonic_suppression: bool = True     # kill octave/fifth ghosts of real notes
    harmonic_ratio: float = 0.28          # ghost must be this much weaker to be cut
    min_confidence: float = 0.14          # drop notes whose mean activation is low
    max_polyphony: int = 0                # 0 = unlimited; else keep N strongest

    # Expression
    pitch_bend: bool = True
    velocity_from_audio: bool = True
    sustain_pedal: bool = False
    expression_cc: bool = False           # CC11 envelope following stem loudness

    # Monophonic (pYIN) settings
    mono_fmin: float = 55.0
    mono_fmax: float = 1200.0
    mono_voiced_prob: float = 0.55
    vibrato_preserve: bool = True

    def with_overrides(self, **kw) -> "StemConfig":
        d = asdict(self)
        d.update({k: v for k, v in kw.items() if v is not None})
        return StemConfig(**d)


def default_stems() -> Dict[str, StemConfig]:
    """The four Demucs stems, each tuned for its instrument class."""
    return {
        "drums": StemConfig(
            name="drums",
            method="drums",
            program=0,
            channel=9,  # GM percussion channel (0-indexed)
            min_midi=27,
            max_midi=87,
            velocity_from_audio=True,
            pitch_bend=False,
        ),
        "bass": StemConfig(
            name="bass",
            method="mono",
            program=GM_FINGERED_BASS,
            min_midi=24,   # C1
            max_midi=67,   # G4
            mono_fmin=30.0,
            mono_fmax=440.0,
            mono_voiced_prob=0.50,
            min_note_ms=70.0,
            pitch_bend=True,
            harmonic_suppression=True,
        ),
        "vocals": StemConfig(
            name="vocals",
            method="mono",
            program=GM_VOICE_OOHS,
            min_midi=36,
            max_midi=88,
            mono_fmin=65.0,
            mono_fmax=1400.0,
            mono_voiced_prob=0.58,
            min_note_ms=80.0,
            pitch_bend=True,
            vibrato_preserve=True,
            expression_cc=True,
        ),
        "other": StemConfig(
            name="other",
            method="poly",
            program=GM_ACOUSTIC_GRAND,
            min_midi=28,
            max_midi=104,
            onset_threshold=0.6,
            frame_threshold=0.45,
            min_note_ms=64.0,
            harmonic_suppression=True,
            min_confidence=0.16,
            max_polyphony=8,
            pitch_bend=False,
            sustain_pedal=True,
        ),
    }


@dataclass
class Config:
    """Top level pipeline configuration."""

    # --- separation -------------------------------------------------------
    separate: bool = True
    separation_model: str = "htdemucs"
    separation_shifts: int = 1        # >1 = slower but cleaner (shift trick)
    separation_overlap: float = 0.25
    separation_segment: Optional[float] = None
    device: str = "auto"              # auto | cpu | cuda

    # --- pre-conditioning -------------------------------------------------
    lossy_repair: bool = True         # compensate for MP3 spectral holes
    lossy_hf_cutoff_probe: bool = True
    normalize: bool = True

    # --- rhythm -----------------------------------------------------------
    detect_tempo: bool = True
    fixed_tempo: Optional[float] = None
    variable_tempo: bool = True       # write a full tempo map, not a single BPM
    quantize: str = "off"             # off | 1/4 | 1/8 | 1/16 | 1/32 | 1/8t | 1/16t
    quantize_strength: float = 0.75   # 0..1 blend toward the grid
    quantize_drums_only: bool = False

    # --- transcription ----------------------------------------------------
    stems: Dict[str, StemConfig] = field(default_factory=default_stems)
    # If separation is off, the whole mix goes through this config instead.
    mixdown_stem: StemConfig = field(
        default_factory=lambda: StemConfig(
            name="mixdown", method="poly", program=GM_ACOUSTIC_GRAND,
            onset_threshold=0.6, frame_threshold=0.45, max_polyphony=10,
        )
    )

    # --- output -----------------------------------------------------------
    ticks_per_beat: int = 480
    write_stems: bool = False         # also dump separated wavs
    write_per_stem_midi: bool = False # one .mid per stem in addition to combined
    merge_to_single_track: bool = False
    key_signature: bool = True
    time_signature: Tuple[int, int] = (4, 4)

    # --- runtime ----------------------------------------------------------
    workers: int = 0                  # 0 -> serial (safest on CPU)
    verbose: bool = True
    cache_dir: Optional[str] = None

    def enabled_stems(self) -> List[StemConfig]:
        return [s for s in self.stems.values() if s.enabled]
