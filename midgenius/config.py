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

    # Basic Pitch decoding thresholds. These are only used when
    # `adaptive_threshold` is off, or as the manual override the CLI sets when
    # the user passes an explicit threshold.
    onset_threshold: float = 0.6
    frame_threshold: float = 0.30
    # Derive the thresholds from the model's own confidence distribution on
    # this material instead of using the fixed numbers above. A fixed value
    # tuned on a punchy track transcribes a sixth of the notes of a soft one.
    adaptive_threshold: bool = True
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
    # Pull octave-too-high detections down to their fundamental. Basic Pitch's
    # signature error is latching onto the 2nd harmonic when the fundamental is
    # weak; each such note is scored wrong twice (phantom high, miss low).
    octave_correction: bool = False
    octave_sub_ratio: float = 0.80
    octave_onset_ratio: float = 0.50
    # Drop octave harmonic ghosts by onset independence: a note an octave above a
    # stronger, simultaneous note is cut when its own onset activation is under
    # `octave_deghost_ratio` of the lower note's (a real octave keeps its attack).
    octave_deghost: bool = False
    octave_deghost_ratio: float = 0.5
    # Envelope-correlation octave deghost (ON by default): drop an octave note
    # whose frame activation is a scaled copy of the note below it (a harmonic),
    # keeping notes with an independent envelope. corr>=threshold => harmonic =>
    # drop. This is the one octave fix that survives full-track validation -
    # confidence, onset strength and a hard harmonic ratio all cut real octaves
    # too, because Basic Pitch's note and onset heads both fire at harmonics;
    # the envelope *shape* is what actually distinguishes a copy from a voice.
    # Measured: aria pitched F1 60.7 -> 61.7 (precision +4) with the dense
    # recording unchanged. --no-octave-deghost restores the old behaviour.
    octave_deghost_env: bool = True
    octave_deghost_corr: float = 0.76
    # Test-time augmentation: extra pitch-shift passes of the model to average,
    # e.g. (-1, 0, 1). Empty = single pass (default). Costs one model run each,
    # so it belongs to the "best" quality preset, not the light default.
    tta_semitones: Tuple[int, ...] = ()
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
            # Basic Pitch (poly), not pYIN. pYIN's Viterbi over a whole track is
            # the single heaviest stage in the pipeline (~30 s on an 8.5 min song,
            # more than half the transcription time), and measured against both
            # reference pairs it was also *less* accurate than the ONNX model on
            # the bass stem - forcing a single f0 through Demucs' bass artefacts
            # loses notes the polyphonic model keeps. Switching cut conversion
            # time roughly in half and raised aria F1 58.7 -> 60.1 with the dense
            # recording essentially unchanged. The min/max MIDI gate below is what
            # keeps it a bass line rather than a second harmony part.
            method="poly",
            program=GM_FINGERED_BASS,
            min_midi=24,   # C1
            max_midi=67,   # G4
            min_note_ms=100.0,
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
            frame_threshold=0.30,
            # 45 ms, down from 64: recovers short notes the longer floor cut,
            # measured +0.7 aria F1 (recall up) with the dense recording neutral.
            min_note_ms=45.0,
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
    # Also transcribe the untouched mix and fold in whatever the stems missed.
    # Separation can strip the attack transients that onset detection needs -
    # see _merge_pitched_sources - so the mix is a safety net, not a luxury.
    transcribe_mix: bool = True
    # Take pitched notes from the whole mix only, using the separated stems just
    # for drums. On clean, well-separated-in-frequency material (synths, chiptune)
    # Demucs' pitched stems carry artefacts that read as phantom notes, and the
    # stem-union over-produces badly; the untouched mix is far cleaner. Measured
    # on a synth track: mix-only precision 50% vs stem-union 40%. Off keeps the
    # full stem-union behaviour, which wins on dense real recordings.
    pitched_from_mix_only: bool = False
    # Mix-primary union: take the mix as the precise base and add back only
    # confident stem notes the mix missed. Keeps mix precision on clean material
    # while still recovering masked notes on dense mixes. Measured on both
    # reference pairs, this beat the old stem-union: aria F1 55.8 -> 58.7 with no
    # regression on the dense real recording (graze 55.1 -> 55.0). The 0.3
    # confidence floor is what preserves graze - a higher floor drops the masked
    # notes the stems exist to recover.
    mix_primary: bool = True
    min_stem_confidence: float = 0.3
    # A mono-tracked stem (vocals) this far below the loudest stem is
    # separation bleed rather than an instrument, and a monophonic tracker
    # turns bleed into confident wrong notes.
    mono_stem_floor_db: float = 20.0
    stems: Dict[str, StemConfig] = field(default_factory=default_stems)
    # If separation is off, the whole mix goes through this config instead.
    mixdown_stem: StemConfig = field(
        default_factory=lambda: StemConfig(
            name="mixdown", method="poly", program=GM_ACOUSTIC_GRAND,
            onset_threshold=0.6, frame_threshold=0.30, max_polyphony=10,
            min_note_ms=45.0,
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
