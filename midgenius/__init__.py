"""MIDGenius - high fidelity audio to MIDI transcription.

A stem-separated, per-instrument transcription pipeline built to address the
classic failure modes of naive audio->MIDI conversion:

* Polyphony & dense mixes  -> source separation before transcription
* "Phantom" notes          -> harmonic/ghost suppression + confidence gating
* Drum confusion           -> dedicated percussion transcriber, not a pitch tracker
* Loss of expressive detail-> velocity curves, pitch bends, sustain pedal, CC11
* MP3 compression loss     -> lossy-codec aware pre-conditioning
"""

__version__ = "1.0.0"

from midgenius.config import Config, StemConfig  # noqa: F401

__all__ = ["Config", "StemConfig", "__version__"]
