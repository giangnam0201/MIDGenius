"""Guess a General MIDI instrument for a stem from how it actually sounds.

The ``other`` stem is a catch-all - brass, guitar, keys, synth pads all land in
it - and playing every one of them back as acoustic grand piano is the most
audible "wrong instrument" error in the output. Nothing here identifies an
instrument by name; it just measures the two acoustic properties that decide
which *family* a General MIDI patch should come from:

* **Sustain** - does a note hold at full level after the attack (bowed, blown,
  organ, pad) or decay away on its own (plucked, struck)? This is the single
  biggest perceptual split: a sustained part rendered as piano dies while the
  record holds, and a piano part rendered as strings smears.
* **Brightness** - spectral centroid relative to the stem's own bandwidth,
  used only to tell a plucked string from a struck one within the decaying
  family. It proved useless for splitting brass from strings (see below).

The mapping is deliberately coarse and conservative: three broad targets, and a
fall back to piano whenever the measurements are inconclusive, because a
neutral piano is a far smaller error than a confidently wrong patch.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

log = logging.getLogger("midgenius.timbre")

# General MIDI programs used as family representatives.
GM_ACOUSTIC_GRAND = 0
GM_ELECTRIC_GUITAR_CLEAN = 27
GM_STRING_ENSEMBLE = 48
GM_BRASS_SECTION = 61          # kept for reference; see guess_program


def _sustain_ratio(y: np.ndarray, sr: int, hop: int = 512) -> float:
    """How much of a note's level survives after its attack, 0..1.

    Measured on the stem's own loud frames: the ratio of the energy a short
    while after each onset to the energy at the onset. Struck/plucked sources
    fall away quickly; bowed, blown and held sources do not.
    """
    import librosa

    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    if rms.size < 8:
        return 0.5
    peaks = librosa.util.peak_pick(env, pre_max=3, post_max=3, pre_avg=5,
                                   post_avg=5, delta=float(np.percentile(env, 75)),
                                   wait=int(0.12 * sr / hop))
    peaks = np.asarray(peaks, dtype=int)
    if peaks.size < 4:
        return 0.5
    lag = max(1, int(round(0.18 * sr / hop)))       # ~180 ms after the attack
    ratios = []
    for p in peaks:
        a, b = p, p + lag
        if b >= rms.size or rms[a] <= 1e-8:
            continue
        ratios.append(float(rms[b] / rms[a]))
    if not ratios:
        return 0.5
    return float(np.clip(np.median(ratios), 0.0, 2.0))


def _brightness(y: np.ndarray, sr: int) -> float:
    """Spectral centroid as a fraction of the stem's own occupied bandwidth."""
    import librosa

    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    if S.size == 0 or S.max() <= 0:
        return 0.5
    freqs = np.fft.rfftfreq(2048, 1.0 / sr)
    power = S.sum(axis=1)
    total = power.sum()
    if total <= 0:
        return 0.5
    centroid = float((freqs * power).sum() / total)
    # Roll-off marks where this stem's content actually ends, so a lowpassed
    # MP3 does not read as "dark" purely because its top octave is missing.
    cum = np.cumsum(power) / total
    idx = int(np.searchsorted(cum, 0.95))
    rolloff = float(freqs[min(idx, len(freqs) - 1)])
    if rolloff <= 0:
        return 0.5
    return float(np.clip(centroid / rolloff, 0.0, 1.0))


def guess_program(y: np.ndarray, sr: int,
                  default: int = GM_ACOUSTIC_GRAND) -> int:
    """Pick a General MIDI program for this stem, or ``default`` if unclear."""
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if y.size < sr:                       # under a second: nothing to judge
        return default
    try:
        sustain = _sustain_ratio(y, sr)
        bright = _brightness(y, sr)
    except Exception as e:                # analysis must never break a conversion
        log.debug("timbre analysis failed (%r), keeping default program", e)
        return default

    # Sustained: the level holds well after the attack, so it needs a patch that
    # holds too - piano would die away underneath a note the record sustains.
    #
    # Brass vs strings is deliberately *not* attempted. Measured across four
    # real stems (ambient pads, game chiptune, pop brass, novelty synth) the
    # brightness figure clustered in 0.22-0.28 with no useful separation, and
    # the brightest of them was the one that should least be brass. A string
    # ensemble is the safer sustained default: it is close for pads and synth,
    # and merely bland - not wrong - under a brass line.
    if sustain >= 0.60:
        program = GM_STRING_ENSEMBLE
    # Clearly decaying and bright: plucked/struck string rather than piano.
    elif sustain <= 0.35 and bright >= 0.34:
        program = GM_ELECTRIC_GUITAR_CLEAN
    else:
        program = default

    log.info("timbre: sustain %.2f brightness %.2f -> GM program %d",
             sustain, bright, program)
    return program
