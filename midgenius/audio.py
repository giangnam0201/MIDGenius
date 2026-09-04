"""Audio loading, resampling and lossy-codec conditioning."""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

log = logging.getLogger("midgenius.audio")


@dataclass
class Audio:
    """Float32 audio, shape (channels, samples)."""

    data: np.ndarray
    sr: int
    path: Optional[str] = None

    @property
    def n_channels(self) -> int:
        return self.data.shape[0]

    @property
    def n_samples(self) -> int:
        return self.data.shape[1]

    @property
    def duration(self) -> float:
        return self.n_samples / float(self.sr)

    def mono(self) -> np.ndarray:
        if self.data.shape[0] == 1:
            return self.data[0]
        return self.data.mean(axis=0)

    def to_stereo(self) -> "Audio":
        if self.n_channels == 2:
            return self
        if self.n_channels == 1:
            return Audio(np.repeat(self.data, 2, axis=0), self.sr, self.path)
        return Audio(np.stack([self.data[0], self.data[1]]), self.sr, self.path)

    def resample(self, target_sr: int) -> "Audio":
        if target_sr == self.sr:
            return self
        return Audio(resample(self.data, self.sr, target_sr), target_sr, self.path)

    def slice(self, start_s: float, end_s: float) -> "Audio":
        a = max(0, int(round(start_s * self.sr)))
        b = min(self.n_samples, int(round(end_s * self.sr)))
        return Audio(self.data[:, a:b].copy(), self.sr, self.path)

    def copy(self) -> "Audio":
        return Audio(self.data.copy(), self.sr, self.path)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load(path: str, sr: Optional[int] = None, mono: bool = False) -> Audio:
    """Load any audio file to float32 (channels, samples).

    Tries libsndfile first (wav/flac/ogg/mp3), then PyAV, then librosa's
    audioread fallback. No external ffmpeg binary is required.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    data, file_sr = _read_any(path)

    if data.ndim == 1:
        data = data[None, :]
    data = np.ascontiguousarray(data.astype(np.float32, copy=False))

    if mono and data.shape[0] > 1:
        data = data.mean(axis=0, keepdims=True)
    if sr is not None and sr != file_sr:
        data = resample(data, file_sr, sr)
        file_sr = sr

    return Audio(data, file_sr, path)


def _read_any(path: str) -> Tuple[np.ndarray, int]:
    errors = []

    # 1) libsndfile
    try:
        import soundfile as sf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, file_sr = sf.read(path, dtype="float32", always_2d=True)
        return y.T, int(file_sr)
    except Exception as e:  # pragma: no cover - depends on libsndfile build
        errors.append("soundfile: %r" % (e,))

    # 2) PyAV (bundled ffmpeg libraries)
    try:
        return _read_pyav(path)
    except Exception as e:  # pragma: no cover
        errors.append("pyav: %r" % (e,))

    # 3) librosa / audioread
    try:
        import librosa
        y, file_sr = librosa.load(path, sr=None, mono=False)
        if y.ndim == 1:
            y = y[None, :]
        return y, int(file_sr)
    except Exception as e:  # pragma: no cover
        errors.append("librosa: %r" % (e,))

    raise RuntimeError("Could not decode %s\n  %s" % (path, "\n  ".join(errors)))


def _read_pyav(path: str) -> Tuple[np.ndarray, int]:
    import av

    with av.open(path) as container:
        stream = container.streams.audio[0]
        stream.thread_type = "AUTO"
        file_sr = stream.codec_context.sample_rate
        resampler = av.audio.resampler.AudioResampler(
            format="fltp", layout=stream.layout, rate=file_sr
        )
        chunks = []
        for frame in container.decode(stream):
            for rframe in resampler.resample(frame):
                chunks.append(rframe.to_ndarray())
        if not chunks:
            raise RuntimeError("no audio frames decoded")
        return np.concatenate(chunks, axis=1), int(file_sr)


def save_wav(path: str, audio: Audio) -> None:
    import soundfile as sf
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    sf.write(path, audio.data.T, audio.sr, subtype="PCM_16")


# --------------------------------------------------------------------------
# resampling
# --------------------------------------------------------------------------

def resample(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """High quality resampling of a (channels, samples) or 1-D array."""
    if orig_sr == target_sr:
        return data
    single = data.ndim == 1
    if single:
        data = data[None, :]
    try:
        import soxr
        out = soxr.resample(data.T, orig_sr, target_sr, quality="VHQ").T
    except Exception:
        import librosa
        out = np.stack([
            librosa.resample(ch, orig_sr=orig_sr, target_sr=target_sr, res_type="soxr_hq")
            for ch in data
        ])
    out = np.ascontiguousarray(np.atleast_2d(out).astype(np.float32))
    return out[0] if single else out


# --------------------------------------------------------------------------
# lossy codec conditioning
# --------------------------------------------------------------------------

def estimate_bandwidth(y: np.ndarray, sr: int, max_probe_s: float = 30.0) -> float:
    """Estimate the lowpass cutoff a lossy encoder imposed, in Hz.

    MP3/AAC encoders discard everything above a bitrate dependent cutoff
    (typically 15-16 kHz at 128 kbps, 19-20 kHz at 320 kbps). Knowing where the
    spectrum dies tells us how much high frequency evidence we actually have,
    which matters for cymbal and hi-hat detection.
    """
    import librosa

    n_fft = 4096
    n = min(len(y), int(max_probe_s * sr))
    if n < n_fft:
        return sr / 2.0
    seg = y[:n]
    S = np.abs(librosa.stft(seg, n_fft=n_fft, hop_length=n_fft // 2))
    if S.size == 0:
        return sr / 2.0
    spec = S.mean(axis=1)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    # Reference level = median energy in the 1-6 kHz band.
    band = (freqs >= 1000) & (freqs <= 6000)
    ref = float(np.median(spec[band])) if band.any() else float(spec.mean())
    if ref <= 0:
        return sr / 2.0

    # Walk down from Nyquist until energy climbs back above -55 dB of ref.
    thresh = ref * (10 ** (-55 / 20.0))
    idx = len(spec) - 1
    while idx > 0 and spec[idx] < thresh:
        idx -= 1
    return float(freqs[min(idx + 1, len(freqs) - 1)])


def repair_lossy(audio: Audio, cutoff_hz: Optional[float] = None,
                 strength: float = 0.6) -> Audio:
    """Reduce artefacts typical of lossy-compressed input.

    Two things actually help transcription:

    1. The codec leaves a low level quantisation-noise bed between real
       partials. It has a near constant per-bin level, so a soft spectral gate
       keyed to a per-bin percentile removes it while leaving partials intact.
       This is what stops the noise bed from being read as note activations.
    2. Below the codec cutoff the signal is trustworthy; above it there is
       essentially nothing. We taper rather than use a cliff edge so the band
       energy features the drum transcriber relies on stay smooth.

    Deliberately conservative: over-processing costs more accuracy than the
    artefacts do.
    """
    import librosa

    sr = audio.sr
    n_fft, hop = 2048, 512
    strength = float(np.clip(strength, 0.0, 1.0))
    out_channels = []

    for ch in audio.data:
        if ch.size < n_fft:
            out_channels.append(ch)
            continue
        S = librosa.stft(ch, n_fft=n_fft, hop_length=hop)
        mag, phase = np.abs(S), np.angle(S)

        floor = np.percentile(mag, 15, axis=1, keepdims=True)
        gain = mag / (mag + floor * (1.0 + 3.0 * strength) + 1e-12)
        mag_clean = mag * (1.0 - strength + strength * gain)

        if cutoff_hz is not None and cutoff_hz < sr / 2.0 * 0.98:
            freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
            lo = cutoff_hz * 0.85
            ramp = np.clip((freqs - lo) / max(cutoff_hz - lo, 1.0), 0.0, 1.0)
            taper = 0.5 * (1.0 + np.cos(np.pi * ramp))
            mag_clean *= taper[:, None]

        out_channels.append(
            librosa.istft(mag_clean * np.exp(1j * phase), hop_length=hop, length=len(ch))
        )

    return Audio(np.stack(out_channels).astype(np.float32), sr, audio.path)


def normalize_peak(audio: Audio, target_db: float = -1.0) -> Audio:
    peak = float(np.abs(audio.data).max())
    if peak <= 1e-9:
        return audio
    target = 10 ** (target_db / 20.0)
    return Audio((audio.data * (target / peak)).astype(np.float32), audio.sr, audio.path)


def rms_envelope(y: np.ndarray, sr: int, hop: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """Return (times, rms) for a mono signal."""
    import librosa
    rms = librosa.feature.rms(y=y, frame_length=hop * 4, hop_length=hop)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    return times, rms
