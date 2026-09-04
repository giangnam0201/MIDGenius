"""Source separation: split a mix into drums / bass / vocals / other.

This is the single most important stage in the pipeline. Polyphonic
transcription of a full mix is a badly posed problem: a piano chord, a bass
note and a kick drum all overlap in the same frequency bins, so the model has
to guess which partial belongs to which source. Separating first turns one hard
problem into four much easier ones, and it is what removes most "phantom"
notes before any note decoding happens.

Backends, in order of preference:

1. ``demucs``    - Hybrid Transformer Demucs (htdemucs), the strongest open
                   4-stem separator available offline.
2. ``torchaudio``- HDemucs (HDEMUCS_HIGH_MUSDB_PLUS) pipeline, no extra deps.
3. ``hpss``      - librosa harmonic/percussive split. Not real separation, but
                   it still beats transcribing the raw mix, and it always works.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import numpy as np

from midgenius.audio import Audio, resample

log = logging.getLogger("midgenius.separation")

STEM_NAMES = ("drums", "bass", "other", "vocals")


def pick_device(requested: str = "auto") -> str:
    if requested and requested != "auto":
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def available_backend() -> str:
    """Best separator installed. Import failures are logged, never silent -
    a silent downgrade to HPSS looks like working software producing bad stems.
    """
    try:
        import demucs.pretrained  # noqa: F401
        return "demucs"
    except Exception as e:
        log.warning("demucs unavailable (%s: %s)", type(e).__name__, e)
    try:
        from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS  # noqa: F401
        return "torchaudio"
    except Exception as e:
        log.warning("torchaudio HDemucs unavailable (%s: %s)", type(e).__name__, e)
    return "hpss"


def separate(
    audio: Audio,
    model_name: str = "htdemucs",
    device: str = "auto",
    shifts: int = 1,
    overlap: float = 0.25,
    segment: Optional[float] = None,
    progress: bool = True,
) -> Dict[str, Audio]:
    """Separate ``audio`` into named stems.

    Returns a dict of stem name -> Audio at the original sample rate. Stems that
    a backend cannot produce are simply absent from the result.
    """
    backend = available_backend()
    log.info("separation backend: %s (device=%s)", backend, pick_device(device))

    if backend == "demucs":
        try:
            return _separate_demucs(audio, model_name, device, shifts, overlap,
                                    segment, progress)
        except Exception as e:
            log.warning("demucs failed (%r), falling back to torchaudio", e)
            backend = "torchaudio"

    if backend == "torchaudio":
        try:
            return _separate_torchaudio(audio, device, segment or 10.0, overlap)
        except Exception as e:
            log.warning("torchaudio HDemucs failed (%r), falling back to HPSS", e)

    return _separate_hpss(audio)


# --------------------------------------------------------------------------
# demucs
# --------------------------------------------------------------------------

def _separate_demucs(audio: Audio, model_name: str, device: str, shifts: int,
                     overlap: float, segment: Optional[float],
                     progress: bool) -> Dict[str, Audio]:
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    dev = pick_device(device)
    model = get_model(model_name)
    model.to(dev)
    model.eval()

    model_sr = int(getattr(model, "samplerate", 44100))
    channels = int(getattr(model, "audio_channels", 2))

    wav = audio.data
    if channels == 2 and wav.shape[0] == 1:
        wav = np.repeat(wav, 2, axis=0)
    elif channels == 1 and wav.shape[0] > 1:
        wav = wav.mean(axis=0, keepdims=True)
    if audio.sr != model_sr:
        wav = resample(wav, audio.sr, model_sr)

    tensor = torch.from_numpy(np.ascontiguousarray(wav))

    # Demucs is trained on loudness-normalised input; match that.
    ref = tensor.mean(0)
    mean, std = ref.mean(), ref.std()
    tensor = (tensor - mean) / (std + 1e-8)

    kwargs = dict(shifts=max(0, int(shifts) - 1), overlap=float(overlap),
                  progress=progress, device=dev)
    if segment:
        kwargs["segment"] = float(segment)

    with torch.no_grad():
        sources = apply_model(model, tensor[None], **kwargs)[0]

    sources = sources * (std + 1e-8) + mean
    out: Dict[str, Audio] = {}
    for name, src in zip(model.sources, sources):
        arr = src.cpu().numpy().astype(np.float32)
        if model_sr != audio.sr:
            arr = resample(arr, model_sr, audio.sr)
        # Length can drift by a sample or two through resampling.
        arr = _fit_length(arr, audio.n_samples)
        out[name] = Audio(arr, audio.sr)
    return out


# --------------------------------------------------------------------------
# torchaudio HDemucs
# --------------------------------------------------------------------------

def _separate_torchaudio(audio: Audio, device: str, segment_s: float,
                         overlap: float) -> Dict[str, Audio]:
    import torch
    from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS as BUNDLE

    dev = pick_device(device)
    model = BUNDLE.get_model().to(dev).eval()
    model_sr = BUNDLE.sample_rate

    wav = audio.data
    if wav.shape[0] == 1:
        wav = np.repeat(wav, 2, axis=0)
    if audio.sr != model_sr:
        wav = resample(wav, audio.sr, model_sr)
    tensor = torch.from_numpy(np.ascontiguousarray(wav)).to(dev)

    ref = tensor.mean(0)
    mean, std = ref.mean(), ref.std()
    tensor = (tensor - mean) / (std + 1e-8)

    out = _chunked_apply(model, tensor, model_sr, segment_s, overlap, dev)
    out = out * (std + 1e-8) + mean

    result: Dict[str, Audio] = {}
    for name, src in zip(model.sources, out):
        arr = src.cpu().numpy().astype(np.float32)
        if model_sr != audio.sr:
            arr = resample(arr, model_sr, audio.sr)
        result[name] = Audio(_fit_length(arr, audio.n_samples), audio.sr)
    return result


def _chunked_apply(model, mix, sr: int, segment_s: float, overlap: float, dev):
    """Overlap-add inference so long tracks fit in memory."""
    import torch

    chunk = int(sr * segment_s)
    stride = max(1, int(chunk * (1.0 - overlap)))
    n = mix.shape[-1]
    n_src = len(model.sources)
    out = torch.zeros(n_src, mix.shape[0], n, device=dev)
    weight_sum = torch.zeros(n, device=dev)
    window = torch.hann_window(chunk, periodic=True, device=dev).clamp_min(1e-3)

    with torch.no_grad():
        for start in range(0, max(n - 1, 1), stride):
            end = min(start + chunk, n)
            seg = mix[:, start:end]
            pad = chunk - seg.shape[-1]
            if pad > 0:
                seg = torch.nn.functional.pad(seg, (0, pad))
            est = model(seg[None])[0]
            w = window[: end - start]
            out[:, :, start:end] += est[:, :, : end - start] * w
            weight_sum[start:end] += w
            if end >= n:
                break
    return out / weight_sum.clamp_min(1e-6)


# --------------------------------------------------------------------------
# HPSS fallback
# --------------------------------------------------------------------------

def _separate_hpss(audio: Audio) -> Dict[str, Audio]:
    """Harmonic/percussive split plus a crossover for a crude bass stem.

    Not source separation, but it gives the drum transcriber a percussive-only
    signal and the pitched transcribers a de-drummed signal, which is most of
    the practical benefit.
    """
    import librosa
    import scipy.signal as sps

    sr = audio.sr
    harm_ch, perc_ch = [], []
    for ch in audio.data:
        h, p = librosa.effects.hpss(ch, margin=(1.0, 3.0))
        harm_ch.append(h)
        perc_ch.append(p)
    harmonic = np.stack(harm_ch).astype(np.float32)
    percussive = np.stack(perc_ch).astype(np.float32)

    sos_lp = sps.butter(4, 250.0, btype="low", fs=sr, output="sos")
    sos_hp = sps.butter(4, 250.0, btype="high", fs=sr, output="sos")
    bass = sps.sosfiltfilt(sos_lp, harmonic, axis=-1).astype(np.float32)
    other = sps.sosfiltfilt(sos_hp, harmonic, axis=-1).astype(np.float32)

    log.warning("using HPSS fallback separation - install `demucs` for real stems")
    return {
        "drums": Audio(percussive, sr),
        "bass": Audio(bass, sr),
        "other": Audio(other, sr),
    }


def _fit_length(arr: np.ndarray, n: int) -> np.ndarray:
    if arr.shape[-1] == n:
        return arr
    if arr.shape[-1] > n:
        return arr[..., :n]
    pad = [(0, 0)] * (arr.ndim - 1) + [(0, n - arr.shape[-1])]
    return np.pad(arr, pad)


def stem_activity(stems: Dict[str, Audio]) -> Dict[str, float]:
    """RMS level of each stem in dBFS - used to skip silent stems."""
    out = {}
    for name, st in stems.items():
        rms = float(np.sqrt(np.mean(st.data.astype(np.float64) ** 2)))
        out[name] = 20.0 * np.log10(rms + 1e-12)
    return out
