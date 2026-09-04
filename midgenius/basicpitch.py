"""Polyphonic pitch posteriorgram inference (Basic Pitch / ICASSP 2022).

We run Spotify's Basic Pitch neural network directly through onnxruntime rather
than going through the ``basic_pitch`` package's own inference path. Two
reasons:

* The package's inference module only accepts *file paths*. We need to feed
  in-memory separated stems without a wav round-trip per stem.
* It hard-depends on TensorFlow and a pinned legacy ``resampy``. The ONNX
  graph shipped inside the wheel needs neither.

The windowing, overlap and frame-time conventions below reproduce the
reference implementation exactly, so the posteriorgrams are bit-comparable.

Model outputs, all in [0, 1]:
    contour (n_frames, 264)  3 bins per semitone pitch salience - drives bends
    note    (n_frames,  88)  per-semitone "note is sounding" probability
    onset   (n_frames,  88)  per-semitone "note starts here" probability
"""

from __future__ import annotations

import logging
import os
import pathlib
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

log = logging.getLogger("midgenius.basicpitch")

# --- constants, matching basic_pitch.constants -----------------------------
FFT_HOP = 256
AUDIO_SAMPLE_RATE = 22050
AUDIO_WINDOW_LENGTH = 2                      # seconds per model window
ANNOTATIONS_FPS = AUDIO_SAMPLE_RATE // FFT_HOP           # 86
ANNOT_N_FRAMES = ANNOTATIONS_FPS * AUDIO_WINDOW_LENGTH   # 172
AUDIO_N_SAMPLES = AUDIO_SAMPLE_RATE * AUDIO_WINDOW_LENGTH - FFT_HOP  # 43844
N_OVERLAPPING_FRAMES = 30
OVERLAP_LEN = N_OVERLAPPING_FRAMES * FFT_HOP
HOP_SIZE = AUDIO_N_SAMPLES - OVERLAP_LEN

ANNOTATIONS_BASE_FREQUENCY = 27.5            # A0
ANNOTATIONS_N_SEMITONES = 88
CONTOURS_BINS_PER_SEMITONE = 3
N_FREQ_BINS_CONTOURS = ANNOTATIONS_N_SEMITONES * CONTOURS_BINS_PER_SEMITONE
MIDI_OFFSET = 21                             # frame bin 0 == MIDI 21 (A0)
MAX_FREQ_IDX = 87

_ONNX_INPUT = "serving_default_input_2:0"
_ONNX_OUTPUTS = {                            # graph output name -> our key
    "note": "StatefulPartitionedCall:1",
    "onset": "StatefulPartitionedCall:2",
    "contour": "StatefulPartitionedCall:0",
}

_SESSION = None
_SESSION_PATH: Optional[str] = None


@dataclass
class Posteriorgram:
    """Model output for one signal, plus the time axis of its frames."""

    note: np.ndarray      # (n_frames, 88)
    onset: np.ndarray     # (n_frames, 88)
    contour: np.ndarray   # (n_frames, 264)
    times: np.ndarray     # (n_frames,) seconds

    @property
    def n_frames(self) -> int:
        return self.note.shape[0]

    @property
    def frame_rate(self) -> float:
        return float(ANNOTATIONS_FPS)


def model_path() -> str:
    """Locate the ONNX weights bundled inside the basic_pitch wheel."""
    env = os.environ.get("MIDGENIUS_BASIC_PITCH_ONNX")
    if env and os.path.exists(env):
        return env
    try:
        import basic_pitch
        root = pathlib.Path(basic_pitch.__file__).parent
        candidate = root / "saved_models" / "icassp_2022" / "nmp.onnx"
        if candidate.exists():
            return str(candidate)
    except Exception:
        pass
    local = pathlib.Path(__file__).parent / "models" / "nmp.onnx"
    if local.exists():
        return str(local)
    raise FileNotFoundError(
        "Basic Pitch ONNX model not found. Install it with "
        "`pip install --no-deps basic-pitch`, or point "
        "MIDGENIUS_BASIC_PITCH_ONNX at an nmp.onnx file."
    )


def get_session(path: Optional[str] = None):
    """Load (and cache) the onnxruntime session."""
    global _SESSION, _SESSION_PATH
    import onnxruntime as ort

    path = path or model_path()
    if _SESSION is not None and _SESSION_PATH == path:
        return _SESSION

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 3
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                 if p in ort.get_available_providers()]
    _SESSION = ort.InferenceSession(path, sess_options=opts, providers=providers)
    _SESSION_PATH = path
    log.info("basic-pitch ONNX loaded (%s)", providers[0])
    return _SESSION


def _window(y: np.ndarray):
    """Yield (1, AUDIO_N_SAMPLES, 1) windows with the reference padding."""
    padded = np.concatenate([np.zeros(OVERLAP_LEN // 2, dtype=np.float32), y])
    for i in range(0, len(padded), HOP_SIZE):
        w = padded[i:i + AUDIO_N_SAMPLES]
        if len(w) < AUDIO_N_SAMPLES:
            w = np.pad(w, (0, AUDIO_N_SAMPLES - len(w)))
        yield w[None, :, None].astype(np.float32)


def _unwrap(stacked: np.ndarray, original_length: int) -> np.ndarray:
    """Drop the overlap guard frames and concatenate windows into one matrix."""
    half = N_OVERLAPPING_FRAMES // 2
    if half > 0:
        stacked = stacked[:, half:-half, :]
    n_out = int(np.floor(original_length * (ANNOTATIONS_FPS / AUDIO_SAMPLE_RATE)))
    flat = stacked.reshape(stacked.shape[0] * stacked.shape[1], stacked.shape[2])
    return flat[:n_out, :]


def frame_times(n_frames: int) -> np.ndarray:
    """Frame index -> seconds, correcting for the per-window time offset.

    Each model window emits 172 frames but only advances the audio by
    (AUDIO_N_SAMPLES / FFT_HOP) frames, so a naive frames-to-time mapping drifts
    forward once per window. The constant below is the reference
    implementation's empirical alignment correction.
    """
    original = np.arange(n_frames) * (FFT_HOP / AUDIO_SAMPLE_RATE)
    window_numbers = np.floor(np.arange(n_frames) / ANNOT_N_FRAMES)
    window_offset = (FFT_HOP / AUDIO_SAMPLE_RATE) * (
        ANNOT_N_FRAMES - (AUDIO_N_SAMPLES / FFT_HOP)
    ) + 0.0018
    return original - window_offset * window_numbers


def predict(y: np.ndarray, sr: int, batch_size: int = 8,
            session=None) -> Posteriorgram:
    """Run Basic Pitch over a mono signal.

    Args:
        y: mono float32 audio.
        sr: its sample rate; resampled to 22050 Hz internally if needed.
        batch_size: windows per ONNX call. Larger is faster but uses more RAM.
    """
    from midgenius.audio import resample

    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if sr != AUDIO_SAMPLE_RATE:
        y = resample(y, sr, AUDIO_SAMPLE_RATE)
    original_length = len(y)
    if original_length < FFT_HOP:
        empty_n = np.zeros((0, ANNOTATIONS_N_SEMITONES), np.float32)
        return Posteriorgram(empty_n, empty_n.copy(),
                             np.zeros((0, N_FREQ_BINS_CONTOURS), np.float32),
                             np.zeros(0))

    sess = session or get_session()
    out_names = [_ONNX_OUTPUTS["note"], _ONNX_OUTPUTS["onset"], _ONNX_OUTPUTS["contour"]]

    acc: Dict[str, list] = {"note": [], "onset": [], "contour": []}
    batch: list = []

    def flush():
        if not batch:
            return
        x = np.concatenate(batch, axis=0)
        note, onset, contour = sess.run(out_names, {_ONNX_INPUT: x})
        acc["note"].append(note)
        acc["onset"].append(onset)
        acc["contour"].append(contour)
        batch.clear()

    for w in _window(y):
        batch.append(w)
        if len(batch) >= batch_size:
            flush()
    flush()

    unwrapped = {k: _unwrap(np.concatenate(v, axis=0), original_length)
                 for k, v in acc.items()}
    n = unwrapped["note"].shape[0]
    return Posteriorgram(
        note=unwrapped["note"].astype(np.float32),
        onset=unwrapped["onset"].astype(np.float32),
        contour=unwrapped["contour"].astype(np.float32),
        times=frame_times(n),
    )


def midi_to_contour_bin(pitch_midi: float) -> float:
    """MIDI pitch -> index into the 264-bin contour axis."""
    hz = 440.0 * 2.0 ** ((pitch_midi - 69.0) / 12.0)
    return 12.0 * CONTOURS_BINS_PER_SEMITONE * np.log2(hz / ANNOTATIONS_BASE_FREQUENCY)
