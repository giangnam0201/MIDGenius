<div align="center">

# 🎵 MIDGenius

### Turn any song into a clean, multi-track MIDI file.

*Separate the mix into stems, then transcribe each instrument with the method that actually suits it — polyphonic model for harmony, dedicated percussion engine for drums, and adaptive strategies that change with the music.*

<br>

[![Stars](https://img.shields.io/github/stars/giangnam0201/MIDGenius?style=for-the-badge&logo=github&color=f5c518)](https://github.com/giangnam0201/MIDGenius/stargazers)
[![Forks](https://img.shields.io/github/forks/giangnam0201/MIDGenius?style=for-the-badge&logo=github&color=6cc644)](https://github.com/giangnam0201/MIDGenius/network/members)
[![Issues](https://img.shields.io/github/issues/giangnam0201/MIDGenius?style=for-the-badge&logo=github&color=8250df)](https://github.com/giangnam0201/MIDGenius/issues)
[![License](https://img.shields.io/github/license/giangnam0201/MIDGenius?style=for-the-badge&color=blue)](LICENSE)

[![CI](https://github.com/giangnam0201/MIDGenius/actions/workflows/ci.yml/badge.svg)](https://github.com/giangnam0201/MIDGenius/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![ONNX Runtime](https://img.shields.io/badge/ONNX-Runtime-005CED?style=flat-square&logo=onnx&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Demucs-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Tests](https://img.shields.io/badge/tests-61%20passing-brightgreen?style=flat-square&logo=pytest&logoColor=white)
![No TensorFlow](https://img.shields.io/badge/no-TensorFlow-lightgrey?style=flat-square)
[![Last commit](https://img.shields.io/github/last-commit/giangnam0201/MIDGenius?style=flat-square)](https://github.com/giangnam0201/MIDGenius/commits)

</div>

```bash
python -m midgenius.cli song.mp3        # → song.mid
```

<div align="center">
<sub><b>🎹 Piano · 🎸 Bass · 🎺 Melody · 🎤 Vocals · 🥁 Drums — each on its own track, on the right instrument.</b></sub>
</div>

---

## 📖 Table of contents

- [Why not just "run a pitch detector"?](#-why-not-just-run-a-pitch-detector)
- [How it works](#-how-it-works)
- [Install](#-install)
- [Usage](#-usage)
- [Measured accuracy](#-measured-accuracy)
- [The drum engine](#-the-drum-engine)
- [Tuning a hard track](#-tuning-a-hard-track)
- [Project layout](#-project-layout)
- [Credits](#-credits)

---

## 🧠 Why not just "run a pitch detector"?

Naive audio→MIDI converters fail in five predictable ways. **Every stage of this pipeline exists to defuse a specific one.**

| ❌ Failure mode | What goes wrong | ✅ What MIDGenius does |
|---|---|---|
| **Polyphony & dense mixes** | One model has to untangle a piano chord, a bass note and a kick sharing the same bins | **Separate stems first** (Demucs), then transcribe each independently — four easy problems instead of one impossible one |
| **"Phantom" notes** | Harmonics get reported as extra notes; dense mixes wash out in weak false positives | Frame-**envelope octave deghost**, harmonic filter, confidence gating, per-instrument range gates, polyphony caps |
| **Drum confusion** | A pitch tracker on a drum kit emits a cloud of nonsense | A **dedicated percussion engine** — per-band onset detection with physically-grounded band-ratio classification. No pitch tracker ever touches drums |
| **Lost expression** | Everything comes out velocity-100, dead straight, no vibrato | Velocities from real attack energy, pitch-bend curves from the f0 contour, CC11 expression, sustain pedal, a **variable tempo map** |
| **MP3 compression loss** | Codec noise reads as notes; the missing top octave hides cymbals | Codec-bandwidth probe + spectral conditioning; the drum stage adapts to whatever high end survived |

---

## ⚙️ How it works

```
 audio ─▶ condition ─▶ separate ─▶ rhythm ─▶ transcribe per stem ─▶ clean ─▶ assemble MIDI
 (MP3…)   (codec fix)  (Demucs)   (tempo)    (right tool each)      (ghosts)   (multi-track)
```

### 🎯 The right tool per source
- **Harmony & bass** → **[Basic Pitch](https://github.com/spotify/basic-pitch)** (Spotify's ICASSP-2022 model) via **ONNX Runtime** — no TensorFlow.
- **Vocals** → Basic Pitch at low thresholds, capped to a couple of voices, to capture the full sung melody.
- **Drums** → a band-onset percussion classifier (see [below](#-the-drum-engine)).

### 🔀 Separation is a means, not an end
Splitting into stems and transcribing each *seems* obvious — but Demucs **generates** each source rather than masking, so outside its training domain the stems aren't a clean partition. On an ambient synth track it routed the **attack transients** of pitched instruments into the *drum* stem; removing that stem (only 26 dB down) **halved** pitched F1.

So the **untouched mix is transcribed too**, and *how* the two are combined is decided **per track**:

| Strategy | Best for | What it does |
|---|---|---|
| **mix-primary** *(default)* | clean synth / instrumental | mix as the precise base; adds back only confident stem notes it missed |
| **union of instruments** | 🎤 produced songs (vocal detected) | keeps bass / melody / vocal / drums as **separate instruments** with full detail |
| **mix-only** | opt-in | pitched notes from the mix alone (`--pitched-from-mix-only`) |

A prominent vocal stem auto-switches to the song strategy — measured against the source (chroma agreement): synth **0.884**, game **0.890**, pop-with-vocals **0.850** — each lands on the better path automatically.

### 📏 Thresholds adapt to the material
Basic Pitch's onset head reports *confidence*, and confidence tracks attack sharpness: a chiptune peaks near 1.0, soft pads near 0.3. One fixed number can't serve both, so the threshold is derived from the distribution of onset peaks **on the track at hand** — the same "normalise to the signal's own scale" idea the percussion stage uses.

### 🪶 Lightweight by design
No TensorFlow. No per-note dynamic programming. Basic Pitch runs as a bundled **ONNX** graph; the bass moved from pYIN to the same ONNX model, roughly **halving conversion time** while *raising* accuracy.

---

## 📦 Install

```bash
pip install -r requirements.txt

# Basic Pitch ships TensorFlow + a legacy resampy pin it doesn't need here;
# only its bundled ONNX weights are used:
pip install --no-deps basic-pitch
```

**Requires Python 3.9+.** No `ffmpeg` binary needed — decoding goes through libsndfile, with PyAV and audioread as fallbacks. GPU optional (CPU separation ≈ real time; everything after is a fraction of that).

> ⚠️ If `resampy` is installed with `setuptools >= 81` it breaks on import and silently drags Demucs down to the HPSS fallback. MIDGenius warns loudly; `pip uninstall resampy` fixes it — nothing here uses it.

---

## 🚀 Usage

```bash
# Basic — auto-detects everything
python -m midgenius.cli song.mp3

# Choose output, snap to a 16th grid
python -m midgenius.cli song.mp3 -o song.mid --quantize 1/16

# Only bass + drums; also save separated audio
python -m midgenius.cli song.mp3 --only bass,drums --write-stems

# Fast path: no separation, whole mix through Basic Pitch
python -m midgenius.cli song.mp3 --no-separate

# Batch a folder
python -m midgenius.cli *.mp3 --outdir midi/
```

<details>
<summary><b>🎛️ Flags that matter</b></summary>

| Flag | Effect |
|---|---|
| `--quality fast\|good\|best` | Speed / accuracy preset |
| `--quantize 1/16` | Snap to a grid (`1/4 … 1/32`, triplets `1/8t`, `1/16t`) |
| `--quantize-strength 0.75` | A **blend**, not a snap — keeps the human push-and-pull |
| `--only` / `--skip` | Pick stems: `drums,bass,vocals,other` |
| `--min-stem-confidence 0.3` | How much stem detail the mix-primary merge trusts |
| `--no-mix-primary` | Use the older stem-union merge |
| `--pitched-from-mix-only` | Take pitched notes from the mix alone |
| `--dense` | **Capture far more notes** — for densely-written material (chiptune, arpeggios) where the default sounds thin. Roughly 2.5× the notes; costs precision on sparse material |
| `--max-polyphony N` | Cap simultaneous voices per stem |
| `--onset-threshold` | Lower ⇒ more notes (and more phantoms) |
| `--no-octave-deghost` | Keep octave harmonic ghosts (diagnostic) |
| `--tempo 128` | Force a fixed BPM instead of detecting |
| `--per-stem-midi` | One `.mid` per stem alongside the combined file |

</details>

<details>
<summary><b>🐍 As a library</b></summary>

```python
from midgenius import Config
from midgenius.pipeline import convert

cfg = Config()
cfg.quantize = "1/16"
cfg.stems["bass"].min_midi = 28          # per-instrument pitch gate
cfg.stems["other"].max_polyphony = 6

result = convert("song.mp3", "song.mid", cfg)
print(result.report())
print(result.n_notes, result.key, result.tempo_map.bpm)
```

</details>

---

## 📊 Measured accuracy

Numbers, not adjectives. Scored with `mir_eval` — a note counts only if its onset is within **50 ms** and its pitch within **50 cents**. `tools/regression.py` reproduces this.

| Track | Precision | Recall | **Pitched F1** | Drums F1 |
|---|:---:|:---:|:---:|:---:|
| **aria** — 8.5 min synth, audio rendered 1:1 from its MIDI | 66.0% | 61.0% | **63.4%** | 52.9% |
| **graze** — 3 min recording vs a separate human arrangement | 65.2% | 48.6% | **55.7%** | 49.7% |

> `aria` is the stricter test: the audio is rendered from the reference, so alignment is exact and **every** discrepancy is the transcriber's fault.

**How close it *sounds*** (`tools/audit.py` renders the MIDI back and compares to the source):

| Metric | Value |
|---|:---:|
| 🎼 Harmony (chroma) agreement | **~0.88–0.90** |
| 🎯 Onset F1 | **~90%** |
| ⏱️ Global time offset | **−12 ms** |
| 📈 Track coverage | **99%** |

> 🔬 **For scale:** multi-instrument transcription of real music is an *open research problem* — state-of-the-art systems live in the **60–70% F1** band, exactly where this sits. Note-level F1 understates the result: rhythm and harmony track the song far more closely than the number suggests, and most remaining loss is exact pitch assignment (largely octave ambiguity baked into the model).

<details>
<summary><b>Reproduce it</b></summary>

```bash
python -m pytest tests -q            # 61 unit tests
python tools/regression.py           # every reference pair, one table
python tools/audit.py song.mp3 out.mid
python tools/evaluate.py song.mp3 out.mid --reference truth.mid --offset 0
```
*(Copyrighted demo tracks aren't shipped — point the tools at your own `audio + reference`.)*

</details>

---

## 🥁 The drum engine

Drums are the one thing a pitch tracker must **never** touch, so they get their own path built on two ideas:

**1. Per-instrument band onset detection.** Instead of detecting onsets once and asking "what was that?", detection happens independently inside each instrument's band (spectral flux from one shared STFT). A kick and a hi-hat on the same sixteenth are two events in two bands — both survive.

**2. Band-ratio contrast, not absolute thresholds.** A snare's thump also lights up the kick band; what separates them is *shape* — a real kick has far more energy below 130 Hz than at 300–1200 Hz (a **16 dB** margin rejects bass/synth low end routed into the drum stem). A snare additionally must show broadband wire-noise above 1.8 kHz — the single test that stops every kick spawning a phantom snare.

Mutually-exclusive voices (hat vs ride, snare vs tom) are arbitrated on decay time and spectral flatness, so one cymbal is never reported twice. → General MIDI percussion on channel 10.

---

## 🎚️ Tuning a hard track

| Symptom | Fix |
|---|---|
| **Sounds thin / hollow** vs the record | `--dense` — the adaptive threshold keeps only the loudest notes on densely-written tracks |
| Missing quiet notes | lower `--onset-threshold` (try `0.3`) |
| Too many phantom notes | raise it, or `--max-polyphony 6` |
| Bass an octave off | `cfg.stems["bass"].min_midi = 28` |
| Machine-gun repeats | raise `--min-note-ms` |
| Timing feels loose | `--quantize 1/16 --quantize-strength 0.85` |
| Separation artefacts | `--quality best` (shift trick) |

<details>
<summary><b>Known limits</b></summary>

- **Octave errors** are the largest remaining pitch error. The envelope deghost removes the *pure* harmonics (a note whose activation is a scaled copy of the octave below); genuinely ambiguous octaves are kept, since no post-hoc signal separates them — Basic Pitch fires at harmonics in every output head. Fully resolving this needs retraining the network.
- **Note lengths** are less reliable than onsets (inherent to frame-threshold decoding).
- **Percussion on melodic material** — Demucs routes plucked/mallet attacks into the drum stem; some are genuinely percussive, some aren't.

</details>

---

## 🗂️ Project layout

```
midgenius/
  audio.py        loading, resampling, codec bandwidth probe & conditioning
  separation.py   Demucs → torchaudio HDemucs → HPSS fallback chain
  basicpitch.py   Basic Pitch ONNX inference (no TensorFlow)
  notes.py        note decoding, phantom / octave suppression, pitch bends
  mono.py         pYIN monophonic transcription
  drums.py        percussion transcription
  dynamics.py     velocity, expression, sustain pedal
  rhythm.py       tempo, beats, downbeats, quantisation, key detection
  midiout.py      MIDI assembly
  pipeline.py     orchestration
  cli.py          command line interface
tests/            61 tests against synthesised ground truth
tools/            benchmark, audit, and reference-scoring harnesses
```

---

## 🙏 Credits

Built on [**Basic Pitch**](https://github.com/spotify/basic-pitch) (Spotify, Apache-2.0), [**Demucs**](https://github.com/adefossez/demucs) (Meta, MIT), and [**librosa**](https://librosa.org).

<div align="center">
<sub>Licensed under <a href="LICENSE">MIT</a>. If MIDGenius saved you some transcription time, consider leaving a ⭐.</sub>
</div>
