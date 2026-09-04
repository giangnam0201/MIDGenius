# MIDGenius

Convert music (MP3, WAV, FLAC, OGG, M4A…) into a **multi-track MIDI file** —
separating the mix into stems first, then transcribing each instrument with the
method that actually suits it.

```bash
python -m midgenius.cli music.mp3
```

---

## Why this isn't just "run a pitch detector"

Naive audio→MIDI converters fail in five predictable ways. Each stage of this
pipeline exists to defuse a specific one.

| Failure mode | What goes wrong | What MIDGenius does |
|---|---|---|
| **Polyphony & dense mixes** | One model has to untangle a piano chord, a bass note and a kick sharing the same frequency bins | **Separate stems first** (Demucs), then transcribe each one independently — four easy problems instead of one impossible one |
| **"Phantom" notes** | Harmonics of real notes get reported as extra notes; dense mixes produce a wash of weak false positives | Harmonic/octave ghost filter, confidence gating, per-instrument pitch-range gates, and a polyphony cap |
| **Drums & percussion confusion** | A pitch tracker aimed at a drum kit emits a cloud of nonsense notes | A **dedicated percussion transcriber** — per-band onset detection with physically-grounded band-ratio classification. No pitch tracker ever touches the drum stem |
| **Loss of expressive detail** | Everything comes out at velocity 100, dead straight, no vibrato | Velocities measured from real attack energy, pitch-bend curves from the f0 contour, CC11 expression, sustain pedal, and a **variable tempo map** |
| **MP3 compression loss** | Codec quantisation noise reads as note activations; the missing top octave hides cymbals | Codec bandwidth probe + spectral conditioning, and the drum stage adapts its bands to whatever high end survived |

### The right tool per source

Using one model for everything is the second-biggest source of errors after
skipping separation. MIDGenius routes each stem to a different transcriber:

- **Bass, vocals** → **pYIN** monophonic f0 tracking. For a genuinely
  monophonic source this beats a polyphonic model by a wide margin, and the
  continuous contour is where vibrato, slides and scoops live.
- **Other / harmonic content** → **Basic Pitch** (Spotify's ICASSP-2022 model),
  run through ONNX Runtime.
- **Drums** → band-onset percussion classifier (below).

---

## Install

```bash
pip install -r requirements.txt

# Basic Pitch ships TensorFlow and a legacy resampy pin it does not need here;
# only its bundled ONNX weights are used:
pip install --no-deps basic-pitch
```

Requires Python 3.9+. No `ffmpeg` binary needed — MP3 decoding goes through
libsndfile, with PyAV and audioread as fallbacks.

GPU is optional. On CPU, separation runs at roughly real time; everything after
it is a small fraction of that.

> **Note:** if `resampy` is installed alongside `setuptools >= 81`, it breaks on
> import (`No module named 'pkg_resources'`) and takes Demucs down with it,
> silently degrading separation to the HPSS fallback. MIDGenius logs a loud
> warning if this happens. `pip uninstall resampy` fixes it; nothing here uses it.

---

## Usage

```bash
# Basic
python -m midgenius.cli song.mp3

# Choose output, snap to a 16th grid
python -m midgenius.cli song.mp3 -o song.mid --quantize 1/16

# Only bass and drums; also save the separated audio
python -m midgenius.cli song.mp3 --only bass,drums --write-stems

# Highest quality (4x shift trick, lower thresholds) — several times slower
python -m midgenius.cli song.mp3 --quality best

# Fast path: no separation at all, whole mix through Basic Pitch
python -m midgenius.cli song.mp3 --no-separate

# Batch
python -m midgenius.cli *.mp3 --outdir midi/
```

### Options that matter

| Flag | Effect |
|---|---|
| `--quality fast\|good\|best` | Speed/accuracy preset |
| `--quantize 1/16` | Snap to a grid (`1/4 … 1/32`, plus triplets `1/8t`, `1/16t`) |
| `--quantize-strength 0.75` | **A blend, not a snap.** 1.0 is machine-tight; 0.75 tightens timing while keeping the human push-and-pull |
| `--only` / `--skip` | Pick stems: `drums,bass,vocals,other` |
| `--onset-threshold` | Lower ⇒ more notes (and more phantoms) |
| `--max-polyphony N` | Cap simultaneous voices per stem |
| `--no-ghost-filter` | Keep harmonic phantom notes (diagnostic) |
| `--no-bends` | Skip pitch-bend expression |
| `--write-stems` | Save separated audio as `.wav` |
| `--per-stem-midi` | One `.mid` per stem in addition to the combined file |
| `--tempo 128` | Force a fixed BPM instead of detecting |

### As a library

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

---

## How the drum transcriber works

Drums are the one thing a pitch tracker must never be pointed at, so they get
their own path built on two ideas:

**1. Per-instrument band onset detection.** Rather than detecting onsets once
and then asking "what was that?", detection happens independently inside each
instrument's characteristic band, using spectral flux computed from a single
shared STFT. A kick and a hi-hat on the same sixteenth are two events in two
bands, and both survive — exclusive classification would keep only one.

**2. Band-ratio contrast, not absolute thresholds.** A snare's low-frequency
thump also lights up the kick band. What separates them is *shape*: a real kick
has far more energy below 130 Hz than at 300–1200 Hz, and a snare does not.
These ratios hold across genres and mastering styles in a way absolute energy
thresholds never do. A snare additionally has to show the broadband noise of its
wires above 1.8 kHz — that single test is what stops every kick from being
doubled by a phantom snare.

Mutually exclusive voices (hat vs. ride, snare vs. tom) that fire on the same
instant are then arbitrated on decay time and spectral flatness, so one cymbal
is never reported twice under two names.

Output is General MIDI percussion on channel 10: kick, snare, closed/open hat,
ride, crash, toms.

---

## Output

A type-1 MIDI file with:

- One track per instrument, each on its own MIDI channel (pitch bend is a
  per-channel control, so sharing would make instruments bend together)
- Drums on channel 10 with a General MIDI key map
- A **variable tempo map** — bar lines land on the music instead of drifting
- Detected key signature and time signature
- Velocities from measured attack energy, pitch bends with an explicit ±2
  semitone RPN, CC11 expression, CC64 sustain

Every run prints a report:

```
  source duration   184.0 s
  tempo             120.2 BPM (variable map)
  key               Am
  separation        demucs
  codec bandwidth   16.6 kHz
  skipped (silent)  vocals

  tracks:
    drums       1683 notes  pitch  36-51   vel  30-127 (avg  85)
    bass         241 notes  pitch  28-52   vel  28-127 (avg  91)
    other        892 notes  pitch  41-89   vel  28-127 (avg  96)
```

---

## Tuning for a difficult track

- **Missing quiet notes** → lower `--onset-threshold` (try `0.3`) and
  `--frame-threshold`
- **Too many phantom notes** → raise both, or set `--max-polyphony 6`
- **Bass an octave off** → tighten its range: `cfg.stems["bass"].min_midi = 28`
- **Machine-gun repeated notes** → raise `--min-note-ms`
- **Timing feels loose** → `--quantize 1/16 --quantize-strength 0.85`
- **Separation artefacts** → `--quality best` (uses the shift trick)

## Layout

```
midgenius/
  audio.py        loading, resampling, codec bandwidth probe & conditioning
  separation.py   Demucs -> torchaudio HDemucs -> HPSS fallback chain
  basicpitch.py   Basic Pitch ONNX inference (no TensorFlow)
  notes.py        note decoding, phantom suppression, pitch bends
  mono.py         pYIN monophonic transcription for bass and vocals
  drums.py        percussion transcription
  dynamics.py     velocity, expression, sustain pedal
  rhythm.py       tempo, beats, downbeats, quantisation, key detection
  midiout.py      MIDI assembly
  pipeline.py     orchestration
  cli.py          command line interface
tests/            46 tests against synthesised ground truth
```

```bash
python -m pytest tests -q
```

## Credits

Built on [Basic Pitch](https://github.com/spotify/basic-pitch) (Spotify,
Apache-2.0), [Demucs](https://github.com/adefossez/demucs) (Meta, MIT), and
[librosa](https://librosa.org).
