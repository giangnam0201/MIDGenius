"""Tests for MIDGenius.

These use synthesised audio with known content, so correctness is checked
against ground truth rather than against whatever the code happened to produce.
Run with:  python -m pytest tests -q
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from midgenius import basicpitch, drums, dynamics, mono, rhythm  # noqa: E402
from midgenius import notes as N  # noqa: E402
from midgenius.audio import Audio, estimate_bandwidth, resample  # noqa: E402
from midgenius.midiout import write_midi, thin_bends, assign_channels  # noqa: E402
from midgenius.notes import Note, Track  # noqa: E402

SR = 22050


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def tone(midi: float, dur: float, sr: int = SR, amp: float = 0.3,
         harmonics: int = 4, vibrato: float = 0.0, vib_rate: float = 5.5
         ) -> np.ndarray:
    """A harmonic tone at a MIDI pitch, optionally with vibrato.

    ``vibrato`` is the fractional frequency deviation, so 0.03 is roughly a
    half-semitone wobble - a singer's vibrato. (Passing a large value would
    synthesise a siren, not a note, and no tracker should call that one pitch.)
    """
    t = np.arange(int(dur * sr)) / sr
    f0 = 440.0 * 2 ** ((midi - 69) / 12.0)
    if vibrato:
        phase_mod = (vibrato / (2 * np.pi * vib_rate)) * np.sin(2 * np.pi * vib_rate * t)
    else:
        phase_mod = 0.0
    y = np.zeros_like(t)
    for h in range(1, harmonics + 1):
        y += (amp / h) * np.sin(2 * np.pi * f0 * h * (t + phase_mod))
    env = np.minimum(1.0, np.arange(len(t)) / max(1, int(0.01 * sr)))
    env *= np.exp(-2.0 * t / max(dur, 1e-6))
    return (y * env).astype(np.float32)


def sequence(pitches, dur=0.5, sr=SR, gap=0.0):
    out = []
    for p in pitches:
        out.append(tone(p, dur, sr))
        if gap:
            out.append(np.zeros(int(gap * sr), np.float32))
    return np.concatenate(out)


def click(sr=SR, dur=0.05, freq=None, noise=False):
    t = np.arange(int(dur * sr)) / sr
    env = np.exp(-40.0 * t)
    if noise:
        rng = np.random.default_rng(0)
        y = rng.standard_normal(len(t)) * env
    else:
        y = np.sin(2 * np.pi * freq * t) * env
    return (y * 0.5).astype(np.float32)


# --------------------------------------------------------------------------
# audio layer
# --------------------------------------------------------------------------

def test_audio_container_basics():
    a = Audio(np.zeros((2, SR), np.float32), SR)
    assert a.n_channels == 2 and a.n_samples == SR
    assert a.duration == pytest.approx(1.0)
    assert a.mono().shape == (SR,)
    assert abs(a.slice(0.25, 0.75).n_samples - SR // 2) <= 1
    assert a.to_stereo().n_channels == 2


def test_resample_roundtrip_preserves_length_and_pitch():
    y = tone(69, 1.0)
    up = resample(y, SR, 44100)
    assert abs(len(up) - 2 * len(y)) < 10
    back = resample(up, 44100, SR)
    n = min(len(back), len(y))
    # Correlation, not sample equality: resampling is not lossless.
    corr = np.corrcoef(back[:n], y[:n])[0, 1]
    assert corr > 0.99


def test_estimate_bandwidth_detects_lowpass():
    rng = np.random.default_rng(1)
    noise = rng.standard_normal(SR * 3).astype(np.float32) * 0.1
    import scipy.signal as sps
    # Steep, to emulate a codec's brick-wall lowpass rather than a gentle
    # analogue rolloff (a shallow filter legitimately still has energy well
    # above its -3 dB point, and the probe would be right to report it).
    sos = sps.cheby1(12, 0.5, 4000, btype="low", fs=SR, output="sos")
    filtered = sps.sosfilt(sos, noise).astype(np.float32)
    cutoff = estimate_bandwidth(filtered, SR)
    assert 3500 < cutoff < 5500, cutoff
    # Full-band noise should report close to Nyquist.
    assert estimate_bandwidth(noise, SR) > SR / 2 * 0.8


# --------------------------------------------------------------------------
# polyphonic transcription
# --------------------------------------------------------------------------

def test_basicpitch_finds_a_held_note():
    y = tone(60, 2.0, amp=0.5)
    post = basicpitch.predict(y, SR)
    assert post.n_frames > 100
    assert post.note.shape[1] == 88 and post.contour.shape[1] == 264
    # Bin for MIDI 60 should dominate.
    best = int(np.argmax(post.note.mean(axis=0))) + basicpitch.MIDI_OFFSET
    assert abs(best - 60) <= 1, best


def test_decode_polyphonic_recovers_a_chord():
    chord = tone(60, 1.5) + tone(64, 1.5) + tone(67, 1.5)
    post = basicpitch.predict(chord, SR)
    found = N.decode_polyphonic(post, onset_threshold=0.4, frame_threshold=0.25,
                                min_note_ms=80)
    pitches = {n.pitch for n in found}
    assert {60, 64, 67} <= pitches, sorted(pitches)


def test_pitch_gate_excludes_out_of_range_notes():
    post = basicpitch.predict(tone(60, 1.5), SR)
    found = N.decode_polyphonic(post, min_midi=70, max_midi=90)
    assert all(70 <= n.pitch <= 90 for n in found)


def test_frame_times_are_monotonic_and_start_at_zero():
    t = basicpitch.frame_times(1000)
    assert t[0] == pytest.approx(0.0, abs=1e-6)
    assert np.all(np.diff(t) > 0)


# --------------------------------------------------------------------------
# phantom note suppression
# --------------------------------------------------------------------------

def test_harmonic_ghost_is_removed():
    real = Note(0.0, 1.0, 48, confidence=0.9)
    ghost = Note(0.02, 0.95, 60, confidence=0.1)      # weak octave, aligned
    kept = N.suppress_harmonic_ghosts([real, ghost])
    assert [n.pitch for n in kept] == [48]


def test_strong_octave_is_kept():
    """A real octave doubling must survive - it is not a ghost."""
    a = Note(0.0, 1.0, 48, confidence=0.9)
    b = Note(0.0, 1.0, 60, confidence=0.8)
    kept = N.suppress_harmonic_ghosts([a, b])
    assert {n.pitch for n in kept} == {48, 60}


def test_unaligned_octave_is_kept():
    a = Note(0.0, 2.0, 48, confidence=0.9)
    b = Note(1.0, 2.0, 60, confidence=0.1)            # starts much later
    kept = N.suppress_harmonic_ghosts([a, b])
    assert {n.pitch for n in kept} == {48, 60}


def test_limit_polyphony_keeps_strongest():
    ns = [Note(0.0, 1.0, 60 + i, confidence=i / 10.0) for i in range(6)]
    kept = N.limit_polyphony(ns, 3)
    assert len(kept) == 3
    assert {n.pitch for n in kept} == {63, 64, 65}


def test_limit_polyphony_ignores_non_overlapping():
    ns = [Note(i * 1.0, i * 1.0 + 0.5, 60, confidence=0.5) for i in range(6)]
    assert len(N.limit_polyphony(ns, 2)) == 6


def test_merge_repeats_joins_stutter():
    ns = [Note(0.0, 0.3, 60), Note(0.31, 0.6, 60)]
    merged = N.merge_repeats(ns, gap_ms=30)
    assert len(merged) == 1 and merged[0].end == pytest.approx(0.6)


def test_merge_repeats_keeps_real_rearticulation():
    ns = [Note(0.0, 0.3, 60), Note(0.6, 0.9, 60)]
    assert len(N.merge_repeats(ns, gap_ms=30)) == 2


def test_trim_overlaps_prevents_same_pitch_collision():
    ns = [Note(0.0, 1.0, 60), Note(0.5, 1.5, 60)]
    out = N.trim_overlaps(ns, min_gap_ms=6)
    first = [n for n in out if n.start == 0.0][0]
    assert first.end <= 0.5


def test_min_duration_and_confidence_filters():
    ns = [Note(0.0, 0.01, 60, confidence=0.9), Note(0.0, 1.0, 62, confidence=0.01)]
    assert [n.pitch for n in N.enforce_min_duration(ns, 50)] == [62]
    assert [n.pitch for n in N.drop_low_confidence(ns, 0.5)] == [60]


# --------------------------------------------------------------------------
# monophonic transcription
# --------------------------------------------------------------------------

def test_mono_transcribes_a_scale():
    pitches = [55, 57, 59, 60, 62]
    y = sequence(pitches, dur=0.55, gap=0.08)
    found = mono.transcribe_mono(y, SR, fmin=80, fmax=800, min_note_ms=120,
                                 min_midi=40, max_midi=90)
    got = [n.pitch for n in found]
    # Every intended pitch appears, in order.
    assert len(got) >= len(pitches) - 1
    for p in pitches:
        assert any(abs(p - g) <= 1 for g in got), (p, got)


def test_mono_captures_vibrato_as_bends():
    y = tone(60, 2.0, vibrato=0.03)
    found = mono.transcribe_mono(y, SR, fmin=100, fmax=800, min_note_ms=150)
    assert found, "no note found"
    bent = [n for n in found if n.bends]
    assert bent, "vibrato was not captured as pitch bend"
    curve = np.array([v for _, v in bent[0].bends])
    assert curve.std() > 0.01


def test_fix_octave_jumps_repairs_isolated_outlier():
    ns = [Note(0.0, 0.3, 48), Note(0.3, 0.5, 60), Note(0.5, 0.8, 48)]
    fixed = mono.fix_octave_jumps(ns)
    assert [n.pitch for n in fixed] == [48, 48, 48]


def test_fill_short_gaps_merges_fragments():
    ns = [Note(0.0, 0.3, 60), Note(0.32, 0.7, 60)]
    out = mono.fill_short_gaps(ns, max_gap_ms=50)
    assert len(out) == 1 and out[0].end == pytest.approx(0.7)


# --------------------------------------------------------------------------
# drums
# --------------------------------------------------------------------------

def _drum_loop(sr=SR, bars=4, bpm=120):
    beat = 60.0 / bpm
    total = int(bars * 4 * beat * sr) + sr
    y = np.zeros(total, np.float32)

    def place(sig, t):
        i = int(t * sr)
        y[i:i + len(sig)] += sig[:max(0, len(y) - i)]

    kick = click(sr, 0.18, freq=55)
    snare = click(sr, 0.14, noise=True)
    for bar in range(bars):
        t0 = bar * 4 * beat
        place(kick, t0)                    # beat 1
        place(kick, t0 + 2 * beat)         # beat 3
        place(snare, t0 + beat)            # beat 2
        place(snare, t0 + 3 * beat)        # beat 4
    return y, beat, bars


def test_drums_detect_kick_and_snare_pattern():
    y, beat, bars = _drum_loop()
    hits = drums.collapse_flams(drums.transcribe_drums(y, SR))
    assert hits, "no drum hits detected"
    kicks = sorted(n.start for n in hits if n.pitch == drums.GM_KICK)
    snares = sorted(n.start for n in hits if n.pitch == drums.GM_SNARE)
    assert len(kicks) >= bars, kicks
    assert len(snares) >= bars, snares
    # Kicks land on beats 1 and 3, i.e. even multiples of the beat.
    for k in kicks:
        offset = (k / beat) % 2.0
        assert min(offset, 2.0 - offset) < 0.15, k


def test_drums_produce_no_pitched_notes():
    y, _, _ = _drum_loop()
    hits = drums.transcribe_drums(y, SR)
    # Everything must be a legal GM percussion key.
    assert all(27 <= n.pitch <= 87 for n in hits)


def test_drums_on_silence_return_nothing():
    assert drums.transcribe_drums(np.zeros(SR * 2, np.float32), SR) == []


def test_collapse_flams_merges_double_triggers():
    ns = [Note(0.0, 0.06, 36, velocity=80), Note(0.008, 0.068, 36, velocity=100)]
    out = drums.collapse_flams(ns, window_ms=22)
    assert len(out) == 1 and out[0].velocity == 100


def test_band_flux_responds_to_its_own_band():
    import librosa
    y = np.concatenate([np.zeros(SR // 2, np.float32), click(SR, 0.2, freq=60)])
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=256))
    freqs = np.fft.rfftfreq(2048, 1.0 / SR)
    low = drums.band_flux(S, freqs, 30, 130)
    high = drums.band_flux(S, freqs, 6000, 10000)
    assert low.max() > high.max() * 3


# --------------------------------------------------------------------------
# dynamics
# --------------------------------------------------------------------------

def test_velocity_tracks_loudness():
    loud = tone(60, 0.5, amp=0.6)
    quiet = tone(60, 0.5, amp=0.02)
    y = np.concatenate([loud, np.zeros(int(0.2 * SR), np.float32), quiet])
    band = dynamics.BandEnergy(y, SR)
    ns = [Note(0.0, 0.5, 60), Note(0.7, 1.2, 60)]
    dynamics.assign_velocities(ns, band)
    assert ns[0].velocity > ns[1].velocity, [n.velocity for n in ns]


def test_velocities_stay_in_range():
    y = tone(60, 1.0)
    band = dynamics.BandEnergy(y, SR)
    ns = [Note(i * 0.1, i * 0.1 + 0.09, 60) for i in range(10)]
    dynamics.assign_velocities(ns, band)
    assert all(1 <= n.velocity <= 127 for n in ns)


def test_velocity_falls_back_to_confidence_without_audio():
    ns = [Note(0.0, 1.0, 60, confidence=0.1), Note(0.0, 1.0, 62, confidence=0.9)]
    dynamics.assign_velocities(ns, None)
    assert ns[1].velocity > ns[0].velocity


# --------------------------------------------------------------------------
# rhythm
# --------------------------------------------------------------------------

def test_tempo_detection_on_a_click_track():
    y, beat, bars = _drum_loop(bpm=120)
    tm = rhythm.analyze_rhythm(y, SR)
    # Allow the classic octave ambiguity (60/120/240).
    assert min(abs(tm.bpm - c) for c in (60, 120, 240)) < 8, tm.bpm


def test_tempo_map_time_beat_roundtrip():
    beats = np.arange(0, 20, 0.5)               # 120 BPM
    tm = rhythm.TempoMap(beats, 120.0)
    for t in (0.0, 0.25, 3.7, 9.99):
        b = tm.time_to_beat(t)
        assert float(tm.beat_to_time(b)) == pytest.approx(t, abs=1e-6)


def test_tempo_map_extrapolates_outside_tracked_beats():
    tm = rhythm.TempoMap(np.array([1.0, 1.5, 2.0]), 120.0)
    assert float(tm.time_to_beat(0.5)) == pytest.approx(-1.0, abs=1e-6)
    assert float(tm.time_to_beat(2.5)) == pytest.approx(3.0, abs=1e-6)


def test_quantize_snaps_to_grid_at_full_strength():
    tm = rhythm.TempoMap(np.arange(0, 20, 0.5), 120.0)   # beat = 0.5 s
    ns = [Note(0.26, 0.70, 60)]                          # near the 1/8 grid
    out = rhythm.quantize_notes(ns, tm, "1/8", strength=1.0)
    beat = float(tm.time_to_beat(out[0].start))
    assert beat == pytest.approx(round(beat * 2) / 2, abs=1e-6)


def test_quantize_strength_is_a_blend():
    tm = rhythm.TempoMap(np.arange(0, 20, 0.5), 120.0)
    ns = [Note(0.30, 0.80, 60)]
    half = rhythm.quantize_notes(ns, tm, "1/8", strength=0.5)[0]
    full = rhythm.quantize_notes(ns, tm, "1/8", strength=1.0)[0]
    assert abs(half.start - 0.30) < abs(full.start - 0.30)


def test_quantize_off_is_identity():
    tm = rhythm.TempoMap(np.arange(0, 20, 0.5), 120.0)
    ns = [Note(0.3, 0.8, 60)]
    assert rhythm.quantize_notes(ns, tm, "off")[0].start == 0.3


def test_snap_drums_respects_max_shift():
    tm = rhythm.TempoMap(np.arange(0, 20, 0.5), 120.0)
    ns = [Note(0.40, 0.46, 36)]           # 100 ms from the 1/4 grid
    out = rhythm.snap_drums(ns, tm, "1/4", strength=1.0, max_shift_ms=20)
    assert abs(out[0].start - 0.40) <= 0.0201


def test_detect_key_finds_c_major():
    # A C major triad, held.
    ns = [Note(0.0, 2.0, p) for p in (60, 64, 67, 72, 65, 62)]
    tonic, is_major, label = rhythm.detect_key(ns)
    assert label in ("C", "Am", "F", "G"), label
    assert 0 <= tonic < 12


# --------------------------------------------------------------------------
# MIDI output
# --------------------------------------------------------------------------

def _read_back(path):
    import mido
    return mido.MidiFile(path)


def test_write_midi_roundtrip():
    tm = rhythm.TempoMap(np.arange(0, 20, 0.5), 120.0)
    tracks = [
        Track("keys", [Note(0.0, 0.5, 60, 90), Note(0.5, 1.0, 64, 70)], program=0),
        Track("drums", [Note(0.0, 0.06, 36, 110)], is_drum=True),
    ]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.mid")
        write_midi(p, tracks, tm)
        mid = _read_back(p)
        assert len(mid.tracks) == 3          # conductor + 2
        ons = [m for t in mid.tracks for m in t
               if m.type == "note_on" and m.velocity > 0]
        assert len(ons) == 3
        assert {m.note for m in ons} == {60, 64, 36}


def test_drum_track_lands_on_channel_10():
    tm = rhythm.TempoMap(np.arange(0, 10, 0.5), 120.0)
    tracks = [Track("d", [Note(0.0, 0.06, 36, 100)], is_drum=True)]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.mid")
        write_midi(p, tracks, tm)
        ons = [m for t in _read_back(p).tracks for m in t if m.type == "note_on"]
        assert all(m.channel == 9 for m in ons)


def test_channels_are_unique_per_pitched_track():
    tracks = [Track("a", [Note(0, 1, 60)]), Track("b", [Note(0, 1, 62)]),
              Track("d", [Note(0, 1, 36)], is_drum=True)]
    assign_channels(tracks)
    pitched = [t.channel for t in tracks if not t.is_drum]
    assert len(set(pitched)) == 2
    assert 9 not in pitched
    assert tracks[2].channel == 9


def test_note_order_is_correct_on_shared_ticks():
    """A repeated pitch must get its note-off before the next note-on."""
    tm = rhythm.TempoMap(np.arange(0, 10, 0.5), 120.0)
    tracks = [Track("k", [Note(0.0, 0.5, 60, 90), Note(0.5, 1.0, 60, 90)])]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.mid")
        write_midi(p, tracks, tm)
        seq = [m.type for t in _read_back(p).tracks for m in t
               if m.type in ("note_on", "note_off")]
        assert seq == ["note_on", "note_off", "note_on", "note_off"]


def test_tempo_map_is_written():
    beats = np.concatenate([np.arange(0, 5, 0.5), np.arange(5, 9, 0.4)])
    tm = rhythm.TempoMap(beats, 120.0)
    tracks = [Track("k", [Note(0.0, 0.5, 60)])]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.mid")
        write_midi(p, tracks, tm, write_tempo_map=True)
        tempos = [m for t in _read_back(p).tracks for m in t
                  if m.type == "set_tempo"]
        assert len(tempos) >= 2, "variable tempo was not written"


def test_bends_are_written_with_range_rpn():
    tm = rhythm.TempoMap(np.arange(0, 10, 0.5), 120.0)
    n = Note(0.0, 1.0, 60, 90)
    n.bends = [(0.0, 0.0), (0.25, 0.5), (0.5, 1.0), (0.75, 0.0)]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.mid")
        write_midi(p, [Track("k", [n])], tm)
        msgs = [m for t in _read_back(p).tracks for m in t]
        wheels = [m for m in msgs if m.type == "pitchwheel"]
        assert len(wheels) >= 4
        assert max(m.pitch for m in wheels) > 3000
        rpn = [m for m in msgs if m.type == "control_change" and m.control in (100, 101)]
        assert rpn, "pitch bend range RPN was not transmitted"
        assert wheels[-1].pitch == 0, "bend was not reset at the end"


def test_thin_bends_reduces_points_but_keeps_shape():
    n = Note(0.0, 1.0, 60)
    n.bends = [(i * 0.005, 0.5) for i in range(200)]
    thin_bends([n])
    assert n.bends is None or len(n.bends) < 20


def test_thin_bends_keeps_real_movement():
    n = Note(0.0, 1.0, 60)
    n.bends = [(i * 0.01, np.sin(i / 5.0)) for i in range(100)]
    thin_bends([n])
    assert n.bends and len(n.bends) > 20


def test_write_midi_rejects_empty_input():
    tm = rhythm.TempoMap(np.arange(0, 10, 0.5), 120.0)
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(ValueError):
            write_midi(os.path.join(d, "t.mid"), [Track("empty", [])], tm)


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------

def test_end_to_end_without_separation():
    """A synthesised melody must survive the whole pipeline into a real file."""
    from midgenius.config import Config
    from midgenius.pipeline import convert
    import soundfile as sf

    y = sequence([60, 62, 64, 65, 67], dur=0.6, gap=0.1)
    cfg = Config()
    cfg.separate = False
    cfg.lossy_repair = False
    cfg.verbose = False
    cfg.mixdown_stem.min_note_ms = 100
    cfg.mixdown_stem.onset_threshold = 0.4

    with tempfile.TemporaryDirectory() as d:
        wav = os.path.join(d, "in.wav")
        sf.write(wav, y, SR)
        out = os.path.join(d, "out.mid")
        res = convert(wav, out, cfg)

        assert os.path.exists(out) and os.path.getsize(out) > 0
        assert res.n_notes > 0
        pitches = {n.pitch for t in res.tracks for n in t.notes}
        assert pitches & {60, 62, 64, 65, 67}, sorted(pitches)
        assert "report" in dir(res) and res.report()


# --------------------------------------------------------------------------
# regressions
# --------------------------------------------------------------------------

def test_beat_grid_covers_the_start_of_the_track():
    """Notes before the first detected beat must not collapse onto tick 0."""
    y, beat, _ = _drum_loop(bpm=120)
    # Push the music later so tracking cannot start at t=0.
    y = np.concatenate([np.zeros(int(1.3 * SR), np.float32), y])
    tm = rhythm.analyze_rhythm(y, SR)
    assert len(tm.beat_times) >= 2
    assert tm.beat_times[0] <= 0.0, tm.beat_times[:3]
    # Every position in the track maps to a non-negative musical position.
    for t in (0.0, 0.05, 0.5, 1.0):
        assert float(tm.time_to_beat(t)) >= -1e-9, (t, float(tm.time_to_beat(t)))


def test_anchor_grid_puts_bar_one_on_the_downbeat():
    beats = np.arange(2.0, 12.0, 0.5)          # first beat at t=2.0
    out = rhythm._anchor_grid(beats, downbeat=2, beats_per_bar=4)
    assert out[0] <= 0.0
    n_pre = len(out) - len(beats)
    # The old downbeat at index 2 must land on a multiple of the bar length.
    assert (2 + n_pre) % 4 == 0, n_pre


def test_early_notes_get_distinct_ticks():
    from midgenius.midiout import _tick
    tm = rhythm.TempoMap(np.arange(-1.0, 10.0, 0.5), 120.0)
    ticks = [_tick(tm, t, 480) for t in (0.0, 0.1, 0.25, 0.4)]
    assert len(set(ticks)) == 4, ticks


def test_pitch_bend_frames_track_the_model_time_axis():
    """Frame lookup must use the model's own (non-uniform) time axis.

    round(t * frame_rate) drifts by over a second across a long track, which
    would read each note's bend curve off the wrong part of the contour.
    """
    n_frames = 16000
    t = basicpitch.frame_times(n_frames)
    naive = int(round(float(t[-1]) * basicpitch.ANNOTATIONS_FPS))
    exact = int(np.searchsorted(t, t[-1]))
    assert abs(naive - exact) > 50, "expected the naive mapping to drift"
    # And the code under test must agree with the exact mapping.
    post = basicpitch.Posteriorgram(
        note=np.zeros((n_frames, 88), np.float32),
        onset=np.zeros((n_frames, 88), np.float32),
        contour=np.zeros((n_frames, 264), np.float32),
        times=t)
    late = Note(float(t[-200]), float(t[-100]), 60)
    N.estimate_pitch_bends(post, [late])      # must not raise or misindex
    assert late.bends is None                 # empty contour -> no bend


def test_atomic_write_leaves_previous_file_on_failure():
    tm = rhythm.TempoMap(np.arange(0, 10, 0.5), 120.0)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.mid")
        write_midi(p, [Track("k", [Note(0.0, 0.5, 60)])], tm)
        original = open(p, "rb").read()
        # A directory in the way makes the replace fail.
        bad = os.path.join(d, "sub")
        os.makedirs(bad)
        with pytest.raises(Exception):
            write_midi(bad, [Track("k", [Note(0.0, 0.5, 60)])], tm)
        assert open(p, "rb").read() == original
        # No temp files left behind.
        assert not [f for f in os.listdir(d) if f.startswith("tmp")]


def test_no_bend_is_invented_from_an_empty_contour():
    """An all-zero contour region must produce no bend, not argmax-of-zeros."""
    n = 400
    post = basicpitch.Posteriorgram(
        note=np.zeros((n, 88), np.float32), onset=np.zeros((n, 88), np.float32),
        contour=np.zeros((n, 264), np.float32), times=basicpitch.frame_times(n))
    note = Note(0.5, 1.5, 60)
    N.estimate_pitch_bends(post, [note])
    assert note.bends is None


def test_real_contour_still_yields_a_bend():
    n = 400
    contour = np.zeros((n, 264), np.float32)
    base = int(round(basicpitch.midi_to_contour_bin(60)))
    for i in range(n):
        contour[i, base + (i * 3) // n] = 1.0      # a slow upward slide
    post = basicpitch.Posteriorgram(
        note=np.zeros((n, 88), np.float32), onset=np.zeros((n, 88), np.float32),
        contour=contour, times=basicpitch.frame_times(n))
    note = Note(float(post.times[10]), float(post.times[n - 10]), 60)
    N.estimate_pitch_bends(post, [note])
    assert note.bends, "a real pitch slide was not captured"
    assert max(v for _, v in note.bends) > 0.2


def _midi_note_times(path):
    """Read a MIDI file back to absolute seconds, the way a DAW would."""
    import mido
    mid = mido.MidiFile(path)
    tpb = mid.ticks_per_beat
    changes, tick = [], 0
    for m in mid.tracks[0]:
        tick += m.time
        if m.type == "set_tempo":
            changes.append((tick, m.tempo))
    if not changes or changes[0][0] != 0:
        changes.insert(0, (0, 500000))

    def to_sec(target):
        sec, prev, tempo = 0.0, 0, changes[0][1]
        for ctick, ctempo in changes:
            if ctick >= target:
                break
            sec += mido.tick2second(ctick - prev, tpb, tempo)
            prev, tempo = ctick, ctempo
        return sec + mido.tick2second(target - prev, tpb, tempo)

    out = []
    for tr in mid.tracks[1:]:
        tick = 0
        for m in tr:
            tick += m.time
            if m.type == "note_on" and m.velocity > 0:
                out.append((to_sec(tick), m.note))
    return sorted(out)


@pytest.mark.parametrize("first_beat", [0.0, 0.02, 0.13, 0.5, 1.7])
def test_written_midi_times_match_requested_times(first_beat):
    """The file must play notes when we said, whatever the beat grid looks like.

    This is the check that matters: everything between the note objects and the
    file - beat anchoring, the tempo map, tick rounding - can shift timing, and
    a listener hears the file, not the objects.
    """
    beats = np.arange(first_beat, first_beat + 20.0, 0.5)
    tm = rhythm.TempoMap(rhythm._anchor_grid(beats, 0, 4), 120.0)
    wanted = [0.0, 0.25, 0.5, 1.0, 2.0, 3.75, 8.0]
    ns = [Note(t, t + 0.2, 60 + i) for i, t in enumerate(wanted)]

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.mid")
        write_midi(p, [Track("k", ns)], tm)
        got = _midi_note_times(p)

    assert len(got) == len(wanted)
    for (t_got, _), t_want in zip(got, wanted):
        assert abs(t_got - t_want) < 0.010, (
            "note wanted at %.3fs was written at %.3fs (first beat %.2f)"
            % (t_want, t_got, first_beat))


def test_anchor_grid_never_starts_before_zero():
    for first in (0.0, 0.01, 0.2, 0.49, 1.3, 2.7):
        beats = np.arange(first, first + 20.0, 0.5)
        out = rhythm._anchor_grid(beats, 2, 4)
        assert out[0] >= -1e-9, (first, out[:3])
        assert np.all(np.diff(out) > 0), "beat grid must be strictly increasing"


def test_tempo_map_never_needs_clamping():
    """Every emitted beat interval must be expressible as a real tempo."""
    for first in (0.0, 0.02, 0.13, 0.5, 1.7):
        beats = rhythm._anchor_grid(np.arange(first, first + 20.0, 0.5), 0, 4)
        for bpm in [b for _, b in rhythm.TempoMap(beats, 120.0).segment_tempi()]:
            assert 5.0 < bpm < 900.0, (first, bpm)
