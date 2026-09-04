"""The MIDGenius pipeline: audio file in, MIDI file out.

Stage order matters, and each stage exists to defuse a specific failure mode:

    load  ->  condition  ->  separate  ->  rhythm  ->  transcribe per stem
          ->  clean  ->  quantise  ->  assemble MIDI

* condition   MP3/AAC artefacts: the codec's quantisation-noise bed reads as
              note activations, and its lowpass tells the drum stage how much
              cymbal evidence actually exists.
* separate    polyphony and dense mixes: four easy problems instead of one hard
              one, and the largest single source of phantom notes removed.
* rhythm      beat grid from the drum stem, so the output has a real tempo map.
* transcribe  the right tool per source: pYIN for monophonic bass and vocals,
              Basic Pitch for polyphonic material, a band-onset classifier for
              percussion. Never a pitch tracker on drums.
* clean       harmonic ghosts, stutter repeats, sub-perceptual fragments, and
              impossible polyphony.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from midgenius import audio as A
from midgenius import basicpitch, drums as D, dynamics, midiout, mono
from midgenius import notes as N
from midgenius import rhythm, separation
from midgenius.config import Config, StemConfig
from midgenius.notes import Note, Track

log = logging.getLogger("midgenius.pipeline")

# A stem quieter than this is silence (an instrumental track's vocal stem, a
# drumless intro's drum stem) and transcribing it only produces noise.
SILENCE_DBFS = -48.0


@dataclass
class Result:
    """What a conversion produced, for reporting and for tests."""

    midi_path: str
    tracks: List[Track] = field(default_factory=list)
    tempo_map: Optional[rhythm.TempoMap] = None
    key: str = ""
    duration: float = 0.0
    codec_cutoff: Optional[float] = None
    backend: str = ""
    stem_levels: Dict[str, float] = field(default_factory=dict)
    skipped: List[str] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)
    stem_paths: Dict[str, str] = field(default_factory=dict)
    per_stem_midi: Dict[str, str] = field(default_factory=dict)

    @property
    def n_notes(self) -> int:
        return sum(len(t.notes) for t in self.tracks)

    def report(self) -> str:
        lines = [
            "MIDGenius transcription report",
            "=" * 62,
            "  source duration   %.1f s" % self.duration,
            "  tempo             %.1f BPM%s" % (
                self.tempo_map.bpm if self.tempo_map else 0.0,
                " (variable map)" if self.tempo_map and self.tempo_map.is_variable else ""),
            "  key               %s" % (self.key or "unknown"),
            "  separation        %s" % self.backend,
        ]
        if self.codec_cutoff:
            lines.append("  codec bandwidth   %.1f kHz" % (self.codec_cutoff / 1000.0))
        if self.stem_levels:
            lines.append("  stem levels       " + "  ".join(
                "%s %.0f dB" % (k, v) for k, v in self.stem_levels.items()))
        if self.skipped:
            lines.append("  skipped (silent)  " + ", ".join(self.skipped))
        lines.append("")
        lines.append("  tracks:")
        lines.append(midiout.summarize(self.tracks))
        lines.append("")
        lines.append("  total notes       %d" % self.n_notes)
        if self.timings:
            lines.append("  timings           " + "  ".join(
                "%s %.1fs" % (k, v) for k, v in self.timings.items()))
        lines.append("  output            %s" % self.midi_path)
        return "\n".join(lines)


def convert(input_path: str, output_path: Optional[str] = None,
            config: Optional[Config] = None) -> Result:
    """Convert an audio file to MIDI. This is the entry point."""
    cfg = config or Config()
    t_start = time.time()
    timings: Dict[str, float] = {}

    if output_path is None:
        output_path = os.path.splitext(input_path)[0] + ".mid"
    out_dir = os.path.dirname(os.path.abspath(output_path)) or "."
    base = os.path.splitext(os.path.basename(output_path))[0]

    # ---- load ------------------------------------------------------------
    t0 = time.time()
    src = A.load(input_path)
    timings["load"] = time.time() - t0
    log.info("loaded %s: %.1fs, %d ch @ %d Hz", os.path.basename(input_path),
             src.duration, src.n_channels, src.sr)

    result = Result(midi_path=output_path, duration=src.duration)

    # ---- condition for lossy input ---------------------------------------
    cutoff = None
    if cfg.lossy_hf_cutoff_probe:
        cutoff = A.estimate_bandwidth(src.mono(), src.sr)
        result.codec_cutoff = cutoff
        if cutoff < src.sr / 2.0 * 0.9:
            log.info("lossy source detected: content ends at %.1f kHz", cutoff / 1000)
    if cfg.lossy_repair and cutoff and cutoff < src.sr / 2.0 * 0.9:
        t0 = time.time()
        src = A.repair_lossy(src, cutoff)
        timings["repair"] = time.time() - t0
    if cfg.normalize:
        src = A.normalize_peak(src)

    # ---- separate --------------------------------------------------------
    stems: Dict[str, A.Audio] = {}
    if cfg.separate:
        t0 = time.time()
        result.backend = separation.available_backend()
        stems = separation.separate(
            src, model_name=cfg.separation_model, device=cfg.device,
            shifts=cfg.separation_shifts, overlap=cfg.separation_overlap,
            segment=cfg.separation_segment, progress=cfg.verbose)
        timings["separate"] = time.time() - t0
        result.stem_levels = separation.stem_activity(stems)
        log.info("stems: %s", ", ".join(
            "%s %.0f dB" % (k, v) for k, v in result.stem_levels.items()))
    else:
        result.backend = "none (whole mix)"
        stems = {"mixdown": src}
        result.stem_levels = separation.stem_activity(stems)

    if cfg.write_stems:
        for name, st in stems.items():
            p = os.path.join(out_dir, "%s_%s.wav" % (base, name))
            A.save_wav(p, st)
            result.stem_paths[name] = p

    # ---- rhythm ----------------------------------------------------------
    t0 = time.time()
    drum_hint = stems["drums"].mono() if "drums" in stems else None
    if drum_hint is not None and _dbfs(drum_hint) < SILENCE_DBFS:
        drum_hint = None
    tempo_map = rhythm.analyze_rhythm(
        src.mono(), src.sr,
        fixed_tempo=cfg.fixed_tempo if not cfg.detect_tempo else None,
        beats_per_bar=cfg.time_signature[0], percussive_hint=drum_hint)
    if cfg.fixed_tempo and not cfg.detect_tempo:
        tempo_map.bpm = cfg.fixed_tempo
    result.tempo_map = tempo_map
    timings["rhythm"] = time.time() - t0

    # ---- transcribe each stem -------------------------------------------
    stem_cfgs = _stem_configs(cfg, stems)
    loudest = max(result.stem_levels.values()) if result.stem_levels else 0.0
    tracks: List[Track] = []
    for name, stem_cfg in stem_cfgs:
        stem = stems.get(name)
        if stem is None or not stem_cfg.enabled:
            continue
        # Pitched-from-mix-only: keep just the drum stem here and let the mix
        # pass supply every pitched note, skipping the artefact-prone pitched
        # stems entirely.
        if cfg.pitched_from_mix_only and stem_cfg.method != "drums":
            continue
        level = result.stem_levels.get(name, -99.0)
        if level < SILENCE_DBFS:
            log.info("skipping %r: silent (%.0f dBFS)", name, level)
            result.skipped.append(name)
            continue
        # A stem far below the loudest one is separation residue, not an
        # instrument. Transcribing it is bad enough; transcribing it with a
        # *monophonic* tracker - which is what the vocals stem gets - turns
        # smeared polyphonic bleed into a stream of confident wrong notes.
        if (stem_cfg.method == "mono" and name != "bass"
                and level < loudest - cfg.mono_stem_floor_db):
            log.info("skipping %r: %.0f dB below the loudest stem, so it is "
                     "bleed rather than a voice", name, loudest - level)
            result.skipped.append(name)
            continue

        t0 = time.time()
        track = transcribe_stem(stem, stem_cfg, cutoff)
        timings["transcribe:" + name] = time.time() - t0
        if track.notes:
            tracks.append(track)
            log.info("%-8s %4d notes (%.1fs)", name, len(track.notes),
                     timings["transcribe:" + name])
        else:
            log.info("%-8s produced no notes", name)

    # ---- transcribe the untouched mix as well ----------------------------
    # Separation is a means, not an end. Demucs *generates* each source rather
    # than masking, and on material outside its training domain it routes the
    # attack transients of pitched instruments (plucks, mallets, koto) into the
    # drum stem - so the pitched stems keep the sustain but lose the attacks
    # that onset detection depends on. Measured against reference
    # transcriptions, per-stem transcription of one such track scored 28% F1
    # where the untouched mix scored 55%.
    #
    # The mix keeps every attack and the full harmonic context; the stems find
    # notes the mix masks. Taking both and de-duplicating beats either alone on
    # average, and - more importantly - has no catastrophic case.
    if cfg.separate and cfg.transcribe_mix:
        t0 = time.time()
        mix_track = transcribe_stem(src, cfg.mixdown_stem, cutoff)
        timings["transcribe:mix"] = time.time() - t0
        log.info("%-8s %4d notes (%.1fs)", "mix", len(mix_track.notes),
                 timings["transcribe:mix"])
        if cfg.mix_primary and mix_track.notes:
            tracks = _merge_mix_primary(tracks, mix_track, cfg.min_stem_confidence)
        else:
            tracks = _merge_pitched_sources(tracks, mix_track)

    if not tracks:
        raise RuntimeError(
            "No notes were transcribed. The input may be silent, or entirely "
            "non-pitched material below the detection thresholds.")

    # ---- quantise --------------------------------------------------------
    if cfg.quantize and cfg.quantize != "off":
        for tr in tracks:
            if tr.is_drum:
                tr.notes = rhythm.snap_drums(tr.notes, tempo_map, cfg.quantize,
                                             strength=min(1.0, cfg.quantize_strength + 0.15))
            elif not cfg.quantize_drums_only:
                tr.notes = rhythm.quantize_notes(tr.notes, tempo_map, cfg.quantize,
                                                 strength=cfg.quantize_strength)
        # Quantisation can push same-pitch notes into each other.
        for tr in tracks:
            if not tr.is_drum:
                tr.notes = N.trim_overlaps(tr.notes)

    # ---- key -------------------------------------------------------------
    pitched = [n for tr in tracks if not tr.is_drum for n in tr.notes]
    if cfg.key_signature and pitched:
        _, _, result.key = rhythm.detect_key(pitched)

    # ---- write -----------------------------------------------------------
    for tr in tracks:
        midiout.thin_bends(tr.notes)

    if cfg.merge_to_single_track:
        tracks = _merge_tracks(tracks)

    midiout.write_midi(
        output_path, tracks, tempo_map,
        ticks_per_beat=cfg.ticks_per_beat,
        time_signature=cfg.time_signature,
        key=result.key if cfg.key_signature else None,
        title=os.path.basename(base),
        write_tempo_map=cfg.variable_tempo)

    if cfg.write_per_stem_midi:
        for tr in tracks:
            p = os.path.join(out_dir, "%s_%s.mid" % (base, tr.name))
            try:
                midiout.write_midi(p, [tr], tempo_map,
                                   ticks_per_beat=cfg.ticks_per_beat,
                                   time_signature=cfg.time_signature,
                                   title=tr.name,
                                   write_tempo_map=cfg.variable_tempo)
                result.per_stem_midi[tr.name] = p
            except ValueError:
                pass

    result.tracks = tracks
    timings["total"] = time.time() - t_start
    result.timings = timings
    return result


# --------------------------------------------------------------------------
# per stem transcription
# --------------------------------------------------------------------------

def transcribe_stem(stem: A.Audio, cfg: StemConfig,
                    codec_cutoff: Optional[float] = None) -> Track:
    """Route one stem to the transcriber suited to it."""
    y = stem.mono()
    sr = stem.sr

    if cfg.method == "drums":
        notes = _transcribe_drums(y, sr, cfg, codec_cutoff)
        return Track(name=cfg.name, notes=notes, program=cfg.program,
                     channel=cfg.channel, is_drum=True)

    if cfg.method == "mono":
        notes = _transcribe_mono(y, sr, cfg)
    else:
        notes = _transcribe_poly(y, sr, cfg)

    track = Track(name=cfg.name, notes=notes, program=cfg.program,
                  channel=cfg.channel, is_drum=False)

    if notes and (cfg.velocity_from_audio or cfg.expression_cc):
        band = _band_energy(y, sr, cfg)
        if cfg.velocity_from_audio:
            dynamics.assign_velocities(notes, band)
        if cfg.expression_cc:
            dynamics.attach_expression(notes, band)
    elif notes:
        dynamics.assign_velocities(notes, None)

    if cfg.sustain_pedal and notes:
        track.sustain = dynamics.detect_sustain(y, sr, notes)

    return track


def _band_energy(y: np.ndarray, sr: int, cfg: StemConfig):
    try:
        return dynamics.BandEnergy(y, sr, fmin_midi=max(21, cfg.min_midi - 2))
    except Exception as e:
        log.debug("band energy unavailable for %s: %r", cfg.name, e)
        return None


def _transcribe_drums(y: np.ndarray, sr: int, cfg: StemConfig,
                      codec_cutoff: Optional[float]) -> List[Note]:
    notes = D.transcribe_drums(y, sr, codec_cutoff=codec_cutoff)
    return D.collapse_flams(notes)


def _transcribe_mono(y: np.ndarray, sr: int, cfg: StemConfig) -> List[Note]:
    notes = mono.transcribe_mono(
        y, sr, fmin=cfg.mono_fmin, fmax=cfg.mono_fmax,
        min_voiced_prob=cfg.mono_voiced_prob, min_note_ms=cfg.min_note_ms,
        min_midi=cfg.min_midi, max_midi=cfg.max_midi,
        keep_bends=cfg.pitch_bend, vibrato_preserve=cfg.vibrato_preserve)
    notes = mono.fill_short_gaps(notes)
    notes = mono.fix_octave_jumps(notes)
    notes = N.enforce_min_duration(notes, cfg.min_note_ms)
    notes = N.trim_overlaps(notes)
    return notes


def _transcribe_poly(y: np.ndarray, sr: int, cfg: StemConfig) -> List[Note]:
    post = basicpitch.predict(y, sr)

    onset_thr, frame_thr = cfg.onset_threshold, cfg.frame_threshold
    if cfg.adaptive_threshold:
        onset_thr, frame_thr = N.adaptive_thresholds(post)
        log.info("%-8s adaptive thresholds: onset %.2f frame %.2f",
                 cfg.name, onset_thr, frame_thr)

    notes = N.decode_polyphonic(
        post,
        onset_threshold=onset_thr,
        frame_threshold=frame_thr,
        min_note_ms=cfg.min_note_ms,
        min_midi=cfg.min_midi, max_midi=cfg.max_midi,
        infer_onsets_flag=cfg.infer_onsets,
        melodia_trick=cfg.melodia_trick,
        energy_tolerance=cfg.energy_tolerance)

    notes = N.drop_low_confidence(notes, cfg.min_confidence)
    if cfg.octave_correction:
        notes = N.correct_octaves(post, notes,
                                  sub_ratio=cfg.octave_sub_ratio,
                                  onset_ratio=cfg.octave_onset_ratio)
    if cfg.harmonic_suppression:
        notes = N.suppress_harmonic_ghosts(notes, ratio=cfg.harmonic_ratio)
    notes = N.merge_repeats(notes)
    notes = N.remove_duplicates(notes)
    notes = N.enforce_min_duration(notes, cfg.min_note_ms)
    if cfg.max_polyphony:
        notes = N.limit_polyphony(notes, cfg.max_polyphony)
    notes = N.trim_overlaps(notes)

    if cfg.pitch_bend and notes:
        N.estimate_pitch_bends(post, notes)
    return notes


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _stem_configs(cfg: Config, stems: Dict[str, A.Audio]) -> List[Tuple[str, StemConfig]]:
    """Pair each available stem with its config, in a stable order."""
    out: List[Tuple[str, StemConfig]] = []
    for name in ("drums", "bass", "other", "vocals", "guitar", "piano"):
        if name in stems:
            sc = cfg.stems.get(name)
            if sc is None:
                sc = StemConfig(name=name, method="poly")
            out.append((name, sc))
    for name in stems:
        if name not in dict(out):
            sc = cfg.stems.get(name, cfg.mixdown_stem)
            out.append((name, StemConfig(**{**sc.__dict__, "name": name})))
    return out


def _covered(index: Dict[int, List[float]], pitch: int, start: float,
             tol: float) -> bool:
    starts = index.get(pitch)
    if not starts:
        return False
    i = int(np.searchsorted(starts, start))
    near = min((abs(starts[j] - start) for j in (i - 1, i)
               if 0 <= j < len(starts)), default=1e9)
    return near <= tol


def _merge_mix_primary(tracks: List[Track], mix_track: Track,
                       min_stem_conf: float, tol: float = 0.05) -> List[Track]:
    """Mix-primary union: the mix is the precise base, stems only add back.

    The stem union over-produces on clean material (Demucs artefacts read as
    notes) but recovers masked notes on dense mixes. Taking the mix as the base
    keeps its precision, and adding only *confident* stem notes the mix missed
    recovers the masked ones without importing the artefact phantoms - the best
    of both, rather than choosing per track.
    """
    drums = [t for t in tracks if t.is_drum]
    pitched = [t for t in tracks if not t.is_drum]

    mix_index: Dict[int, List[float]] = {}
    for n in mix_track.notes:
        mix_index.setdefault(n.pitch, []).append(n.start)
    for v in mix_index.values():
        v.sort()

    added = 0
    kept_stem: List[Note] = []
    for tr in pitched:
        for n in tr.notes:
            if n.confidence < min_stem_conf:
                continue
            if _covered(mix_index, n.pitch, n.start, tol):
                continue
            kept_stem.append(n)
            added += 1

    harmony = Track(name="harmony", program=(pitched[0].program if pitched else 0),
                    is_drum=False)
    harmony.notes = list(mix_track.notes) + kept_stem
    harmony.sort()
    harmony.notes = N.trim_overlaps(harmony.notes)
    log.info("mix-primary: %d mix notes + %d confident stem notes recovered",
             len(mix_track.notes), added)
    return drums + [harmony]


def _merge_pitched_sources(tracks: List[Track], mix_track: Track,
                           tol: float = 0.05) -> List[Track]:
    """Fold mix-derived notes into the stem tracks, without duplicating.

    The stems decide instrument routing (bass on its own track, drums on
    channel 10). The mix contributes whatever the stems lost. A mix note is
    kept only when no stem already has that pitch at that moment, so the two
    sources add coverage rather than doubling every note.
    """
    if not mix_track.notes:
        return tracks

    existing: Dict[int, List[float]] = {}
    for tr in tracks:
        if tr.is_drum:
            continue
        for n in tr.notes:
            existing.setdefault(n.pitch, []).append(n.start)
    for starts in existing.values():
        starts.sort()

    fresh: List[Note] = []
    for n in mix_track.notes:
        starts = existing.get(n.pitch)
        if starts:
            i = int(np.searchsorted(starts, n.start))
            near = min((abs(starts[j] - n.start)
                        for j in (i - 1, i) if 0 <= j < len(starts)),
                       default=1e9)
            if near <= tol:
                continue
        fresh.append(n)

    if not fresh:
        return tracks

    log.info("%-8s %4d notes added that the stems missed", "mix", len(fresh))
    harmony = next((t for t in tracks
                    if not t.is_drum and t.name in ("other", "harmony")), None)
    if harmony is not None:
        harmony.notes.extend(fresh)
        harmony.sort()
        harmony.notes = N.trim_overlaps(harmony.notes)
    else:
        mix_track.notes = fresh
        mix_track.name = "harmony"
        tracks.append(mix_track)
    return tracks


def _dbfs(y: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.asarray(y, np.float64) ** 2)))
    return 20.0 * np.log10(rms + 1e-12)


def _merge_tracks(tracks: List[Track]) -> List[Track]:
    """Collapse pitched tracks into one, keeping drums separate."""
    drum = [t for t in tracks if t.is_drum]
    pitched = [t for t in tracks if not t.is_drum]
    if len(pitched) <= 1:
        return tracks
    merged = Track(name="music", program=pitched[0].program, is_drum=False)
    for t in pitched:
        merged.notes.extend(t.notes)
        merged.sustain.extend(t.sustain)
    # Bends are per channel; merging voices makes them wrong.
    for n in merged.notes:
        n.bends = None
    merged.sort()
    merged.sustain.sort(key=lambda s: s[0])
    return [merged] + drum
