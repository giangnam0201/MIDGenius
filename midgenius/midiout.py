"""MIDI file assembly.

Everything the earlier stages recovered - notes, velocities, bends, expression,
pedal, tempo, key - has to survive the trip into a .mid file, and a few details
decide whether a DAW reads it the way we meant:

* Note positions are written in *beats*, converted to ticks through the tempo
  map, so bar lines land on the music instead of drifting.
* Each pitched stem gets its own MIDI channel, because pitch bend is a
  per-channel control. Two instruments sharing a channel would bend together.
* Bends are written as 14-bit values against an explicit +/-2 semitone range,
  and every channel is reset to centre at the end so a bend cannot leak into
  whatever plays next.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from midgenius.notes import Note, Track
from midgenius.rhythm import TempoMap

log = logging.getLogger("midgenius.midiout")

DRUM_CHANNEL = 9          # GM channel 10, zero-indexed
PITCH_BEND_RANGE = 2.0    # semitones, matching the RPN we transmit
BEND_CENTER = 8192
MAX_BEND = 8191


def _tick(tempo_map: TempoMap, t: float, tpb: int) -> int:
    return max(0, int(round(float(tempo_map.time_to_beat(t)) * tpb)))


def assign_channels(tracks: Sequence[Track]) -> None:
    """Give every track a channel, in place, keeping channel 10 for drums."""
    used = {t.channel for t in tracks if t.channel is not None}
    free = [c for c in range(16) if c != DRUM_CHANNEL and c not in used]
    for t in tracks:
        if t.channel is not None:
            continue
        if t.is_drum:
            t.channel = DRUM_CHANNEL
        elif free:
            t.channel = free.pop(0)
        else:
            # More than 15 pitched tracks: reuse, and drop bends on the
            # duplicates so they cannot detune each other.
            t.channel = 0
            for n in t.notes:
                n.bends = None
            log.warning("out of MIDI channels - track %r shares channel 1", t.name)


def write_midi(
    path: str,
    tracks: Sequence[Track],
    tempo_map: TempoMap,
    ticks_per_beat: int = 480,
    time_signature: Tuple[int, int] = (4, 4),
    key: Optional[str] = None,
    title: Optional[str] = None,
    write_tempo_map: bool = True,
) -> None:
    """Write a type-1 (multi-track) MIDI file."""
    import mido

    tracks = [t for t in tracks if t.notes]
    if not tracks:
        raise ValueError("nothing to write: all tracks are empty")

    assign_channels(tracks)

    mid = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)

    # --- conductor track: tempo, time signature, key ----------------------
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("track_name", name=title or "MIDGenius", time=0))
    meta.append(mido.MetaMessage(
        "time_signature", numerator=time_signature[0],
        denominator=time_signature[1], time=0))
    if key:
        try:
            meta.append(mido.MetaMessage("key_signature", key=key, time=0))
        except Exception:
            log.debug("mido rejected key signature %r", key)

    tempo_events: List[Tuple[int, float]] = []
    if write_tempo_map and tempo_map.is_variable:
        for t_sec, bpm in tempo_map.segment_tempi():
            tempo_events.append((_tick(tempo_map, t_sec, ticks_per_beat), bpm))
    if not tempo_events:
        tempo_events = [(0, float(tempo_map.bpm or 120.0))]
    if tempo_events[0][0] != 0:
        tempo_events.insert(0, (0, tempo_events[0][1]))

    prev = 0
    for tick, bpm in tempo_events:
        meta.append(mido.MetaMessage(
            "set_tempo", tempo=mido.bpm2tempo(float(bpm)), time=max(0, tick - prev)))
        prev = tick
    meta.append(mido.MetaMessage("end_of_track", time=1))
    mid.tracks.append(meta)

    # --- one MIDI track per instrument ------------------------------------
    for track in tracks:
        mid.tracks.append(_build_track(mido, track, tempo_map, ticks_per_beat))

    _save_atomic(mid, path)
    log.info("wrote %s (%d tracks, %d notes)", path, len(tracks),
             sum(len(t.notes) for t in tracks))


def _save_atomic(mid, path: str) -> None:
    """Write via a temp file, then replace.

    Transcribing a full track costs minutes; losing all of it because the
    destination was open in a DAW or a player is not an acceptable ending. A
    failed write also leaves any previous file intact rather than truncated.
    """
    import os
    import tempfile

    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)

    fd, tmp = tempfile.mkstemp(suffix=".mid", dir=directory)
    os.close(fd)
    try:
        mid.save(tmp)
        os.replace(tmp, path)
    except PermissionError as e:
        _cleanup(tmp)
        raise PermissionError(
            "cannot write %s - it is open in another program (a DAW or player). "
            "Close it, or pass a different -o path." % path) from e
    except Exception:
        _cleanup(tmp)
        raise


def _cleanup(tmp: str) -> None:
    import os
    try:
        os.remove(tmp)
    except OSError:
        pass


def _build_track(mido, track: Track, tempo_map: TempoMap, tpb: int):
    """Assemble one MIDI track: program, notes, bends, CC."""
    ch = track.channel if track.channel is not None else 0
    events: List[Tuple[int, int, object]] = []   # (tick, priority, message)

    def add(tick: int, priority: int, msg) -> None:
        events.append((max(0, tick), priority, msg))

    mt = mido.MidiTrack()
    mt.append(mido.MetaMessage("track_name", name=track.name[:32], time=0))

    if not track.is_drum:
        mt.append(mido.Message("program_change", channel=ch,
                               program=int(np.clip(track.program, 0, 127)), time=0))
        # Declare the pitch-bend range explicitly (RPN 0). Without this a
        # synth's default range is anyone's guess and every bend is wrong.
        if any(n.bends for n in track.notes):
            for control, value in ((101, 0), (100, 0), (6, int(PITCH_BEND_RANGE)), (38, 0)):
                mt.append(mido.Message("control_change", channel=ch,
                                       control=control, value=value, time=0))

    notes = sorted(track.notes, key=lambda n: (n.start, n.pitch))
    for n in notes:
        t_on = _tick(tempo_map, n.start, tpb)
        t_off = max(t_on + 1, _tick(tempo_map, n.end, tpb))
        pitch = int(np.clip(n.pitch, 0, 127))
        vel = int(np.clip(n.velocity, 1, 127))

        # Priority orders events that land on the same tick: bends and control
        # changes must be in place *before* the note-on they belong to, and
        # note-offs must precede note-ons so a repeated pitch retriggers.
        if n.bends and not track.is_drum:
            for bt, semis in n.bends:
                add(_tick(tempo_map, bt, tpb), 1,
                    mido.Message("pitchwheel", channel=ch,
                                 pitch=_bend_value(semis), time=0))
        if n.expression and not track.is_drum:
            for et, level in n.expression:
                add(_tick(tempo_map, et, tpb), 1,
                    mido.Message("control_change", channel=ch, control=11,
                                 value=int(np.clip(round(level * 127), 0, 127)),
                                 time=0))

        add(t_on, 2, mido.Message("note_on", channel=ch, note=pitch,
                                  velocity=vel, time=0))
        add(t_off, 0, mido.Message("note_off", channel=ch, note=pitch,
                                   velocity=0, time=0))

    for t_sec, down in track.sustain:
        add(_tick(tempo_map, t_sec, tpb), 1,
            mido.Message("control_change", channel=ch, control=64,
                         value=127 if down else 0, time=0))

    if any(n.bends for n in notes) and not track.is_drum:
        last = max((_tick(tempo_map, n.end, tpb) for n in notes), default=0)
        add(last + 2, 3, mido.Message("pitchwheel", channel=ch, pitch=0, time=0))

    events.sort(key=lambda e: (e[0], e[1]))
    prev = 0
    for tick, _, msg in events:
        msg.time = max(0, tick - prev)
        prev = tick
        mt.append(msg)

    mt.append(mido.MetaMessage("end_of_track", time=1))
    return mt


def _bend_value(semitones: float) -> int:
    """Semitone offset -> 14-bit pitch wheel value in mido's -8192..8191 space."""
    frac = float(np.clip(semitones / PITCH_BEND_RANGE, -1.0, 1.0))
    return int(np.clip(round(frac * MAX_BEND), -BEND_CENTER, MAX_BEND))


def thin_bends(notes: Iterable[Note], min_interval_ms: float = 18.0,
               min_delta: float = 0.02) -> None:
    """Drop redundant bend points, in place.

    Basic Pitch and pYIN both emit a value every ~12 ms. Writing all of them
    produces tens of thousands of messages that some hardware and older DAWs
    choke on, for a curve the ear cannot distinguish from a thinned one.

    A point is kept only if it says something new: it must differ from the last
    kept value *and* be far enough after it. A held bend therefore collapses to
    its first point, since MIDI holds the last value sent, while a vibrato keeps
    every peak and trough.
    """
    step = min_interval_ms / 1000.0
    for n in notes:
        if not n.bends:
            continue
        kept: List[Tuple[float, float]] = [n.bends[0]]
        last_t, last_v = n.bends[0]
        for t, v in n.bends[1:]:
            if (t - last_t) < step or abs(v - last_v) < min_delta:
                continue
            kept.append((t, v))
            last_t, last_v = t, v
        # Always land on the final value so the curve ends where it should.
        if n.bends[-1] != kept[-1]:
            kept.append(n.bends[-1])
        n.bends = kept if len(kept) > 1 else None


def summarize(tracks: Sequence[Track]) -> str:
    lines = []
    for t in tracks:
        if not t.notes:
            continue
        pitches = [n.pitch for n in t.notes]
        vels = [n.velocity for n in t.notes]
        span = max(n.end for n in t.notes) - min(n.start for n in t.notes)
        bends = sum(1 for n in t.notes if n.bends)
        lines.append(
            "  %-10s %5d notes  pitch %3d-%-3d  vel %3d-%-3d (avg %3d)  "
            "%5.1fs  %d with bends" % (
                t.name, len(t.notes), min(pitches), max(pitches),
                min(vels), max(vels), int(np.mean(vels)), span, bends))
    return "\n".join(lines)
