from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import torch
from note_seq import midi_io
from note_seq.protobuf.music_pb2 import NoteSequence
import yaml

from .mt3_midi_tokenizer import MT3MIDITokenizer


TechniqueId = int


@dataclass
class MIDIProcessorConfig:
    start_time: float = 5.0
    window_seconds: float = 10.0
    time_step_seconds: float = 0.01
    cc_frame_hop: float = 0.04  # 25 Hz
    technique_lead: float = 0.01  # 10 ms lead before note onset
    playable_note_min_pitch: int = 55
    midi_note_min_pitch: Optional[int] = None
    midi_note_max_pitch: Optional[int] = None
    default_cc1_value: Optional[int] = None
    default_technique_id: Optional[int] = None


def has_cc1_information(sequence: NoteSequence) -> bool:
    return any(cc.control_number == 1 for cc in sequence.control_changes)


def has_low_pitch_annotation(
    sequence: NoteSequence,
    playable_note_min_pitch: int = 55,
) -> bool:
    return any(int(note.pitch) < int(playable_note_min_pitch) for note in sequence.notes)


def load_technique_map(config_path: Path) -> Dict[int, TechniqueId]:
    """
    Load pitch->technique id mapping from ks_config.yaml.
    Mapping: sustain(36)=1, tremolo(37)=2, trill_major(38)=3, trill_minor(39)=4,
    staccato(40)=5, spiccato(41)=6, ricochet(42)=7, pizzicato(43)=8,
    harmonic(44)=9, legato_bow(48)=10, legato_slur(49)=11, legato_portamento(50)=12.
    style_normal to style_sultasto are ignored.
    """
    with config_path.open("r") as f:
        data = yaml.safe_load(f)
    order = [
        ("sustain", 36),
        ("tremolo", 37),
        ("trill_major", 38),
        ("trill_minor", 39),
        ("staccato", 40),
        ("spiccato", 41),
        ("ricochet", 42),
        ("pizzicato", 43),
        ("harmonic", 44),
        ("legato_bow", 48),
        ("legato_slur", 49),
        ("legato_portamento", 50),
    ]
    tech_map: Dict[int, TechniqueId] = {}
    for idx, (_name, midi_num) in enumerate(order, start=1):
        if data.get(_name) is not None:
            tech_map[int(data[_name])] = idx
    return tech_map


def extract_cc1_frames(sequence: NoteSequence, cfg: MIDIProcessorConfig) -> torch.Tensor:
    """
    Extract CC1, normalize to [0,1], sample with ZOH at 25 Hz over the window.
    Returns tensor of shape (F,1).
    """
    start = cfg.start_time
    end = start + cfg.window_seconds
    hop = cfg.cc_frame_hop
    num_frames = int(math.ceil(cfg.window_seconds / hop))

    # collect cc1 events
    cc_events = [(cc.time, cc.control_value) for cc in sequence.control_changes if cc.control_number == 1]
    cc_events.sort(key=lambda x: x[0])

    # When eval MIDI contains no CC1 at all, keep the condition active with a
    # musically neutral default instead of collapsing to the null branch.
    if not cc_events and cfg.default_cc1_value is not None:
        current_val = int(max(0, min(int(cfg.default_cc1_value), 127)))
    else:
        current_val = 0
        for t, v in cc_events:
            if t <= start:
                current_val = v
            else:
                break

    frames = np.zeros((num_frames, 1), dtype=np.float32)
    event_idx = 0
    for i in range(num_frames):
        t = start + i * hop
        while event_idx < len(cc_events) and cc_events[event_idx][0] <= t and cc_events[event_idx][0] < end:
            current_val = cc_events[event_idx][1]
            event_idx += 1
        frames[i, 0] = float(current_val) / 127.0
    return torch.from_numpy(frames)


def match_techniques(
    sequence: NoteSequence,
    tech_map: Dict[int, TechniqueId],
    start_time: float,
    window_seconds: float,
    lead: float,
    playable_note_min_pitch: int = 55,
    default_technique_id: Optional[int] = None,
) -> Dict[Tuple[float, int], TechniqueId]:
    """
    Match technique events to note onsets within the window.
    Returns dict keyed by (note_onset, pitch) -> technique_id.
    """
    end_time = start_time + window_seconds
    has_any_annotation = has_low_pitch_annotation(
        sequence, playable_note_min_pitch=playable_note_min_pitch
    )
    technique_events = [note for note in sequence.notes if note.pitch in tech_map]
    note_events = [
        note for note in sequence.notes if int(note.pitch) >= int(playable_note_min_pitch)
    ]

    if (not has_any_annotation) and default_technique_id is not None:
        tech_for_note: Dict[Tuple[float, int], TechniqueId] = {}
        for note in note_events:
            if note.start_time >= end_time or note.end_time <= start_time:
                continue
            tech_for_note[(float(note.start_time), int(note.pitch))] = int(default_technique_id)
        return tech_for_note

    # index technique events by time for quick lookup
    tech_times = [(n.start_time, tech_map[n.pitch]) for n in technique_events if start_time - lead <= n.start_time <= end_time]
    tech_times.sort(key=lambda x: x[0])

    tech_for_note: Dict[Tuple[float, int], TechniqueId] = {}
    for note in note_events:
        if note.start_time >= end_time or note.end_time <= start_time:
            continue
        onset = note.start_time
        # desired technique onset ~ onset - lead
        target_min = max(start_time, onset - 2 * lead)
        target_max = onset + lead
        best = None
        for t, tid in tech_times:
            if t < target_min:
                continue
            if t > target_max:
                break
            best = tid
            break
        if best is not None:
            tech_for_note[(onset, int(note.pitch))] = best
    return tech_for_note


def process_midi(
    midi_path: str | Path,
    tech_map: Dict[int, TechniqueId],
    cfg: Optional[MIDIProcessorConfig] = None,
    sequence: Optional[NoteSequence] = None,
) -> Tuple[List[int], List[int], List[int], List[int], torch.Tensor, List[int]]:
    """
    Process a MIDI into (note_tokens, technique_seq, velocity_seq, pos_midi, cc_frames, tech_onset_seq).
    technique_seq is aligned to note_tokens (same length), with technique ids
    placed at note-on token positions; zeros elsewhere.
    tech_onset_seq is one technique id per note onset (in time order), for use when
    MIDI roll conditioning is enabled and technique context is onset-only.
    velocity_seq is similar, containing velocity bins (1-16) at note-on positions.
    cc_frames has shape (F,1) at 25 Hz over the window.
    """
    cfg = cfg or MIDIProcessorConfig()

    if sequence is None:
        midi_path = Path(midi_path)
        sequence = midi_io.midi_file_to_note_sequence(str(midi_path))

    tokenizer = MT3MIDITokenizer(window_seconds=cfg.window_seconds, time_step_seconds=cfg.time_step_seconds)

    # Prepare technique matches and CC
    tech_matches = match_techniques(
        sequence,
        tech_map,
        cfg.start_time,
        cfg.window_seconds,
        cfg.technique_lead,
        playable_note_min_pitch=cfg.playable_note_min_pitch,
        default_technique_id=cfg.default_technique_id,
    )
    cc_frames = extract_cc1_frames(sequence, cfg)

    # Tokenize notes (filter out technique pitches)
    # Optimized: pass ignore_pitches to _build_events instead of copying sequence
    ignore_pitches = set(tech_map.keys())
    tied_notes, events = tokenizer._build_events(
        sequence,
        cfg.start_time,
        ignore_pitches=ignore_pitches,
        min_pitch=cfg.midi_note_min_pitch
        if cfg.midi_note_min_pitch is not None
        else cfg.playable_note_min_pitch,
        max_pitch=cfg.midi_note_max_pitch,
    )

    # Rebuild tokens manually to keep alignment with filtered events
    token_list: List[int] = []
    technique_seq: List[int] = []
    velocity_seq: List[int] = []
    pos_midi: List[int] = []
    tech_onset_seq: List[int] = []  # One technique per note onset (for midi-roll mode)

    def _frame_pos_from_steps(t_idx_steps: int) -> int:
        # Map absolute MIDI steps to the 25Hz audio frame grid (generic to cfg values).
        seconds = t_idx_steps * cfg.time_step_seconds
        return int(round(seconds / cfg.cc_frame_hop))

    def _append_token(token_id: int, t_idx_steps: int, tech_id: int = 0, vel_id: int = 0) -> None:
        token_list.append(token_id)
        technique_seq.append(tech_id)
        velocity_seq.append(vel_id)
        pos_midi.append(_frame_pos_from_steps(t_idx_steps))
    # tied section
    for pitch, vel in sorted(tied_notes):
        vel_val = max(0, min(vel, 127))
        vel_bin = (vel_val // 8) + 1
        _append_token(tokenizer.vocab.note_id(pitch), t_idx_steps=0, vel_id=vel_bin)
        _append_token(tokenizer.vocab.on_id, t_idx_steps=0)

    if tokenizer.vocab.include_end_tie:
        _append_token(tokenizer.vocab.end_tie_id, t_idx_steps=0)
    last_time = cfg.start_time
    t_idx_steps = 0
    for event_time, is_on, pitch, velocity in events:
        delta = event_time - last_time
        if delta >= cfg.time_step_seconds:
            steps_total = max(0, int(round(delta / cfg.time_step_seconds)))
            while steps_total > 0:
                cur_steps = min(tokenizer.vocab.num_time_bins, steps_total)
                t_idx_steps += cur_steps
                _append_token(tokenizer.vocab.time_id(cur_steps), t_idx_steps=t_idx_steps)
                steps_total -= cur_steps
            last_time = event_time
        note_tech = 0
        note_vel = 0
        if is_on:
            key = (event_time if event_time >= cfg.start_time else cfg.start_time, pitch)
            tech_id = tech_matches.get(key)
            if tech_id is not None:
                note_tech = tech_id
            tech_onset_seq.append(note_tech)

            # Velocity bin logic: 0-127 -> 1-16
            vel_val = max(0, min(velocity, 127))
            note_vel = (vel_val // 8) + 1

        _append_token(
            tokenizer.vocab.note_id(pitch),
            t_idx_steps=t_idx_steps,
            tech_id=note_tech,
            vel_id=note_vel,
        )
        _append_token(
            tokenizer.vocab.on_id if is_on else tokenizer.vocab.off_id,
            t_idx_steps=t_idx_steps,
        )

    _append_token(tokenizer.vocab.eos_id, t_idx_steps=t_idx_steps)
    tokens = token_list

    return tokens, technique_seq, velocity_seq, pos_midi, cc_frames, tech_onset_seq
