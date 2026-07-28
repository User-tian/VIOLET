from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple

import torch
from note_seq import midi_io
from note_seq.protobuf.music_pb2 import NoteSequence
from torch import nn


@dataclass(frozen=True)
class MT3MIDIVocabulary:
    """
    Flat vocabulary matching MT3/RenderBox MIDILike events for a single instrument.
    """
    num_pitches: int = 128
    num_time_bins: int = 512
    include_end_tie: bool = True

    @property
    def note_offset(self) -> int:
        return 1

    @property
    def on_off_offset(self) -> int:
        return self.note_offset + self.num_pitches

    @property
    def time_offset(self) -> int:
        return self.on_off_offset + 2

    @property
    def end_tie_id(self) -> int:
        return self.time_offset + self.num_time_bins if self.include_end_tie else -1

    @property
    def eos_id(self) -> int:
        return (self.end_tie_id + 1) if self.include_end_tie else (self.time_offset + self.num_time_bins)

    @property
    def size(self) -> int:
        return self.eos_id + 1

    def note_id(self, pitch: int) -> int:
        pitch = max(0, min(pitch, self.num_pitches - 1))
        return self.note_offset + pitch

    @property
    def on_id(self) -> int:
        return self.on_off_offset

    @property
    def off_id(self) -> int:
        return self.on_off_offset + 1

    def time_id(self, steps: int) -> int:
        steps = max(1, min(steps, self.num_time_bins))
        return self.time_offset + (steps - 1)


def _sinusoidal_positional_encoding(max_len: int, dim: int) -> torch.Tensor:
    position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(max_len, dim, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class MT3MIDITokenizer:
    def __init__(
        self,
        vocab: MT3MIDIVocabulary | None = None,
        window_seconds: float = 10.0,
        time_step_seconds: float = 0.01,
    ) -> None:
        self.vocab = vocab or MT3MIDIVocabulary()
        self.window_seconds = window_seconds
        self.time_step_seconds = time_step_seconds

    def encode(self, midi_path: str | Path, start_time: float = 5.0) -> List[int]:
        sequence = self._load_sequence(midi_path)

        # 1. Get Separated Lists: Tied Notes vs New Events
        tied_notes, events = self._build_events(sequence, start_time)

        tokens: List[int] = []

        # 2. Build Tie Section (Active notes state declaration)
        tied_notes.sort()
        for pitch, _velocity in tied_notes:
            tokens.append(self.vocab.note_id(pitch))
            tokens.append(self.vocab.on_id)  # Declare as ON

        # 3. Trigger End Tie Section
        if self.vocab.include_end_tie:
            tokens.append(self.vocab.end_tie_id)

        # 4. Process Window Events
        last_time = start_time

        for event_time, is_on, pitch, velocity in events:
            delta = event_time - last_time

            if delta >= self.time_step_seconds:
                tokens.extend(self._time_tokens(delta))
                last_time = event_time

            tokens.append(self.vocab.note_id(pitch))
            tokens.append(self.vocab.on_id if is_on else self.vocab.off_id)

        tokens.append(self.vocab.eos_id)
        return tokens

    def _time_tokens(self, delta_seconds: float) -> List[int]:
        steps_total = max(0, int(round(delta_seconds / self.time_step_seconds)))
        if steps_total == 0:
            return []

        tokens: List[int] = []
        remaining = steps_total
        while remaining > 0:
            step = min(remaining, self.vocab.num_time_bins)
            tokens.append(self.vocab.time_id(step))
            remaining -= step
        return tokens

    def _build_events(
        self,
        sequence: NoteSequence,
        start_time: float,
        ignore_pitches: Optional[Set[int]] = None,
        min_pitch: Optional[int] = None,
        max_pitch: Optional[int] = None,
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[float, bool, int, int]]]:
        """
        Identify active (tied) notes and future events.

        Returns:
            tied_notes: List of (pitch, velocity) active at start_time.
            events: List of (time, is_on, pitch, velocity) within the window.
        """
        end_time = start_time + self.window_seconds

        tied_notes: List[Tuple[int, int]] = []
        events: List[Tuple[float, bool, int, int]] = []

        for note in sequence.notes:
            # Skip ignored pitches (e.g. technique keyswitches)
            if ignore_pitches and note.pitch in ignore_pitches:
                continue
            if min_pitch is not None and note.pitch < min_pitch:
                continue
            if max_pitch is not None and note.pitch > max_pitch:
                continue

            # Skip notes that end before window starts or start after window ends
            if note.end_time <= start_time or note.start_time >= end_time:
                continue

            # CHECK: Is this a Tied Note?
            # A note is tied if it started BEFORE this window but sustains INTO it.
            if note.start_time < start_time:
                tied_notes.append((int(note.pitch), int(note.velocity)))
                # We still need the Note Off event for this note later in the window
                offset = min(note.end_time, end_time)
                events.append((offset, False, int(note.pitch), int(note.velocity)))
            else:
                # Regular note starting inside the window
                onset = note.start_time
                offset = min(note.end_time, end_time)

                events.append((onset, True, int(note.pitch), int(note.velocity)))
                events.append((offset, False, int(note.pitch), int(note.velocity)))

        # Sort Events:
        # Primary key: Time
        # Secondary key: Note Off (False) before Note On (True) at same time.
        #   We use `is_on` directly: False < True.
        #   So sorting by `(time, is_on, pitch)` puts Off before On.
        events.sort(key=lambda x: (x[0], x[1], x[2]))

        return tied_notes, events

    @staticmethod
    def _load_sequence(midi_path: str | Path) -> NoteSequence:
        path = Path(midi_path)
        if not path.exists():
            raise FileNotFoundError(f"MIDI file not found: {path}")
        return midi_io.midi_file_to_note_sequence(str(path))