#!/usr/bin/env python3
"""Extract monophonic voices from the polyphonic MIDI used to build CSV-TD."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pretty_midi


@dataclass
class _Voice:
    instrument: pretty_midi.Instrument
    notes: list[pretty_midi.Note] = field(default_factory=list)


def _midi_files(folder: Path) -> Iterable[Path]:
    return sorted(
        path for path in folder.iterdir() if path.suffix.lower() in {".mid", ".midi"}
    )


def _copy_performance_events(
    source: pretty_midi.PrettyMIDI,
    target: pretty_midi.Instrument,
) -> None:
    """Copy controller and pitch-bend events without duplicating shared tracks."""
    controls: dict[tuple[int, float, int], pretty_midi.ControlChange] = {}
    bends: dict[tuple[float, int], pretty_midi.PitchBend] = {}
    for instrument in source.instruments:
        for control in instrument.control_changes:
            controls[(control.number, control.time, control.value)] = control
        for bend in instrument.pitch_bends:
            bends[(bend.time, bend.pitch)] = bend

    target.control_changes = [
        deepcopy(control)
        for control in sorted(controls.values(), key=lambda event: event.time)
    ]
    target.pitch_bends = [
        deepcopy(bend) for bend in sorted(bends.values(), key=lambda event: event.time)
    ]


def split_into_monophonic_voices(
    midi: pretty_midi.PrettyMIDI,
    overlap_tolerance: float = 0.01,
) -> list[pretty_midi.PrettyMIDI]:
    """Greedily separate notes using temporal compatibility and closest pitch.

    Notes that overlap by at most ``overlap_tolerance`` seconds are treated as
    boundary imprecision and clipped at the next onset. Longer overlaps start a
    separate voice. Among compatible voices, closest pitch preserves voice
    continuity.
    """
    source_notes = [
        (deepcopy(note), instrument)
        for instrument in midi.instruments
        if not instrument.is_drum
        for note in instrument.notes
    ]
    source_notes.sort(key=lambda item: (item[0].start, -item[0].pitch, item[0].end))

    voices: list[_Voice] = []
    for note, source_instrument in source_notes:
        compatible = [
            voice
            for voice in voices
            if note.start >= voice.notes[-1].end - overlap_tolerance
            and note.start > voice.notes[-1].start
        ]
        if compatible:
            voice = min(
                compatible,
                key=lambda candidate: (
                    abs(note.pitch - candidate.notes[-1].pitch),
                    candidate.notes[-1].end,
                ),
            )
            if note.start < voice.notes[-1].end:
                voice.notes[-1].end = note.start
        else:
            voice = _Voice(source_instrument)
            voices.append(voice)
        voice.notes.append(note)

    outputs: list[pretty_midi.PrettyMIDI] = []
    for voice_number, voice in enumerate(voices, start=1):
        output = deepcopy(midi)
        output.instruments = []
        instrument = pretty_midi.Instrument(
            program=voice.instrument.program,
            is_drum=False,
            name=f"voice_{voice_number}",
        )
        instrument.notes = voice.notes
        _copy_performance_events(midi, instrument)
        output.instruments.append(instrument)
        outputs.append(output)
    return outputs


def split_file(input_path: Path, output_folder: Path) -> list[Path]:
    midi = pretty_midi.PrettyMIDI(str(input_path))
    voices = split_into_monophonic_voices(midi)
    if not voices:
        return []

    output_folder.mkdir(parents=True, exist_ok=True)
    output_paths = []
    for index, voice in enumerate(voices, start=1):
        suffix = "" if len(voices) == 1 else f"_voice{index}"
        output_path = output_folder / f"{input_path.stem}{suffix}.mid"
        voice.write(str(output_path))
        output_paths.append(output_path)
    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split a folder of MIDI files into monophonic voices."
    )
    parser.add_argument("input_folder", type=Path)
    parser.add_argument("output_folder", type=Path)
    args = parser.parse_args()

    if not args.input_folder.is_dir():
        parser.error(f"input folder does not exist: {args.input_folder}")

    written = 0
    for input_path in _midi_files(args.input_folder):
        output_paths = split_file(input_path, args.output_folder)
        written += len(output_paths)
        print(f"{input_path.name}: wrote {len(output_paths)} voice(s)")
    print(f"Wrote {written} monophonic MIDI file(s) to {args.output_folder}")


if __name__ == "__main__":
    main()