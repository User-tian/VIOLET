#!/usr/bin/env python3
"""Generate the monophonic MIDI variants used to construct CSV-TD."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Iterable

import numpy as np
import pretty_midi


@dataclass(frozen=True)
class _NoteRef:
    note: pretty_midi.Note
    instrument: pretty_midi.Instrument

    @property
    def key(self) -> tuple[float, float, int]:
        return (self.note.start, self.note.end, self.note.pitch)


def _midi_files(folder: Path) -> Iterable[Path]:
    return sorted(
        path for path in folder.iterdir() if path.suffix.lower() in {".mid", ".midi"}
    )


def _all_notes(midi: pretty_midi.PrettyMIDI) -> list[_NoteRef]:
    notes = [
        _NoteRef(note, instrument)
        for instrument in midi.instruments
        for note in instrument.notes
    ]
    notes.sort(key=lambda item: item.note.start)
    return notes


def _copy_performance_events(
    source: pretty_midi.PrettyMIDI,
    target: pretty_midi.Instrument,
) -> None:
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
        deepcopy(event) for event in sorted(bends.values(), key=lambda event: event.time)
    ]


def _build_output(
    source: pretty_midi.PrettyMIDI,
    notes: list[_NoteRef],
    name: str,
) -> pretty_midi.PrettyMIDI:
    output = deepcopy(source)
    output.instruments = []
    template = notes[0].instrument if notes else source.instruments[0]
    instrument = pretty_midi.Instrument(
        program=template.program,
        is_drum=template.is_drum,
        name=name,
    )
    instrument.notes = [deepcopy(item.note) for item in notes]
    _copy_performance_events(source, instrument)
    output.instruments.append(instrument)
    return output


def analyze_polyphony(
    midi: pretty_midi.PrettyMIDI,
    sample_interval: float = 0.01,
    min_segment_duration: float = 0.1,
) -> dict[str, float | int]:
    """Measure the percentage of sampled time points containing multiple notes."""
    notes = _all_notes(midi)
    analysis: dict[str, float | int] = {
        "total_notes": len(notes),
        "max_simultaneous_notes": 0,
        "polyphony_percentage": 0.0,
        "sustained_polyphony_segments": 0,
    }
    if not notes:
        return analysis

    time_points = np.arange(
        min(item.note.start for item in notes),
        max(item.note.end for item in notes),
        sample_interval,
    )
    polyphonic_samples = 0
    consecutive_samples = 0
    sustained_segments = 0
    for time_point in time_points:
        active = sum(item.note.start <= time_point < item.note.end for item in notes)
        if active > 1:
            polyphonic_samples += 1
            consecutive_samples += 1
            analysis["max_simultaneous_notes"] = max(
                int(analysis["max_simultaneous_notes"]), active
            )
        else:
            if consecutive_samples * sample_interval >= min_segment_duration:
                sustained_segments += 1
            consecutive_samples = 0
    if consecutive_samples * sample_interval >= min_segment_duration:
        sustained_segments += 1

    analysis["sustained_polyphony_segments"] = sustained_segments
    if len(time_points):
        analysis["polyphony_percentage"] = 100.0 * polyphonic_samples / len(time_points)
    return analysis


def _overlap_is_allowed(previous: pretty_midi.Note, current: pretty_midi.Note) -> bool:
    overlap = min(previous.end, current.end) - max(previous.start, current.start)
    if overlap <= 0:
        return True
    previous_duration = previous.end - previous.start
    current_duration = current.end - current.start
    previous_ratio = overlap / previous_duration if previous_duration > 0 else 0.0
    current_ratio = overlap / current_duration if current_duration > 0 else 0.0
    contained = current.start >= previous.start and current.end <= previous.end
    return previous_ratio < 0.4 and current_ratio < 0.4 and not contained


def _split_two_closest_pitch(
    midi: pretty_midi.PrettyMIDI,
) -> tuple[pretty_midi.PrettyMIDI, pretty_midi.PrettyMIDI]:
    """Legacy selective closest-pitch splitter shared by two output families."""
    notes = _all_notes(midi)
    if not notes:
        empty = deepcopy(midi)
        empty.instruments = []
        return empty, deepcopy(empty)

    first_onset = notes[0].note.start
    concurrent = [item for item in notes if abs(item.note.start - first_onset) <= 0.03]
    highest = max(concurrent, key=lambda item: item.note.pitch)
    lowest = min(concurrent, key=lambda item: item.note.pitch)
    branches = [[highest], [lowest]]
    processed = {highest.key, lowest.key}

    for item in notes:
        if item.key in processed:
            continue
        allowed = [
            _overlap_is_allowed(branch[-1].note, item.note) for branch in branches
        ]
        if not any(allowed):
            continue
        if all(allowed):
            distances = [
                (item.note.pitch - branch[-1].note.pitch) ** 2 for branch in branches
            ]
            branch_index = 0 if distances[0] <= distances[1] else 1
        else:
            branch_index = 0 if allowed[0] else 1
        branches[branch_index].append(item)

    return (
        _build_output(midi, branches[0], f"{branches[0][0].instrument.name}_part1"),
        _build_output(midi, branches[1], f"{branches[1][0].instrument.name}_part2"),
    )


def split_into_two_parts_naive(
    midi: pretty_midi.PrettyMIDI,
) -> tuple[pretty_midi.PrettyMIDI, pretty_midi.PrettyMIDI]:
    """Produce the legacy ``_1`` and ``_2`` high-polyphony branches."""
    return _split_two_closest_pitch(midi)


def split_with_closest_pitch(
    midi: pretty_midi.PrettyMIDI,
) -> tuple[pretty_midi.PrettyMIDI, pretty_midi.PrettyMIDI]:
    """Produce the legacy ``_closest_1`` and ``_closest_2`` branches."""
    return _split_two_closest_pitch(midi)


def split_with_highest_pitch(
    midi: pretty_midi.PrettyMIDI,
    sample_interval: float = 0.01,
) -> tuple[pretty_midi.PrettyMIDI, pretty_midi.PrettyMIDI]:
    """Split notes selected as highest on a 10 ms grid from the remainder."""
    notes = _all_notes(midi)
    if not notes:
        empty = deepcopy(midi)
        empty.instruments = []
        return empty, deepcopy(empty)

    time_points = np.arange(
        min(item.note.start for item in notes),
        max(item.note.end for item in notes) + sample_interval,
        sample_interval,
    )
    highest_keys: set[tuple[float, float, int]] = set()
    remaining_keys: set[tuple[float, float, int]] = set()
    for time_point in time_points:
        active = [item for item in notes if item.note.start <= time_point < item.note.end]
        if len(active) > 1:
            highest = max(active, key=lambda item: item.note.pitch)
            highest_keys.add(highest.key)
            remaining_keys.update(item.key for item in active if item is not highest)
        elif active:
            highest_keys.add(active[0].key)

    highest_notes = [item for item in notes if item.key in highest_keys]
    remaining_notes = [
        item
        for item in notes
        if item.key not in highest_keys and item.key in remaining_keys
    ]
    source_name = notes[0].instrument.name
    return (
        _build_output(midi, highest_notes, f"{source_name}_highest"),
        _build_output(midi, remaining_notes, f"{source_name}_remaining"),
    )


def remove_redundant_notes(
    midi: pretty_midi.PrettyMIDI,
    coverage_threshold: float = 0.95,
) -> pretty_midi.PrettyMIDI:
    """Remove short notes whose duration is covered by other notes."""
    output = deepcopy(midi)
    for instrument in output.instruments:
        notes = sorted(instrument.notes, key=lambda note: note.end - note.start)
        removed: set[int] = set()
        for index, note in enumerate(notes):
            intervals = sorted(
                (max(note.start, other.start), min(note.end, other.end))
                for other_index, other in enumerate(notes)
                if other_index != index
                and other_index not in removed
                and other.start < note.end
                and other.end > note.start
            )
            merged: list[list[float]] = []
            for start, end in intervals:
                if merged and start <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], end)
                else:
                    merged.append([start, end])
            duration = note.end - note.start
            coverage = sum(end - start for start, end in merged)
            if duration > 0 and coverage / duration >= coverage_threshold:
                removed.add(index)
        instrument.notes = [
            note for index, note in enumerate(notes) if index not in removed
        ]
    return output


def adjust_to_violin_range(
    midi: pretty_midi.PrettyMIDI,
    rng: random.Random,
    min_pitch: int = 55,
    max_pitch: int = 103,
) -> None:
    """Shift each branch into violin range, dropping outliers if no shift fits."""
    for instrument in midi.instruments:
        notes = list(instrument.notes)
        if all(min_pitch <= note.pitch <= max_pitch for note in notes):
            continue
        while notes:
            pitches = [note.pitch for note in notes]
            min_shift = min_pitch - min(pitches)
            max_shift = max_pitch - max(pitches)
            if min_shift <= max_shift:
                shift = rng.choice(range(min_shift, max_shift + 1))
                for note in notes:
                    note.pitch += shift
                break
            low_distance = abs(min(pitches) - min_pitch)
            high_distance = abs(max(pitches) - max_pitch)
            outlier = max(pitches) if high_distance > low_distance else min(pitches)
            del notes[pitches.index(outlier)]
        instrument.notes = notes


def create_split_variants(
    midi: pretty_midi.PrettyMIDI,
    high_threshold: float = 80.0,
    low_threshold: float = 15.0,
    rng: random.Random | None = None,
) -> tuple[dict[str, pretty_midi.PrettyMIDI], dict[str, float | int]]:
    """Route a MIDI through the legacy non-Partitura Stage 1 algorithms."""
    if low_threshold > high_threshold:
        raise ValueError("low threshold cannot exceed high threshold")
    rng = rng or random.Random()
    analysis = analyze_polyphony(midi)
    rate = float(analysis["polyphony_percentage"])
    variants: dict[str, pretty_midi.PrettyMIDI] = {}

    if rate > high_threshold:
        first, second = split_into_two_parts_naive(midi)
        variants["_1"] = first
        variants["_2"] = second

    if rate >= low_threshold:
        closest_first, closest_second = split_with_closest_pitch(midi)
        variants["_closest_1"] = closest_first
        if closest_second.instruments and closest_second.instruments[0].notes:
            variants["_closest_2"] = closest_second
        highest, _remaining = split_with_highest_pitch(midi)
        variants["_highest"] = highest
    else:
        variants[""] = deepcopy(midi)

    for suffix, variant in list(variants.items()):
        adjust_to_violin_range(variant, rng)
        variants[suffix] = remove_redundant_notes(variant)
    return variants, analysis


def split_file(
    input_path: Path,
    output_folder: Path,
    high_threshold: float = 80.0,
    low_threshold: float = 15.0,
    rng: random.Random | None = None,
) -> tuple[list[Path], dict[str, float | int]]:
    midi = pretty_midi.PrettyMIDI(str(input_path))
    variants, analysis = create_split_variants(
        midi,
        high_threshold=high_threshold,
        low_threshold=low_threshold,
        rng=rng,
    )
    output_folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix, variant in variants.items():
        output_path = output_folder / f"{input_path.stem}{suffix}.mid"
        variant.write(str(output_path))
        paths.append(output_path)
    return paths, analysis


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate rate-dependent monophonic MIDI variants."
    )
    parser.add_argument("input_folder", type=Path)
    parser.add_argument("output_folder", type=Path)
    parser.add_argument("--high-threshold", type=float, default=80.0)
    parser.add_argument("--low-threshold", type=float, default=15.0)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    if not args.input_folder.is_dir():
        parser.error(f"input folder does not exist: {args.input_folder}")
    if args.low_threshold > args.high_threshold:
        parser.error("--low-threshold cannot exceed --high-threshold")

    rng = random.Random(args.seed)
    written = 0
    for input_path in _midi_files(args.input_folder):
        paths, analysis = split_file(
            input_path,
            args.output_folder,
            high_threshold=args.high_threshold,
            low_threshold=args.low_threshold,
            rng=rng,
        )
        written += len(paths)
        rate = float(analysis["polyphony_percentage"])
        print(f"{input_path.name}: polyphony={rate:.1f}%, wrote {len(paths)} file(s)")
    print(f"Wrote {written} monophonic MIDI file(s) to {args.output_folder}")


if __name__ == "__main__":
    main()