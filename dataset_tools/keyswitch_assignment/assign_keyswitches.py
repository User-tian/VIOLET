#!/usr/bin/env python3
"""Assign note-level violin techniques and encode them as MIDI keyswitches."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import random
from typing import Iterable

import pretty_midi
import yaml


PLAYABLE_MIN_PITCH = 55  # G3
KEYSWITCH_LEAD_SECONDS = 0.01
KEYSWITCH_DURATION_SECONDS = 0.01
DEFAULT_KEYSWITCH_CONFIG = Path(__file__).resolve().parents[2] / "configs/ks_config.yaml"
DEFAULT_PROBABILITY_CONFIG = Path(__file__).with_name("technique_probabilities.yaml")

# Quarter-note multipliers and sampling weights used by the data-generation code.
OVERLAP_DURATIONS = {
    "half_32nd": (0.0625, 0.60),
    "32nd": (0.125, 0.27),
    "16th": (0.25, 0.11),
    "eighth": (0.5, 0.02),
}


def _midi_files(folder: Path) -> Iterable[Path]:
    return sorted(
        path for path in folder.iterdir() if path.suffix.lower() in {".mid", ".midi"}
    )


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"expected a non-empty mapping in {path}")
    return data


def load_keyswitches(path: Path = DEFAULT_KEYSWITCH_CONFIG) -> dict[str, int]:
    data = _load_yaml(path)
    return {str(name): int(pitch) for name, pitch in data.items()}


def load_probabilities(
    path: Path = DEFAULT_PROBABILITY_CONFIG,
) -> dict[str, dict[str, float]]:
    data = _load_yaml(path)
    probabilities = {
        str(bucket): {str(name): float(weight) for name, weight in weights.items()}
        for bucket, weights in data.items()
    }
    if any(weight < 0 for weights in probabilities.values() for weight in weights.values()):
        raise ValueError(f"probability weights must be non-negative: {path}")
    return probabilities


def _duration_bucket(duration: float) -> str:
    if duration <= 0.15:
        return "<=0.15"
    if duration <= 0.40:
        return "0.15-0.40"
    if duration <= 0.80:
        return "0.40-0.80"
    if duration <= 2.00:
        return "0.80-2.00"
    return ">2.00"


def _weighted_choice(weights: dict[str, float], rng: random.Random) -> str:
    positive = [(name, weight) for name, weight in weights.items() if weight > 0]
    if not positive:
        raise ValueError("at least one technique must have positive weight")
    names, values = zip(*positive)
    return rng.choices(names, weights=values, k=1)[0]


def _select_technique(
    duration: float,
    legato: bool,
    probabilities: dict[str, dict[str, float]],
    rng: random.Random,
) -> str:
    weights = probabilities[_duration_bucket(duration)]
    selected = {
        name: weight
        for name, weight in weights.items()
        if name.startswith("legato_") is legato
    }
    if legato and not any(selected.values()):
        selected = {
            name: sum(bucket.get(name, 0.0) for bucket in probabilities.values())
            for name in sorted(
                {name for bucket in probabilities.values() for name in bucket}
            )
            if name.startswith("legato_")
        }
    return _weighted_choice(selected, rng)


def _overlap_seconds(midi: pretty_midi.PrettyMIDI, rng: random.Random) -> float:
    names = list(OVERLAP_DURATIONS)
    selected = rng.choices(
        names,
        weights=[OVERLAP_DURATIONS[name][1] for name in names],
        k=1,
    )[0]
    tempo_values = midi.get_tempo_changes()[1]
    tempo = float(tempo_values[0]) if len(tempo_values) else 120.0
    return (60.0 / tempo) * OVERLAP_DURATIONS[selected][0]


def _create_overlaps(
    midi: pretty_midi.PrettyMIDI,
    probability: float,
    rng: random.Random,
) -> None:
    for instrument in midi.instruments:
        notes = sorted(instrument.notes, key=lambda note: (note.start, note.pitch))
        for previous, current in zip(notes, notes[1:]):
            if abs(previous.end - current.start) < 0.001 and rng.random() < probability:
                current.start = max(0.0, current.start - _overlap_seconds(midi, rng))


def assign_keyswitches(
    midi: pretty_midi.PrettyMIDI,
    keyswitches: dict[str, int],
    probabilities: dict[str, dict[str, float]],
    rng: random.Random,
    overlap_probability: float = 0.8,
    create_overlaps: bool = True,
) -> pretty_midi.PrettyMIDI:
    """Return a copy with one technique keyswitch for every playable note."""
    output = deepcopy(midi)
    for instrument in output.instruments:
        instrument.notes = [
            note for note in instrument.notes if note.pitch >= PLAYABLE_MIN_PITCH
        ]

    if create_overlaps:
        _create_overlaps(output, overlap_probability, rng)

    for instrument in output.instruments:
        notes = sorted(instrument.notes, key=lambda note: (note.start, note.pitch))
        techniques: list[str] = []
        annotations: list[pretty_midi.Note] = []
        for index, note in enumerate(notes):
            legato = False
            if index > 0:
                previous = notes[index - 1]
                previous_technique = techniques[-1]
                legato = (
                    note.pitch != previous.pitch
                    and note.start < previous.end - 0.001
                    and (
                        previous_technique == "sustain"
                        or previous_technique.startswith("legato_")
                    )
                )

            technique = _select_technique(
                note.end - note.start,
                legato,
                probabilities,
                rng,
            )
            if technique not in keyswitches:
                raise KeyError(f"keyswitch pitch is not configured for {technique!r}")
            techniques.append(technique)
            start = max(0.0, note.start - KEYSWITCH_LEAD_SECONDS)
            annotations.append(
                pretty_midi.Note(
                    velocity=100,
                    pitch=keyswitches[technique],
                    start=start,
                    end=start + KEYSWITCH_DURATION_SECONDS,
                )
            )

        instrument.notes = sorted(notes + annotations, key=lambda note: (note.start, note.pitch))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add duration-conditioned technique keyswitches to monophonic MIDI."
    )
    parser.add_argument("input_folder", type=Path)
    parser.add_argument("output_folder", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--overlap-probability", type=float, default=0.8)
    parser.add_argument("--no-overlaps", action="store_true")
    parser.add_argument("--keyswitch-config", type=Path, default=DEFAULT_KEYSWITCH_CONFIG)
    parser.add_argument(
        "--probability-config", type=Path, default=DEFAULT_PROBABILITY_CONFIG
    )
    args = parser.parse_args()

    if not args.input_folder.is_dir():
        parser.error(f"input folder does not exist: {args.input_folder}")
    if not 0.0 <= args.overlap_probability <= 1.0:
        parser.error("--overlap-probability must be between 0 and 1")

    keyswitches = load_keyswitches(args.keyswitch_config)
    probabilities = load_probabilities(args.probability_config)
    rng = random.Random(args.seed)
    args.output_folder.mkdir(parents=True, exist_ok=True)

    written = 0
    for input_path in _midi_files(args.input_folder):
        midi = pretty_midi.PrettyMIDI(str(input_path))
        annotated = assign_keyswitches(
            midi,
            keyswitches,
            probabilities,
            rng,
            overlap_probability=args.overlap_probability,
            create_overlaps=not args.no_overlaps,
        )
        annotated.write(str(args.output_folder / f"{input_path.stem}.mid"))
        written += 1
    print(f"Wrote {written} annotated MIDI file(s) to {args.output_folder}")


if __name__ == "__main__":
    main()