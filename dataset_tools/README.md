# CSV-TD Data Pipeline

This folder documents the two symbolic preprocessing stages used to build **CSV-TD**
(Controlled Synthetic Violin with Techniques and Dynamics), the curated dataset behind
VIOLET. The implementations are compact provenance references, not a supported
end-to-end dataset reproduction workflow. Training and inference do not depend on them.

The end-to-end flow has three stages:

```
 MID_FiLD MIDI (with human-written CC1 dynamics)
        │
        ▼
 1. split_symphony/        extract monophonic solo lines suitable for rendering
        │
        ▼
 2. keyswitch_assignment/  insert note-level technique keyswitches (below violin range)
        │                  using duration-based probabilistic heuristics
        ▼
 3. Kontakt rendering      commercial virtual instrument → 48 kHz stereo WAV
        │
        ▼
 CSV-TD (MIDI ↔ audio pairs + technique + CC1 dynamics)
```

| Stage | Folder | What it does |
|-------|--------|--------------|
| 1 | [`split_symphony/`](split_symphony/) | Measure polyphony and generate closest-pitch/highest-pitch solo-line variants. |
| 2 | [`keyswitch_assignment/`](keyswitch_assignment/) | Generate overlaps, assign note-level techniques, and write keyswitches below the violin range. |
| 3 | See main [`README.md`](../README.md) for Batched offline MIDI2Audio renderer | Render annotated MIDI with the commercial Kontakt instrument used for CSV-TD. |

The keyswitch numbering used across stages 2–3 matches [`../configs/ks_config.yaml`](../configs/ks_config.yaml)
and the technique-ID mapping in [`../src/data/components/midi_processor.py`](../src/data/components/midi_processor.py).
