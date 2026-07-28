# Stage 2 - Keyswitch Assignment

Given a monophonic line from [Stage 1](../split_symphony/), this stage assigns a **playing
technique to each note** and writes it as a **MIDI keyswitch below the playable violin
range** (G3–A7), so the keyswitch pitches never collide with actual notes.

[`assign_keyswitches.py`](assign_keyswitches.py) contains the compact annotation logic.
Techniques are assigned with duration-based probabilistic heuristics that mirror idiomatic
violin writing:

- **short notes** are more likely to receive **spiccato, staccato, or pizzicato**;
- **long notes** are more likely to receive **legato, trill, or harmonic**.

**Keyswitch mapping.** Keyswitch MIDI numbers follow
[`../../configs/ks_config.yaml`](../../configs/ks_config.yaml), and the resulting
technique IDs (1–12) follow the mapping in
[`../../src/data/components/midi_processor.py`](../../src/data/components/midi_processor.py).
Bowing-style modifiers (`style_normal`, `style_sordino`, `style_sulpont`, `style_sultasto`)
can be emitted as keyswitches for the renderer but are **not** among the 12 note-level
technique classes used for conditioning.

For connected notes, the pipeline first samples a short overlap. A different-pitch overlap
may receive one of the three legato techniques only when the preceding note is sustain or
legato. Every annotation is a 10 ms MIDI note at, or up to 10 ms before, the corresponding
playable-note onset. Existing controller data, including CC1, is preserved.

The historical weights are in
[`technique_probabilities.yaml`](technique_probabilities.yaml). The code omits renderer
integration, synthetic dynamics replacement, and quality-control utilities because they are
not needed to explain the note-labeling method or to run VIOLET.

**Output.** Technique-annotated MIDI (notes + keyswitches + CC1) ready for rendering.

Optional reference invocation:

```bash
pip install -r dataset_tools/requirements.txt
python dataset_tools/keyswitch_assignment/assign_keyswitches.py INPUT_DIR OUTPUT_DIR --seed 0
```
