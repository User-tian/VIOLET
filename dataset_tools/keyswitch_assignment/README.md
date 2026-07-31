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

For connected notes, the pipeline first samples a short overlap. A different-pitch overlap
may receive one of the three legato techniques only when the preceding note is sustain or
legato. The overlap is generated for boundary-connected notes with probability 0.8. Its
duration is sampled from half-32nd, 32nd, 16th, and eighth-note values and converted to
seconds using the MIDI's initial tempo. Notes covered at least 95% by other notes are then
removed before labeling.

Every keyswitch starts 10 ms before its playable note, except at the beginning of a file
where it shares the note onset. Its duration is the tempo-dependent equivalent of 10 MIDI
ticks, using the historical 480-ticks-per-quarter assumption. For example, this is about
10.42 ms at 120 BPM. If a note overlaps the previous note but cannot be legato because the
previous technique is detached, and normal sampling selects sustain, its onset is moved to
10 ticks after the previous note's end. This removes the overlap so the renderer does not
interpret a sustain onset as a legato transition. Existing controller data, including CC1,
is preserved.

Rounded heuristic approximations of the historical weights are in
[`technique_probabilities.yaml`](technique_probabilities.yaml). The code omits renderer
integration, synthetic dynamics replacement, and quality-control utilities because they are
not needed to explain the note-labeling method or to run VIOLET.

**Output.** Technique-annotated MIDI (notes + keyswitches + CC1) ready for rendering.

Optional reference invocation:

```bash
pip install -r dataset_tools/requirements.txt
python dataset_tools/keyswitch_assignment/assign_keyswitches.py INPUT_DIR OUTPUT_DIR --seed 0
```
