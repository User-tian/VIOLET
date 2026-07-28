# Stage 1 - Split Symphony

Source material for CSV-TD is [MID_FiLD](https://github.com/your-link), which provides
MIDI notes together with **human-written dynamics curves (CC1)**. Much of this material is
polyphonic or multi-track, which a solo-violin instrument cannot render faithfully.

[`split_symphony.py`](split_symphony.py) is a compact reference for the line-extraction
step. It flattens non-drum notes and assigns each note to the compatible voice with the
closest preceding pitch. An overlap longer than 10 ms starts another voice; smaller
boundary overlaps are clipped. MIDI timing metadata, controller events (including CC1),
and pitch bends are copied to every extracted voice.

**Output.** One MIDI file per extracted monophonic voice, with CC1 dynamics retained.

The legacy development script also contained alternative splitters, tempo augmentation,
trimming, logging, and quality-control experiments. Those are intentionally omitted here;
this folder records the core transformation and is not required for VIOLET training or
inference.

Optional reference invocation:

```bash
pip install -r dataset_tools/requirements.txt
python dataset_tools/split_symphony/split_symphony.py INPUT_DIR OUTPUT_DIR
```
