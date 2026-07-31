# Stage 1 - Split Symphony

Source material for CSV-TD is
[MID-FiLD](https://github.com/pozalabs/MID-FiLD) ([Ryu et al., 2024](https://doi.org/10.1609/aaai.v38i1.27774)),
which provides MIDI notes together with **human-written dynamics curves (CC1)**. Much of this
material is polyphonic or multi-track, which a solo-violin instrument cannot render faithfully.

[`split_symphony.py`](split_symphony.py) first samples the MIDI every 10 ms and computes the percentage
of time points with more than one active note. The default 15% and 80% thresholds route
the file through the same set of extraction methods used for CSV-TD:

| Polyphony rate | Outputs |
|---|---|
| `< 15%` | Original filename after note cleanup and violin-range adjustment. |
| `15-80%` | `_closest_1`, `_closest_2` (when nonempty), and `_highest`. |
| `> 80%` | `_1`, `_2`, `_closest_1`, `_closest_2`, and `_highest`. |

The selective closest-pitch method seeds two branches with the highest and lowest notes at
the first onset, then assigns compatible notes by pitch distance, following the practice of [Dong et al., 2021](https://github.com/salu133445/arranger). Both entry points are kept
because high-polyphony files contributed both `_1`/`_2` and `_closest_1`/`_closest_2`
variants to the dataset. The highest-pitch method selects the highest active note on the
same 10 ms grid. It computes a remaining-note branch internally, but the historical batch
pipeline only saved `_highest`.

Each saved variant is cleaned of notes covered at least 95% by other notes and shifted into
the G3-A7 violin range when necessary. MIDI timing metadata, controller events (including
CC1), and pitch bends are retained. 

## Usage

From the repository root, install the dataset-tool dependencies:

```bash
pip install -r dataset_tools/requirements.txt
```

Run Stage 1 on every `.mid` or `.midi` file in an input directory:

```bash
python dataset_tools/split_symphony/split_symphony.py INPUT_DIR OUTPUT_DIR
```

The equivalent command with the default thresholds made explicit and a reproducible random
seed is:

```bash
python dataset_tools/split_symphony/split_symphony.py \
	INPUT_DIR OUTPUT_DIR \
	--low-threshold 15 \
	--high-threshold 80 \
	--seed 0
```

The output directory is created automatically. Run the script with `--help` to see all
available arguments.

