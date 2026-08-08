# VIOLET: High-Fidelity Violin Synthesis with Techniques and Dynamics

Official implementation of the ISMIR 2026 paper
**"VIOLET: High-Fidelity Violin Synthesis with Techniques and Dynamics."**

VIOLET is a latent-diffusion framework for controllable violin synthesis. It uses a
Diffusion Transformer (DiT) with a rectified-flow objective to synthesize high-fidelity
48 kHz audio from **MIDI notes**, **note-level playing techniques**, and **continuous
dynamics** curves. The three control signals are time-aligned and injected into the DiT
backbone via Adaptive Layer Normalization (AdaLN).

<a href="https://www.python.org/"><img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10+-blue.svg"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.8-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>

## Links

- 🎧 **Demo page:** coming soon
- 📄 **Paper:** arXiv link coming soon
- 💾 **Checkpoints:** coming soon
- 📦 **Dataset:** coming soon
- 🎼 [**Subjective-study excerpts**](subjective_study/)
- 🎹 **Batched offline MIDI2Audio renderer:** coming soon

## Release checklist

Keep this list limited to public release metadata and assets; track implementation work in
GitHub Issues.

- [ ] After this repository is public, update arXiv and add the paper link.
- [ ] Publish the demo page and restore its link.
- [ ] Publish the VIOLET and DACVAE checkpoints and add their link.
- [ ] Publish CSV-TD and add its dataset link.
- [ ] Publish MidiForge and restore its link.
- [x] Upload the [excerpts used in the subjective study](subjective_study/).
- [ ] Document the separate distribution terms for datasets and checkpoints.

## Checkpoints

Pre-trained checkpoints will be linked above when the release is public. Download and place
them at the paths shown below. For a different location, override
`model.ema_ckpt_path` or `encoder.finetuned_ckpt` as appropriate.

| Model | Description | Default path |
|-------|-------------|--------------|
| VIOLET (Full) | DiT latent-diffusion model trained on all corpora | `pretrained_checkpoint/ema_snapshots/ema_prof_99515` |
| DACVAE (violin) | Fine-tuned DACVAE decoder for 48 kHz violin | `dacvae_ft/weights.pth` |

## Setup

### Install dependencies

We use [uv](https://github.com/astral-sh/uv) to manage the Python 3.10 environment.

```bash
# clone project
git clone https://github.com/User-tian/VIOLET
cd VIOLET

# create a Python 3.10 virtual environment with uv (install uv first)
uv venv EVS --python 3.10
source EVS/bin/activate

# install the PyTorch stack first (CUDA 12.8 wheels):
#   torch==2.8.0+cu128, torchvision==0.23.0+cu128, torchaudio==2.8.0+cu128
uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128

# install the remaining requirements
uv pip install -r requirements.txt
uv pip install git+https://github.com/facebookresearch/dacvae.git
uv pip install --upgrade "protobuf>=4.25,<6"
```

If you are not using CUDA 12.8, keep the same version trio from `requirements.txt` and
change only the PyTorch wheel index URL to the one matching your system from the official
[PyTorch install selector](https://pytorch.org/get-started/locally/).

`requirements.txt` pins the Python 3.10 training environment for this repo.

## How to run

VIOLET uses [hydra](https://hydra.cc/) config management. Experiment configs live under
[configs/experiment/](configs/experiment/). Entry points are `src/train.py` and
`src/eval.py`.

### Training

Train the DiT latent-diffusion model (DDP, mixed precision) on 2 GPUs:

```bash
uv run python src/train.py \
  trainer=ddp.yaml trainer.devices=2 \
  experiment=violin_synthesis/violin_synthesis.yaml \
  +trainer.precision=bf16-mixed +trainer.accumulate_grad_batches=4
```

Single GPU / resume from a checkpoint:

```bash
uv run python src/train.py \
  experiment=violin_synthesis/violin_synthesis.yaml \
  +trainer.precision=bf16-mixed ckpt_path="/path/to/ckpt.ckpt"
```

> On some RTX 4090 nodes, prepend `NCCL_P2P_DISABLE=1` to avoid DDP hangs
> ([reference](https://discuss.pytorch.org/t/ddp-training-on-rtx-4090-ada-cu118/168366)).

**Leading-silence augmentation (optional).** If your MIDI/audio pairs sometimes start with a note
at `t=0` rather than a brief silence, set `data.leading_silence_prob` > 0 (with
`data.leading_silence_delta_ms` controlling how much silence to insert, default `30`ms).
`leading_silence_prob` defaults to `0` (off). This randomly prepends a short silence during
training so the model learns a clean onset instead of assuming every clip starts mid-note —
useful because inference-time MIDI doesn't always start exactly at `t=0` either.

**Silent-pair augmentation.** `data.silence_pair_prob` is `0.03` in the training experiment.
The collator uses stochastic rounding per batch, so 3% of training samples are silent in
expectation even when a small batch cannot contain a fractional number of examples.

### Inference / Evaluation

`data=eval_midi` selects the [`configs/data/eval_midi.yaml`](configs/data/eval_midi.yaml) config,
which recursively globs `.mid`/`.midi` files (with keyswitch-encoded techniques, see
[below](#the-12-playing-techniques)) from a directory you supply — it defaults to a relative
`Eval-midi/` folder, which is not shipped in this repo. Point it at your own MIDI files by
overriding `data.data_dir`:

```bash
CUDA_VISIBLE_DEVICES=0 python src/eval.py \
  data=eval_midi data.data_dir=/path/to/your/midi_dir \
  +trainer.precision=32 \
  experiment=violin_synthesis_inference/violin_synthesis_inference.yaml
```

This experiment evaluates the EMA model configured by `model.ema_ckpt_path`; do not pass a
Lightning `ckpt_path`. To select another EMA snapshot, override
`model.ema_ckpt_path=/path/to/ema_prof_STEP`.

For a bundled input, set `data.data_dir=midi_example`. The example's provenance and
CC BY 4.0 attribution are documented in [`midi_example/README.md`](midi_example/README.md).

If a MIDI file has no technique keyswitches or CC1 (dynamics) automation, `EvalMidiDataset` falls
back to fixed defaults rather than erroring: technique `1` (sustain) and a CC1 value of 100/127
(≈0.79 after normalization). This keeps unannotated MIDI usable at inference time, but means the
model always receives *some* technique/dynamics signal for such input — see `default_cc1_value`
/`default_technique_id` in [`src/data/components/midi_processor.py`](src/data/components/midi_processor.py)
if that default isn't representative of your MIDI.

For durations longer than the model's fixed context window (i.e. 10s), enable the tiled
overlap-add long-audio sampler, which renders each window independently and crossfades the
resulting audio (not a latent-space blend):

```bash
CUDA_VISIBLE_DEVICES=0 python src/eval.py \
  data=eval_midi data.data_dir=/path/to/your/midi_dir \
  +trainer.precision=32 ++long_audio.enabled=true \
  experiment=violin_synthesis_inference/violin_synthesis_inference.yaml
```

By default, long-audio rendering follows each MIDI through its actual end and adds a one-second
tail. Set `long_audio.duration_seconds` only when a fixed output duration is intentional.

## Method overview

VIOLET is trained in two stages:

1. **DACVAE fine-tuning.** We fine-tune the decoder of DACVAE (the VAE variant of the
   Descript Audio Codec) on violin recordings. The codec is then frozen and provides a
   25 Hz latent representation (40 ms per frame) for 48 kHz mono audio. We fine-tuned using
   the training/fine-tuning instructions from
   [descript-audio-codec](https://github.com/descriptinc/descript-audio-codec) (the
   `descript-audiotools` recipe DAC itself is trained with), using the **audio portion of
   all four datasets** below (CSV-TD, MOSA, MUSC, MOSA_VPT) as the fine-tuning corpus.
2. **Latent diffusion (DiT).** A Diffusion Transformer is trained with a rectified-flow
   objective to generate audio latents. MIDI, technique, and dynamics conditions are
   embedded, aligned to the latent frame rate, and injected via AdaLN-Zero. At inference,
   we use a compositional classifier-free guidance scheme with a Euler rectified-flow
   sampler.

## Datasets

VIOLET is trained on a mix of synthetic and real violin corpora. The centerpiece is
**CSV-TD**, a dataset we curated specifically for this task; we augment it with two real
recorded corpora (**MOSA**, **MUSC**) and one synthetic augmentation set (**MOSA_VPT**).

| Dataset | Type | # Pairs | Sample rate | Annotation | Duration |
|---------|------|--------:|-------------|------------|---------:|
| CSV-TD (train) | Synthetic | 6,108 | 48 kHz | Stereo, technique + dynamics | 35.4 h |
| CSV-TD (test)  | Synthetic |   686 | 48 kHz | Stereo, technique + dynamics |  3.7 h |
| MOSA_VPT       | Synthetic | 1,864 | 48 kHz | Stereo, technique             | 75.6 h |
| MOSA           | Real      |   461 | 44.1 kHz | Mono, none                   | 18.9 h |
| MUSC           | Real      |   939 | 48 kHz | Stereo, none                  | 30.9 h |

During training we mix these with a curriculum that starts synthetic-heavy
(CSV-TD : MOSA_VPT : MUSC : MOSA = 60 : 20 : 10 : 10) and shifts toward real recordings
(40 : 10 : 25 : 25) to improve natural transitions and fidelity.

### CSV-TD — Controlled Synthetic Violin with Techniques and Dynamics

To our knowledge, no public violin dataset provides the aligned **MIDI notes**,
**note-level playing techniques**, and **continuous dynamics** that VIOLET needs to learn
controllable synthesis. We therefore built **CSV-TD** by rendering symbolically-controlled
MIDI through a commercial virtual instrument. Because we own the exact symbolic controls
used to drive the renderer, every audio frame comes with perfectly-aligned MIDI, technique,
and dynamics annotations with no forced alignment or transcription required.

The creation pipeline has four stages (tools live in [`dataset_tools/`](dataset_tools/)):

1. **Source MIDI + dynamics.** We start from
  [MID-FiLD](https://github.com/pozalabs/MID-FiLD) ([Ryu et al., 2024](https://doi.org/10.1609/aaai.v38i1.27774))
  as the source material because it ships **human-written dynamics curves** (MIDI CC1),
  giving expressive, musically-plausible loudness shaping rather than synthetic envelopes.
2. **Monophonic line extraction** ([`dataset_tools/split_symphony/`](dataset_tools/split_symphony/)).
  We measure each file's polyphony rate and use closest-pitch and highest-pitch extraction
  methods to create multiple solo-line variants while preserving the original CC1 dynamics.
3. **Technique annotation** ([`dataset_tools/keyswitch_assignment/`](dataset_tools/keyswitch_assignment/)).
  We generate short overlaps for connected notes, assign a technique to every note, and
  encode it as a **MIDI keyswitch below the violin range** (the playable range is G3–A7, so keyswitches never collide with real notes). Labels
   are assigned with **duration-based probabilistic heuristics** that mirror idiomatic writing:
   short notes are more likely to be *spiccato / staccato / pizzicato*, while long notes are more
   likely to be *legato / trill / harmonic*.
4. **Offline rendering.** We built a **JUCE-based offline rendering framework for Kontakt**
  and rendered the annotated MIDI to **48 kHz stereo** audio using **Joshua Bell Violin**
  (Embertone), a high-quality commercial solo-violin instrument. The project-specific renderer
  and commercial sample library are not redistributed.

**Dynamics representation.** Continuous dynamics come from **MIDI CC1**. CC1 events (0–127)
are min-max normalized to [0, 1] and expanded with a zero-order hold, producing a
piecewise-constant, frame-aligned dynamics curve used as a conditioning signal. Note that in the training set, CC#1 and CC#11 are both present with exact same values. In the test set we condition on one of 4 normalized dynamics patterns and only CC#1 is present.

### The 12 playing techniques

CSV-TD annotates each note with one of **12 note-level playing techniques**. The technique
condition is a binary pianoroll `R_tech ∈ {0,1}^(12×T)` aligned to the MIDI notes, where each
row is one technique and each event spans the duration of its note. The mapping below matches
the keyswitches in [`configs/ks_config.yaml`](configs/ks_config.yaml) and the technique IDs in
[`src/data/components/midi_processor.py`](src/data/components/midi_processor.py):

| ID | Technique | Keyswitch (MIDI #) | Family |
|---:|-----------|-------------------:|--------|
| 1  | Sustain             | 36 | Bowed, sustained |
| 2  | Tremolo             | 37 | Bowed, articulation |
| 3  | Trill (major)       | 38 | Ornament |
| 4  | Trill (minor)       | 39 | Ornament |
| 5  | Staccato            | 40 | Bowed, short |
| 6  | Spiccato            | 41 | Bowed, short (off-string) |
| 7  | Ricochet            | 42 | Bowed, bouncing |
| 8  | Pizzicato           | 43 | Plucked |
| 9  | Harmonic            | 44 | Harmonic |
| 10 | Legato (bow)        | 48 | Legato |
| 11 | Legato (slur)       | 49 | Legato |
| 12 | Legato (portamento) | 50 | Legato |

### The 7 techniques used for evaluation

While the CSV-TD training set contains all 12 technique labels, model development and
evaluation focus on the **7 most common/representative techniques**:

> **harmonic, pizzicato, slur legato, spiccato, staccato, major trill, minor trill.**

Objective evaluation is run on the **CSV-TD test set** (686 pairs, 3.7 h). For subjective
evaluation we curated **7 single-technique excerpts** plus **3 multi-technique excerpts**
that each combine several techniques. The released single-technique set contains two
harmonic excerpts and one each for pizzicato, slur legato, spiccato, staccato, and trill;
its score metadata uses `trill` as an umbrella label for the two trill conditioning
classes. The [scores, MIDI files, and metadata](subjective_study/) are included in this
repository.

### Additional corpora

- **MOSA** — 19 h of professionally recorded solo violin by 15 expert players; MIDI–audio
  alignments are auto-generated and manually checked. No technique/dynamics annotations.
- **MUSC** — 31 h (939 pairs) of solo violin recordings (Wohlfahrt, Kayser, and Paganini
  etudes) with aligned MIDI and audio, but no technique/dynamics annotations.
- **MOSA_VPT** — a synthetic augmentation of MOSA covering four techniques (sustain,
  harmonic, spiccato, pizzicato); we use the 48 kHz version (76 h) as extra
  technique-supervised training data.

## Citation

```bibtex
@inproceedings{violet2026,
  title     = {{VIOLET}: High-Fidelity Violin Synthesis with Techniques and Dynamics},
  author    = {Baotong Tian and Cynthia Lu and Vincent K. M. Cheung and Ting-Kang Wang and Jonathan Churchill and Zhiyao Duan},
  booktitle = {Proc. of the 27th Int. Society for Music Information Retrieval Conf.},
  year      = {2026},
  address   = {Abu Dhabi, UAE}
}
```

## Acknowledgments

This codebase builds on ideas and tooling from:
- [DAC / DACVAE by Descript](https://github.com/facebookresearch/dacvae)
- [DiT by Peebles & Xie](https://github.com/facebookresearch/DiT)
- [Stable Audio](https://github.com/Stability-AI/stable-audio-tools)
- [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)

## License

The source code is released under the [MIT License](LICENSE). Linked datasets, pretrained
checkpoints, commercial sample libraries, and other third-party assets retain their own terms
and are not relicensed by this repository.
