# VIOLET Subjective-Study Excerpts

This directory contains the 10 one-page score excerpts used in the VIOLET
subjective listening study: 7 single-technique excerpts and 3 multi-technique
excerpts. Each score is accompanied by JSON metadata and one or more MIDI files
with technique and dynamics controls. Audio renders, listener responses, and
study results are not included here.

## Contents

### Single-technique excerpts

| Technique | Excerpt | Composer | Year | Difficulty | Files |
|---|---|---|---:|---|---|
| Harmonic | *Csárdás* | Vittorio Monti | 1904 | Advanced | [Score](single_technique/harmonic/czardas.pdf), [MIDI](single_technique/harmonic/czardas.mid), [metadata](single_technique/harmonic/czardas.json) |
| Harmonic | *Harmonic Etude* | Cynthia Lu | 2026 | Easy-intermediate | [Score](single_technique/harmonic/harmonic_etude.pdf), [MIDI](single_technique/harmonic/harmonic_etude.mid), [metadata](single_technique/harmonic/harmonic_etude.json) |
| Pizzicato | *Two Guitars* | Traditional | 19th century | Intermediate | [Score](single_technique/pizzicato/two_guitars.pdf), [MIDI 1](single_technique/pizzicato/two_guitars_1.mid), [MIDI 2](single_technique/pizzicato/two_guitars_2.mid), [metadata](single_technique/pizzicato/two_guitars.json) |
| Slur legato | *Wohlfahrt Op. 45 Book 2 No. 41* | Franz Wohlfahrt | 1877 | Easy-intermediate | [Score](single_technique/slur_legato/wohlfahrt_op_45_no_41.pdf), [MIDI 1](single_technique/slur_legato/wohlfahrt_op_45_no_41_1.mid), [MIDI 2](single_technique/slur_legato/wohlfahrt_op_45_no_41_2.mid), [metadata](single_technique/slur_legato/wohlfahrt_op_45_no_41.json) |
| Spiccato | *Kayser Op. 20 No. 9* | Heinrich Ernst Kayser | 1848 | Intermediate | [Score](single_technique/spiccato/kayser_op_20_no_9.pdf), [MIDI 1](single_technique/spiccato/kayser_op_20_no_9_1.mid), [MIDI 2](single_technique/spiccato/kayser_op_20_no_9_2.mid), [metadata](single_technique/spiccato/kayser_op_20_no_9.json) |
| Staccato | *Kayser Op. 20 No. 27* | Heinrich Ernst Kayser | 1848 | Easy | [Score](single_technique/staccato/kayser_op_20_no_27.pdf), [MIDI 1](single_technique/staccato/kayser_op_20_no_27_1.mid), [MIDI 2](single_technique/staccato/kayser_op_20_no_27_2.mid), [metadata](single_technique/staccato/kayser_op_20_no_27.json) |
| Trill | *Kayser Op. 20 No. 17* | Heinrich Ernst Kayser | 1848 | Intermediate | [Score](single_technique/trill/kayser_op_20_no_17.pdf), [MIDI 1](single_technique/trill/kayser_op_20_no_17_1.mid), [MIDI 2](single_technique/trill/kayser_op_20_no_17_2.mid), [metadata](single_technique/trill/kayser_op_20_no_17.json) |

The model represents major and minor trills as separate conditioning classes.
The released score metadata uses the umbrella label `trill` and does not assign
a major/minor class at the excerpt level.

### Multi-technique excerpts

| Excerpt | Composer | Year | Techniques | Difficulty | Files |
|---|---|---:|---|---|---|
| *Violin Concerto No. 1 in A minor, BWV 1041* | Johann Sebastian Bach | 1720 | Slur legato, staccato, trill | Advanced | [Score](multi_technique/bach_violin_concerto_bwv_1041/score.pdf), [MIDI](multi_technique/bach_violin_concerto_bwv_1041/excerpt.mid), [metadata](multi_technique/bach_violin_concerto_bwv_1041/metadata.json) |
| *Gavotte from Mignon* | Ambroise Thomas | 1866 | Slur legato, spiccato, trill, pizzicato | Easy | [Score](multi_technique/gavotte_from_mignon/score.pdf), [MIDI](multi_technique/gavotte_from_mignon/excerpt.mid), [metadata](multi_technique/gavotte_from_mignon/metadata.json) |
| *La Cinquantaine* | Jean Gabriel-Marie | 1887 | Slur legato, spiccato, trill, harmonic | Easy-intermediate | [Score](multi_technique/la_cinquantaine/score.pdf), [MIDI](multi_technique/la_cinquantaine/excerpt.mid), [metadata](multi_technique/la_cinquantaine/metadata.json) |

## MIDI files

The 15 MIDI assets are Standard MIDI files (format 0 or 1) at 480 ticks per
quarter note. They preserve the technique keyswitch notes and MIDI CC1 dynamics
used by the VIOLET rendering pipeline. The low keyswitch pitches are control
events rather than performed violin notes; see the
[keyswitch configuration](../configs/ks_config.yaml) before using the files with
another synthesizer.

Some single-technique score pages have two associated MIDI excerpts. Files
ending in `_1.mid` and `_2.mid` are separate passages, with numbering retained
from the source collection.

## Metadata

Each JSON file contains the following fields:

| Field | Type | Description |
|---|---|---|
| `title` | string | Work or excerpt title |
| `composer` | string | Composer attribution |
| `year` | integer or string | Composition year or period |
| `techniques_covered` | array of strings | Technique labels represented by the excerpt |
| `difficulty_level` | string | Broad performance-difficulty category |
| `score_file` | string | Score filename relative to the metadata file |
| `midi_files` | array of strings | MIDI filenames relative to the metadata file |

Technique labels use snake_case and follow the identifiers used by the VIOLET
evaluation pipeline where applicable.

## License

The score PDFs, MIDI files, and metadata in this directory are released under the
[Creative Commons Attribution 4.0 International License](LICENSE.md). Retain
the title and composer credits when sharing or adapting an excerpt.

The historical compositions are in the public domain. *Harmonic Etude* is
copyright © 2026 Cynthia Lu and is included under the same CC BY 4.0 license.
