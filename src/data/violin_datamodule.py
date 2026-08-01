from typing import Any, Dict, Optional, List, Tuple, Mapping, Set
import glob
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, ConcatDataset, IterableDataset
from src.models.components.dacvae_wrapper import FTDACVAE
from torch_audiomentations import (
    ApplyImpulseResponse,
    Compose,
    HighPassFilter,
    LowPassFilter,
    BandPassFilter,
    # Gain,
    PolarityInversion,
    Shift,
    OneOf,
)
from audiomentations import PitchShift
from note_seq import midi_io
from src.data.components.mt3_midi_tokenizer import MT3MIDIVocabulary

from src.data.components.midi_processor import (
    MIDIProcessorConfig,
    load_technique_map,
    match_techniques,
    process_midi,
)
from src.data.audio_processing_utils import (
    get_audio_info,
    load_waveform,
)


def build_cropped_binary_midi_roll(
    sequence,
    start_time: float,
    window_seconds: float,
    step_seconds: float,
    min_pitch: int,
    max_pitch: int,
    ignore_pitches: Optional[Set[int]] = None,
    pitch_shift: int = 0,
) -> torch.Tensor:
    """Build a cropped binary roll with shape [num_pitches, num_frames]."""
    num_frames = int(np.ceil(window_seconds / step_seconds))
    num_pitches = max_pitch - min_pitch + 1
    roll = np.zeros((num_pitches, num_frames), dtype=np.float32)
    end_time = start_time + window_seconds
    for note in sequence.notes:
        if ignore_pitches and int(note.pitch) in ignore_pitches:
            continue
        shifted_pitch = int(note.pitch) + int(pitch_shift)
        if shifted_pitch < min_pitch or shifted_pitch > max_pitch:
            continue

        onset = max(float(note.start_time), start_time)
        offset = min(float(note.end_time), end_time)
        if offset <= onset:
            continue

        start_idx = int(np.floor((onset - start_time) / step_seconds))
        end_idx = int(np.ceil((offset - start_time) / step_seconds))
        start_idx = max(0, min(start_idx, num_frames))
        end_idx = max(0, min(end_idx, num_frames))
        if end_idx > start_idx:
            roll[shifted_pitch - min_pitch, start_idx:end_idx] = 1.0

    return torch.from_numpy(roll)


def build_cropped_tech_roll(
    sequence,
    tech_map: Dict[int, int],
    start_time: float,
    window_seconds: float,
    step_seconds: float,
    num_techniques: int = 13,
    technique_lead: float = 0.01,
    tech_note_duration_seconds: Optional[float] = None,
    mosa_technique_id: Optional[int] = None,
    playable_note_min_pitch: int = 55,
    default_technique_id: Optional[int] = None,
) -> torch.Tensor:
    """Build a binary technique roll with shape [num_techniques, num_frames].

    Each row corresponds to a technique ID (0 = no technique, 1-12 = techniques).
    For each note, the matched technique channel is filled for the note's duration.
    Notes without a matched technique get channel 0 filled.

    For MOSA_VPT, *mosa_technique_id* overrides technique matching: all notes
    receive the given technique ID. When *tech_note_duration_seconds* is set,
    each technique event uses that fixed duration from note onset instead of the
    note's full duration.
    """
    num_frames = int(np.ceil(window_seconds / step_seconds))
    roll = np.zeros((num_techniques, num_frames), dtype=np.float32)
    end_time = start_time + window_seconds

    if tech_note_duration_seconds is not None and tech_note_duration_seconds <= 0.0:
        raise ValueError("tech_note_duration_seconds must be positive when provided.")

    tech_pitches = set(tech_map.keys())

    if mosa_technique_id is None:
        tech_for_note = match_techniques(
            sequence,
            tech_map,
            start_time,
            window_seconds,
            technique_lead,
            playable_note_min_pitch=playable_note_min_pitch,
            default_technique_id=default_technique_id,
        )
    else:
        tech_for_note = None

    for note in sequence.notes:
        if int(note.pitch) in tech_pitches or int(note.pitch) < int(playable_note_min_pitch):
            continue
        onset = max(float(note.start_time), start_time)
        if tech_note_duration_seconds is None:
            offset = min(float(note.end_time), end_time)
        else:
            offset = min(float(note.start_time) + tech_note_duration_seconds, end_time)
        if offset <= onset:
            continue

        start_idx = int(np.floor((onset - start_time) / step_seconds))
        end_idx = int(np.ceil((offset - start_time) / step_seconds))
        start_idx = max(0, min(start_idx, num_frames))
        end_idx = max(0, min(end_idx, num_frames))
        if end_idx <= start_idx:
            continue

        if mosa_technique_id is not None:
            tid = mosa_technique_id
        elif tech_for_note is not None:
            tid = tech_for_note.get((float(note.start_time), int(note.pitch)), 0)
        else:
            tid = 0

        tid = max(0, min(tid, num_techniques - 1))
        roll[tid, start_idx:end_idx] = 1.0

    return torch.from_numpy(roll)


def shift_note_token_pitches(
    note_tokens: List[int],
    semitones: int,
    vocab: Optional[MT3MIDIVocabulary] = None,
) -> List[int]:
    """Shift only pitch-bearing note tokens and keep timing/control events untouched."""
    if semitones == 0:
        return list(note_tokens)

    vocab = vocab or MT3MIDIVocabulary()
    shifted_tokens: List[int] = []
    note_min = vocab.note_offset
    note_max = vocab.note_offset + vocab.num_pitches

    for token in note_tokens:
        if note_min <= token < note_max:
            pitch = token - vocab.note_offset
            shifted_pitch = max(0, min(pitch + semitones, vocab.num_pitches - 1))
            shifted_tokens.append(vocab.note_id(shifted_pitch))
        else:
            shifted_tokens.append(token)

    return shifted_tokens


def apply_leading_silence_fix(
    waveform: torch.Tensor,
    sequence,
    start_time: float,
    sr: int,
    technique_pitches: Set[int],
    delta_ms: int = 30,
) -> torch.Tensor:
    """Zero-out audio before the first effective MIDI note onset in the crop window.

    When the cropped window starts with silence (no MIDI activity), the original
    audio may contain noise/room tone that misleads the model.  This function:
      1. Skips the fix if any non-keyswitch note straddles the crop start boundary
         (a note was "cut in half" – it's already sounding at time 0 of the window).
      2. Finds the first non-keyswitch note whose onset falls inside the window.
      3. If that onset is > 0 relative to the window, blanks the audio up to
         (onset − delta_ms) and applies a linear fade-in over the final delta_ms.

    Args:
        waveform: Audio tensor of shape [1, T].
        sequence: note_seq.NoteSequence for the full file.
        start_time: Absolute start time (seconds) of the crop window.
        sr: Sample rate in Hz.
        technique_pitches: Set of MIDI pitches used as keyswitch markers (ignored).
        delta_ms: Length of the linear fade-in ramp preceding the first onset (ms).

    Returns:
        Modified waveform [1, T] (operates on a clone; original is not mutated).
    """
    # Guard: any non-keyswitch note that started before the window but is still
    # sounding at start_time means there is already content at t=0.
    for note in sequence.notes:
        if int(note.pitch) in technique_pitches:
            continue
        if float(note.start_time) < start_time < float(note.end_time):
            return waveform  # note cut in half – skip

    # Find the first effective note onset within the window.
    first_onset_abs = None
    for note in sequence.notes:
        if int(note.pitch) in technique_pitches:
            continue
        if float(note.start_time) >= start_time:
            t = float(note.start_time)
            if first_onset_abs is None or t < first_onset_abs:
                first_onset_abs = t

    if first_onset_abs is None:
        return waveform  # no notes in window

    first_onset_rel = first_onset_abs - start_time
    if first_onset_rel <= 0.0:
        return waveform  # first note at or before window start

    T = waveform.shape[1]
    window_length = T / sr
    if first_onset_rel >= window_length:
        return waveform  # first onset is beyond the crop window — nothing to fix

    delta = delta_ms / 1000.0
    fade_start_rel = max(0.0, first_onset_rel - delta)
    fade_start_sample = int(fade_start_rel * sr)
    onset_sample = int(first_onset_rel * sr)

    # Clone to avoid mutating the original tensor.
    waveform = waveform.clone()

    # Absolute silence before the fade region.
    if fade_start_sample > 0:
        waveform[:, :fade_start_sample] = 0.0

    # Linear fade-in of the real audio over [fade_start_sample, onset_sample).
    fade_len = onset_sample - fade_start_sample
    if fade_len > 0:
        ramp = torch.linspace(0.0, 1.0, fade_len, device=waveform.device)
        waveform[:, fade_start_sample:onset_sample] *= ramp

    return waveform


class ViolinWaveformDataset(Dataset):
    """
    Loads paired WAV+MIDI, applies waveform augmentations, and tokenizes MIDI.
    """

    def __init__(
        self,
        data_dir: str,
        split: str,
        dataset_name: str = "VIOLET",
        target_sample_rate: int = 48000,
        ks_config_path: str = "configs/ks_config.yaml",
        rir_dir: Optional[str] = None,
        audio_start_time: float = 0.0,
        audio_window_seconds: Optional[float] = None,
        pitch_augmentation: bool = True,
        max_files: Optional[int] = None,
        mosa_include_normal: bool = True,
        mosa_technique_folders: Optional[List[str]] = None,
        is_real_dataset: bool = False,
        midi_roll_min_pitch: int = 55,
        midi_roll_max_pitch: int = 105,
        num_techniques: int = 13,
        tech_roll_note_duration_seconds: Optional[float] = None,
        leading_silence_prob: float = 0.0,
        leading_silence_delta_ms: int = 30,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.dataset_name = dataset_name.lower()
        self.is_real_dataset = is_real_dataset
        self.target_sample_rate = target_sample_rate
        self.pitch_augmentation = pitch_augmentation
        self.pitch_shift_prob = 0.5
        self.micro_pitch_shift = 0.1
        self.audio_start_time = audio_start_time
        self.audio_window_seconds = audio_window_seconds
        self.ks_config_path = ks_config_path
        self.mosa_include_normal = mosa_include_normal
        self.midi_roll_min_pitch = midi_roll_min_pitch
        self.midi_roll_max_pitch = midi_roll_max_pitch
        self.num_techniques = num_techniques
        self.tech_roll_note_duration_seconds = tech_roll_note_duration_seconds
        self.leading_silence_prob = leading_silence_prob
        self.leading_silence_delta_ms = leading_silence_delta_ms

        if self.dataset_name == "mosa_vpt":
            self.audio_root = os.path.join(self.data_dir, "pt_aug")
            self.midi_root = os.path.join(self.data_dir, "midi")
            # Technique folders present in pt_aug.
            # We expand MOSA_VPT so that each (midi, technique) audio becomes its own training pair.
            # By default we include "normal" (treated as technique_id=0).
            if mosa_technique_folders is not None:
                self.mosa_technique_folders = list(mosa_technique_folders)
            else:
                discovered: List[str] = []
                if os.path.isdir(self.audio_root):
                    for name in sorted(os.listdir(self.audio_root)):
                        p = os.path.join(self.audio_root, name)
                        if os.path.isdir(p):
                            discovered.append(name)
                # Fallback to known set if directory listing is unavailable
                if not discovered:
                    discovered = ["normal", "flageolet", "legato", "spiccato", "pizzicato"]
                if not self.mosa_include_normal and "normal" in discovered:
                    discovered = [d for d in discovered if d != "normal"]
                self.mosa_technique_folders = discovered
            # mapping to ks_config technique ids (default 0 = none)
            self.mosa_technique_to_id = {
                "normal": 0,
                "flageolet": 9,   # harmonic
                "legato": 10,     # bow legato
                "spiccato": 6,
                "pizzicato": 8,
                "staccato": 5
            }
        else:
            self.audio_root = os.path.join(self.data_dir, f"{self.split}_audio")
            self.midi_root = os.path.join(self.data_dir, self.split)

        # Handle MUSC and MOSA_real
        if self.dataset_name == "musc":
            # MUSC: audio/Composer/File.wav, midi/Composer/File.mid
            # We assume data_dir points to MUSC root
            self.audio_root = os.path.join(self.data_dir, "audio")
            self.midi_root = os.path.join(self.data_dir, "midi")
        elif self.dataset_name == "mosa_real":
            # MOSA_real: audio/File.wav, midi/File.mid
            self.audio_root = os.path.join(self.data_dir, "audio")
            self.midi_root = os.path.join(self.data_dir, "midi")
        elif self.dataset_name != "mosa_vpt":
            # VIOLET (default)
            self.audio_root = os.path.join(self.data_dir, f"{self.split}_audio")
            self.midi_root = os.path.join(self.data_dir, self.split)

        # Each entry is (audio_path, midi_path, technique_id_or_None)
        self.pairs: List[Tuple[str, str, Optional[int]]] = self._discover_pairs()
        if max_files:
            self.pairs = self.pairs[:max_files]

        # Technique mapping for pitch-shift bounds
        self.technique_mapping = load_technique_map(Path(ks_config_path))
        self.technique_pitches = set(self.technique_mapping.keys())
        self.midi_cfg = MIDIProcessorConfig(
            start_time=self.audio_start_time,
            window_seconds=self.audio_window_seconds if self.audio_window_seconds else 10.0,
            time_step_seconds=0.01,
            cc_frame_hop=0.04,
            technique_lead=0.01,
            playable_note_min_pitch=55,
            midi_note_min_pitch=self.midi_roll_min_pitch,
            midi_note_max_pitch=self.midi_roll_max_pitch,
        )

        # Caches
        self.midi_cache = {}
        self.audio_meta_cache = {}

    def _discover_pairs(self) -> List[Tuple[str, str]]:
        """
        Pair audio and MIDI by shared stem. Audio lives under `<split>_audio/`,
        MIDI under `<split>/`, preserving subdirectory structure.
        """
        if self.dataset_name == "mosa_vpt":
            audio_exts = [".wav", ".flac"]
            midi_exts = [".mid", ".midi"]
            pairs: List[Tuple[str, str, Optional[int]]] = []

            for mext in midi_exts:
                for midi_path in glob.glob(
                    os.path.join(self.midi_root, f"**/*{mext}"), recursive=True
                ):
                    rel = os.path.relpath(midi_path, self.midi_root)
                    stem, _ = os.path.splitext(rel)
                    for folder in self.mosa_technique_folders:
                        for aext in audio_exts:
                            candidate = os.path.join(
                                self.audio_root, folder, stem + aext
                            )
                            if os.path.exists(candidate):
                                technique_id = self.mosa_technique_to_id.get(folder, 0)
                                # One training pair per available (midi, technique) audio
                                pairs.append((candidate, midi_path, technique_id))
                                break
            pairs.sort(key=lambda x: (x[1], x[0]))
            return pairs

        audio_exts = [".wav", ".flac"]
        midi_exts = [".mid", ".midi"]
        pairs: List[Tuple[str, str, Optional[int]]] = []

        for ext in audio_exts:
            for audio_path in glob.glob(
                os.path.join(self.audio_root, f"**/*{ext}"), recursive=True
            ):
                rel = os.path.relpath(audio_path, self.audio_root)
                stem, _ = os.path.splitext(rel)
                midi_path = None
                for mext in midi_exts:
                    candidate = os.path.join(self.midi_root, stem + mext)
                    if os.path.exists(candidate):
                        midi_path = candidate
                        break
                if midi_path:
                    pairs.append((audio_path, midi_path, None))

        pairs.sort()
        return pairs

    def _discover_rirs(self, rir_dir: str) -> List[str]:
        if not os.path.exists(rir_dir):
            return []
        return sorted(glob.glob(os.path.join(rir_dir, "**/*.wav"), recursive=True))

    def __len__(self) -> int:
        return len(self.pairs)


    def _crop_data(self, embeddings: torch.Tensor, midi_path: str) -> Tuple[torch.Tensor, MIDIProcessorConfig]:
        """
        Deprecated.
        """
        pass

    def _apply_base_augmentations(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Deprecated.
        """
        pass

    def _apply_reverb(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.reverb_transform is None:
            return waveform
        return self.reverb_transform(
            waveform.unsqueeze(0), sample_rate=self.target_sample_rate
        ).squeeze(0)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.dataset_name == "mosa_vpt":
            audio_path, midi_path, technique_id = self.pairs[idx]
            instrument_id = 1
            mosa_folder = os.path.basename(os.path.dirname(audio_path)).lower()
        else:
            audio_path, midi_path, _ = self.pairs[idx]
            technique_id = None
            instrument_id = 0
            mosa_folder = None

        # Cache Audio Metadata
        if audio_path not in self.audio_meta_cache:
            info = get_audio_info(audio_path)
            self.audio_meta_cache[audio_path] = (info.sample_rate, info.num_frames)
        src_sr, src_frames = self.audio_meta_cache[audio_path]

        # 1. Load waveform (optimized)
        window_sec = self.audio_window_seconds
        start_arg = None

        if self.split != "train" and window_sec is not None:
            # metadata = torchaudio.info(audio_path) # Cached
            duration_sec = src_frames / src_sr
            start_time = self.audio_start_time
            if start_time + window_sec > duration_sec:
                start_time = 0.0
            start_arg = start_time

        waveform, start_time = load_waveform(
            audio_path,
            start=start_arg,
            tar_sr=self.target_sample_rate,
            tar_len=window_sec,
            return_start=True,
            known_sr=src_sr,
            known_frames=src_frames
        )
        # load_waveform returns [T], make it [1, T]
        waveform = waveform.unsqueeze(0)

        if window_sec is None:
             window_sec = waveform.shape[-1] / self.target_sample_rate

        # 3. No encoding here - just return waveform
        # waveform = self._apply_pitch_shift(waveform, apply_pitch_shift)

        # 4. MIDI tokenization (aligned window)
        midi_cfg = MIDIProcessorConfig(
            start_time=start_time,
            window_seconds=window_sec,
            time_step_seconds=self.midi_cfg.time_step_seconds,
            cc_frame_hop=self.midi_cfg.cc_frame_hop,
            technique_lead=self.midi_cfg.technique_lead,
            playable_note_min_pitch=self.midi_cfg.playable_note_min_pitch,
            midi_note_min_pitch=self.midi_roll_min_pitch,
            midi_note_max_pitch=self.midi_roll_max_pitch,
            default_cc1_value=self.midi_cfg.default_cc1_value,
            default_technique_id=self.midi_cfg.default_technique_id,
        )

        # Cache MIDI Sequence
        if midi_path not in self.midi_cache:
            self.midi_cache[midi_path] = midi_io.midi_file_to_note_sequence(str(midi_path))
        sequence = self.midi_cache[midi_path]

        # Direction 2: leading-silence fix — blank pre-onset audio and fade in.
        if (
            self.split == "train"
            and self.leading_silence_prob > 0.0
            and random.random() < self.leading_silence_prob
        ):
            waveform = apply_leading_silence_fix(
                waveform,
                sequence,
                start_time,
                self.target_sample_rate,
                self.technique_pitches,
                delta_ms=self.leading_silence_delta_ms,
            )

        note_tokens, tech_seq, velocity_seq, pos_midi, cc_frames, tech_onset_seq = process_midi(
            midi_path,
            self.technique_mapping,
            midi_cfg,
            sequence=sequence
        )

        # --- Pitch Shift Logic ---
        vocab = MT3MIDIVocabulary()
        fixed_midi_shift = 12 if mosa_folder == "flageolet" else 0
        if fixed_midi_shift != 0:
            # MOSA_VPT flageolet is rendered one octave lower than VIOLET harmonic.
            # Shift only the conditioning pitch content and keep event timing unchanged.
            note_tokens = shift_note_token_pitches(note_tokens, fixed_midi_shift, vocab=vocab)
        micro_shift = 0.0
        int_shift = 0

        # For paired training data, enabled pitch augmentation shifts audio and MIDI
        # together. The fixed MOSA_VPT flageolet octave lift above is a separate
        # MIDI-only normalization step to match VIOLET harmonic semantics.
        if (
            self.pitch_augmentation
            and self.split == "train"
            and random.random() < 0.5
        ):
            # Identify pitches
            current_pitches = []
            # note_tokens is List[int] here
            for t in note_tokens:
                if vocab.note_offset <= t < vocab.note_offset + vocab.num_pitches:
                    current_pitches.append(t - vocab.note_offset)

            # Calculate integer shift
            if current_pitches:
                min_p = min(current_pitches)
                max_p = max(current_pitches)

                # Violin range G3(55) - A7(105)
                valid_min = 55 - min_p
                valid_max = 105 - max_p

                # Intersect with [-3, 3]
                lower = max(-3, valid_min)
                upper = min(3, valid_max)

                if lower <= upper:
                    int_shift = random.randint(lower, upper)

            # Apply to Audio (Integer part only here)
            if int_shift != 0:
                # waveform is [1, T] Tensor. audiomentations expects numpy [T] or [C, T]
                # Convert to numpy
                wav_np = waveform.numpy()

                # Apply pitch shift (CPU)
                # audiomentations PitchShift: min_semitones, max_semitones
                augmentor = PitchShift(
                    min_semitones=int_shift,
                    max_semitones=int_shift,
                    p=1.0
                )

                # output is numpy array
                shifted_np = augmentor(samples=wav_np, sample_rate=self.target_sample_rate)

                # Convert back to Tensor
                waveform = torch.from_numpy(shifted_np)

            # Apply to MIDI
            if int_shift != 0:
                new_tokens = []
                for t in note_tokens:
                    if vocab.note_offset <= t < vocab.note_offset + vocab.num_pitches:
                        new_tokens.append(t + int_shift)
                    else:
                        new_tokens.append(t)
                note_tokens = new_tokens

        if self.dataset_name == "mosa_vpt":
            # Override techniques: all notes share the chosen technique derived from folder
            # Only apply technique to Note On events (where velocity > 0)
            # Note tokens are 1-128.
            tech_seq = [
                technique_id if (v > 0 and 1 <= t <= 128) else 0
                for t, v in zip(note_tokens, velocity_seq)
            ]
            # Rebuild tech_onset_seq from overridden tech_seq (at note-on positions; onset-only mode)
            tech_onset_seq = [
                tech_seq[i] for i in range(len(tech_seq))
                if velocity_seq[i] > 0 and 1 <= note_tokens[i] <= 128
            ]

        # Velocity conditioning is not used by the model; always zero it out.
        velocity_seq = [0] * len(velocity_seq)

        # Onset-only technique sequence (one per note onset), aligned to the MIDI-roll grid.
        tech_for_tokens = tech_onset_seq

        # Build cropped binary roll from sequence with pitch_shift applied (aligned with audio augmentation)
        midi_roll = build_cropped_binary_midi_roll(
            sequence=sequence,
            start_time=start_time,
            window_seconds=window_sec,
            step_seconds=self.midi_cfg.time_step_seconds,
            min_pitch=self.midi_roll_min_pitch,
            max_pitch=self.midi_roll_max_pitch,
            ignore_pitches=self.technique_pitches,
            pitch_shift=fixed_midi_shift + int_shift,
        )

        # Build technique pianoroll.
        mosa_tid = technique_id if self.dataset_name == "mosa_vpt" else None
        tech_roll = build_cropped_tech_roll(
            sequence=sequence,
            tech_map=self.technique_mapping,
            start_time=start_time,
            window_seconds=window_sec,
            step_seconds=self.midi_cfg.time_step_seconds,
            num_techniques=self.num_techniques,
            technique_lead=self.midi_cfg.technique_lead,
            tech_note_duration_seconds=self.tech_roll_note_duration_seconds,
            mosa_technique_id=mosa_tid,
            playable_note_min_pitch=midi_cfg.playable_note_min_pitch,
            default_technique_id=midi_cfg.default_technique_id,
        )

        note_tokens = torch.LongTensor(note_tokens)
        tech_tokens = torch.LongTensor(tech_for_tokens)
        velocity_tokens = torch.LongTensor(velocity_seq)
        pos_midi = torch.LongTensor(pos_midi)
        cc_tokens = cc_frames.float()  # [F,1]

        item = {
            "waveform": waveform,  # [1, T]
            "midi_tokens": note_tokens,
            "tech_tokens": tech_tokens,
            "velocity_tokens": velocity_tokens,
            "pos_midi": pos_midi,
            "cc_tokens": cc_tokens,
            "audio_path": audio_path,
            "midi_path": midi_path,
            "start_time": start_time,
            "pitch_shift": torch.tensor(micro_shift, dtype=torch.float32),
            "instrument_id": torch.tensor(instrument_id, dtype=torch.int64),
            "is_real": torch.tensor(self.is_real_dataset, dtype=torch.bool),
            "midi_roll": midi_roll,
            "tech_roll": tech_roll,
        }
        return item


class EvalMidiDataset(Dataset):
    """
    Loads only MIDI files for evaluation/generation.
    Uses first 10s by default.
    Globs recursively for .mid and .midi files.
    """
    def __init__(
        self,
        data_dir: str,
        target_sample_rate: int = 48000,
        ks_config_path: str = "configs/ks_config.yaml",
        audio_window_seconds: float = 10.0,
        midi_roll_min_pitch: int = 55,
        midi_roll_max_pitch: int = 105,
        num_techniques: int = 13,
        tech_roll_note_duration_seconds: Optional[float] = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.target_sample_rate = target_sample_rate
        self.audio_window_seconds = audio_window_seconds
        self.ks_config_path = ks_config_path
        self.midi_roll_min_pitch = midi_roll_min_pitch
        self.midi_roll_max_pitch = midi_roll_max_pitch
        self.num_techniques = num_techniques
        self.tech_roll_note_duration_seconds = tech_roll_note_duration_seconds

        # Glob all midi files recursively
        self.midi_files = []
        for ext in ["**/*.mid", "**/*.midi"]:
            self.midi_files.extend(glob.glob(os.path.join(data_dir, ext), recursive=True))
        self.midi_files = sorted(list(set(self.midi_files)))
        if not self.midi_files:
             print(f"Warning: No MIDI files found in {data_dir}")
        else:
             print(f"Found {len(self.midi_files)} MIDI files in {data_dir}")

        self.technique_mapping = load_technique_map(Path(ks_config_path))
        self.midi_cfg = MIDIProcessorConfig(
            start_time=0.0,
            window_seconds=self.audio_window_seconds,
            time_step_seconds=0.01,
            cc_frame_hop=0.04,
            technique_lead=0.01,
            playable_note_min_pitch=55,
            midi_note_min_pitch=self.midi_roll_min_pitch,
            midi_note_max_pitch=self.midi_roll_max_pitch,
            default_cc1_value=100,
            default_technique_id=1,
        )

        # Caches
        self.midi_cache = {}

    def __len__(self) -> int:
        return len(self.midi_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        midi_path = self.midi_files[idx]

        # Cache MIDI Sequence
        if midi_path not in self.midi_cache:
            try:
                self.midi_cache[midi_path] = midi_io.midi_file_to_note_sequence(str(midi_path))
            except Exception as e:
                print(f"Error loading MIDI {midi_path}: {e}")
                # Return a dummy item or raise error. Ideally return None and collator handles it,
                # but let's try to return a safe fallback or let it fail.
                raise e

        sequence = self.midi_cache[midi_path]

        # Process MIDI
        # Note: process_midi handles truncation/windowing via midi_cfg
        note_tokens, tech_seq, velocity_seq, pos_midi, cc_frames, tech_onset_seq = process_midi(
            midi_path,
            self.technique_mapping,
            self.midi_cfg,
            sequence=sequence
        )

        # Onset-only technique sequence (one per note onset), aligned to the MIDI-roll grid.
        tech_for_tokens = tech_onset_seq

        # Create tensors
        note_tokens = torch.LongTensor(note_tokens)
        tech_tokens = torch.LongTensor(tech_for_tokens)
        velocity_tokens = torch.LongTensor(velocity_seq)
        pos_midi = torch.LongTensor(pos_midi)
        if isinstance(cc_frames, torch.Tensor):
            cc_tokens = cc_frames.float()
        else:
            cc_tokens = torch.tensor(cc_frames).float()

        # Dummy waveform [1, T]
        # T = sample_rate * window_seconds
        T = int(self.target_sample_rate * self.audio_window_seconds)
        waveform = torch.zeros(1, T)

        item = {
            "waveform": waveform,
            "midi_tokens": note_tokens,
            "tech_tokens": tech_tokens,
            "velocity_tokens": velocity_tokens,
            "pos_midi": pos_midi,
            "cc_tokens": cc_tokens,
            "audio_path": midi_path, # Use midi path for filename purposes
            "midi_path": midi_path,
            "start_time": 0.0,
            "pitch_shift": torch.tensor(0.0, dtype=torch.float32),
            "instrument_id": torch.tensor(0, dtype=torch.int64),
            "is_real": torch.tensor(False, dtype=torch.bool),
        }
        item["midi_roll"] = build_cropped_binary_midi_roll(
            sequence=sequence,
            start_time=0.0,
            window_seconds=self.audio_window_seconds,
            step_seconds=self.midi_cfg.time_step_seconds,
            min_pitch=self.midi_roll_min_pitch,
            max_pitch=self.midi_roll_max_pitch,
            ignore_pitches=set(self.technique_mapping.keys()),
            pitch_shift=0,
        )
        item["tech_roll"] = build_cropped_tech_roll(
            sequence=sequence,
            tech_map=self.technique_mapping,
            start_time=0.0,
            window_seconds=self.audio_window_seconds,
            step_seconds=self.midi_cfg.time_step_seconds,
            num_techniques=self.num_techniques,
            technique_lead=self.midi_cfg.technique_lead,
            tech_note_duration_seconds=self.tech_roll_note_duration_seconds,
            playable_note_min_pitch=self.midi_cfg.playable_note_min_pitch,
            default_technique_id=self.midi_cfg.default_technique_id,
        )
        return item


class ViolinCollator:
    def __init__(self, silence_pair_prob: float = 0.0):
        self.silence_pair_prob = silence_pair_prob

    @staticmethod
    def _make_silent(item: Dict[str, Any]) -> Dict[str, Any]:
        """Return a copy of *item* with audio and MIDI conditioning zeroed out."""
        silent = dict(item)
        silent["waveform"] = torch.zeros_like(item["waveform"])
        silent["midi_tokens"] = torch.zeros(0, dtype=torch.long)
        silent["tech_tokens"] = torch.zeros(0, dtype=torch.long)
        silent["velocity_tokens"] = torch.zeros(0, dtype=torch.long)
        silent["pos_midi"] = torch.zeros(0, dtype=torch.long)
        silent["cc_tokens"] = torch.zeros_like(item["cc_tokens"])
        if "midi_roll" in item:
            silent["midi_roll"] = torch.zeros_like(item["midi_roll"])
        if "tech_roll" in item:
            silent["tech_roll"] = torch.zeros_like(item["tech_roll"])
        return silent

    def collate(self, minibatch: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        minibatch = [x for x in minibatch if x is not None]
        if len(minibatch) == 0:
            return None

        # Direction 1: inject ~3% silent MIDI-audio pairs.
        if self.silence_pair_prob > 0.0:
            n_silent = round(len(minibatch) * self.silence_pair_prob)
            if n_silent > 0:
                silent_indices = random.sample(range(len(minibatch)), n_silent)
                for idx in silent_indices:
                    minibatch[idx] = self._make_silent(minibatch[idx])

        batch: Dict[str, Any] = {}

        # Embedding padding (instead of waveform)
        if "embeddings" in minibatch[0]:
            embeddings = [item["embeddings"] for item in minibatch]
            # Embeddings: [D, T]. Pad T.
            max_len = max(e.shape[-1] for e in embeddings)
            dims = embeddings[0].shape[0]
            padded_emb = torch.zeros(len(embeddings), dims, max_len)
            emb_lengths = []
            for i, e in enumerate(embeddings):
                length = e.shape[-1]
                padded_emb[i, :, :length] = e
                emb_lengths.append(length)
            batch["embeddings"] = padded_emb
            batch["embeddings_length"] = torch.tensor(emb_lengths, dtype=torch.long)

        # Waveform padding (instead of embeddings)
        waveforms = [item["waveform"] for item in minibatch]
        # Waveform: [1, T]
        max_len = max(w.shape[-1] for w in waveforms)
        # Pad with 0s (silence)
        padded_wav = torch.zeros(len(waveforms), 1, max_len)
        wav_lengths = []
        for i, w in enumerate(waveforms):
            length = w.shape[-1]
            padded_wav[i, :, :length] = w
            wav_lengths.append(length)
        batch["waveform"] = padded_wav
        batch["waveform_length"] = torch.tensor(wav_lengths, dtype=torch.long)

        def _pad_1d(key: str, dtype=torch.long) -> Tuple[torch.Tensor, torch.Tensor]:
            tensors = [item[key] for item in minibatch]
            max_len_local = max(t.shape[0] for t in tensors)
            padded = torch.zeros(len(tensors), max_len_local, dtype=dtype)
            lengths = []
            for i, t in enumerate(tensors):
                length = t.shape[0]
                padded[i, :length] = t
                lengths.append(length)
            return padded, torch.tensor(lengths, dtype=torch.long)

        batch["midi_tokens"], batch["midi_length"] = _pad_1d("midi_tokens")
        batch["tech_tokens"], batch["tech_length"] = _pad_1d("tech_tokens")
        batch["velocity_tokens"], batch["velocity_length"] = _pad_1d("velocity_tokens")
        batch["pos_midi"], batch["pos_midi_length"] = _pad_1d("pos_midi")
        # cc_tokens are [F,1] float; pad along frame dimension
        cc_tensors = [item["cc_tokens"] for item in minibatch]  # each [F,1]
        cc_max_len = max(t.shape[0] for t in cc_tensors)
        cc_padded = torch.zeros(len(cc_tensors), cc_max_len, cc_tensors[0].shape[1])
        cc_lengths = []
        for i, t in enumerate(cc_tensors):
            length = t.shape[0]
            cc_padded[i, :length, :] = t
            cc_lengths.append(length)
        batch["cc_tokens"] = cc_padded
        batch["cc_length"] = torch.tensor(cc_lengths, dtype=torch.long)
        if "midi_roll" in minibatch[0]:
            midi_rolls = [item["midi_roll"] for item in minibatch]  # [P, L]
            midi_roll_max_len = max(t.shape[1] for t in midi_rolls)
            midi_roll_padded = torch.zeros(
                len(midi_rolls),
                midi_rolls[0].shape[0],
                midi_roll_max_len,
                dtype=torch.float32,
            )
            midi_roll_lengths = []
            for i, t in enumerate(midi_rolls):
                length = t.shape[1]
                midi_roll_padded[i, :, :length] = t
                midi_roll_lengths.append(length)
            batch["midi_roll"] = midi_roll_padded
            batch["midi_roll_length"] = torch.tensor(midi_roll_lengths, dtype=torch.long)

        if "tech_roll" in minibatch[0]:
            tech_rolls = [item["tech_roll"] for item in minibatch]  # [C, L]
            tech_roll_max_len = max(t.shape[1] for t in tech_rolls)
            tech_roll_padded = torch.zeros(
                len(tech_rolls),
                tech_rolls[0].shape[0],
                tech_roll_max_len,
                dtype=torch.float32,
            )
            for i, t in enumerate(tech_rolls):
                tech_roll_padded[i, :, :t.shape[1]] = t
            batch["tech_roll"] = tech_roll_padded

        batch["pitch_shift"] = torch.stack([item["pitch_shift"] for item in minibatch])
        batch["instrument_id"] = torch.stack([item["instrument_id"] for item in minibatch])
        batch["is_real"] = torch.stack([item["is_real"] for item in minibatch])
        batch["audio_path"] = [item["audio_path"] for item in minibatch]
        batch["midi_path"] = [item["midi_path"] for item in minibatch]

        return batch


class ViolinDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/VIOLET",
        mosa_dir: Optional[str] = None,
        batch_size: int = 4,
        num_workers: int = 4,
        pin_memory: bool = True,
        dataset_name: str = "VIOLET",
        target_sample_rate: int = 48000,
        audio_start_time: float = 0.0,
        audio_window_seconds: Optional[float] = None,
        ks_config_path: str = "configs/ks_config.yaml",
        rir_dir: Optional[str] = None,
        pitch_augmentation: bool = True,
        max_files: Optional[int] = None,
        mosa_include_normal: bool = True,
        mosa_technique_folders: Optional[List[str]] = None,
        # New datasets
        musc_dir: Optional[str] = None,
        mosa_real_dir: Optional[str] = None,
        # Curriculum
        stage1_steps: int = 100000,
        stage_transition_steps: int = 0,
        stage1_ratio: float = 0.8,
        stage2_ratio: float = 0.3,
        stage1_ratios: Optional[Dict[str, float]] = None,
        stage2_ratios: Optional[Dict[str, float]] = None,

        non_blocking: bool = True, # Ignored, but accepted for compatibility
        prefetch_factor: Optional[int] = None,
        persistent_workers: bool = True,  # Keep workers alive between epochs
        dacvae_posterior_mode: str = "sample",
        dacvae_ckpt: str = "facebook/dacvae-watermarked",
        dacvae_ft_ckpt: Optional[str] = None,
        dacvae_use_ft: bool = False,
        midi_roll_min_pitch: int = 55,
        midi_roll_max_pitch: int = 105,
        num_techniques: int = 13,
        tech_roll_note_duration_seconds: Optional[float] = None,
        silence_pair_prob: float = 0.0,
        leading_silence_prob: float = 0.0,
        leading_silence_delta_ms: int = 30,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        if self.hparams.dataset_name == "eval_midi":
            print(f"Loading EvalMidiDataset from {self.hparams.data_dir}...")
            self.data_val = EvalMidiDataset(
                data_dir=self.hparams.data_dir,
                target_sample_rate=self.hparams.target_sample_rate,
                ks_config_path=self.hparams.ks_config_path,
                audio_window_seconds=self.hparams.audio_window_seconds if self.hparams.audio_window_seconds else 10.0,
                midi_roll_min_pitch=self.hparams.midi_roll_min_pitch,
                midi_roll_max_pitch=self.hparams.midi_roll_max_pitch,
                num_techniques=self.hparams.num_techniques,
                tech_roll_note_duration_seconds=self.hparams.tech_roll_note_duration_seconds,
            )
            # For compatibility if train/test are accessed
            self.data_train = self.data_val
            # (Though usually eval only uses val/test)

            # Initialize DACVAE (needed for model setup usually)
            if not hasattr(self, "dacvae"):
                self.dacvae = FTDACVAE.load_with_finetuned_weights(
                    base_ckpt=self.hparams.dacvae_ckpt,
                    finetuned_ckpt=self.hparams.dacvae_ft_ckpt,
                    use_finetuned=self.hparams.dacvae_use_ft,
                    posterior_mode=self.hparams.dacvae_posterior_mode,
                )
                self.dacvae.eval()
            return

        if not self.data_train:
            # --- COLLECT ALL DATASETS ---
            datasets_dict: Dict[str, Dataset] = {}

            # Primary dataset (VIOLET)
            main_train = ViolinWaveformDataset(
                data_dir=self.hparams.data_dir,
                split="train",
                dataset_name=self.hparams.dataset_name,
                target_sample_rate=self.hparams.target_sample_rate,
                ks_config_path=self.hparams.ks_config_path,
                rir_dir=self.hparams.rir_dir,
                audio_start_time=self.hparams.audio_start_time,
                audio_window_seconds=self.hparams.audio_window_seconds,
                pitch_augmentation=self.hparams.pitch_augmentation,
                max_files=self.hparams.max_files,
                mosa_include_normal=self.hparams.mosa_include_normal,
                mosa_technique_folders=self.hparams.mosa_technique_folders,
                is_real_dataset=False,
                midi_roll_min_pitch=self.hparams.midi_roll_min_pitch,
                midi_roll_max_pitch=self.hparams.midi_roll_max_pitch,
                num_techniques=self.hparams.num_techniques,
                tech_roll_note_duration_seconds=self.hparams.tech_roll_note_duration_seconds,
                leading_silence_prob=self.hparams.leading_silence_prob,
                leading_silence_delta_ms=self.hparams.leading_silence_delta_ms,
            )
            datasets_dict['violet'] = main_train

            # Optional MOSA_VPT
            if self.hparams.mosa_dir:
                mosa_train = ViolinWaveformDataset(
                    data_dir=self.hparams.mosa_dir,
                    split="train",
                    dataset_name="MOSA_VPT",
                    target_sample_rate=self.hparams.target_sample_rate,
                    ks_config_path=self.hparams.ks_config_path,
                    rir_dir=self.hparams.rir_dir,
                    audio_start_time=self.hparams.audio_start_time,
                    audio_window_seconds=self.hparams.audio_window_seconds,
                    pitch_augmentation=self.hparams.pitch_augmentation,
                    max_files=self.hparams.max_files,
                    mosa_include_normal=self.hparams.mosa_include_normal,
                    mosa_technique_folders=self.hparams.mosa_technique_folders,
                    is_real_dataset=False,
                    midi_roll_min_pitch=self.hparams.midi_roll_min_pitch,
                    midi_roll_max_pitch=self.hparams.midi_roll_max_pitch,
                    num_techniques=self.hparams.num_techniques,
                    tech_roll_note_duration_seconds=self.hparams.tech_roll_note_duration_seconds,
                    leading_silence_prob=self.hparams.leading_silence_prob,
                    leading_silence_delta_ms=self.hparams.leading_silence_delta_ms,
                )
                datasets_dict['mosa_vpt'] = mosa_train

            # --- REAL DATASETS ---
            if self.hparams.musc_dir:
                musc_train = ViolinWaveformDataset(
                    data_dir=self.hparams.musc_dir,
                    split="train",
                    dataset_name="musc",
                    target_sample_rate=self.hparams.target_sample_rate,
                    ks_config_path=self.hparams.ks_config_path,
                    rir_dir=self.hparams.rir_dir,
                    audio_start_time=self.hparams.audio_start_time,
                    audio_window_seconds=self.hparams.audio_window_seconds,
                    pitch_augmentation=self.hparams.pitch_augmentation,
                    max_files=self.hparams.max_files,
                    is_real_dataset=True,
                    midi_roll_min_pitch=self.hparams.midi_roll_min_pitch,
                    midi_roll_max_pitch=self.hparams.midi_roll_max_pitch,
                    num_techniques=self.hparams.num_techniques,
                    tech_roll_note_duration_seconds=self.hparams.tech_roll_note_duration_seconds,
                    leading_silence_prob=self.hparams.leading_silence_prob,
                    leading_silence_delta_ms=self.hparams.leading_silence_delta_ms,
                )
                datasets_dict['musc'] = musc_train

            if self.hparams.mosa_real_dir:
                mosa_real_train = ViolinWaveformDataset(
                    data_dir=self.hparams.mosa_real_dir,
                    split="train",
                    dataset_name="mosa_real",
                    target_sample_rate=self.hparams.target_sample_rate,
                    ks_config_path=self.hparams.ks_config_path,
                    rir_dir=self.hparams.rir_dir,
                    audio_start_time=self.hparams.audio_start_time,
                    audio_window_seconds=self.hparams.audio_window_seconds,
                    pitch_augmentation=self.hparams.pitch_augmentation,
                    max_files=self.hparams.max_files,
                    is_real_dataset=True,
                    midi_roll_min_pitch=self.hparams.midi_roll_min_pitch,
                    midi_roll_max_pitch=self.hparams.midi_roll_max_pitch,
                    num_techniques=self.hparams.num_techniques,
                    tech_roll_note_duration_seconds=self.hparams.tech_roll_note_duration_seconds,
                    leading_silence_prob=self.hparams.leading_silence_prob,
                    leading_silence_delta_ms=self.hparams.leading_silence_delta_ms,
                )
                datasets_dict['mosa_real'] = mosa_real_train

            # --- DETERMINE RATIOS ---
            real_keys = [k for k in datasets_dict if k in ["musc", "mosa_real"]]
            synth_keys = [k for k in datasets_dict if k in ["violet", "mosa_vpt"]]
            has_real = len(real_keys) > 0

            def _resolve_stage_ratios(
                explicit_ratios: Optional[Mapping[str, float]],
                real_ratio: float,
            ) -> Dict[str, float]:
                # Mode A: explicit per-dataset ratios.
                if explicit_ratios is not None:
                    ratios: Dict[str, float] = {}
                    for key, value in dict(explicit_ratios).items():
                        if key in datasets_dict:
                            ratios[key] = float(value)
                    total = sum(max(v, 0.0) for v in ratios.values())
                    if total > 0.0:
                        return {k: max(v, 0.0) / total for k, v in ratios.items()}

                # Mode B: legacy real:synth ratio.
                ratios = {}
                p_real = float(real_ratio) if has_real else 0.0
                p_real = max(0.0, min(1.0, p_real))
                p_synth = 1.0 - p_real

                if len(synth_keys) == 0 and real_keys:
                    p_real, p_synth = 1.0, 0.0
                elif len(real_keys) == 0 and synth_keys:
                    p_real, p_synth = 0.0, 1.0

                if real_keys:
                    for key in real_keys:
                        ratios[key] = p_real / len(real_keys)
                if synth_keys:
                    for key in synth_keys:
                        ratios[key] = p_synth / len(synth_keys)
                return ratios

            s1_ratios = _resolve_stage_ratios(self.hparams.stage1_ratios, self.hparams.stage1_ratio)
            s2_ratios = _resolve_stage_ratios(self.hparams.stage2_ratios, self.hparams.stage2_ratio)

            # --- CREATE DATASET ---
            # If explicit ratios provided OR multiple datasets exist (and we want curriculum behavior aka has_real),
            # use UnifiedViolinDataset.

            use_unified = (
                (self.hparams.stage1_ratios is not None)
                or (self.hparams.stage2_ratios is not None)
                or has_real
            )

            if use_unified:
                from src.data.components.unified_dataset import UnifiedViolinDataset
                self.data_train = UnifiedViolinDataset(
                    datasets=datasets_dict,
                    batch_size=self.hparams.batch_size,
                    stage1_steps=self.hparams.stage1_steps,
                    stage_transition_steps=self.hparams.stage_transition_steps,
                    stage1_ratios=s1_ratios,
                    stage2_ratios=s2_ratios,
                )
            else:
                # Fallback to ConcatDataset (Standard Training)
                # Only include synth datasets as per legacy logic (no real implies only synth)
                synth_ds = [d for k, d in datasets_dict.items() if k in ['violet', 'mosa_vpt']]
                self.data_train = ConcatDataset(synth_ds) if len(synth_ds) > 1 else synth_ds[0]

            # Validation: keep primary dataset test split
            self.data_val = ViolinWaveformDataset(
                data_dir=self.hparams.data_dir,
                split="test",
                dataset_name=self.hparams.dataset_name,
                target_sample_rate=self.hparams.target_sample_rate,
                ks_config_path=self.hparams.ks_config_path,
                rir_dir=self.hparams.rir_dir,
                audio_start_time=self.hparams.audio_start_time,
                audio_window_seconds=self.hparams.audio_window_seconds,
                pitch_augmentation=self.hparams.pitch_augmentation,
                max_files=self.hparams.max_files,
                mosa_include_normal=self.hparams.mosa_include_normal,
                mosa_technique_folders=self.hparams.mosa_technique_folders,
                midi_roll_min_pitch=self.hparams.midi_roll_min_pitch,
                midi_roll_max_pitch=self.hparams.midi_roll_max_pitch,
                num_techniques=self.hparams.num_techniques,
                tech_roll_note_duration_seconds=self.hparams.tech_roll_note_duration_seconds,
            )

        # Initialize DACVAE
        if not hasattr(self, "dacvae"):
            self.dacvae = FTDACVAE.load_with_finetuned_weights(
                base_ckpt=self.hparams.dacvae_ckpt,
                finetuned_ckpt=self.hparams.dacvae_ft_ckpt,
                use_finetuned=self.hparams.dacvae_use_ft,
                posterior_mode=self.hparams.dacvae_posterior_mode,
            )
            self.dacvae.eval()

        # Initialize augmentations (only for training)
        if not hasattr(self, "augmentations"):
            self.augmentations = Compose(
                transforms=[
                    # --- 1. Robustness & Generalization ---
                    # Randomly flip the phase
                    PolarityInversion(p=0.5),
                    # Very light loudness jitter (+/- 2 dB)
                    # Gain(min_gain_in_db=-2.0, max_gain_in_db=2.0, p=0.5),

                    # --- 2. Timbre Variation (Body Resonances) ---
                    OneOf([
                        # "darker" (keep subtle by cutting only very high band)
                        LowPassFilter(
                            min_cutoff_freq=12000,
                            max_cutoff_freq=20000
                        ),
                        # "thinner" (subtle low-end trim)
                        HighPassFilter(
                            min_cutoff_freq=20,
                            max_cutoff_freq=120
                        ),
                        # specific body resonances
                        BandPassFilter(
                            min_center_frequency=150,
                            max_center_frequency=8000,
                            min_bandwidth_fraction=0.05,
                            max_bandwidth_fraction=0.25
                        ),
                    ], p=0.5),
                ],
                output_type="tensor"
            )

        # Initialize micro pitch shift (only for training)
        # if not hasattr(self, "micro_pitch_shift"):
        #     self.micro_pitch_shift = PitchShift(
        #         min_transpose_semitones=-0.15,
        #         max_transpose_semitones=0.15,
        #         p=0.5,
        #         sample_rate=self.hparams.target_sample_rate
        #     )
    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        if batch is None:
            return None


        # Optimization for eval_midi: skip audio processing since we only have dummy audio
        if self.hparams.dataset_name == "eval_midi":
            # Still perform DACVAE device/dtype check to ensure dacvae is on correct device if needed later
            # (though strictly not needed if we skip encoding)
            return batch


        # Extract waveforms: [B, 1, T]
        waveforms = batch["waveform"]
        # Apply augmentations (only in training)
        if self.trainer.training:
            if hasattr(self, "augmentations"):
                # Waveforms must be [B, C, T]. Our data is [B, 1, T] which is correct.
                waveforms = self.augmentations(
                    waveforms, sample_rate=self.hparams.target_sample_rate
                )

            # Apply micro pitch shift
            # if hasattr(self, "micro_pitch_shift"):
            #     # Move to device if needed (PitchShift is an nn.Module)
            #     self.micro_pitch_shift.to(waveforms.device)
            #     waveforms = self.micro_pitch_shift(waveforms)
        # DACVAE Encoding
        # Align DACVAE device/dtype with incoming batch to avoid CPU/GPU mismatch.
        model_device = next(self.dacvae.parameters()).device
        target_device = waveforms.device
        if model_device != target_device:
            self.dacvae.to(target_device)

        target_dtype = next(self.dacvae.parameters()).dtype
        if waveforms.dtype != target_dtype:
            waveforms = waveforms.to(dtype=target_dtype, non_blocking=self.hparams.pin_memory)

        with torch.no_grad():
            # DACVAE encode expects [B, 1, T]
            encoded = self.dacvae.encode(waveforms) # (B, D, T_latent)
        # Update batch
        batch["audio_latents"] = encoded
        batch["audio_latents_length"] = torch.tensor(encoded.shape[-1], dtype=torch.long).repeat(len(encoded))
        if "midi_roll" in batch:
            batch["midi_roll"] = batch["midi_roll"].to(
                encoded.device, non_blocking=self.hparams.pin_memory
            )

        if "waveform" in batch:
            del batch["waveform"]
        del batch["waveform_length"]

        return batch

    def train_dataloader(self):
        if isinstance(self.data_train, IterableDataset) and hasattr(self.data_train, "configure_schedule"):
            trainer = getattr(self, "trainer", None)
            accumulate_grad_batches = getattr(trainer, "accumulate_grad_batches", 1)
            try:
                accumulate_grad_batches = int(accumulate_grad_batches)
            except (TypeError, ValueError):
                accumulate_grad_batches = 1
            initial_optimizer_step = getattr(trainer, "global_step", 0)
            try:
                initial_optimizer_step = int(initial_optimizer_step)
            except (TypeError, ValueError):
                initial_optimizer_step = 0
            self.data_train.configure_schedule(
                accumulate_grad_batches=accumulate_grad_batches,
                initial_optimizer_step=initial_optimizer_step,
            )

        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            collate_fn=ViolinCollator(silence_pair_prob=self.hparams.silence_pair_prob).collate,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False if isinstance(self.data_train, IterableDataset) else True,
            persistent_workers=self.hparams.persistent_workers and self.hparams.num_workers > 0,
            prefetch_factor=self.hparams.prefetch_factor if self.hparams.num_workers > 0 else None,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            collate_fn=ViolinCollator().collate,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=self.hparams.persistent_workers and self.hparams.num_workers > 0,
            prefetch_factor=self.hparams.prefetch_factor if self.hparams.num_workers > 0 else None,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            collate_fn=ViolinCollator().collate,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=self.hparams.persistent_workers and self.hparams.num_workers > 0,
            prefetch_factor=self.hparams.prefetch_factor if self.hparams.num_workers > 0 else None,
        )
