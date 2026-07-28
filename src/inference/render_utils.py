from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence

import torch


DEFAULT_EDGE_SILENCE_SECONDS = 0.04
DEFAULT_MAX_RENDER_ATTEMPTS = 4
DEFAULT_PLAYABLE_NOTE_MIN_PITCH = 55
DEFAULT_RETRY_SEED_STRIDE = 1009
DEFAULT_SILENCE_ANALYSIS_HOP_SECONDS = 0.025
DEFAULT_SILENCE_ANALYSIS_WINDOW_SECONDS = 0.1
DEFAULT_SILENCE_PEAK_RMS_THRESHOLD = 1e-2  # ~= -40 dBFS


def make_seeded_noise(
    shape: Sequence[int],
    seed: Optional[int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if seed is None:
        return torch.randn(*shape, device=device, dtype=dtype)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    noise = torch.randn(*shape, generator=generator, dtype=torch.float32)
    return noise.to(device=device, dtype=dtype)


def retry_seed(
    base_seed: int,
    sample_index: int,
    attempt: int,
    seed_stride: int = DEFAULT_RETRY_SEED_STRIDE,
) -> int:
    return int(base_seed) + int(sample_index) * int(seed_stride) + int(attempt)


def _iter_playable_notes(sequence: Any, playable_note_min_pitch: int):
    for note in sequence.notes:
        if int(note.pitch) >= int(playable_note_min_pitch):
            yield note


def window_has_playable_activity(
    sequence: Any,
    start_time: float,
    window_seconds: float,
    playable_note_min_pitch: int = DEFAULT_PLAYABLE_NOTE_MIN_PITCH,
) -> bool:
    end_time = start_time + window_seconds
    for note in _iter_playable_notes(sequence, playable_note_min_pitch):
        note_start = float(note.start_time)
        note_end = float(note.end_time)
        if note_end > start_time and note_start < end_time:
            return True
    return False


def first_effective_note_onset_seconds(
    sequence: Any,
    start_time: float,
    window_seconds: float,
    playable_note_min_pitch: int = DEFAULT_PLAYABLE_NOTE_MIN_PITCH,
) -> Optional[float]:
    end_time = start_time + window_seconds
    first_onset = None

    for note in _iter_playable_notes(sequence, playable_note_min_pitch):
        note_start = float(note.start_time)
        note_end = float(note.end_time)
        if note_end <= start_time or note_start >= end_time:
            continue
        if note_start <= start_time:
            return 0.0
        onset_rel = note_start - start_time
        if first_onset is None or onset_rel < first_onset:
            first_onset = onset_rel

    return first_onset


def apply_eval_output_silence_mask(
    waveform: torch.Tensor,
    sample_rate: int,
    sequence: Any,
    start_time: float,
    window_seconds: float,
    edge_silence_seconds: float = DEFAULT_EDGE_SILENCE_SECONDS,
    playable_note_min_pitch: int = DEFAULT_PLAYABLE_NOTE_MIN_PITCH,
) -> torch.Tensor:
    masked = waveform.clone()
    edge_samples = min(int(round(edge_silence_seconds * sample_rate)), masked.shape[-1])
    if edge_samples <= 0:
        return masked

    masked[..., -edge_samples:] = 0.0
    onset_rel = first_effective_note_onset_seconds(
        sequence=sequence,
        start_time=start_time,
        window_seconds=window_seconds,
        playable_note_min_pitch=playable_note_min_pitch,
    )
    if onset_rel is None or onset_rel >= edge_silence_seconds:
        masked[..., :edge_samples] = 0.0
    return masked


def _mono_waveform(waveform: torch.Tensor) -> torch.Tensor:
    audio = waveform.detach().float().cpu()
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        return audio.mean(dim=0)
    return audio.reshape(-1)


def compute_peak_rms(
    waveform: torch.Tensor,
    sample_rate: int,
    analysis_window_seconds: float = DEFAULT_SILENCE_ANALYSIS_WINDOW_SECONDS,
    analysis_hop_seconds: float = DEFAULT_SILENCE_ANALYSIS_HOP_SECONDS,
) -> float:
    mono = _mono_waveform(waveform)
    if mono.numel() == 0:
        return 0.0

    frame = max(1, int(round(analysis_window_seconds * sample_rate)))
    hop = max(1, int(round(analysis_hop_seconds * sample_rate)))
    if mono.numel() <= frame:
        return float(mono.square().mean().sqrt().item())

    peak_rms = 0.0
    for start in range(0, mono.numel() - frame + 1, hop):
        frame_rms = float(mono[start : start + frame].square().mean().sqrt().item())
        if frame_rms > peak_rms:
            peak_rms = frame_rms
    return peak_rms


def render_window_diagnostics(
    waveform: torch.Tensor,
    sample_rate: int,
    sequence: Any,
    start_time: float,
    window_seconds: float,
    playable_note_min_pitch: int = DEFAULT_PLAYABLE_NOTE_MIN_PITCH,
    edge_silence_seconds: float = DEFAULT_EDGE_SILENCE_SECONDS,
    peak_rms_threshold: float = DEFAULT_SILENCE_PEAK_RMS_THRESHOLD,
    analysis_window_seconds: float = DEFAULT_SILENCE_ANALYSIS_WINDOW_SECONDS,
    analysis_hop_seconds: float = DEFAULT_SILENCE_ANALYSIS_HOP_SECONDS,
) -> Dict[str, Any]:
    expected_activity = window_has_playable_activity(
        sequence=sequence,
        start_time=start_time,
        window_seconds=window_seconds,
        playable_note_min_pitch=playable_note_min_pitch,
    )
    onset_rel = first_effective_note_onset_seconds(
        sequence=sequence,
        start_time=start_time,
        window_seconds=window_seconds,
        playable_note_min_pitch=playable_note_min_pitch,
    )

    mono = _mono_waveform(waveform)
    edge_samples = int(round(edge_silence_seconds * sample_rate))
    start_trim = edge_samples if onset_rel is None or onset_rel >= edge_silence_seconds else 0
    end_trim = edge_samples
    if mono.numel() > start_trim + end_trim:
        analysis = mono[start_trim : mono.numel() - end_trim]
    else:
        analysis = mono

    peak_rms = compute_peak_rms(
        waveform=analysis,
        sample_rate=sample_rate,
        analysis_window_seconds=analysis_window_seconds,
        analysis_hop_seconds=analysis_hop_seconds,
    )
    peak_rms_db = 20.0 * math.log10(max(peak_rms, 1e-12))

    return {
        "expected_activity": expected_activity,
        "first_effective_onset_seconds": onset_rel,
        "peak_rms": peak_rms,
        "peak_rms_db": peak_rms_db,
        "near_silence": bool(expected_activity and peak_rms < peak_rms_threshold),
    }
