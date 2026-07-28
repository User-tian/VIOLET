"""Tiled long-form generation aligned with eval (Hydra / ViolinDiffusionModule).

Each window is denoised independently to a full waveform, then windows are
crossfaded together in waveform space (see ``get_blending_weights`` /
``_blend_window_audios``) — not a latent-space MultiDiffusion blend.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchaudio
from lightning import LightningDataModule
from note_seq import midi_io
from tqdm import tqdm

from src.data.components.midi_processor import MIDIProcessorConfig, load_technique_map, process_midi
from src.data.violin_datamodule import build_cropped_binary_midi_roll, build_cropped_tech_roll
from src.inference.render_utils import (
    DEFAULT_EDGE_SILENCE_SECONDS,
    DEFAULT_MAX_RENDER_ATTEMPTS,
    DEFAULT_PLAYABLE_NOTE_MIN_PITCH,
    apply_eval_output_silence_mask,
    make_seeded_noise,
    render_window_diagnostics,
    retry_seed,
)
if TYPE_CHECKING:
    from src.models.violin_diffusion_module import ViolinDiffusionModule


def get_blending_weights(
    window_size: int,
    start_positions: List[int],
    device: torch.device,
) -> torch.Tensor:
    """Build per-window blend weights with flat exposed boundaries.

    Windows are Hann-tapered so overlaps crossfade smoothly, except the first
    window's leading edge and the last window's trailing edge (no neighbor to
    blend with), which are forced to weight 1.0 so the taper doesn't silence
    the very start/end of the full clip.
    """
    n_windows = len(start_positions)
    if n_windows == 0:
        raise ValueError("start_positions must contain at least one window")
    if n_windows == 1:
        return torch.ones(1, 1, window_size, device=device)

    window = torch.hann_window(window_size, periodic=False, device=device)
    blend_weights = window.unsqueeze(0).repeat(n_windows, 1)

    first_exposed = int(start_positions[1] - start_positions[0])
    first_exposed = max(0, min(window_size, first_exposed))
    blend_weights[0, :first_exposed] = 1.0

    last_overlap = int(start_positions[-2] + window_size - start_positions[-1])
    last_overlap = max(0, min(window_size, last_overlap))
    blend_weights[-1, last_overlap:] = 1.0

    return blend_weights.unsqueeze(1)


def _pad_1d_windows(
    windows: List[dict],
    key: str,
    N: int,
    device: torch.device,
    dtype: torch.dtype = torch.long,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Match ``ViolinCollator._pad_1d``: pad each window's 1D token row to its own max length."""
    tensors = [w[key] for w in windows]
    max_len = max(int(t.shape[0]) for t in tensors)
    padded = torch.zeros(N, max_len, dtype=dtype, device=device)
    lengths = torch.zeros(N, dtype=torch.long, device=device)
    for i, t in enumerate(tensors):
        t = t.to(device=device, dtype=dtype)
        L = int(t.shape[0])
        padded[i, :L] = t
        lengths[i] = L
    return padded, lengths


def _batch_window_dicts(
    windows: List[dict],
    device: torch.device,
    dtype: Optional[torch.dtype],
) -> Dict[str, torch.Tensor]:
    """Collate window dicts like ``ViolinCollator.collate`` (separate lengths per modality)."""
    N = len(windows)

    midi_tokens, midi_length = _pad_1d_windows(windows, "midi_tokens", N, device)
    tech_tokens, tech_length = _pad_1d_windows(windows, "tech_tokens", N, device)
    velocity_tokens, velocity_length = _pad_1d_windows(windows, "velocity_tokens", N, device)
    pos_midi, pos_midi_length = _pad_1d_windows(windows, "pos_midi", N, device)

    cc_tensors = [w["cc_tokens"] for w in windows]
    cc_max_len = max(int(t.shape[0]) for t in cc_tensors)
    cc_c = int(cc_tensors[0].shape[1])
    cc_tokens = torch.zeros(N, cc_max_len, cc_c, dtype=torch.float32, device=device)
    cc_length = torch.zeros(N, dtype=torch.long, device=device)
    for i, t in enumerate(cc_tensors):
        t = t.to(device=device, dtype=torch.float32)
        L = int(t.shape[0])
        cc_tokens[i, :L, :] = t
        cc_length[i] = L
    if dtype is not None:
        cc_tokens = cc_tokens.to(dtype)

    out: Dict[str, torch.Tensor] = {
        "midi_tokens": midi_tokens,
        "tech_tokens": tech_tokens,
        "velocity_tokens": velocity_tokens,
        "pos_midi": pos_midi,
        "cc_tokens": cc_tokens,
        "midi_length": midi_length,
        "tech_length": tech_length,
        "velocity_length": velocity_length,
        "pos_midi_length": pos_midi_length,
        "cc_length": cc_length,
    }

    if windows[0].get("midi_roll") is not None:
        mr = [w["midi_roll"] for w in windows]
        max_t = max(int(t.shape[1]) for t in mr)
        P = int(mr[0].shape[0])
        stacked = torch.zeros(N, P, max_t, dtype=torch.float32, device=device)
        midi_roll_length = torch.zeros(N, dtype=torch.long, device=device)
        for i, t in enumerate(mr):
            t = t.to(device=device, dtype=torch.float32)
            lt = int(t.shape[1])
            stacked[i, :, :lt] = t
            midi_roll_length[i] = lt
        out["midi_roll"] = stacked
        out["midi_roll_length"] = midi_roll_length

    if windows[0].get("tech_roll") is not None:
        tr = [w["tech_roll"] for w in windows]
        max_t = max(int(t.shape[1]) for t in tr)
        C = int(tr[0].shape[0])
        stacked = torch.zeros(N, C, max_t, dtype=torch.float32, device=device)
        for i, t in enumerate(tr):
            t = t.to(device=device, dtype=torch.float32)
            lt = int(t.shape[1])
            stacked[i, :, :lt] = t
        out["tech_roll"] = stacked

    return out


def prepare_eval_aligned_windows(
    midi_path: str,
    technique_map: dict,
    technique_pitch_keys: set,
    sequence: Any,
    total_duration: float,
    window_seconds: float,
    overlap: float,
    frame_rate: float,
    use_velocity_token: bool,
    use_midi_roll: bool,
    use_tech_roll: bool,
    midi_roll_min_pitch: int,
    midi_roll_max_pitch: int,
    num_techniques: int,
    tech_roll_note_duration_seconds: Optional[float],
) -> Tuple[List[dict], int, int, int]:
    cc_frame_hop = 1.0 / frame_rate
    window_frames = int(round(window_seconds * frame_rate))
    step_frames = int(window_frames * (1 - overlap))
    total_frames = int(math.ceil(total_duration * frame_rate))
    total_frames = max(total_frames, window_frames)

    midi_total_duration = sequence.total_time
    if total_duration > midi_total_duration + 1.0:
        print(
            f"[WARNING] Requested duration ({total_duration:.1f}s) exceeds MIDI "
            f"duration ({midi_total_duration:.1f}s). Trailing windows may be silent."
        )

    positions: List[int] = []
    current = 0
    while current + window_frames <= total_frames:
        positions.append(current)
        current += step_frames

    if len(positions) == 0 or (positions[-1] + window_frames) < total_frames:
        positions.append(max(0, total_frames - window_frames))

    print(
        f"MIDI duration: {midi_total_duration:.1f}s | Target: {total_duration:.1f}s | "
        f"Total frames: {total_frames} | Window: {window_frames} frames "
        f"({window_seconds}s) | Step: {step_frames} frames | Num windows: {len(positions)}"
    )

    windows: List[dict] = []
    for start_frame in tqdm(positions, desc="Processing MIDI windows"):
        start_time = start_frame / frame_rate

        midi_cfg = MIDIProcessorConfig(
            start_time=start_time,
            window_seconds=window_seconds,
            time_step_seconds=0.01,
            cc_frame_hop=cc_frame_hop,
            technique_lead=0.01,
            playable_note_min_pitch=DEFAULT_PLAYABLE_NOTE_MIN_PITCH,
            midi_note_min_pitch=midi_roll_min_pitch if use_midi_roll else None,
            midi_note_max_pitch=midi_roll_max_pitch if use_midi_roll else None,
            default_cc1_value=100,
            default_technique_id=1,
        )

        note_tokens, tech_seq, velocity_seq, pos_midi, cc_frames, tech_onset_seq = process_midi(
            midi_path, technique_map, midi_cfg, sequence=sequence
        )

        tech_for_tokens = tech_onset_seq if use_midi_roll else tech_seq

        if not use_velocity_token:
            velocity_seq = [0] * len(velocity_seq)

        if cc_frames.shape[0] < window_frames:
            pad_size = window_frames - cc_frames.shape[0]
            cc_frames = F.pad(cc_frames, (0, 0, 0, pad_size), value=0.0)
        elif cc_frames.shape[0] > window_frames:
            cc_frames = cc_frames[:window_frames]

        wdict: dict = {
            "midi_tokens": torch.LongTensor(note_tokens),
            "tech_tokens": torch.LongTensor(tech_for_tokens),
            "velocity_tokens": torch.LongTensor(velocity_seq),
            "pos_midi": torch.LongTensor(pos_midi),
            "cc_tokens": cc_frames.float(),
            "midi_length": len(note_tokens),
            "start_frame": start_frame,
        }

        if use_midi_roll:
            wdict["midi_roll"] = build_cropped_binary_midi_roll(
                sequence=sequence,
                start_time=start_time,
                window_seconds=window_seconds,
                step_seconds=midi_cfg.time_step_seconds,
                min_pitch=midi_roll_min_pitch,
                max_pitch=midi_roll_max_pitch,
                ignore_pitches=technique_pitch_keys,
                pitch_shift=0,
            )

        if use_tech_roll:
            wdict["tech_roll"] = build_cropped_tech_roll(
                sequence=sequence,
                tech_map=technique_map,
                start_time=start_time,
                window_seconds=window_seconds,
                step_seconds=midi_cfg.time_step_seconds,
                num_techniques=num_techniques,
                technique_lead=midi_cfg.technique_lead,
                tech_note_duration_seconds=tech_roll_note_duration_seconds,
                playable_note_min_pitch=midi_cfg.playable_note_min_pitch,
                default_technique_id=midi_cfg.default_technique_id,
            )

        windows.append(wdict)

    return windows, total_frames, window_frames, step_frames


def _data_hparams(dm: LightningDataModule) -> Any:
    return dm.hparams


@torch.no_grad()
def _render_single_window_audio(
    model: ViolinDiffusionModule,
    window: dict,
    seed: int,
) -> torch.Tensor:
    device = next(model.net.parameters()).device
    model_dtype = next(model.net.parameters()).dtype
    raw_batch = _batch_window_dicts([window], device=device, dtype=model_dtype)
    initial_noise = make_seeded_noise(
        (1, model.latent_dim, model.generated_frame_length),
        seed=seed,
        device=device,
        dtype=torch.float32,
    )
    latents = model.synthesize_from_noise(initial_noise, raw_batch)
    return model.codec_model.decode(latents.float()).cpu()


def _blend_window_audios(
    window_audios: List[torch.Tensor],
    start_positions_samples: List[int],
    total_samples: int,
) -> torch.Tensor:
    if not window_audios:
        raise ValueError("window_audios must contain at least one element")

    device = window_audios[0].device
    window_samples = int(window_audios[0].shape[-1])
    weights = get_blending_weights(window_samples, start_positions_samples, device=device)
    audio_sum = torch.zeros(1, total_samples, dtype=window_audios[0].dtype, device=device)
    weight_sum = torch.zeros_like(audio_sum)

    for idx, wav in enumerate(window_audios):
        start = start_positions_samples[idx]
        end = start + window_samples
        audio_sum[:, start:end] += wav[0] * weights[idx]
        weight_sum[:, start:end] += weights[idx]

    weight_sum = torch.clamp(weight_sum, min=1e-8)
    return audio_sum / weight_sum


@torch.no_grad()
def long_audio_test_step(model: ViolinDiffusionModule, batch: Any, la: dict) -> None:
    """Tiled long-form generation inside ``trainer.test()`` (called from ``test_step``)."""
    if "midi_path" not in batch or not isinstance(batch["midi_path"], list):
        raise ValueError(
            "long_audio requires batch['midi_path'] "
            "(use data=eval_midi, data=eval_midi_etudes_10s, or data=eval_midi_etudes_30s)."
        )
    dm = model.trainer.datamodule
    hp = _data_hparams(dm)

    use_midi_roll = getattr(hp, "midi_note_representation", None) == "pianoroll" or bool(
        getattr(hp, "use_midi_roll_condition", False)
    )
    tech_m = getattr(hp, "tech_condition_method", None)
    use_tech_roll = tech_m in ("concat", "adaln")

    ks_path = Path(getattr(hp, "ks_config_path", "configs/ks_config.yaml"))
    if not ks_path.is_absolute():
        root = os.environ.get("PROJECT_ROOT", ".")
        ks_path = Path(root) / ks_path
    technique_map = load_technique_map(ks_path)
    technique_pitch_keys = set(technique_map.keys())

    use_velocity_token = bool(getattr(hp, "use_velocity_token", True))
    midi_roll_min_pitch = int(getattr(hp, "midi_roll_min_pitch", 55))
    midi_roll_max_pitch = int(getattr(hp, "midi_roll_max_pitch", 105))
    num_techniques = int(getattr(hp, "num_techniques", 13))
    tech_roll_note_duration_seconds = getattr(hp, "tech_roll_note_duration_seconds", None)
    if tech_roll_note_duration_seconds is not None:
        tech_roll_note_duration_seconds = float(tech_roll_note_duration_seconds)

    frame_rate = float(model.cc_frame_rate)
    window_frames_cfg = int(model.generated_frame_length)
    window_seconds = window_frames_cfg / frame_rate

    net_wf = int(model.net.input_size)
    if net_wf != window_frames_cfg:
        print(
            f"[WARN] model.net.input_size={net_wf} vs generated_frame_length={window_frames_cfg}; "
            f"using net.input_size for tiling."
        )
        window_frames_cfg = net_wf
        window_seconds = window_frames_cfg / frame_rate

    duration_cfg = la.get("duration_seconds")
    tail_seconds = max(0.0, float(la.get("tail_seconds", 1.0)))
    overlap = float(la.get("overlap", 0.5))
    max_bw = la.get("max_batch_windows")
    if max_bw is not None:
        max_bw = int(max_bw)

    custom_out = la.get("output_dir")
    if custom_out is not None and str(custom_out).strip():
        out_root = str(custom_out)
    else:
        out_root = os.path.join(model.logger.save_dir, "long_samples")
    os.makedirs(out_root, exist_ok=True)

    override_mp = la.get("midi_path")
    if override_mp is not None and str(override_mp).strip():
        override_mp = str(override_mp)

    B = batch["midi_tokens"].shape[0]
    single_out = la.get("output_path")

    if model.codec_model is None:
        raise RuntimeError("codec_model is None; on_test_start should have loaded DACVAE")

    for i in range(B):
        midi_path = batch["midi_path"][i]
        if override_mp:
            if os.path.abspath(str(midi_path)) != os.path.abspath(override_mp) and str(
                override_mp
            ) not in str(midi_path):
                continue

        sequence = midi_io.midi_file_to_note_sequence(str(midi_path))
        if duration_cfg is None:
            base_duration = float(sequence.total_time)
        else:
            base_duration = float(duration_cfg)
        eff_duration = base_duration + tail_seconds
        eff_duration = max(eff_duration, window_seconds)

        windows, total_frames, window_frames, _ = prepare_eval_aligned_windows(
            midi_path=str(midi_path),
            technique_map=technique_map,
            technique_pitch_keys=technique_pitch_keys,
            sequence=sequence,
            total_duration=eff_duration,
            window_seconds=window_seconds,
            overlap=overlap,
            frame_rate=frame_rate,
            use_velocity_token=use_velocity_token,
            use_midi_roll=use_midi_roll,
            use_tech_roll=use_tech_roll,
            midi_roll_min_pitch=midi_roll_min_pitch,
            midi_roll_max_pitch=midi_roll_max_pitch,
            num_techniques=num_techniques,
            tech_roll_note_duration_seconds=tech_roll_note_duration_seconds,
        )

        if window_frames != window_frames_cfg:
            raise RuntimeError(
                f"Window frame mismatch: prepared {window_frames} vs model {window_frames_cfg}"
            )

        base_seed = (
            int(la["seed"])
            if la.get("seed") is not None
            else int(getattr(model, "_eval_seed", torch.initial_seed()) or torch.initial_seed())
        )
        sample_index = i
        window_audios: List[torch.Tensor] = []
        window_attempts: List[int] = []
        window_seeds: List[int] = []
        start_positions_samples: List[int] = []

        for window_idx, window in enumerate(windows):
            start_time = float(window["start_frame"]) / frame_rate
            start_positions_samples.append(int(round(start_time * model.audio_sample_rate)))

            window_selected_wav = None
            window_selected_diag = None
            window_best_score = -float("inf")
            window_selected_attempt = 0
            window_selected_seed = retry_seed(
                base_seed,
                sample_index * max(len(windows), 1) + window_idx,
                0,
            )

            for attempt in range(DEFAULT_MAX_RENDER_ATTEMPTS):
                attempt_seed = retry_seed(
                    base_seed,
                    sample_index * max(len(windows), 1) + window_idx,
                    attempt,
                )
                wav_attempt = _render_single_window_audio(model, window, attempt_seed)
                diag = render_window_diagnostics(
                    waveform=wav_attempt[0],
                    sample_rate=model.audio_sample_rate,
                    sequence=sequence,
                    start_time=start_time,
                    window_seconds=window_seconds,
                    playable_note_min_pitch=DEFAULT_PLAYABLE_NOTE_MIN_PITCH,
                    edge_silence_seconds=DEFAULT_EDGE_SILENCE_SECONDS,
                )

                score = float(diag["peak_rms"]) if diag["expected_activity"] else float("inf")
                if score > window_best_score:
                    window_best_score = score
                    window_selected_wav = wav_attempt
                    window_selected_diag = diag
                    window_selected_attempt = attempt
                    window_selected_seed = attempt_seed

                if not diag["near_silence"]:
                    window_selected_wav = wav_attempt
                    window_selected_diag = diag
                    window_selected_attempt = attempt
                    window_selected_seed = attempt_seed
                    break

                if attempt + 1 < DEFAULT_MAX_RENDER_ATTEMPTS:
                    print(
                        f"[retry] {Path(str(midi_path)).stem}: near-silent 10s window at "
                        f"{start_time:.2f}s (peak RMS {diag['peak_rms_db']:.2f} dBFS); "
                        f"rerendering with seed "
                        f"{retry_seed(base_seed, sample_index * max(len(windows), 1) + window_idx, attempt + 1)}"
                    )

            masked_window = apply_eval_output_silence_mask(
                waveform=window_selected_wav[0],
                sample_rate=model.audio_sample_rate,
                sequence=sequence,
                start_time=start_time,
                window_seconds=window_seconds,
                edge_silence_seconds=DEFAULT_EDGE_SILENCE_SECONDS,
                playable_note_min_pitch=DEFAULT_PLAYABLE_NOTE_MIN_PITCH,
            ).unsqueeze(0)
            window_audios.append(masked_window)
            window_attempts.append(window_selected_attempt + 1)
            window_seeds.append(window_selected_seed)

        total_samples = max(
            int(round(eff_duration * model.audio_sample_rate)),
            start_positions_samples[-1] + int(window_audios[-1].shape[-1]),
        )
        blended_wav = _blend_window_audios(
            window_audios=window_audios,
            start_positions_samples=start_positions_samples,
            total_samples=total_samples,
        )
        wav = apply_eval_output_silence_mask(
            waveform=blended_wav,
            sample_rate=model.audio_sample_rate,
            sequence=sequence,
            start_time=0.0,
            window_seconds=eff_duration,
            edge_silence_seconds=DEFAULT_EDGE_SILENCE_SECONDS,
            playable_note_min_pitch=DEFAULT_PLAYABLE_NOTE_MIN_PITCH,
        )
        base = Path(str(midi_path)).stem
        if B == 1 and single_out is not None and str(single_out).strip():
            out_path = str(single_out)
        else:
            out_path = os.path.join(out_root, f"{base}_long.wav")
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        torchaudio.save(out_path, wav.float(), model.audio_sample_rate)
        print(
            f"Saved long audio → {out_path} "
            f"(base_duration={base_duration:.2f}s, tail={tail_seconds:.2f}s, "
            f"window attempts={window_attempts}, window seeds={window_seeds})"
        )
