#!/usr/bin/env python
"""
Long-form audio generation using Tiled MultiDiffusion for Rectified Flow models.

Generates audio sequences longer than the model's training window (10s / 250 frames)
by splitting generation into overlapping windows and blending velocity predictions
with Hanning weights at each ODE step (MultiDiffusion approach).

Usage:
    python tools/generate_long_audio.py \
        --midi_path /path/to/long_midi.mid \
        --ema_ckpt_path /path/to/ema_snapshot \
        --output_path output_long.wav \
        --duration 30.0 \
        --num_steps 30 \
        --cfg_scale 1.0 \
        --use_heun

Requirements:
    - A trained DiT model checkpoint (EMA snapshot pickle)
    - DACVAE codec (base + optional finetuned weights)
    - A MIDI file with the desired performance
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

# ── Project root setup ────────────────────────────────────────────────────────
import pyrootutils

ROOT = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from note_seq import midi_io

from src.data.components.midi_processor import (
    MIDIProcessorConfig,
    load_technique_map,
    process_midi,
)
from src.models.components.dacvae_wrapper import FTDACVAE
from src.models.components.diffusion import ReFlow


# ═══════════════════════════════════════════════════════════════════════════════
#  Blending Utilities
# ═══════════════════════════════════════════════════════════════════════════════


def get_blending_weights(window_size: int, device: torch.device) -> torch.Tensor:
    """
    Create a Hanning window for smooth blending of overlapping windows.

    Center of the window has weight 1.0, edges taper towards 0.0.
    Shape: ``(1, 1, window_size)`` for broadcasting with ``(B, C, T)`` latents.
    """
    window = torch.hann_window(window_size, periodic=False, device=device)
    return window.view(1, 1, -1)


# ═══════════════════════════════════════════════════════════════════════════════
#  MIDI Pre-processing
# ═══════════════════════════════════════════════════════════════════════════════


def prepare_window_conditions(
    midi_path: str,
    technique_map: dict,
    total_duration: float,
    window_seconds: float = 10.0,
    overlap: float = 0.5,
    frame_rate: float = 25.0,
    use_velocity: bool = False,
) -> Tuple[List[dict], int, int, int]:
    """
    Pre-process a MIDI file into overlapping windows of conditioning tokens.

    For each window, calls :func:`process_midi` with the appropriate time range
    so that ``midi_tokens``, ``tech_tokens``, ``velocity_tokens``, and ``cc_tokens``
    are correctly aligned to the local window.

    Args:
        midi_path: Path to MIDI file.
        technique_map: Pitch → TechniqueId mapping loaded from ``ks_config.yaml``.
        total_duration: Total desired output duration in seconds.
        window_seconds: Duration of each window in seconds (must match training, default 10s).
        overlap: Overlap ratio between adjacent windows (0.5 = 50%).
        frame_rate: Latent frame rate in Hz (25.0 for DACVAE at 48 kHz).
        use_velocity: Whether to include velocity conditioning.

    Returns:
        ``(windows, total_frames, window_frames, step_frames)`` where *windows* is a
        list of per-window conditioning dicts, each containing ``midi_tokens``,
        ``tech_tokens``, ``velocity_tokens``, ``cc_tokens``, ``midi_length``, and
        ``start_frame``.
    """
    cc_frame_hop = 1.0 / frame_rate  # 0.04 s for 25 Hz
    window_frames = int(window_seconds * frame_rate)
    step_frames = int(window_frames * (1 - overlap))
    total_frames = int(math.ceil(total_duration * frame_rate))

    # Ensure at least one full window
    total_frames = max(total_frames, window_frames)

    # Parse MIDI once
    sequence = midi_io.midi_file_to_note_sequence(str(midi_path))
    midi_total_duration = sequence.total_time

    if total_duration > midi_total_duration + 1.0:
        print(
            f"[WARNING] Requested duration ({total_duration:.1f}s) exceeds MIDI "
            f"duration ({midi_total_duration:.1f}s). Trailing windows may be silent."
        )

    # ── Compute window start positions ────────────────────────────────────
    positions: List[int] = []
    current = 0
    while current + window_frames <= total_frames:
        positions.append(current)
        current += step_frames

    # Ensure the very last frames are covered
    if len(positions) == 0 or (positions[-1] + window_frames) < total_frames:
        positions.append(max(0, total_frames - window_frames))

    print(
        f"MIDI duration: {midi_total_duration:.1f}s | Target: {total_duration:.1f}s | "
        f"Total frames: {total_frames} | Window: {window_frames} frames "
        f"({window_seconds}s) | Step: {step_frames} frames | "
        f"Num windows: {len(positions)}"
    )

    # ── Process each window ───────────────────────────────────────────────
    windows: List[dict] = []
    for start_frame in tqdm(positions, desc="Processing MIDI windows"):
        start_time = start_frame / frame_rate

        midi_cfg = MIDIProcessorConfig(
            start_time=start_time,
            window_seconds=window_seconds,
            cc_frame_hop=cc_frame_hop,
        )

        note_tokens, tech_seq, velocity_seq, pos_midi, cc_frames, _ = process_midi(
            midi_path, technique_map, midi_cfg, sequence=sequence
        )

        if not use_velocity:
            velocity_seq = [0] * len(velocity_seq)

        # Ensure CC frames match window_frames exactly (guard against rounding)
        if cc_frames.shape[0] < window_frames:
            pad_size = window_frames - cc_frames.shape[0]
            cc_frames = F.pad(cc_frames, (0, 0, 0, pad_size), value=0.0)
        elif cc_frames.shape[0] > window_frames:
            cc_frames = cc_frames[:window_frames]

        windows.append(
            {
                "midi_tokens": torch.LongTensor(note_tokens),
                "tech_tokens": torch.LongTensor(tech_seq),
                "velocity_tokens": torch.LongTensor(velocity_seq),
                "pos_midi": torch.LongTensor(pos_midi),
                "cc_tokens": cc_frames.float(),  # (window_frames, 1)
                "midi_length": len(note_tokens),
                "start_frame": start_frame,
            }
        )

    return windows, total_frames, window_frames, step_frames


# ═══════════════════════════════════════════════════════════════════════════════
#  Batching Utility
# ═══════════════════════════════════════════════════════════════════════════════


def batch_windows(
    windows: List[dict], device: torch.device, dtype: Optional[torch.dtype] = None
) -> dict:
    """
    Collate a list of per-window conditioning dicts into a single batched dict.

    Variable-length MIDI/tech/velocity token sequences are zero-padded to the
    maximum length. ``cc_tokens`` are already fixed-length (window_frames).
    If ``dtype`` is set, float tensors (cc_tokens) are cast to match the model.
    """
    N = len(windows)

    max_midi_len = max(w["midi_length"] for w in windows)

    midi_tokens = torch.zeros(N, max_midi_len, dtype=torch.long, device=device)
    tech_tokens = torch.zeros(N, max_midi_len, dtype=torch.long, device=device)
    velocity_tokens = torch.zeros(N, max_midi_len, dtype=torch.long, device=device)
    pos_midi = torch.zeros(N, max_midi_len, dtype=torch.long, device=device)
    midi_length = torch.zeros(N, dtype=torch.long, device=device)

    for i, w in enumerate(windows):
        L = w["midi_length"]
        midi_tokens[i, :L] = w["midi_tokens"]
        tech_tokens[i, :L] = w["tech_tokens"]
        velocity_tokens[i, :L] = w["velocity_tokens"]
        pos_midi[i, :L] = w["pos_midi"]
        midi_length[i] = L

    # CC tokens are fixed length per window; cast to model dtype if needed
    cc_tokens = torch.stack([w["cc_tokens"] for w in windows]).to(device)  # (N, F, 1)
    if dtype is not None:
        cc_tokens = cc_tokens.to(dtype)

    return {
        "midi_tokens": midi_tokens,
        "tech_tokens": tech_tokens,
        "velocity_tokens": velocity_tokens,
        "pos_midi": pos_midi,
        "cc_tokens": cc_tokens,
        "midi_length": midi_length,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Tiled MultiDiffusion Generation
# ═══════════════════════════════════════════════════════════════════════════════


def _compute_blended_velocity(
    x_global: torch.Tensor,
    net: torch.nn.Module,
    diffusion: ReFlow,
    batched_cond: dict,
    start_positions: List[int],
    window_frames: int,
    blend_weights: torch.Tensor,
    sigma: torch.Tensor,
    cfg_scale: float,
    max_batch_windows: Optional[int],
    device: torch.device,
    model_dtype: torch.dtype,
) -> torch.Tensor:
    """
    Run the DiT on every window, then blend predictions with Hanning weights.

    If ``max_batch_windows`` is set, windows are processed in chunks to save
    VRAM.  Otherwise all windows are batched in a single forward pass.
    """
    N_windows = len(start_positions)

    velocity_sum = torch.zeros_like(x_global)
    weight_sum = torch.zeros_like(x_global)

    chunk_size = N_windows if max_batch_windows is None else max_batch_windows

    for chunk_start in range(0, N_windows, chunk_size):
        chunk_end = min(chunk_start + chunk_size, N_windows)
        chunk_indices = list(range(chunk_start, chunk_end))

        # Slice latent windows from global state
        x_slices = torch.cat(
            [
                x_global[:, :, start_positions[j] : start_positions[j] + window_frames]
                for j in chunk_indices
            ],
            dim=0,
        )  # (n_chunk, C, window_frames)

        # Slice conditioning for this chunk
        chunk_cond = {
            "midi_tokens": batched_cond["midi_tokens"][chunk_start:chunk_end],
            "tech_tokens": batched_cond["tech_tokens"][chunk_start:chunk_end],
            "velocity_tokens": batched_cond["velocity_tokens"][chunk_start:chunk_end],
            "pos_midi": batched_cond["pos_midi"][chunk_start:chunk_end],
            "cc_tokens": batched_cond["cc_tokens"][chunk_start:chunk_end],
            "midi_length": batched_cond["midi_length"][chunk_start:chunk_end],
        }

        # Prepare sigma batch in model dtype to avoid dtype mismatches
        sigma_batch = torch.full(
            (x_slices.shape[0],),
            fill_value=sigma,
            device=device,
            dtype=model_dtype,
        )

        # Predict velocity via denoise_fn (handles CFG internally)
        v_pred = diffusion.denoise_fn(
            x_slices,
            net=net,
            sigmas=sigma_batch,
            inference=True,
            cond_scale=cfg_scale,
            **chunk_cond,
        )

        # Accumulate weighted predictions
        for k, j in enumerate(chunk_indices):
            start = start_positions[j]
            end = start + window_frames
            velocity_sum[:, :, start:end] += v_pred[k : k + 1] * blend_weights
            weight_sum[:, :, start:end] += blend_weights

    # Normalise (avoid div-by-zero for uncovered frames)
    weight_sum = torch.clamp(weight_sum, min=1e-8)
    return velocity_sum / weight_sum


@torch.no_grad()
def generate_long_audio_tiled(
    net: torch.nn.Module,
    diffusion: ReFlow,
    windows: List[dict],
    total_frames: int,
    window_frames: int,
    latent_channels: int = 128,
    num_steps: int = 30,
    cfg_scale: float = 1.0,
    use_heun: bool = True,
    device: torch.device = torch.device("cuda"),
    max_batch_windows: Optional[int] = None,
) -> torch.Tensor:
    """
    Tiled MultiDiffusion for long-form audio with Rectified Flow.

    At each ODE step the global latent is split into overlapping windows
    (matching the model's trained context length).  Each window receives its
    own MIDI / technique / CC conditioning.  The predicted velocity fields are
    blended with a Hanning window and a single Euler (or Heun) step is taken on
    the global latent.

    Args:
        net:                DiT backbone (``net.in_channels``, ``net.input_size`` used).
        diffusion:          :class:`ReFlow` instance that wraps ``net`` via ``denoise_fn``.
        windows:            Pre-computed window conditions from :func:`prepare_window_conditions`.
        total_frames:       Total number of latent frames to generate.
        window_frames:      Frames per window (must equal ``net.input_size``).
        latent_channels:    Latent channel dim (must equal ``net.in_channels``).
        num_steps:          Number of ODE solver steps (sigma schedule length).
        cfg_scale:          Classifier-Free Guidance scale (1.0 = no guidance).
        use_heun:           Use Heun 2nd-order correction (doubles NFE, better quality).
        device:             Computation device.
        max_batch_windows:  Process at most this many windows per forward pass (None = all).

    Returns:
        Global latent tensor ``(1, latent_channels, total_frames)`` in [-1, 1].
    """
    B = 1  # single-sample generation

    # Match model dtype (e.g. float16 for mixed precision)
    model_dtype = next(net.parameters()).dtype

    # ── Time schedule (Linear: 1.0 → 0.0, matching LinearSchedule) ────────
    # Use model dtype to avoid mixed-precision dtype mismatches in t-embedding
    sigmas = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=model_dtype)

    # Initialise global latent:  x = sigma[0] * noise  (= 1.0 * noise)
    x_global = torch.randn(B, latent_channels, total_frames, device=device, dtype=model_dtype)
    x_global = sigmas[0] * x_global

    # Blending mask (Hanning) — must match model dtype for accumulation
    blend_weights = get_blending_weights(window_frames, device).to(model_dtype)

    # Pre-batch conditions (cc_tokens cast to model dtype)
    start_positions = [w["start_frame"] for w in windows]
    batched_cond = batch_windows(windows, device, dtype=model_dtype)

    N_windows = len(windows)
    nfe_per_step = 2 if use_heun else 1
    print(
        f"\nTiled MultiDiffusion: {total_frames} frames, {N_windows} windows, "
        f"{num_steps} ODE steps ({'Heun' if use_heun else 'Euler'}), "
        f"CFG={cfg_scale}, ~{num_steps * nfe_per_step * N_windows} total NFE"
    )

    # ── ODE solver loop ───────────────────────────────────────────────────
    for i in tqdm(range(num_steps), desc="ODE Steps"):
        sigma_curr = sigmas[i]
        sigma_next = sigmas[i + 1]
        dt = sigma_next - sigma_curr  # negative (1 → 0)

        # Blended velocity at current state
        v_global = _compute_blended_velocity(
            x_global,
            net,
            diffusion,
            batched_cond,
            start_positions,
            window_frames,
            blend_weights,
            sigma_curr,
            cfg_scale,
            max_batch_windows,
            device,
            model_dtype,
        )

        if use_heun and sigma_next != 0:
            # ── Heun (2nd-order) correction ───────────────────────────────
            x_euler = x_global + dt * v_global

            v_global_next = _compute_blended_velocity(
                x_euler,
                net,
                diffusion,
                batched_cond,
                start_positions,
                window_frames,
                blend_weights,
                sigma_next,
                cfg_scale,
                max_batch_windows,
                device,
                model_dtype,
            )

            x_global = x_global + 0.5 * dt * (v_global + v_global_next)
        else:
            # ── Euler step ────────────────────────────────────────────────
            x_global = x_global + dt * v_global

    return x_global.clamp(-1.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
#  Model / Codec Loading
# ═══════════════════════════════════════════════════════════════════════════════


def load_model_and_codec(
    ema_ckpt_path: str,
    codec_base_ckpt: str = "facebook/dacvae-watermarked",
    codec_ft_ckpt: Optional[str] = None,
    codec_use_ft: bool = True,
    codec_posterior_mode: str = "mean",
    device: str = "cuda",
    force_fp32: bool = True,
) -> Tuple[torch.nn.Module, ReFlow, FTDACVAE]:
    """
    Load the DiT backbone (from an EMA pickle), a :class:`ReFlow` diffusion
    wrapper, and the DACVAE codec for decoding latents to audio.

    Args:
        ema_ckpt_path:        Path to the EMA snapshot (pickled ``nn.Module``).
        codec_base_ckpt:      HuggingFace hub ID or local path to base DACVAE.
        codec_ft_ckpt:        Optional path to finetuned DACVAE weights.
        codec_use_ft:         Whether to load finetuned weights.
        codec_posterior_mode:  ``"mean"`` or ``"sample"`` for the DACVAE posterior.
        device:               Target device.

    Returns:
        ``(net, diffusion, codec_model)``
    """
    # ── Load EMA model ────────────────────────────────────────────────────
    print(f"Loading EMA model from {ema_ckpt_path} ...")
    with open(ema_ckpt_path, "rb") as f:
        net = pickle.load(f)
    net = net.to(device=device)
    if force_fp32:
        net = net.float()
    net.eval()
    print(
        f"  → DiT loaded: in_channels={net.in_channels}, "
        f"input_size={net.input_size}"
    )

    # ── ReFlow diffusion (default: for_edm=False) ────────────────────────
    diffusion = ReFlow()
    diffusion.to(device)

    # ── DACVAE codec ──────────────────────────────────────────────────────
    print(f"Loading DACVAE codec (base={codec_base_ckpt}) ...")
    codec_model = FTDACVAE.load_with_finetuned_weights(
        base_ckpt=codec_base_ckpt,
        finetuned_ckpt=codec_ft_ckpt,
        use_finetuned=codec_use_ft,
        posterior_mode=codec_posterior_mode,
    )
    codec_model.eval()
    codec_model.to(device)

    return net, diffusion, codec_model


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate long-form audio via Tiled MultiDiffusion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Required ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--midi_path",
        type=str,
        required=True,
        help="Path to the input MIDI file.",
    )
    parser.add_argument(
        "--ema_ckpt_path",
        type=str,
        required=True,
        help="Path to the EMA checkpoint pickle file (pickled nn.Module).",
    )

    # ── Output ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--output_path",
        type=str,
        default="output_long.wav",
        help="Output WAV file path.",
    )

    # ── Generation parameters ─────────────────────────────────────────────
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Target duration in seconds. Defaults to the MIDI file duration.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=100,
        help="Number of ODE solver steps.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Window overlap ratio (0.5 = 50%% overlap).",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=1.0,
        help="Classifier-Free Guidance scale (1.0 = no guidance).",
    )
    parser.add_argument(
        "--use_heun",
        action="store_true",
        default=True,
        help="Use Heun 2nd-order ODE solver (better quality, 2× NFE).",
    )
    parser.add_argument(
        "--no_heun",
        action="store_true",
        help="Disable Heun correction; use plain Euler stepping.",
    )

    # ── Codec / model paths ───────────────────────────────────────────────
    parser.add_argument(
        "--codec_base_ckpt",
        type=str,
        default="facebook/dacvae-watermarked",
        help="Base DACVAE checkpoint (HF hub ID or local path).",
    )
    parser.add_argument(
        "--codec_ft_ckpt",
        type=str,
        default="dacvae_ft/weights.pth",
        help="Finetuned DACVAE weights path.",
    )
    parser.add_argument(
        "--codec_use_ft",
        action="store_true",
        default=True,
        help="Use finetuned DACVAE weights.",
    )
    parser.add_argument(
        "--no_codec_ft",
        action="store_true",
        help="Disable finetuned DACVAE weights.",
    )
    parser.add_argument(
        "--codec_posterior_mode",
        type=str,
        default="mean",
        choices=["mean", "sample"],
        help="DACVAE posterior mode.",
    )

    parser.add_argument(
        "--force_fp32",
        action="store_true",
        default=True,
        help="Force model weights to float32 for inference stability/precision.",
    )
    parser.add_argument(
        "--keep_model_dtype",
        action="store_true",
        help="Keep EMA model dtype (e.g., bf16/fp16). Overrides --force_fp32.",
    )

    # ── Audio / latent parameters ─────────────────────────────────────────
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=48000,
        help="Audio sample rate (must match codec).",
    )
    parser.add_argument(
        "--frame_rate",
        type=float,
        default=25.0,
        help="Latent frame rate in Hz (DACVAE at 48 kHz → 25 Hz).",
    )

    # ── Technique / conditioning ──────────────────────────────────────────
    parser.add_argument(
        "--ks_config",
        type=str,
        default="configs/ks_config.yaml",
        help="Path to ks_config.yaml for technique mapping.",
    )
    parser.add_argument(
        "--use_velocity",
        action="store_true",
        default=False,
        help="Include velocity conditioning (must match training config).",
    )

    # ── Performance ───────────────────────────────────────────────────────
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Computation device.",
    )
    parser.add_argument(
        "--max_batch_windows",
        type=int,
        default=None,
        help="Max windows per forward pass (limit VRAM; None = all at once).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    args = parser.parse_args()

    # Resolve mutually exclusive flags
    if args.no_heun:
        args.use_heun = False
    if args.no_codec_ft:
        args.codec_use_ft = False
    if args.keep_model_dtype:
        args.force_fp32 = False

    return args


def main() -> None:
    args = parse_args()

    # ── Reproducibility ───────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── Load models ───────────────────────────────────────────────────────
    net, diffusion, codec_model = load_model_and_codec(
        ema_ckpt_path=args.ema_ckpt_path,
        codec_base_ckpt=args.codec_base_ckpt,
        codec_ft_ckpt=args.codec_ft_ckpt,
        codec_use_ft=args.codec_use_ft,
        codec_posterior_mode=args.codec_posterior_mode,
        device=args.device,
        force_fp32=args.force_fp32,
    )

    # ── Read model geometry ───────────────────────────────────────────────
    latent_channels: int = net.in_channels  # 128
    window_frames: int = net.input_size  # 250
    frame_rate: float = args.frame_rate  # 25.0
    window_seconds: float = window_frames / frame_rate  # 10.0

    print(
        f"\nModel geometry: latent_channels={latent_channels}, "
        f"window_frames={window_frames}, frame_rate={frame_rate} Hz, "
        f"window_duration={window_seconds:.1f}s"
    )

    # ── Determine target duration ─────────────────────────────────────────
    if args.duration is None:
        sequence = midi_io.midi_file_to_note_sequence(str(args.midi_path))
        args.duration = sequence.total_time
        print(f"Auto-detected MIDI duration: {args.duration:.1f}s")

    if args.duration < window_seconds:
        print(
            f"[INFO] Duration ({args.duration:.1f}s) < window ({window_seconds:.1f}s); "
            f"padding to one full window."
        )
        args.duration = window_seconds

    # ── Technique mapping ─────────────────────────────────────────────────
    ks_config_path = Path(args.ks_config)
    if not ks_config_path.is_absolute():
        ks_config_path = ROOT / ks_config_path
    technique_map = load_technique_map(ks_config_path)

    # ── Prepare MIDI conditioning for each window ─────────────────────────
    windows, total_frames, window_frames, step_frames = prepare_window_conditions(
        midi_path=args.midi_path,
        technique_map=technique_map,
        total_duration=args.duration,
        window_seconds=window_seconds,
        overlap=args.overlap,
        frame_rate=frame_rate,
        use_velocity=args.use_velocity,
    )

    # ── Tiled MultiDiffusion generation ───────────────────────────────────
    print("\n" + "=" * 60)
    print("  Starting Tiled MultiDiffusion Generation")
    print("=" * 60)

    gen_latents = generate_long_audio_tiled(
        net=net,
        diffusion=diffusion,
        windows=windows,
        total_frames=total_frames,
        window_frames=window_frames,
        latent_channels=latent_channels,
        num_steps=args.num_steps,
        cfg_scale=args.cfg_scale,
        use_heun=args.use_heun,
        device=torch.device(args.device),
        max_batch_windows=args.max_batch_windows,
    )

    # ── Decode latents → audio ────────────────────────────────────────────
    print("\nDecoding latents to audio ...")
    with torch.no_grad():
        # Codec expects float32; cast if latents are half-precision
        audio = codec_model.decode(gen_latents.float())  # (1, 1, T_audio)
    audio = audio.cpu()

    # ── Save output ───────────────────────────────────────────────────────
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    torchaudio.save(args.output_path, audio.squeeze(0).float(), args.sample_rate)

    actual_duration = audio.shape[-1] / args.sample_rate
    print(f"\nSaved {actual_duration:.1f}s audio → {args.output_path}")
    print("Done!")


if __name__ == "__main__":
    main()
