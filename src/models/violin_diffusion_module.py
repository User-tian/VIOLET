from typing import Any, Optional, List, Dict, Tuple
import typing
import json
import os
import io
from PIL import Image
import torch
import torchaudio
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from lightning import LightningModule
from torchmetrics import MeanMetric, MinMetric
import copy, pickle
from einops import rearrange
import wandb
from note_seq import midi_io
from src.models.components.dacvae_wrapper import FTDACVAE
from src.models.components.utils import extend_dim, to_batch
from torch.utils.data import Dataset as TorchDataset
from src.inference.render_utils import (
    DEFAULT_EDGE_SILENCE_SECONDS,
    DEFAULT_MAX_RENDER_ATTEMPTS,
    DEFAULT_PLAYABLE_NOTE_MIN_PITCH,
    apply_eval_output_silence_mask,
    make_seeded_noise,
    render_window_diagnostics,
    retry_seed,
)


class ViolinDiffusionModule(LightningModule):
    """
    LightningModule for Violin Synthesis using Latent Diffusion.
    Wires together the DiT backbone, rectified-flow diffusion process, and
    multi-modal (MIDI/technique/dynamics) conditioning for training and inference.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        noise_scheduler: torch.nn.Module,
        noise_distribution: torch.nn.Module,
        sampler: torch.nn.Module,
        diffusion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        latent_dim: int,
        generated_frame_length: int, # Latent sequence length
        audio_sample_rate: int,
        use_ema: bool = True,
        use_phema: bool = False,
        num_ema_snapshot_item: Optional[int] = 96000,
        ema_resume_path: Optional[str] = None,  # Path to EMA snapshot for resuming training
        ema_resume_nitem: Optional[int] = None,  # cur_nitem value when EMA was saved (for correct decay)
        total_test_samples: Optional[int] = None,
        ema_ckpt_path: Optional[str] = None,
        codec_model: Optional[torch.nn.Module] = None, # For validation decoding
        codec_ckpt: str = "facebook/dacvae-watermarked",
        codec_posterior_mode: str = "sample",
        codec_ft_ckpt: Optional[str] = None,
        codec_use_ft: bool = False,
        log_audio_every_n_epochs: int = 5,
        log_audio_every_n_steps: Optional[int] = None,  # If set, log train anchor audio every N steps (for step-based training)
        log_attention_every_n_steps: Optional[int] = None,  # If set, log attention weights to wandb every N steps (DiT with serial attn)
        log_attention_weights: bool = True,  # If True, request return_attn_weights when logging attention (disable to save compute)
        attention_log_num_blocks: int = 4,  # Number of blocks to visualize for attention logging
        cc_frame_rate: float = 25.0, # Default frame rate for CC tokens
        log_target_audio_paths: Optional[list] = None, # List of substrings to match in audio_path for validation logging
        save_test_conditioning_debug: bool = False,
        # If True during trainer.test(), save ``net.midi_harmonic_extractor`` outputs (B, cqt_bins, T)
        # under ``{logger.save_dir}/test_samples/{stem}_harmonic.pt`` (same ``target_len`` as DiT forward).
        save_test_harmonic_debug: bool = False,

        # Curriculum Learning
        stage1_steps: int = 100000,
        lr_scheduler_interval: str = "epoch",
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.generated_frame_length = generated_frame_length
        self.audio_sample_rate = audio_sample_rate

        self.stage1_steps = stage1_steps
        self.lr_scheduler_interval = lr_scheduler_interval
        self.log_audio_every_n_epochs = log_audio_every_n_epochs
        self.log_audio_every_n_steps = log_audio_every_n_steps
        self.log_attention_every_n_steps = log_attention_every_n_steps
        self.log_attention_weights = log_attention_weights
        self.attention_log_num_blocks = max(int(attention_log_num_blocks), 1)
        self.cc_frame_rate = cc_frame_rate
        self.log_target_audio_paths = log_target_audio_paths or []
        self.save_test_conditioning_debug = bool(save_test_conditioning_debug)
        self.save_test_harmonic_debug = bool(save_test_harmonic_debug)

        self.optimizer = optimizer
        self.scheduler = scheduler

        # diffusion components
        self.net = net
        self.use_ema = use_ema
        self.use_phema = use_phema
        self.cur_nitem = 0
        self.num_ema_snapshot_item = num_ema_snapshot_item
        self.ema_resume_path = ema_resume_path
        self.ema_resume_nitem = ema_resume_nitem
        self.ema_ckpt_path = ema_ckpt_path

        self.sampler = sampler
        self.diffusion = diffusion
        self.noise_distribution = noise_distribution # for training
        self.noise_scheduler = noise_scheduler()     # for sampling

        self.total_test_samples = total_test_samples
        self.codec_ckpt = codec_ckpt
        self.codec_posterior_mode = codec_posterior_mode
        self.codec_ft_ckpt = codec_ft_ckpt
        self.codec_use_ft = codec_use_ft

        # For validation decoding (optional)
        if codec_model is not None:
            self.codec_model = codec_model
        else:
            self.codec_model = None

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.val_loss_best = MinMetric()

        # Load technique map for logging
        self.tech_map_inv = {
            1: "sustain", 2: "tremolo", 3: "trill_major", 4: "trill_minor",
            5: "staccato", 6: "spiccato", 7: "ricochet", 8: "pizzicato",
            9: "harmonic", 10: "legato_bow", 11: "legato_slur", 12: "legato_portamento",
            0: "none"
        }

        self.validation_step_outputs = []
        self.train_anchor_batch = None
        # Cache matched validation samples on CPU to avoid repeated GPU-heavy logging inside validation_step
        # (prevents long-run CUDA allocator fragmentation / apparent "leak" after many epochs)
        self._val_log_cache: dict[str, dict[str, Any]] = {}
        # Track static W&B artifacts already logged per stage/sample.
        # We only need to log these once because they do not change over time.
        self._wandb_static_logged_keys: set[str] = set()
        self._wandb_step_metric_defined = False

    def load_state_dict(self, state_dict, strict=True):
        """
        Override load_state_dict to handle incompatible keys between training and inference configs.

        This is particularly important when the model architecture changes slightly between versions.
        """
        # Filter out keys that don't exist in the current model
        model_keys = set(self.state_dict().keys())
        checkpoint_keys = set(state_dict.keys())

        # Keys in checkpoint but not in model (unexpected keys)
        unexpected_keys = checkpoint_keys - model_keys
        # Keys in model but not in checkpoint (missing keys)
        missing_keys = model_keys - checkpoint_keys

        # Filter out keys we know are safe to ignore (e.g. optional components not
        # present in every config, such as an older checkpoint from a component that
        # has since been dropped from the architecture).
        safe_to_ignore_prefixes: list[str] = []

        filtered_unexpected = []
        for key in unexpected_keys:
            if not any(key.startswith(prefix) for prefix in safe_to_ignore_prefixes):
                filtered_unexpected.append(key)

        filtered_missing = []
        for key in missing_keys:
            if not any(key.startswith(prefix) for prefix in safe_to_ignore_prefixes):
                filtered_missing.append(key)

        # Remove unexpected keys from state_dict
        for key in unexpected_keys:
            if any(key.startswith(prefix) for prefix in safe_to_ignore_prefixes):
                state_dict.pop(key, None)

        # If there are still unexpected/missing keys, handle based on strict flag
        if filtered_unexpected or filtered_missing:
            if strict:
                error_msgs = []
                if filtered_unexpected:
                    error_msgs.append(f'Unexpected key(s): {", ".join(filtered_unexpected)}')
                if filtered_missing:
                    error_msgs.append(f'Missing key(s): {", ".join(filtered_missing)}')
                raise RuntimeError(f"Error(s) loading state_dict:\n\t" + "\n\t".join(error_msgs))
            else:
                # Just warn
                if filtered_unexpected:
                    print(f"Warning: Unexpected keys in state_dict: {filtered_unexpected}")
                if filtered_missing:
                    print(f"Warning: Missing keys in state_dict: {filtered_missing}")

        # Call parent load_state_dict with filtered state_dict (non-strict to allow missing optional components)
        return super().load_state_dict(state_dict, strict=False)

    def setup(self, stage: str):
        if self.codec_model is None:
            # Try to share from datamodule to save memory
            if hasattr(self.trainer, "datamodule") and hasattr(self.trainer.datamodule, "dacvae"):
                print("Sharing DACVAE from Datamodule")
                self.codec_model = self.trainer.datamodule.dacvae
            else:
                print(f"Loading DACVAE from {self.codec_ckpt}...")
                self.codec_model = FTDACVAE.load_with_finetuned_weights(
                    base_ckpt=self.codec_ckpt,
                    finetuned_ckpt=self.codec_ft_ckpt,
                    use_finetuned=self.codec_use_ft,
                    posterior_mode=self.codec_posterior_mode,
                )
                self.codec_model.eval()
                for p in self.codec_model.parameters():
                    p.requires_grad = False
        elif self.codec_model is not None:
            self.codec_model.eval()
            for p in self.codec_model.parameters():
                p.requires_grad = False

    def on_train_start(self):
        if self.global_rank != 0:
            return
        # Create a fixed anchor batch from training data for consistent visualization
        # We try to balance between VIOLET and MOSA if possible.
        if self.trainer.train_dataloader is not None:
             # We need to access the underlying dataset
             # The dataloader is likely a CombinedLoader or standard DataLoader
             # self.trainer.train_dataloader might be a generator if reloading, but usually accessible.
             # Better to access datamodule directly
             dm = self.trainer.datamodule
             if dm is not None and dm.data_train is not None:
                 dataset = dm.data_train
                 # Identify indices
                 # dataset might be ConcatDataset [VIOLET, MOSA] or just VIOLET

                 anchor_indices = []
                 if isinstance(dataset, torch.utils.data.ConcatDataset):
                     # Assume [0] is VIOLET, [1] is MOSA
                     # Get 5 from VIOLET
                     len_violet = len(dataset.datasets[0])
                     indices_violet = np.random.RandomState(42).choice(len_violet, 5, replace=False).tolist()
                     anchor_indices.extend(indices_violet)

                     # Get 5 from MOSA
                     len_mosa = len(dataset.datasets[1])
                     # MOSA indices in ConcatDataset start at len_violet
                     indices_mosa = np.random.RandomState(42).choice(len_mosa, 5, replace=False) + len_violet
                     anchor_indices.extend(indices_mosa.tolist())
                     anchor_set = None  # set below via Subset
                 else:
                     from src.data.components.unified_dataset import UnifiedViolinDataset
                     if isinstance(dataset, UnifiedViolinDataset):
                         # IterableDataset has no len(); build anchor from underlying real/synth datasets
                         rng = np.random.RandomState(42)
                         n_real = min(5, len(dataset.real_dataset))
                         n_synth = min(5, len(dataset.synth_dataset))
                         anchor_samples = []
                         if n_real > 0:
                             idx_real = rng.choice(len(dataset.real_dataset), n_real, replace=False)
                             anchor_samples.extend([dataset.real_dataset[int(i)] for i in idx_real])
                         if n_synth > 0:
                             idx_synth = rng.choice(len(dataset.synth_dataset), n_synth, replace=False)
                             anchor_samples.extend([dataset.synth_dataset[int(i)] for i in idx_synth])
                         class _AnchorListDataset(TorchDataset):
                             def __init__(self, items):
                                 self.items = items
                             def __len__(self):
                                 return len(self.items)
                             def __getitem__(self, i):
                                 return self.items[i]
                         anchor_set = _AnchorListDataset(anchor_samples)
                         anchor_indices = list(range(len(anchor_samples)))
                     else:
                         anchor_indices = np.random.RandomState(42).choice(len(dataset), 10, replace=False).tolist()
                         anchor_set = None  # set below

                 # Create subset and loader (anchor_set already set for UnifiedViolinDataset)
                 from torch.utils.data import Subset, DataLoader
                 from src.data.violin_datamodule import ViolinCollator
                 if anchor_set is None:
                     anchor_set = Subset(dataset, anchor_indices)
                 # We use batch_size = len(anchor_indices) to get all in one batch
                 self.anchor_loader = DataLoader(
                     anchor_set,
                     batch_size=len(anchor_indices),
                     shuffle=False,
                     num_workers=0,
                     collate_fn=ViolinCollator().collate,
                     pin_memory=True
                 )

                # Fetch the batch immediately (batch comes from collator only;
                # it does not go through datamodule.on_after_batch_transfer)
                 for batch in self.anchor_loader:
                    # Calculate audio latents manually since on_after_batch_transfer is skipped
                    # Collator outputs "waveform"; "audio" is only set by the datamodule hook
                    waveforms = batch['waveform']

                     # Ensure codec model is ready
                    if self.codec_model is None:
                        # This should be initialized in setup, but double check
                        self.codec_model = FTDACVAE.load_with_finetuned_weights(
                            base_ckpt=self.codec_ckpt,
                            finetuned_ckpt=self.codec_ft_ckpt,
                            use_finetuned=self.codec_use_ft,
                            posterior_mode=self.codec_posterior_mode,
                        )
                        self.codec_model.eval()

                    # We need to move codec to the same device as where we want to process
                    # Since this is on_train_start, accelerator should be ready.
                    # We can use self.device
                    if next(self.codec_model.parameters()).device != self.device:
                        self.codec_model.to(self.device)

                    waveforms = waveforms.to(self.device)

                    with torch.no_grad():
                        encoded = self.codec_model.encode(waveforms) # (B, D, T_latent)

                    # Store back in batch on CPU (will move to GPU when logging)
                    batch['audio_latents'] = encoded.cpu()
                    batch['audio_latents_length'] = torch.tensor(encoded.shape[-1], dtype=torch.long).repeat(len(encoded))

                    # Remove raw waveform to save memory if desired, but we keep it here just in case
                    # del batch["waveform"]

                    self.train_anchor_batch = batch
                    break

    @torch.no_grad()
    def synthesize_from_noise(self, initial_noise, conditioning, ema_model=None, **sampling_kwargs):
        # conditioning is a dict of {midi_tokens, tech_tokens, cc_tokens}
        # initial_noise: (B, C, H, W) or (B, C, T)

        # Sampler expects (x, classes, ...)
        # We pass conditioning via kwargs to sampler -> denoise_fn -> net

        net_kwargs = self._prepare_conditioning_kwargs(conditioning)

        # Wrap denoise_fn to handle kwargs?
        # The sampler calls: fn(x, sigma * s_in, **extra_args)
        # We need to ensure extra_args are passed.

        # We can pass them as 'classes' if sampler supports arbitrary args, or modify sampler.
        # Standard sampler in this repo:
        # src/models/components/sampler_rf.py: forward(self, noise, classes, fn, net, sigmas)
        # It calls: denoised = fn(x, sigma, classes) -> net(x, sigma, classes)

        # DiT.forward signature: (x, t, classes, ..., midi_tokens, ...)
        # So passing conditioning as kwargs to net is fine IF sampler supports it.
        # The sampler in repo:
        # d = fn(x_hat, sigma_hat * s_in, **extra_args)
        # So we can pass 'extra_args'.

        latents = self.sampler(
            initial_noise,
            classes=None,
            fn=self.diffusion.denoise_fn,
            net=self.net,
            sigmas=self.noise_scheduler.to(self.device),
            **net_kwargs,
            **sampling_kwargs,
        )

        return latents

    @staticmethod
    def _length_mask(lengths: torch.Tensor, max_len: int, device: torch.device) -> torch.Tensor:
        return torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)

    def _sample_has_nonzero(self, values: Optional[torch.Tensor], lengths: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        if values is None:
            return None

        nonzero = values != 0
        if lengths is not None:
            valid_mask = self._length_mask(lengths.to(device=values.device), values.shape[1], values.device)
            while valid_mask.ndim < nonzero.ndim:
                valid_mask = valid_mask.unsqueeze(-1)
            nonzero = nonzero & valid_mask

        reduce_dims = tuple(range(1, nonzero.ndim))
        return nonzero.any(dim=reduce_dims)

    def _prepare_conditioning_kwargs(self, batch: dict) -> dict:
        # Samples with no real technique/CC annotation get rewritten to the null-conditioning
        # id/zero (not left at real label 0) so the model can tell "unannotated" apart from
        # "annotated as none"; the *_keep_mask tensors record which samples were rewritten.
        kwargs = {}
        for key in (
            "midi_tokens",
            "tech_tokens",
            "velocity_tokens",
            "pos_midi",
            "cc_tokens",
            "midi_roll",
            "tech_roll",
            "midi_length",
            "tech_length",
        ):
            if key in batch and batch[key] is not None:
                kwargs[key] = batch[key]

        tech_present = None
        if "tech_tokens" in kwargs:
            tech_present = self._sample_has_nonzero(kwargs["tech_tokens"], batch.get("tech_length"))
        if "tech_roll" in kwargs:
            tech_roll_present = self._sample_has_nonzero(kwargs["tech_roll"])
            tech_present = tech_roll_present if tech_present is None else (tech_present | tech_roll_present)

        if tech_present is not None:
            missing_tech = ~tech_present
            if missing_tech.any():
                if "tech_tokens" in kwargs:
                    tech_vocab_size = getattr(self.net, "tech_vocab_size", 21)
                    null_tech_id = tech_vocab_size - 1
                    missing_tech_expanded = rearrange(missing_tech, "b -> b 1").expand_as(kwargs["tech_tokens"])
                    kwargs["tech_tokens"] = torch.where(
                        missing_tech_expanded,
                        torch.full_like(kwargs["tech_tokens"], null_tech_id),
                        kwargs["tech_tokens"],
                    )
                if "tech_roll" in kwargs:
                    missing_tech_roll = rearrange(missing_tech, "b -> b 1 1").expand_as(kwargs["tech_roll"])
                    kwargs["tech_roll"] = torch.where(
                        missing_tech_roll,
                        torch.zeros_like(kwargs["tech_roll"]),
                        kwargs["tech_roll"],
                    )
                kwargs["tech_roll_keep_mask"] = rearrange(tech_present, "b -> b 1 1").to(dtype=torch.float32)

        if "cc_tokens" in kwargs:
            cc_present = self._sample_has_nonzero(kwargs["cc_tokens"], batch.get("cc_length"))
            if cc_present is not None:
                missing_cc = ~cc_present
                if missing_cc.any():
                    missing_cc_expanded = rearrange(missing_cc, "b -> b 1 1").expand_as(kwargs["cc_tokens"])
                    kwargs["cc_tokens"] = torch.where(
                        missing_cc_expanded,
                        torch.zeros_like(kwargs["cc_tokens"]),
                        kwargs["cc_tokens"],
                    )
                kwargs["cc_keep_mask"] = rearrange(cc_present, "b -> b 1 1").to(dtype=torch.float32)

        return kwargs

    def _eval_base_seed(self) -> int:
        base_seed = getattr(self, "_eval_seed", None)
        if base_seed is None:
            return int(torch.initial_seed())
        return int(base_seed)

    def _describe_generation(self, **sampling_kwargs) -> Dict[str, Any]:
        resolved = dict(sampling_kwargs)
        cond_scale = (
            resolved["cond_scale"]
            if "cond_scale" in resolved
            else getattr(self.sampler, "cond_scale", None)
        )
        w_tech = (
            float(resolved["w_tech"])
            if "w_tech" in resolved
            else float(getattr(self.sampler, "w_tech", 0.0))
        )
        w_cc = (
            float(resolved["w_cc"])
            if "w_cc" in resolved
            else float(getattr(self.sampler, "w_cc", 0.0))
        )

        active_output_equals_branch = None
        if cond_scale is None:
            if abs(w_tech) < 1e-8 and abs(w_cc) < 1e-8:
                active_output_equals_branch = "m"
            elif abs(w_tech - 1.0) < 1e-8 and abs(w_cc) < 1e-8:
                active_output_equals_branch = "mt"
            elif abs(w_tech - 1.0) < 1e-8 and abs(w_cc - 1.0) < 1e-8:
                active_output_equals_branch = "full"
            effective_guidance_mode = "compositional_cfg"
        else:
            cond_scale_value = float(cond_scale)
            if abs(cond_scale_value) < 1e-8:
                active_output_equals_branch = "m"
            elif abs(cond_scale_value - 1.0) < 1e-8:
                active_output_equals_branch = "full"
            effective_guidance_mode = "classic_cfg"

        return {
            "effective_guidance_mode": effective_guidance_mode,
            "cond_scale": cond_scale,
            "w_tech": w_tech,
            "w_cc": w_cc,
            "active_output_equals_branch": active_output_equals_branch,
        }

    def _default_conditioning_debug_branch_specs(self) -> dict:
        generation = self._describe_generation()
        if generation["effective_guidance_mode"] == "compositional_cfg":
            return {
                "m": {"cond_scale": None, "w_tech": 0.0, "w_cc": 0.0},
                "mt": {"cond_scale": None, "w_tech": 1.0, "w_cc": 0.0},
                "full": {"cond_scale": None, "w_tech": 1.0, "w_cc": 1.0},
            }
        return {
            "m": {"cond_scale": 0.0},
            "full": {"cond_scale": 1.0},
        }

    @torch.no_grad()
    def _render_test_audio(
        self,
        conditioning: dict,
        seed: int,
        debug_branch_specs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        initial_noise = make_seeded_noise(
            (1, self.latent_dim, self.generated_frame_length),
            seed=seed,
            device=self.device,
            dtype=torch.float32,
        )
        gen_latents = self.synthesize_from_noise(initial_noise, conditioning)

        debug_branch_audio: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            audio_gen = self.codec_model.decode(gen_latents).cpu()
            if debug_branch_specs:
                for branch_name, branch_kwargs in debug_branch_specs.items():
                    branch_latents = self.synthesize_from_noise(
                        initial_noise.clone(),
                        conditioning,
                        **branch_kwargs,
                    )
                    debug_branch_audio[branch_name] = self.codec_model.decode(branch_latents).cpu()
        return audio_gen, debug_branch_audio

    def forward(self, batch: dict):
        """
        Forward pass for diffusion training.

        Args:
            batch: Dictionary containing 'audio_latents', 'midi_tokens', etc.

        Returns:
            loss (scalar)
        """
        # batch: 'audio_latents', 'midi_tokens', ...
        audio_latents = batch['audio_latents'] # (B, D, T)
        # Ensure shape matches DiT expectation
        # DiT expects (B, D, T).
        # Removed unsqueeze(2) as we moved to 1D DiT

        # Sample noise
        sigmas = self.noise_distribution(num_samples=audio_latents.shape[0],
                                         device=audio_latents.device)

        # Prepare conditioning and infer missing technique / CC annotations per sample.
        kwargs = self._prepare_conditioning_kwargs(batch)

        # compute loss
        loss = self.diffusion(
            audio_latents,
            self.net,
            classes=None,  # No global class label
            sigmas=sigmas,
            **kwargs,
        )

        loss_mean = loss.mean()

        return loss_mean

    def on_fit_start(self):
        if self.global_rank == 0:
            logger = getattr(self, "logger", None)
            exp = getattr(logger, "experiment", None) if logger is not None else None
            if exp is not None and hasattr(exp, "define_metric") and not self._wandb_step_metric_defined:
                try:
                    exp.define_metric("global_step")
                    exp.define_metric("*", step_metric="global_step")
                    self._wandb_step_metric_defined = True
                except Exception as e:
                    print(f"Warning: failed to define W&B step metric: {e}")

        if self.use_ema:
             # Import based on repo structure
             from .phema import PowerFunctionEMA, TraditionalEMA
             if self.use_phema:
                 self.ema_prof = PowerFunctionEMA(self.net.to(self.device), stds=[0.050, 0.100])
             else:
                 self.ema_prof = TraditionalEMA(self.net.to(self.device), halflife_Mimg=0.3, rampup_ratio=0.09)

             # Resume EMA from snapshot if provided
             if self.ema_resume_path is not None:
                 print(f"Loading EMA snapshot from {self.ema_resume_path}...")
                 with open(self.ema_resume_path, 'rb') as f:
                     ema_snapshot = pickle.load(f)

                 # Load state into ema_prof
                 if self.use_phema:
                     # For PowerFunctionEMA, we need to copy weights to each ema
                     for ema in self.ema_prof.emas:
                         ema.load_state_dict(ema_snapshot.state_dict())
                 else:
                     # For TraditionalEMA
                     self.ema_prof.ema.load_state_dict(ema_snapshot.state_dict())

                 del ema_snapshot

                 # Restore cur_nitem so the EMA rampup schedule continues correctly.
                 # cur_nitem accumulates `batch_size` per micro-batch, so at optimizer step G it
                 # equals G * batch_size * accumulate_grad_batches. Snapshots are named by G
                 # (``ema_prof{suffix}_{global_step}``), so when ema_resume_nitem is not given we
                 # recover G from the filename and rescale by the effective per-step item count.
                 if self.ema_resume_nitem is not None:
                     self.cur_nitem = self.ema_resume_nitem
                     print(f"Restored cur_nitem to {self.cur_nitem}")
                 else:
                     step = self._parse_step_from_snapshot_path(self.ema_resume_path)
                     items_per_step = self._effective_items_per_step()
                     if step is not None and items_per_step is not None:
                         self.cur_nitem = step * items_per_step
                         print(
                             f"Inferred cur_nitem={self.cur_nitem} (= global_step {step} "
                             f"x {items_per_step} items/step) from snapshot filename "
                             f"'{os.path.basename(self.ema_resume_path)}'."
                         )
                     elif step is not None:
                         self.cur_nitem = step
                         print(
                             f"Warning: recovered global_step={step} from the snapshot filename but "
                             "could not determine batch_size x accumulate_grad_batches; falling back to "
                             "cur_nitem=global_step (EMA rampup may be off). Pass ema_resume_nitem to override."
                         )
                     else:
                         print(
                             "Warning: ema_resume_nitem not provided and could not be parsed from the "
                             "snapshot filename; EMA rampup schedule may be off."
                         )

    @staticmethod
    def _parse_step_from_snapshot_path(path: str) -> Optional[int]:
        """Extract the trailing ``{global_step}`` integer from an ``ema_prof..._{step}`` filename."""
        stem = os.path.splitext(os.path.basename(str(path)))[0]
        tail = stem.split("_")[-1]
        try:
            return int(tail)
        except (TypeError, ValueError):
            return None

    def _effective_items_per_step(self) -> Optional[int]:
        """How much cur_nitem grows per optimizer step: batch_size * accumulate_grad_batches.

        Matches training-time accumulation (per-rank ``batch_size`` counted every micro-batch,
        ``accumulate_grad_batches`` micro-batches per optimizer step); world size is not included
        because ``cur_nitem`` is not scaled by it during training either.
        """
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return None
        dm = getattr(trainer, "datamodule", None)
        batch_size = getattr(getattr(dm, "hparams", None), "batch_size", None) if dm is not None else None
        if batch_size is None:
            return None
        accum = getattr(trainer, "accumulate_grad_batches", 1) or 1
        try:
            return int(batch_size) * int(accum)
        except (TypeError, ValueError):
            return None

    def _log_metrics_with_trainer_step(self, metrics: dict) -> None:
        """Send custom logger payloads at the same step source Lightning uses."""
        logger = getattr(self, "logger", None)
        if logger is None:
            return

        step = int(self.global_step)
        payload = dict(metrics)
        payload["global_step"] = step
        if hasattr(logger, "log_metrics"):
            logger.log_metrics(payload, step=step)
            return

        exp = getattr(logger, "experiment", None)
        if exp is None or not hasattr(exp, "log"):
            return

        try:
            exp.log(payload, step=step)
        except TypeError:
            exp.log(payload)

    def model_step(self, batch: Any):
        return self.forward(batch)

    def on_validation_epoch_start(self) -> None:
        self._val_log_cache = {}
        self.val_loss.reset()
        # Clear validation outputs to prevent memory accumulation over many epochs
        self.validation_step_outputs.clear()

    @torch.no_grad()
    def validation_step(self, batch: Any, batch_idx: int):
        loss = self.model_step(batch)
        self.val_loss(loss)

        # Collect matched validation samples on CPU (so we can log all 10 targets even if
        # they appear in different batches, without repeatedly doing sampling+decode on GPU).
        if self.log_target_audio_paths and ("audio_path" in batch):
            for i, path in enumerate(batch["audio_path"]):
                matched_target = None
                for target in self.log_target_audio_paths:
                    if target in path:
                        matched_target = target
                        break
                if matched_target is None:
                    continue
                if matched_target in self._val_log_cache:
                    continue

                # Store a single-sample slice on CPU
                sample: Dict[str, Any] = {"audio_path": path}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        sample[k] = v[i : i + 1].detach().cpu()
                sample["custom_key"] = os.path.splitext(os.path.basename(path))[0]
                self._val_log_cache[matched_target] = sample

        # Default behavior: log first batch ONLY if no targets specified
        # if not self.log_target_audio_paths and batch_idx == 0:
        #      custom_keys = None
        #      if 'audio_path' in batch:
        #          custom_keys = []
        #          for path in batch['audio_path']:
        #              stem = os.path.splitext(os.path.basename(path))[0]
        #              custom_keys.append(stem)
        #      self.log_validation_samples(batch, stage="val", custom_keys=custom_keys)

        return {"loss": loss}

    def on_validation_epoch_end(self) -> None:
        val_loss = self.val_loss.compute()
        self.log("val/loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.val_loss_best(val_loss)
        self.log("val/loss_best", self.val_loss_best.compute(), prog_bar=True, sync_dist=True)

        if not self.trainer.sanity_checking and self.global_rank == 0:
            val_loss_scalar = float(val_loss.detach().cpu()) if isinstance(val_loss, torch.Tensor) else float(val_loss)
            self._log_metrics_with_trainer_step({"val/loss_step": val_loss_scalar})

        if self.global_rank != 0:
            return

        # Check logging frequency.
        # Prefer step-based gating when configured (for step-based training runs),
        # and fall back to epoch-based gating otherwise.
        if self.log_audio_every_n_steps is not None:
            if self.global_step <= 0 or self.global_step % self.log_audio_every_n_steps != 0:
                return
        else:
            # We need to check (current_epoch + 1) to align with "every N epochs" semantics
            if (self.current_epoch + 1) % self.log_audio_every_n_epochs != 0:
                return

        # Log exactly the configured target samples (up to 10), in config order.
        if not self.log_target_audio_paths:
            return
        if not self._val_log_cache:
            return

        def _pad_and_cat(tensors: typing.List[torch.Tensor]) -> torch.Tensor:
            """Pad tensors (on the right) to the max shape (excluding batch dim) then cat on dim 0.

            This is needed because our cached validation samples may come from different
            validation batches with different padding lengths (e.g., midi_tokens length).
            """
            if len(tensors) == 1:
                return tensors[0]

            # All cached tensors are stored as 1-sample batches: (1, ...)
            ndim = tensors[0].ndim
            if any(t.ndim != ndim for t in tensors):
                # Fallback: if ranks disagree, just try raw cat (will raise a clearer error)
                return torch.cat(tensors, dim=0)

            # Compute target sizes for dims 1..ndim-1
            target_sizes = [max(t.shape[d] for t in tensors) for d in range(1, ndim)]

            padded = []
            for t in tensors:
                pad = []
                # torch.nn.functional.pad expects pads for last dim first: (l_last, r_last, l_prev, r_prev, ...)
                for d in range(ndim - 1, 0, -1):
                    missing = target_sizes[d - 1] - t.shape[d]
                    pad.extend([0, int(max(missing, 0))])
                if any(pad):
                    t = torch.nn.functional.pad(t, pad, value=0)
                padded.append(t)

            return torch.cat(padded, dim=0)

        # Build a batch in the same deterministic order as log_target_audio_paths
        samples = []
        custom_keys = []
        for target in self.log_target_audio_paths:
            if target in self._val_log_cache:
                s = self._val_log_cache[target]
                samples.append(s)
                custom_keys.append(s.get("custom_key", target))
        if not samples:
            return

        # Stack tensors into a single batch, keep audio_path as list
        batch: dict[str, Any] = {"audio_path": [s["audio_path"] for s in samples]}
        tensor_keys = set().union(*[set(s.keys()) for s in samples])
        tensor_keys.discard("audio_path")
        tensor_keys.discard("custom_key")
        for k in tensor_keys:
            if all((k in s and isinstance(s[k], torch.Tensor)) for s in samples):
                batch[k] = _pad_and_cat([s[k] for s in samples])

        self.log_validation_samples(batch, stage="val", custom_keys=custom_keys)

    @torch.no_grad()
    def log_validation_samples(self, batch, stage="val", custom_keys=None, suffix=""):
        if self.global_rank != 0:
            return

        # Generate using conditioning from batch
        # Iterate over all samples in the batch (since batch is small, e.g. 10 for anchor)
        # But to be safe and avoid OOM, let's process one by one or keep it vectorized if small.
        # Anchor batch size is ~10. Real batch size ~16.
        # Processing 10 samples might be heavy. Let's do it vectorized but careful.

        B = batch['midi_tokens'].shape[0]
        # Limit to 10 max if batch is large
        if B > 10:
            B = 10

        conditioning = {
            "midi_tokens": batch["midi_tokens"][:B],
            "tech_tokens": batch["tech_tokens"][:B],
            "velocity_tokens": batch["velocity_tokens"][:B],
            "pos_midi": batch["pos_midi"][:B],
            "cc_tokens": batch["cc_tokens"][:B],
            "midi_roll": batch["midi_roll"][:B] if "midi_roll" in batch else None,
            "tech_roll": batch["tech_roll"][:B] if "tech_roll" in batch else None,
        }
        if "midi_length" in batch:
            conditioning["midi_length"] = batch["midi_length"][:B]
        if "tech_length" in batch:
            conditioning["tech_length"] = batch["tech_length"][:B]

        # Noise shape: (B, D, T)
        D = self.latent_dim
        T = self.generated_frame_length

        # 1. Ground Truth (Real Audio)
        # Note: 'audio_latents' is what we have in batch, not raw audio.
        # But we can decode the latents to get "Reconstruction" (or "Ground Truth" approx)
        # If we want *actual* raw audio, we'd need to modify DataModule to pass it through.
        # For now, let's treat "Ground Truth" as decoded latents from the real audio.
        # Or "Reconstruction" branch if available.

        # Move required tensors to device (validation cache stores CPU tensors by design)
        def _to_dev(x: Any) -> Any:
            if isinstance(x, torch.Tensor):
                return x.to(self.device, non_blocking=True)
            return x

        for k in list(conditioning.keys()):
            conditioning[k] = _to_dev(conditioning[k])

        # 2. Reconstruction (Decoded Real Latents)
        real_latents = _to_dev(batch["audio_latents"][:B])

        # Shared noise for fair comparison across condition variants
        initial_noise = torch.randn((B, D, T), device=self.device)

        # Conditioning variants (avoid cloning big tensors)
        cond_no_cc = dict(conditioning)
        cond_no_cc["cc_tokens"] = torch.zeros_like(conditioning["cc_tokens"])

        cond_no_tech = dict(conditioning)
        cond_no_tech["tech_tokens"] = torch.zeros_like(conditioning["tech_tokens"])
        if conditioning.get("tech_roll") is not None:
            cond_no_tech["tech_roll"] = torch.zeros_like(conditioning["tech_roll"])

        # Decode and Log
        if self.codec_model:
            try:
                # Ensure codec is on the same device
                if next(self.codec_model.parameters()).device != self.device:
                    self.codec_model.to(self.device)

                # Iterative decoding to save memory
                def decode_to_np(z):
                    # z: (B, D, T)
                    # Loop over batch
                    wavs = []
                    for i in range(z.shape[0]):
                        # Slice one sample: (1, D, T)
                        z_i = z[i:i+1]
                        # Decode
                        with torch.backends.cudnn.flags(enabled=False):
                            wav_i = self.codec_model.decode(z_i) # (1, 1, T_audio)
                        # Move to CPU immediately
                        wavs.append(wav_i.cpu())

                    # Concatenate on CPU
                    wav_batch = torch.cat(wavs, dim=0) # (B, 1, T_audio)
                    return wav_batch.squeeze(1).float().detach().numpy() # (B, T_audio)

                gen_latents = self.synthesize_from_noise(initial_noise, conditioning)
                audio_gen = decode_to_np(gen_latents)
                del gen_latents

                gen_latents_no_cc = self.synthesize_from_noise(initial_noise, cond_no_cc)
                audio_gen_no_cc = decode_to_np(gen_latents_no_cc)
                del gen_latents_no_cc

                gen_latents_no_tech = self.synthesize_from_noise(initial_noise, cond_no_tech)
                audio_gen_no_tech = decode_to_np(gen_latents_no_tech)
                del gen_latents_no_tech

                sample_ids = []
                static_log_flags = []
                static_sample_keys = []  # Keep static artifacts once per stage/sample.
                for i in range(B):
                    if custom_keys:
                        sample_id = f"_{custom_keys[i]}"
                        # Keep static artifacts once per stage (train/val) per sample
                        static_sample_key = f"{stage}/sample_{custom_keys[i]}"
                    else:
                        sample_id = f"_{i}"
                        # Without custom_keys, include stage to avoid conflating train sample 0 with val sample 0
                        static_sample_key = f"{stage}/sample_{i}"
                    sample_id += suffix
                    sample_ids.append(sample_id)
                    static_sample_keys.append(static_sample_key)

                    static_log_flags.append(static_sample_key not in self._wandb_static_logged_keys)

                # Full precision logging path (no autocast / no approximations)
                audio_recon = decode_to_np(real_latents) if any(static_log_flags) else None

                log_dict = {}
                import matplotlib.pyplot as plt

                for i in range(B):
                    sample_id = sample_ids[i]
                    sample_key = f"{stage}/sample{sample_id}"
                    should_log_static = static_log_flags[i]

                    sample_log = {
                        f"{sample_key}/audio_gen": wandb.Audio(audio_gen[i], sample_rate=self.audio_sample_rate, caption="Generated"),
                        f"{sample_key}/audio_gen_no_cc": wandb.Audio(audio_gen_no_cc[i], sample_rate=self.audio_sample_rate, caption="No CC"),
                        f"{sample_key}/audio_gen_no_tech": wandb.Audio(audio_gen_no_tech[i], sample_rate=self.audio_sample_rate, caption="No Tech"),
                    }

                    if should_log_static:
                        # Visualize CC Curve
                        cc_np = conditioning['cc_tokens'][i, :, 0].cpu().detach().numpy() # (F,)
                        # Create time axis
                        num_frames = cc_np.shape[0]
                        time_axis = np.arange(num_frames) / self.cc_frame_rate

                        fig, ax = plt.subplots(figsize=(6, 2))
                        ax.plot(time_axis, cc_np)
                        ax.set_ylim(0, 1)
                        ax.set_title("CC1 Curve")
                        ax.set_xlabel("Time (s)")
                        plt.tight_layout()
                        cc_img = wandb.Image(fig)
                        plt.close(fig) # close to prevent display

                        # Map Techniques
                        note_seq = conditioning['midi_tokens'][i].cpu().detach().numpy()
                        tech_seq = conditioning['tech_tokens'][i].cpu().detach().numpy()

                        tech_list = []
                        for n, t in zip(note_seq, tech_seq):
                            if t != 0:
                                tech_name = self.tech_map_inv.get(t, str(t))
                                tech_list.append(tech_name)
                        tech_str = ", ".join(tech_list)

                        sample_log.update({
                            f"{sample_key}/audio_recon": wandb.Audio(audio_recon[i], sample_rate=self.audio_sample_rate, caption="Reconstruction"),
                            f"{sample_key}/cc_curve": cc_img,
                            f"{sample_key}/techniques": wandb.Html(f"<p>{tech_str}</p>"),
                        })
                        self._wandb_static_logged_keys.add(static_sample_keys[i])

                    log_dict.update(sample_log)

                self._log_metrics_with_trainer_step(log_dict)
            except Exception as e:
                print(f"Error logging audio: {e}")
            finally:
                # Best-effort cleanup to reduce long-run CUDA fragmentation
                # This helps prevent memory accumulation over many validation epochs
                try:
                    # Clear any GPU tensors that might still be referenced
                    del initial_noise
                except (NameError, UnboundLocalError):
                    pass
                try:
                    del real_latents, conditioning, cond_no_cc, cond_no_tech
                except (NameError, UnboundLocalError):
                    pass
                if self.device.type == "cuda":
                    # Clear cache without synchronization to avoid training slowdown
                    # empty_cache() is sufficient for preventing fragmentation
                    torch.cuda.empty_cache()

    def _log_train_anchor_audio(self):
        """Log training anchor batch (GT + generated) to wandb. Called from epoch or step hooks."""
        if self.global_rank != 0 or self.train_anchor_batch is None:
            return
        anchor_batch = self.train_anchor_batch
        custom_keys = None
        if 'audio_path' in anchor_batch:
            custom_keys = [os.path.splitext(os.path.basename(p))[0] for p in anchor_batch['audio_path']]
        self.log_validation_samples(anchor_batch, stage="train", custom_keys=custom_keys)

    def _log_attention_weights(self, batch: dict):
        """Log attention weights from DiT blocks to wandb (self-attn and cross-attn for serial mode)."""
        if not hasattr(self.net, "attention_mode") or self.global_rank != 0:
            return
        if not hasattr(self.logger, "experiment") or not hasattr(self.logger.experiment, "log"):
            return
        if "audio_latents" not in batch or not isinstance(batch["audio_latents"], torch.Tensor):
            return

        out = None
        all_attn_weights = None
        try:
            audio_latents = batch["audio_latents"][:1].to(self.device, non_blocking=True)

            sigmas = self.noise_distribution(num_samples=1, device=self.device)
            t_padded = extend_dim(sigmas, dim=audio_latents.ndim)
            z1 = torch.randn_like(audio_latents, device=self.device)
            zt = (1 - t_padded) * audio_latents + t_padded * z1
            zt = zt.to(audio_latents.dtype)

            kwargs = {}
            for k in ("midi_tokens", "tech_tokens", "velocity_tokens", "pos_midi", "cc_tokens", "midi_roll", "tech_roll", "midi_length", "tech_length"):
                if k in batch and batch[k] is not None:
                    v = batch[k]
                    kwargs[k] = v[:1].to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v

            block_indices = None
            if hasattr(self.net, "blocks"):
                depth = len(self.net.blocks)
                if depth > 0:
                    n_blocks = min(self.attention_log_num_blocks, depth)
                    if n_blocks < depth:
                        block_indices = sorted(set(np.linspace(0, depth - 1, num=n_blocks, dtype=int).tolist()))
                    else:
                        block_indices = list(range(depth))

            autocast_enabled = self.device.type == "cuda"
            autocast_dtype = torch.bfloat16 if autocast_enabled and torch.cuda.is_bf16_supported() else torch.float16
            with torch.inference_mode():
                with torch.autocast(device_type=self.device.type, dtype=autocast_dtype, enabled=autocast_enabled):
                    out = self.net(
                        zt,
                        sigmas,
                        return_attn_weights=self.log_attention_weights,
                        attn_block_indices=block_indices,
                        **kwargs,
                    )

            if not isinstance(out, (list, tuple)) or len(out) != 2:
                return
            _, all_attn_weights = out
            if not all_attn_weights:
                return

            log_dict = {}
            for bi, block_attn in all_attn_weights.items():
                for attn_type, attn in block_attn.items():
                    attn_mean = attn[0].float().mean(dim=0).cpu().numpy()
                    fig, ax = plt.subplots(figsize=(6, 5))
                    im = ax.imshow(attn_mean, aspect="auto", cmap="viridis")
                    ax.set_xlabel("Key position")
                    ax.set_ylabel("Query position")
                    ax.set_title(f"Block {bi} {attn_type}")
                    plt.colorbar(im, ax=ax)
                    plt.tight_layout()
                    buf = io.BytesIO()
                    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
                    plt.close(fig)
                    buf.seek(0)
                    log_dict[f"attention/block{bi}_{attn_type}"] = wandb.Image(Image.open(buf).copy())
                    buf.close()

            if log_dict:
                self._log_metrics_with_trainer_step(log_dict)
        finally:
            if out is not None:
                del out
            if all_attn_weights is not None:
                del all_attn_weights
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Step-based train anchor audio and attention logging."""
        if self.global_rank != 0 or self.global_step <= 0:
            return
        do_audio_log = self.log_audio_every_n_steps is not None and self.global_step % self.log_audio_every_n_steps == 0
        do_attention_log = (
            self.log_attention_weights
            and self.log_attention_every_n_steps is not None
            and self.global_step % self.log_attention_every_n_steps == 0
        )

        if do_audio_log:
            self._log_train_anchor_audio()
        if do_attention_log:
            batch_for_attn = self.train_anchor_batch if self.train_anchor_batch is not None else batch
            if batch_for_attn is not None and "audio_latents" in batch_for_attn:
                self._log_attention_weights(batch_for_attn)
        if self.device.type == "cuda" and (do_audio_log or do_attention_log):
            torch.cuda.empty_cache()

    def on_train_epoch_end(self):
        # Clear validation outputs to prevent accumulation (negligible overhead)
        if hasattr(self, 'validation_step_outputs'):
            self.validation_step_outputs.clear()

        # Periodic CUDA cache clearing every 50 epochs (infrequent to avoid slowdown)
        # Only when memory fragmentation might be an issue after many epochs
        if self.device.type == "cuda" and self.current_epoch > 0 and self.current_epoch % 50 == 0:
            torch.cuda.empty_cache()

        # Log training samples: epoch-based (when log_audio_every_n_steps is None) or skipped (step-based handles it)
        if self.log_audio_every_n_steps is None:
            # Moved from training_step to avoid memory spike (Graph + Inference)
            # NOTE: current_epoch is 0-indexed, check after epoch completion
            if ((self.current_epoch + 1) % self.log_audio_every_n_epochs == 0):
                self._log_train_anchor_audio()

    def on_before_optimizer_step(self, optimizer):
        # Log Gradient Norm
        # This hook is called before optimizer step, so gradients are available.
        # Compute total norm for generator/diffusion model
        total_norm = 0.0
        for p in self.net.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        # Store grad norm to log in training_step with proper batch_size
        self._grad_norm = total_norm

    def training_step(self, batch: Any, batch_idx: int):
        batch_size = batch["audio_latents"].shape[0] if isinstance(batch, dict) and "audio_latents" in batch else None
        audio_latents = batch.get('audio_latents')


        loss = self.model_step(batch)
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)

        # Log gradient norm (set by on_before_optimizer_step hook)
        if hasattr(self, '_grad_norm'):
            self.log("train/grad_norm", self._grad_norm, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True, batch_size=batch_size)

        # Log generator learning rate
        opt = self.optimizers()
        if opt is not None:
            if isinstance(opt, list):
                opt_g = opt[0]
            else:
                opt_g = opt
            for i, param_group in enumerate(opt_g.param_groups):
                lr = param_group.get('lr', 0)
                self.log(f"train/lr_generator", lr, on_step=True, on_epoch=False, prog_bar=False, sync_dist=True, batch_size=batch_size)
                break

        # Log AMP Scale
        if self.trainer.precision_plugin is not None:
             if hasattr(self.trainer.precision_plugin, 'scaler'):
                 scaler = self.trainer.precision_plugin.scaler
                 if scaler is not None:
                     self.log("train/amp_scale", scaler.get_scale(), on_step=True, on_epoch=False, prog_bar=False, sync_dist=True, batch_size=batch_size)

        # EMA update
        if self.use_ema:
            # Use boundary-crossing check so snapshots trigger even when resuming with non-aligned cur_nitem
            _inc = batch_size
            _crossed = (int(self.cur_nitem + _inc) // self.num_ema_snapshot_item) > (int(self.cur_nitem) // self.num_ema_snapshot_item)
            if _crossed and self.trainer.global_rank == 0 and self.global_step > 0:
                ema_list = self.ema_prof.get()
                ema_list = ema_list if isinstance(ema_list, list) else [(ema_list, '')]
                for ema_net, ema_suffix in ema_list:
                    ema_snapshot_path = os.path.join(self.logger.save_dir, 'ema_snapshots')
                    os.makedirs(ema_snapshot_path, exist_ok=True)
                    device_prev = next(ema_net.parameters()).device
                    ema_net.to("cpu")
                    try:
                        ema_snapshot = copy.deepcopy(ema_net).eval().requires_grad_(False).to(torch.float16)
                        with open(os.path.join(ema_snapshot_path, f'ema_prof{ema_suffix}_{self.global_step}'), 'wb') as f:
                            pickle.dump(ema_snapshot, f)
                        del ema_snapshot
                    finally:
                        ema_net.to(device_prev)

            self.cur_nitem += batch_size
            self.ema_prof.update(self.cur_nitem, batch_size)

        return {"loss": loss}


    def on_test_start(self):
        if self.use_ema and self.ema_ckpt_path is not None:
            print(f"Loading EMA weights from {self.ema_ckpt_path}...")
            with open(self.ema_ckpt_path, "rb") as f:
                ema_net = pickle.load(f)
                target_dtype = next(self.net.parameters()).dtype
                self.net = ema_net.to(device=self.device, dtype=target_dtype)

        if self.codec_model is None:
            print(f"Loading DACVAE from {self.codec_ckpt}...")
            self.codec_model = FTDACVAE.load_with_finetuned_weights(
                base_ckpt=self.codec_ckpt,
                finetuned_ckpt=self.codec_ft_ckpt,
                use_finetuned=self.codec_use_ft,
                posterior_mode=self.codec_posterior_mode,
            )
            self.codec_model.eval()
            self.codec_model.to(self.device)

    def test_step(self, batch: Any, batch_idx: int):
        # Generate samples for the batch
        # We rely on the sampler (configured in hydra) to handle cond_scale

        # Filter samples based on log_target_audio_paths if specified (only during testing)
        # This allows generating only specific target samples instead of the entire test set
        if self.trainer.testing and self.log_target_audio_paths and 'audio_path' in batch:
            # Find which samples in this batch match our target paths
            batch_indices_to_process = []
            for i, path in enumerate(batch['audio_path']):
                for target in self.log_target_audio_paths:
                    if target in path:
                        batch_indices_to_process.append(i)
                        break

            # If no matches in this batch, skip it entirely
            if not batch_indices_to_process:
                return

            # Filter batch to only include matched samples
            filtered_batch = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    filtered_batch[k] = v[batch_indices_to_process]
                elif isinstance(v, list):
                    filtered_batch[k] = [v[i] for i in batch_indices_to_process]
                else:
                    filtered_batch[k] = v
            batch = filtered_batch

        la = getattr(self, "_long_audio_cfg", None) or {}
        if bool(la.get("enabled", False)):
            from src.inference.long_audio import long_audio_test_step

            long_audio_test_step(self, batch, la)
            return

        # Create output directory
        audio_save_dir = os.path.join(self.logger.save_dir, 'test_samples')
        os.makedirs(audio_save_dir, exist_ok=True)

        B = batch['midi_tokens'].shape[0]

        # Prepare conditioning
        conditioning = {
            "midi_tokens": batch["midi_tokens"],
            "tech_tokens": batch["tech_tokens"],
            "velocity_tokens": batch["velocity_tokens"],
            "pos_midi": batch["pos_midi"],
            "cc_tokens": batch["cc_tokens"],
            "midi_roll": batch.get("midi_roll"),
            "tech_roll": batch.get("tech_roll"),
        }
        if "midi_length" in batch:
            conditioning["midi_length"] = batch["midi_length"]
        if "tech_length" in batch:
            conditioning["tech_length"] = batch["tech_length"]

        if "audio_path" in batch:
            sample_stems = [
                os.path.splitext(os.path.basename(p))[0] for p in batch["audio_path"]
            ]
        else:
            sample_stems = [f"test_{batch_idx}_{i}" for i in range(B)]

        if self.save_test_harmonic_debug and batch.get("midi_roll") is not None:
            extractor = getattr(self.net, "midi_harmonic_extractor", None)
            if extractor is not None:
                mr = batch["midi_roll"].to(
                    device=self.device, dtype=next(self.net.parameters()).dtype
                )
                interp = bool(getattr(self.net, "midi_roll_interp_to_latent", True))
                target_len = self.generated_frame_length if interp else None
                with torch.no_grad():
                    harmonic = extractor(mr, target_len=target_len)
                for i in range(harmonic.shape[0]):
                    out_pt = os.path.join(
                        audio_save_dir, f"{sample_stems[i]}_harmonic.pt"
                    )
                    torch.save(
                        {
                            "harmonic": harmonic[i].detach().cpu().float(),
                            "cqt_bins": int(harmonic.shape[1]),
                            "time": int(harmonic.shape[2]),
                            "target_len": target_len,
                            "midi_roll_frames": int(mr.shape[2]),
                            "audio_path": batch.get("audio_path", [None] * B)[i]
                            if isinstance(batch.get("audio_path"), list)
                            else None,
                        },
                        out_pt,
                    )

        generation_info = self._describe_generation()
        debug_branch_specs = None
        if self.save_test_conditioning_debug:
            debug_branch_specs = self._default_conditioning_debug_branch_specs()

        if self.codec_model:
            # Move codec to device if needed
            if next(self.codec_model.parameters()).device != self.device:
                self.codec_model.to(self.device)
        else:
            return

        # Save to disk (stems match ``sample_stems`` above)
        filenames = sample_stems

        tech_tokens = batch.get("tech_tokens")
        tech_roll = batch.get("tech_roll")
        cc_tokens = batch.get("cc_tokens")
        debug_manifest_path = os.path.join(audio_save_dir, "conditioning_debug.jsonl")
        base_seed = self._eval_base_seed()
        for i in range(B):
            sample_conditioning = {}
            for key, value in conditioning.items():
                if isinstance(value, torch.Tensor) and value.shape[0] == B:
                    sample_conditioning[key] = value[i : i + 1]
                else:
                    sample_conditioning[key] = value

            midi_path = (
                batch.get("midi_path", [None] * B)[i]
                if isinstance(batch.get("midi_path"), list)
                else None
            )
            sequence = midi_io.midi_file_to_note_sequence(str(midi_path)) if midi_path else None
            selected_seed = retry_seed(base_seed, batch_idx * max(B, 1) + i, 0)
            selected_attempt = 0
            selected_diag = None

            selected_wav, _ = self._render_test_audio(
                sample_conditioning,
                seed=selected_seed,
                debug_branch_specs=None,
            )

            if sequence is not None:
                window_seconds = selected_wav.shape[-1] / self.audio_sample_rate
                best_score = -float("inf")
                best_payload = (selected_wav, selected_seed, 0, None)

                for attempt in range(DEFAULT_MAX_RENDER_ATTEMPTS):
                    attempt_seed = retry_seed(base_seed, batch_idx * max(B, 1) + i, attempt)
                    if attempt == 0:
                        wav_attempt = selected_wav
                    else:
                        wav_attempt, _ = self._render_test_audio(
                            sample_conditioning,
                            seed=attempt_seed,
                            debug_branch_specs=None,
                        )

                    diag = render_window_diagnostics(
                        waveform=wav_attempt[0],
                        sample_rate=self.audio_sample_rate,
                        sequence=sequence,
                        start_time=0.0,
                        window_seconds=window_seconds,
                        playable_note_min_pitch=DEFAULT_PLAYABLE_NOTE_MIN_PITCH,
                        edge_silence_seconds=DEFAULT_EDGE_SILENCE_SECONDS,
                    )
                    score = float(diag["peak_rms"]) if diag["expected_activity"] else float("inf")
                    if score > best_score:
                        best_score = score
                        best_payload = (wav_attempt, attempt_seed, attempt, diag)
                    if not diag["near_silence"]:
                        selected_wav = wav_attempt
                        selected_seed = attempt_seed
                        selected_attempt = attempt
                        selected_diag = diag
                        break
                    if attempt + 1 < DEFAULT_MAX_RENDER_ATTEMPTS:
                        print(
                            f"[retry] {filenames[i]}: near-silent render "
                            f"({diag['peak_rms_db']:.2f} dBFS peak RMS), "
                            f"rerendering with seed "
                            f"{retry_seed(base_seed, batch_idx * max(B, 1) + i, attempt + 1)}"
                        )
                else:
                    selected_wav, selected_seed, selected_attempt, selected_diag = best_payload

                selected_wav = apply_eval_output_silence_mask(
                    waveform=selected_wav[0],
                    sample_rate=self.audio_sample_rate,
                    sequence=sequence,
                    start_time=0.0,
                    window_seconds=window_seconds,
                    edge_silence_seconds=DEFAULT_EDGE_SILENCE_SECONDS,
                    playable_note_min_pitch=DEFAULT_PLAYABLE_NOTE_MIN_PITCH,
                ).unsqueeze(0)
            else:
                selected_wav = selected_wav.clone()
                edge_samples = min(
                    int(round(DEFAULT_EDGE_SILENCE_SECONDS * self.audio_sample_rate)),
                    selected_wav.shape[-1],
                )
                if edge_samples > 0:
                    selected_wav[..., -edge_samples:] = 0.0

            debug_branch_audio = {}
            if self.save_test_conditioning_debug:
                selected_wav, debug_branch_audio = self._render_test_audio(
                    sample_conditioning,
                    seed=selected_seed,
                    debug_branch_specs=debug_branch_specs,
                )
                if sequence is not None:
                    selected_wav = apply_eval_output_silence_mask(
                        waveform=selected_wav[0],
                        sample_rate=self.audio_sample_rate,
                        sequence=sequence,
                        start_time=0.0,
                        window_seconds=selected_wav.shape[-1] / self.audio_sample_rate,
                        edge_silence_seconds=DEFAULT_EDGE_SILENCE_SECONDS,
                        playable_note_min_pitch=DEFAULT_PLAYABLE_NOTE_MIN_PITCH,
                    ).unsqueeze(0)
                else:
                    edge_samples = min(
                        int(round(DEFAULT_EDGE_SILENCE_SECONDS * self.audio_sample_rate)),
                        selected_wav.shape[-1],
                    )
                    if edge_samples > 0:
                        selected_wav[..., -edge_samples:] = 0.0

            wav = selected_wav[0]
            save_path = os.path.join(audio_save_dir, f"{filenames[i]}.wav")
            # wav is (1, T)
            torchaudio.save(save_path, wav.float(), self.audio_sample_rate)

            if not self.save_test_conditioning_debug:
                continue

            saved_branch_audio = {}
            for branch_name, branch_audio in debug_branch_audio.items():
                branch_path = os.path.join(audio_save_dir, f"{filenames[i]}__{branch_name}.wav")
                branch_wav = branch_audio[0]
                if sequence is not None:
                    branch_wav = apply_eval_output_silence_mask(
                        waveform=branch_wav,
                        sample_rate=self.audio_sample_rate,
                        sequence=sequence,
                        start_time=0.0,
                        window_seconds=branch_wav.shape[-1] / self.audio_sample_rate,
                        edge_silence_seconds=DEFAULT_EDGE_SILENCE_SECONDS,
                        playable_note_min_pitch=DEFAULT_PLAYABLE_NOTE_MIN_PITCH,
                    )
                else:
                    edge_samples = min(
                        int(round(DEFAULT_EDGE_SILENCE_SECONDS * self.audio_sample_rate)),
                        branch_wav.shape[-1],
                    )
                    if edge_samples > 0:
                        branch_wav[..., -edge_samples:] = 0.0
                torchaudio.save(branch_path, branch_wav.float(), self.audio_sample_rate)
                saved_branch_audio[branch_name] = os.path.basename(branch_path)

            tech_ids = []
            if tech_tokens is not None:
                tech_ids = sorted({int(t) for t in tech_tokens[i].detach().cpu().tolist() if int(t) != 0})

            record = {
                "filename": filenames[i],
                "midi_path": midi_path,
                "saved_audio": os.path.basename(save_path),
                "saved_branch_audio": saved_branch_audio,
                "effective_guidance_mode": generation_info["effective_guidance_mode"],
                "active_output_equals_branch": generation_info["active_output_equals_branch"],
                "effective_cond_scale": generation_info["cond_scale"],
                "effective_w_tech": generation_info["w_tech"],
                "effective_w_cc": generation_info["w_cc"],
                "sampler_cond_scale": getattr(self.sampler, "cond_scale", None),
                "sampler_w_tech": float(getattr(self.sampler, "w_tech", 0.0)),
                "sampler_w_cc": float(getattr(self.sampler, "w_cc", 0.0)),
                "tech_token_ids": tech_ids,
                "technique_names": [self.tech_map_inv.get(tid, str(tid)) for tid in tech_ids],
                "tech_tokens_nonzero": bool(tech_tokens is not None and tech_tokens[i].detach().cpu().ne(0).any().item()),
                "tech_roll_nonzero": bool(tech_roll is not None and tech_roll[i].detach().cpu().ne(0).any().item()),
                "cc_nonzero": bool(cc_tokens is not None and cc_tokens[i].detach().cpu().ne(0).any().item()),
                "render_seed": int(selected_seed),
                "render_attempt": int(selected_attempt + 1),
                "silence_peak_rms_db": None if selected_diag is None else float(selected_diag["peak_rms_db"]),
            }
            with open(debug_manifest_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        # Diffusion model optimizer
        optimizer_g = self.optimizer(params=list(self.net.parameters()))

        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer_g)
            return {
                "optimizer": optimizer_g,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": self.lr_scheduler_interval,
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer_g}
