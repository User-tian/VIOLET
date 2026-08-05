import math
import torch
import torch.nn as nn
from einops import rearrange
from .operator_utils import prob_mask_like
from .utils import exists
import torch.nn.functional as F
import numpy as np


class L2NormalizationLayer(nn.Module):
    def __init__(self, dim=1, eps=1e-12):
        super(L2NormalizationLayer, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim, eps=self.eps)

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations (standard sinusoidal timestep embedding).
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    in DiT: class_channels== time_embed_dim
    """
    def __init__(self,
                 num_classes,
                 class_embed_dim,
                 model_channels,
                 class_channels,
                 ):
        super().__init__()

        assert num_classes is None or class_embed_dim is None, "Provide either num_classes or class_embed_dim, not both."
        self.num_classes = num_classes
        self.null_classes_emb = nn.Parameter(torch.randn(1, model_channels))

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, model_channels)

        elif class_embed_dim is not None:
            # only embedding is provided
            self.class_embed_norm = L2NormalizationLayer()
            self.label_emb = nn.Linear(class_embed_dim, model_channels)
            nn.init.normal_(self.null_classes_emb, 0, 1 / model_channels ** 0.5)

        self.class_to_cond = nn.Sequential(
                nn.LayerNorm(model_channels),
                nn.Linear(model_channels, class_channels),
                nn.SiLU(),
                nn.Linear(class_channels, class_channels)
            )

    def forward(self, classes, cond_drop_prob):

        if self.num_classes is None:
            classes = self.class_embed_norm(classes)

        classes_emb = self.label_emb(classes)

        if cond_drop_prob > 0:

            label_keep_mask = prob_mask_like((classes.shape[0],), 1 - cond_drop_prob, device=classes.device)

            classes_emb = torch.where(
                rearrange(label_keep_mask, 'b -> b 1'),
                classes_emb,
                self.null_classes_emb
            )

        classes_emb = self.class_to_cond(classes_emb)

        return classes_emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed_from_grid_torch(embed_dim, pos):
    """
    Pure PyTorch version of get_1d_sincos_pos_embed_from_grid.
    Use this in forward() to avoid .numpy() which breaks under torch.func.jvp
    (tensors in JVP trace don't have storage and cannot be converted to numpy).

    embed_dim: output dimension for each position
    pos: (M,) tensor of positions
    out: (M, D) tensor
    """
    assert embed_dim % 2 == 0
    device = pos.device
    dtype = pos.dtype if pos.dtype.is_floating_point else torch.float32
    half_dim = embed_dim // 2
    omega = torch.arange(half_dim, device=device, dtype=dtype)
    omega = omega / (embed_dim / 2.0)
    omega = 1.0 / (10000.0 ** omega)
    pos = pos.reshape(-1).to(dtype)
    out = torch.einsum('m,d->md', pos, omega)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    emb = torch.cat([emb_sin, emb_cos], dim=1)
    return emb

class MidiHarmonicExtractor(nn.Module):
    """Extract sparse CQT-like MIDI harmonics from cropped binary roll.
    Uses strict left-padding to prevent future MIDI onset energy from
    leaking into present frames (causal convolutions)."""

    def __init__(self, top_k=5, cqt_bins=84, input_pitches=51):
        super().__init__()
        self.top_k = top_k
        self.cqt_bins = cqt_bins

        # Removed symmetric padding (padding=0). We will handle it manually in forward()
        self.conv1 = nn.Conv1d(input_pitches, 256, kernel_size=5, padding=0, stride=2)
        self.conv2 = nn.Conv1d(256, 512, kernel_size=3, padding=0, stride=2)
        self.conv3 = nn.Conv1d(512, cqt_bins, kernel_size=3, padding=0)

    def forward(self, midi_roll, target_len=None):
        # midi_roll: (B, 51, L) at 10ms resolution

        # 1. Causal Pad & Conv1
        # kernel=5 -> pad 4 on the left, 0 on the right
        x = F.pad(midi_roll, (4, 0))
        x = F.silu(self.conv1(x))

        # 2. Causal Pad & Conv2
        # kernel=3 -> pad 2 on the left, 0 on the right
        x = F.pad(x, (2, 0))
        x = F.silu(self.conv2(x))

        # 3. Causal Pad & Conv3
        # kernel=3 -> pad 2 on the left, 0 on the right
        x = F.pad(x, (2, 0))
        x = F.relu(self.conv3(x))

        k = min(self.top_k, x.shape[1])
        top_values, top_indices = torch.topk(x, k, dim=1)
        sparse = torch.zeros_like(x)
        sparse.scatter_(1, top_indices, top_values)

        # "nearest" (not linear) interpolation to avoid blurring the sparse top-k spikes
        if target_len is not None and sparse.shape[-1] != target_len:
            sparse = F.interpolate(sparse, size=target_len, mode="nearest")

        return sparse


class CausalWarmupWrapper(nn.Module):
    """Warms up a causal extractor with prepended silence, then re-aligns output.

    *warmup_frames* is in the same units as the MIDI pianoroll time axis (input frames to the extractor).
    """

    def __init__(
        self,
        extractor: nn.Module,
        warmup_frames: int = 50,
        downsample_factor: int = 4,
    ):
        super().__init__()
        self.extractor = extractor
        self.warmup_frames = int(warmup_frames)
        self.downsample_factor = int(downsample_factor)

        if self.warmup_frames < 0:
            raise ValueError("warmup_frames must be non-negative.")
        if self.downsample_factor <= 0:
            raise ValueError("downsample_factor must be positive.")

        base_pad_frames = max(0, self.warmup_frames)
        remainder = base_pad_frames % self.downsample_factor
        if remainder != 0:
            self.pad_frames_in = base_pad_frames + (self.downsample_factor - remainder)
        else:
            self.pad_frames_in = base_pad_frames
        self.drop_frames_out = self.pad_frames_in // self.downsample_factor

        # Preserve commonly accessed extractor metadata for downstream code.
        self.cqt_bins = getattr(extractor, "cqt_bins", None)
        self.top_k = getattr(extractor, "top_k", None)

    def forward(self, midi_roll: torch.Tensor, target_len: int = None) -> torch.Tensor:
        padded_midi = (
            F.pad(midi_roll, (self.pad_frames_in, 0))
            if self.pad_frames_in > 0
            else midi_roll
        )
        raw_extracted = self.extractor(padded_midi, target_len=None)
        aligned_conditioning = (
            raw_extracted[:, :, self.drop_frames_out:]
            if self.drop_frames_out > 0
            else raw_extracted
        )

        if target_len is not None and aligned_conditioning.shape[-1] != target_len:
            aligned_conditioning = F.interpolate(
                aligned_conditioning,
                size=target_len,
                mode="nearest",
            )

        return aligned_conditioning


_VALID_PIANO_ROLL_TEMPORAL_MODES = frozenset({"nearest", "linear", "avg_pool"})


def resize_piano_roll_temporal(x: torch.Tensor, target_len: int, mode: str) -> torch.Tensor:
    """Resize last dimension of ``(B, C, L)`` to ``target_len``.

    * ``nearest`` / ``linear``: ``F.interpolate`` on the time axis.
    * ``avg_pool``: ``F.adaptive_avg_pool1d`` (smooth pooling to exact length).
    """
    if mode not in _VALID_PIANO_ROLL_TEMPORAL_MODES:
        raise ValueError(
            f"temporal resize mode must be one of {sorted(_VALID_PIANO_ROLL_TEMPORAL_MODES)}, got {mode!r}"
        )
    if x.shape[-1] == target_len:
        return x
    if mode == "avg_pool":
        return F.adaptive_avg_pool1d(x, target_len)
    if mode == "nearest":
        return F.interpolate(x, size=target_len, mode="nearest")
    return F.interpolate(x, size=target_len, mode="linear", align_corners=False)


class TechRollExtractor(nn.Module):
    """Temporal pooling for technique pianoroll to align with latent time resolution.

    Input:  (B, num_techniques, L)  -- binary roll at e.g. 10 ms resolution
    Output: (B, out_channels, T)    -- features at latent resolution

    Uses a lightweight linear projection per-frame (1x1 conv) followed by
    temporal resizing to match the target length (see ``temporal_resize_mode``).
    No harmonic expansion is needed because techniques are already a compact
    one-hot-like representation.
    """

    def __init__(
        self,
        num_techniques: int = 13,
        out_channels: int = 13,
        temporal_resize_mode: str = "nearest",
    ):
        super().__init__()
        if temporal_resize_mode not in _VALID_PIANO_ROLL_TEMPORAL_MODES:
            raise ValueError(
                f"temporal_resize_mode must be one of {sorted(_VALID_PIANO_ROLL_TEMPORAL_MODES)}, "
                f"got {temporal_resize_mode!r}"
            )
        self.num_techniques = num_techniques
        self.out_channels = out_channels
        self.temporal_resize_mode = temporal_resize_mode
        self.proj = nn.Conv1d(num_techniques, out_channels, kernel_size=1)

    def forward(self, tech_roll, target_len=None):
        # tech_roll: (B, num_techniques, L)
        x = self.proj(tech_roll)
        if target_len is not None and x.shape[-1] != target_len:
            # Backward compat: if temporal_resize_mode is missing (old checkpoint), fall back to nearest
            mode = getattr(self, "temporal_resize_mode", "nearest")
            x = resize_piano_roll_temporal(x, target_len, mode)
        return x


class CCEmbedder(nn.Module):
    """
    Embeds CC tokens (continuous frame conditioning).
    Projects and potentially upsamples to match latent time dimension.
    When cc_keep is provided, dropped samples use a learned null embedding.
    """
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.proj = nn.Linear(input_dim, hidden_size)
        self.act = nn.SiLU()
        self.norm = nn.LayerNorm(hidden_size)
        # Learned null vector for classifier-free / missing-CC conditioning.
        self.null_cc_embed = nn.Parameter(torch.randn(1, 1, hidden_size))
        nn.init.normal_(self.null_cc_embed, mean=0.0, std=hidden_size ** -0.5)
    def forward(self, cc_tokens, target_len=None, cc_keep=None):
        # cc_tokens: (B, L_cc, D_in)
        # cc_keep: (B, 1, 1) optional mask for cond_drop / missing conditioning
        x = self.proj(cc_tokens)
        x = self.act(x)
        x = self.norm(x)

        if target_len is not None and target_len != x.shape[1]:
            # Interpolate to target_len
            x = x.transpose(1, 2)  # (B, D, L_cc)
            x = F.interpolate(x, size=target_len, mode='linear', align_corners=False)
            x = x.transpose(1, 2)  # (B, T, D)

        if cc_keep is not None:
            # Keep present samples; replace dropped/missing samples with learned null.
            null_cc = self.null_cc_embed.to(dtype=x.dtype)
            x = torch.where(cc_keep.bool(), x, null_cc)

        return x
