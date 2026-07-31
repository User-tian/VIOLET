import numpy as np
from functools import partial
from typing import Optional, Callable
import torch
from torch import Tensor
from torch.nn import functional as F
import torch.nn as nn
from .utils import to_2tuple, default, exists
from .attention_utils import Attention
try:
    from torch import _assert
except ImportError:
    def _assert(condition: bool, message: str):
        assert condition, message
from .conditioner import (
    TimestepEmbedder,
    LabelEmbedder,
    TextEmbedder,
    MidiHarmonicExtractor,
    CausalWarmupWrapper,
    TechRollExtractor,
    CCEmbedder,
    prob_mask_like,
    get_1d_sincos_pos_embed_from_grid,
    resize_piano_roll_temporal,
)
from einops import rearrange

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


#################################################################################
#                                   DiT Configs
# DiT_XL_2: DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

# DiT_XL_4: DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

# DiT_XL_8: DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

# DiT_L_2: DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

# DiT_L_4: DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

# DiT_L_8: DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

# DiT_B_2: DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

# DiT_B_4: DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

# DiT_B_8: DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

# DiT_S_2: DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

# DiT_S_4: DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

# DiT_S_8: DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)

#################################################################################

def modulate(x, shift, scale):
    if shift.ndim == 2:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    return x * (1 + scale) + shift

# ViT layers
class AudioTokenEmbed(nn.Module):
    def __init__(self, in_dim, hidden_size):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # x: (B, T, D_lat)
        return self.norm(self.proj(x))  # (B, T, hidden)

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class GLU(nn.Module):
    """Gated Linear Unit (SwiGLU variant with configurable activation)."""
    def __init__(self, dim_in, dim_out, activation):
        super().__init__()
        self.act = activation
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x = self.proj(x)
        x, gate = x.chunk(2, dim=-1)
        return x * self.act(gate)


class GatedMlp(nn.Module):
    """Gated MLP (SwiGLU) as used in stable-audio-tools FeedForward."""
    def __init__(self, in_features, hidden_features=None, out_features=None, zero_init_output=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.glu = GLU(in_features, hidden_features, nn.SiLU())
        self.fc2 = nn.Linear(hidden_features, out_features)
        if zero_init_output:
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.glu(x)
        x = self.fc2(x)
        return x

#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size,
                 num_heads, mlp_ratio=4.0,
                 use_self_text_cond=True,
                 use_qk_l2norm=False, use_rope=True,
                 use_absolute_timing_rope_cross_attn: bool = False,
                 use_rope_in_cross_attn: bool = True,
                 attention_mode: str = "standard",
                 self_rope_rotary_frac: float = 1.0,
                 cross_rope_rotary_frac: float = 1.0,
                 rope_cross_attn_apply_to_v: bool = False,
                 rope_max_seq_len: Optional[int] = None,
                 use_midi_roll_adaln: bool = False,
                 use_tech_roll_adaln: bool = False):
        super().__init__()
        if attention_mode not in {"standard", "serial"}:
            raise ValueError(f"Unsupported attention_mode: {attention_mode}")
        self.attention_mode = attention_mode

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        if self.attention_mode == "serial":
            self.self_attn = Attention(
                dim=hidden_size,
                heads=num_heads,
                context_dim=hidden_size,
                use_self_text_cond=False,
                use_qk_l2norm=use_qk_l2norm,
                use_rope=use_rope,
                use_rope_in_cross_attn=False,
                use_absolute_timing_rope_cross_attn=False,
                self_rope_rotary_frac=self_rope_rotary_frac,
                rope_max_seq_len=rope_max_seq_len,
            )
            self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.cross_attn = Attention(
                dim=hidden_size,
                heads=num_heads,
                context_dim=hidden_size,
                use_self_text_cond=False,
                use_qk_l2norm=use_qk_l2norm,
                use_rope=use_rope,
                use_rope_in_cross_attn=use_rope_in_cross_attn,
                use_absolute_timing_rope_cross_attn=use_absolute_timing_rope_cross_attn,
                cross_rope_rotary_frac=cross_rope_rotary_frac,
                rope_cross_attn_apply_to_v=rope_cross_attn_apply_to_v,
                rope_max_seq_len=rope_max_seq_len,
            )
            mod_channels = 9 * hidden_size
        else:
            self.attn = Attention(dim=hidden_size,
                                  heads=num_heads,
                                  context_dim=hidden_size,
                                  use_self_text_cond=use_self_text_cond,
                                  use_qk_l2norm=use_qk_l2norm,
                                  use_rope=use_rope,
                                  use_absolute_timing_rope_cross_attn=use_absolute_timing_rope_cross_attn,
                                  use_rope_in_cross_attn=use_rope_in_cross_attn,
                                  self_rope_rotary_frac=self_rope_rotary_frac,
                                  cross_rope_rotary_frac=cross_rope_rotary_frac,
                                  rope_cross_attn_apply_to_v=rope_cross_attn_apply_to_v,
                                  rope_max_seq_len=rope_max_seq_len)
            mod_channels = 6 * hidden_size

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = GatedMlp(in_features=hidden_size,
                            hidden_features=mlp_hidden_dim,
                            out_features=hidden_size,
                            zero_init_output=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, mod_channels, bias=True)
        )

        # Additional per-frame modulations (CC / piano-roll conditions)
        self.cc_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, mod_channels, bias=True)
        )
        if use_midi_roll_adaln:
            self.midi_roll_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, mod_channels, bias=True)
            )
        if use_tech_roll_adaln:
            self.tech_roll_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, mod_channels, bias=True)
            )

    def forward(self, x, c,
                context=None,
                context_mask=None,
                pos_audio: Optional[Tensor] = None,
                pos_midi: Optional[Tensor] = None,
                cc_feats=None,
                midi_roll_feats=None,
                tech_roll_feats=None,
                cc_keep=None,
                return_attn_weights: bool = False):

        # Global modulation
        if self.attention_mode == "serial":
            (
                shift_msa,
                scale_msa,
                gate_msa,
                shift_cross,
                scale_cross,
                gate_cross,
                shift_mlp,
                scale_mlp,
                gate_mlp,
            ) = self.adaLN_modulation(c).chunk(9, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)

        # Convert global modulation to time-axis form, then add temporal condition modulations.
        if self.attention_mode == "serial":
            shift_msa = shift_msa.unsqueeze(1)
            scale_msa = scale_msa.unsqueeze(1)
            gate_msa = gate_msa.unsqueeze(1)
            shift_cross = shift_cross.unsqueeze(1)
            scale_cross = scale_cross.unsqueeze(1)
            gate_cross = gate_cross.unsqueeze(1)
            shift_mlp = shift_mlp.unsqueeze(1)
            scale_mlp = scale_mlp.unsqueeze(1)
            gate_mlp = gate_mlp.unsqueeze(1)
        else:
            shift_msa = shift_msa.unsqueeze(1)
            scale_msa = scale_msa.unsqueeze(1)
            gate_msa = gate_msa.unsqueeze(1)
            shift_mlp = shift_mlp.unsqueeze(1)
            scale_mlp = scale_mlp.unsqueeze(1)
            gate_mlp = gate_mlp.unsqueeze(1)

        # Local (per-timestep) modulation: cc_feats, midi_roll_feats, tech_roll_feats
        # have time dim aligned with x (B, T, D); modulation is applied per position.
        modulation_sources = []
        if cc_feats is not None:
            modulation_sources.append((cc_feats, self.cc_modulation))
        if midi_roll_feats is not None and hasattr(self, 'midi_roll_modulation'):
            modulation_sources.append((midi_roll_feats, self.midi_roll_modulation))
        if tech_roll_feats is not None and hasattr(self, 'tech_roll_modulation'):
            modulation_sources.append((tech_roll_feats, self.tech_roll_modulation))

        for src_feats, modulation in modulation_sources:
            if self.attention_mode == "serial":
                (
                    shift_msa_src,
                    scale_msa_src,
                    gate_msa_src,
                    shift_cross_src,
                    scale_cross_src,
                    gate_cross_src,
                    shift_mlp_src,
                    scale_mlp_src,
                    gate_mlp_src,
                ) = modulation(src_feats).chunk(9, dim=2)
            else:
                (
                    shift_msa_src,
                    scale_msa_src,
                    gate_msa_src,
                    shift_mlp_src,
                    scale_mlp_src,
                    gate_mlp_src,
                ) = modulation(src_feats).chunk(6, dim=2)

            shift_msa = shift_msa + shift_msa_src
            scale_msa = scale_msa + scale_msa_src
            gate_msa = gate_msa + gate_msa_src
            if self.attention_mode == "serial":
                shift_cross = shift_cross + shift_cross_src
                scale_cross = scale_cross + scale_cross_src
                gate_cross = gate_cross + gate_cross_src
            shift_mlp = shift_mlp + shift_mlp_src
            scale_mlp = scale_mlp + scale_mlp_src
            gate_mlp = gate_mlp + gate_mlp_src

        block_attn = {}
        if self.attention_mode == "serial":
            out_sa = self.self_attn(
                modulate(self.norm1(x), shift_msa, scale_msa),
                context=None,
                context_mask=None,
                return_attn_weights=return_attn_weights,
            )
            if return_attn_weights:
                out_sa, block_attn["self_attn"] = out_sa
            x = x + gate_msa * out_sa
            if context is not None:
                out_ca = self.cross_attn(
                    modulate(self.norm_cross(x), shift_cross, scale_cross),
                    context,
                    context_mask,
                    pos_audio=pos_audio,
                    pos_midi=pos_midi,
                    return_attn_weights=return_attn_weights,
                )
                if return_attn_weights:
                    out_ca, block_attn["cross_attn"] = out_ca
                x = x + gate_cross * out_ca
        else:
            out_attn = self.attn(
                modulate(self.norm1(x), shift_msa, scale_msa),
                context,
                context_mask,
                pos_audio=pos_audio,
                pos_midi=pos_midi,
                return_attn_weights=return_attn_weights,
            )
            if return_attn_weights:
                out_attn, block_attn["attn"] = out_attn
            x = x + gate_msa * out_attn
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        if return_attn_weights:
            return x, block_attn
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels, use_midi_roll_adaln: bool = False, use_tech_roll_adaln: bool = False):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        # Additional per-frame modulations (CC / piano-roll conditions)
        self.cc_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        if use_midi_roll_adaln:
            self.midi_roll_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 2 * hidden_size, bias=True)
            )
        if use_tech_roll_adaln:
            self.tech_roll_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 2 * hidden_size, bias=True)
            )

    def forward(self, x, c, cc_feats=None, midi_roll_feats=None, tech_roll_feats=None, cc_keep=None):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)

        if cc_feats is not None:
            shift_cc, scale_cc = self.cc_modulation(cc_feats).chunk(2, dim=2)
            shift = shift + shift_cc
            scale = scale + scale_cc
        if midi_roll_feats is not None and hasattr(self, 'midi_roll_modulation'):
            shift_midi, scale_midi = self.midi_roll_modulation(midi_roll_feats).chunk(2, dim=2)
            shift = shift + shift_midi
            scale = scale + scale_midi
        if tech_roll_feats is not None and hasattr(self, 'tech_roll_modulation'):
            shift_tech, scale_tech = self.tech_roll_modulation(tech_roll_feats).chunk(2, dim=2)
            shift = shift + shift_tech
            scale = scale + scale_tech

        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=256,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        cond_drop_prob=0.1,
        midi_cond_drop_prob=None,
        tech_cond_drop_prob=None,
        cc_cond_drop_prob=None,
        num_classes=None,
        class_embed_dim=None,
        label_cond=False,
        text_cond=False,
        text_embed_dim=512,
        max_text_len=128,
        use_self_text_cond=True,
        use_rope_in_cross_attn=True,
        use_absolute_timing_rope_cross_attn: bool = False,
        attention_mode: str = "standard",
        self_rope_rotary_frac: float = 1.0,
        cross_rope_rotary_frac: float = 1.0,
        rope_cross_attn_apply_to_v: bool = False,
        use_qk_l2norm=False,

        # Violin Synthesis conditioning: MIDI-roll (harmonic CQT-like) and technique-roll,
        # both applied via per-timestep AdaLN modulation. This is the only combination
        # supported here (see configs/experiment/violin_synthesis*); the token/crossattn/
        # concat condition paths and the "downsample" MIDI extractor variant, which were
        # never used by any shipped config, have been removed.
        use_cc=False,
        audio_use_sinusoidal_pos_emb=True,
        midi_roll_input_pitches: int = 51,
        midi_roll_top_k: int = 5,
        midi_roll_cqt_bins: int = 84,
        midi_roll_projection: str = "conv1d",
        midi_roll_interp_to_latent: bool = True,
        midi_roll_warmup_frames: int = 0,  # pianoroll input frames (same grid as conditioning)
        midi_roll_downsample_factor: int = 4,
        tech_roll_temporal_resize_mode: str = "nearest",
        tech_roll_num_techniques: int = 13,
        tech_roll_out_channels: int = 13,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.input_size = input_size
        self.num_heads = num_heads
        self.cond_drop_prob = cond_drop_prob
        self.midi_cond_drop_prob = midi_cond_drop_prob if midi_cond_drop_prob is not None else cond_drop_prob
        self.tech_cond_drop_prob = tech_cond_drop_prob if tech_cond_drop_prob is not None else cond_drop_prob
        self.cc_cond_drop_prob = cc_cond_drop_prob if cc_cond_drop_prob is not None else cond_drop_prob
        self.num_classes = num_classes
        self.label_cond = label_cond

        self.use_cc = use_cc
        self.midi_roll_interp_to_latent = midi_roll_interp_to_latent
        self.midi_roll_warmup_frames = midi_roll_warmup_frames
        self.tech_roll_temporal_resize_mode = tech_roll_temporal_resize_mode
        self.audio_use_sinusoidal_pos_emb = audio_use_sinusoidal_pos_emb
        self.attention_mode = attention_mode

        self.x_embedder = AudioTokenEmbed(in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size, hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, class_embed_dim, hidden_size, hidden_size) if label_cond else None
        self.text_conditioner = TextEmbedder(hidden_size, text_embed_dim, max_text_len) if text_cond else None

        self.tech_roll_extractor = TechRollExtractor(
            num_techniques=tech_roll_num_techniques,
            out_channels=tech_roll_out_channels,
            temporal_resize_mode=tech_roll_temporal_resize_mode,
        )
        self.tech_roll_frame_embedder = CCEmbedder(tech_roll_out_channels, hidden_size)

        if midi_roll_projection != "conv1d":
            raise ValueError("Only conv1d projection is currently supported for midi roll conditioning.")
        midi_harmonic_extractor = MidiHarmonicExtractor(
            top_k=midi_roll_top_k,
            cqt_bins=midi_roll_cqt_bins,
            input_pitches=midi_roll_input_pitches,
        )
        if midi_roll_warmup_frames > 0:
            midi_harmonic_extractor = CausalWarmupWrapper(
                extractor=midi_harmonic_extractor,
                warmup_frames=midi_roll_warmup_frames,
                downsample_factor=midi_roll_downsample_factor,
            )
        self.midi_harmonic_extractor = midi_harmonic_extractor
        self.midi_roll_frame_embedder = CCEmbedder(midi_roll_cqt_bins, hidden_size)

        if use_cc:
            # CC embedder for temporal FiLM (1 channel input)
            self.cc_frame_embedder = CCEmbedder(1, hidden_size)

        num_patches = self.input_size
        # Fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                     use_self_text_cond=use_self_text_cond,
                     use_qk_l2norm=use_qk_l2norm, use_rope=True,
                     use_absolute_timing_rope_cross_attn=use_absolute_timing_rope_cross_attn,
                     use_rope_in_cross_attn=use_rope_in_cross_attn,
                     attention_mode=attention_mode,
                     self_rope_rotary_frac=self_rope_rotary_frac,
                     cross_rope_rotary_frac=cross_rope_rotary_frac,
                     rope_cross_attn_apply_to_v=rope_cross_attn_apply_to_v,
                     use_midi_roll_adaln=True,
                     use_tech_roll_adaln=True) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, self.out_channels, use_midi_roll_adaln=True, use_tech_roll_adaln=True)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos = np.arange(self.input_size)
        pos_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], pos)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        nn.init.xavier_uniform_(self.tech_roll_extractor.proj.weight)
        if self.tech_roll_extractor.proj.bias is not None:
            nn.init.constant_(self.tech_roll_extractor.proj.bias, 0)

        # Initialize label embedding table:
        if self.label_cond:
            nn.init.normal_(self.y_embedder.label_emb.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            # Initialize CC modulation
            if hasattr(block, 'cc_modulation'):
                nn.init.constant_(block.cc_modulation[-1].weight, 0)
                nn.init.constant_(block.cc_modulation[-1].bias, 0)
            if hasattr(block, 'midi_roll_modulation'):
                nn.init.constant_(block.midi_roll_modulation[-1].weight, 0)
                nn.init.constant_(block.midi_roll_modulation[-1].bias, 0)
            if hasattr(block, 'tech_roll_modulation'):
                nn.init.constant_(block.tech_roll_modulation[-1].weight, 0)
                nn.init.constant_(block.tech_roll_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        if hasattr(self.final_layer, 'cc_modulation'):
             nn.init.constant_(self.final_layer.cc_modulation[-1].weight, 0)
             nn.init.constant_(self.final_layer.cc_modulation[-1].bias, 0)
        if hasattr(self.final_layer, 'midi_roll_modulation'):
             nn.init.constant_(self.final_layer.midi_roll_modulation[-1].weight, 0)
             nn.init.constant_(self.final_layer.midi_roll_modulation[-1].bias, 0)
        if hasattr(self.final_layer, 'tech_roll_modulation'):
             nn.init.constant_(self.final_layer.tech_roll_modulation[-1].weight, 0)
             nn.init.constant_(self.final_layer.tech_roll_modulation[-1].bias, 0)

    def forward(self,
                x: Tensor,
                t: Tensor,
                classes:Optional[Tensor] = None,         # class labels or class embeddings
                text_embeds:Optional[Tensor] = None,         # text embeddings
                text_mask:Optional[Tensor] = None,
                cc_tokens: Optional[Tensor] = None,
                midi_roll: Optional[Tensor] = None,
                tech_roll: Optional[Tensor] = None,
                cc_keep_mask: Optional[Tensor] = None,
                tech_roll_keep_mask: Optional[Tensor] = None,
                cond_drop_prob=None,
                midi_cond_drop_prob: Optional[float] = None,
                tech_cond_drop_prob: Optional[float] = None,
                cc_cond_drop_prob: Optional[float] = None,
                return_attn_weights: bool = False,
                attn_block_indices: Optional[list] = None,
                **kwargs):
        """
        Forward pass of DiT.
        x: (N, C, T) tensor of audio inputs
        time: (N,) tensor of diffusion timesteps
        classes: (N,) tensor of class labels
        """
        # Resolve cond_drop: if passed as scalar (e.g. 0/1 for CFG), use for all; else use stored per-modality probs
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)
        if isinstance(cond_drop_prob, (int, float)):
            midi_drop = tech_drop = cc_drop = float(cond_drop_prob)
        else:
            midi_drop = self.midi_cond_drop_prob
            tech_drop = self.tech_cond_drop_prob
            cc_drop = self.cc_cond_drop_prob

        if midi_cond_drop_prob is not None:
            midi_drop = float(midi_cond_drop_prob)
        if tech_cond_drop_prob is not None:
            tech_drop = float(tech_cond_drop_prob)
        if cc_cond_drop_prob is not None:
            cc_drop = float(cc_cond_drop_prob)

        # Transpose from (B, C, T) to (B, T, C) for Linear embedder
        x = x.transpose(1, 2)
        x = self.x_embedder(x)
        midi_roll_feats = None
        tech_roll_feats = None
        midi_feat = None
        if exists(midi_roll):
            midi_roll = midi_roll.to(device=x.device, dtype=x.dtype)
            target_len = x.shape[1] if self.midi_roll_interp_to_latent else None
            midi_feat = self.midi_harmonic_extractor(midi_roll, target_len=target_len)
            if target_len is None and midi_feat.shape[-1] != x.shape[1]:
                midi_feat = resize_piano_roll_temporal(midi_feat, x.shape[1], "nearest")

        if midi_feat is not None:
            # Local adaLN: feats interpolated to latent time T for per-timestep modulation
            midi_keep = None
            if midi_drop > 0:
                keep_mask = prob_mask_like((x.shape[0],), 1 - midi_drop, device=x.device)
                midi_keep = rearrange(keep_mask, 'b -> b 1 1').type(x.dtype)
            midi_roll_feats = self.midi_roll_frame_embedder(
                midi_feat.transpose(1, 2),
                target_len=x.shape[1],
                cc_keep=midi_keep,
            )
        if exists(tech_roll):
            # Local adaLN: feats interpolated to latent time T for per-timestep modulation
            tech_keep = None
            if tech_roll_keep_mask is not None:
                tech_keep = tech_roll_keep_mask.to(device=x.device, dtype=x.dtype)
            if tech_drop > 0:
                keep_mask = prob_mask_like((x.shape[0],), 1 - tech_drop, device=x.device)
                sampled_keep = rearrange(keep_mask, 'b -> b 1 1').type(x.dtype)
                tech_keep = sampled_keep if tech_keep is None else tech_keep * sampled_keep
            tech_feat = self.tech_roll_extractor(
                tech_roll.to(device=x.device, dtype=x.dtype),
                target_len=x.shape[1],
            )
            tech_roll_feats = self.tech_roll_frame_embedder(
                tech_feat.transpose(1, 2),
                target_len=x.shape[1],
                cc_keep=tech_keep,
            )
        if self.audio_use_sinusoidal_pos_emb:
            x = x + self.pos_embed  # (N, T, D)
        t = self.t_embedder(t)                   # (N, D)

        if exists(classes):
            c = self.y_embedder(classes, cond_drop_prob)    # (N, D)
            c = c + t
        else:
            c = t

        # Precompute CC features for modulation if present
        cc_feats = None
        cc_keep = None
        if self.use_cc and exists(cc_tokens):
            T = x.shape[1]
            if cc_keep_mask is not None:
                cc_keep = cc_keep_mask.to(device=x.device, dtype=x.dtype)
            if cc_drop > 0:
                keep_mask = prob_mask_like((x.shape[0],), 1 - cc_drop, device=x.device)
                sampled_keep = rearrange(keep_mask, 'b -> b 1 1').type(x.dtype)
                cc_keep = sampled_keep if cc_keep is None else cc_keep * sampled_keep
            cc_feats = self.cc_frame_embedder(cc_tokens, target_len=T, cc_keep=cc_keep)  # (B, T, D), blends null when dropped


        # Collect conditioning contexts (Cross-Attention)
        contexts = []
        masks = []

        # Text condition
        if exists(text_embeds):
            text_context, text_mask = self.text_conditioner(text_embeds, text_mask, cond_drop_prob)
            contexts.append(text_context)
            if text_mask is not None:
                masks.append(text_mask)
            else:
                # Assuming no mask means all valid
                masks.append(torch.ones((text_context.shape[0], text_context.shape[1]), device=text_context.device).bool())

        if len(contexts) > 0:
            context = torch.cat(contexts, dim=1)
            # Combine masks
            if len(masks) > 0:
                context_mask = torch.cat(masks, dim=1)
            else:
                context_mask = None
        else:
            context = None
            context_mask = None

        # Absolute-timing RoPE cross-attn was only ever wired to the (now removed) MIDI
        # token cross-attention context, so there is no source of positions left to use.
        pos_audio_ctx = None
        pos_midi_ctx = None

        all_attn_weights = {}
        attn_block_index_set = None
        if return_attn_weights and attn_block_indices is not None:
            attn_block_index_set = {
                int(i) for i in attn_block_indices
                if 0 <= int(i) < len(self.blocks)
            }
        for i, block in enumerate(self.blocks):
            collect_block_attn = return_attn_weights and (
                attn_block_index_set is None or i in attn_block_index_set
            )
            out = block(
                x,
                c,
                context,
                context_mask,
                pos_audio=pos_audio_ctx,
                pos_midi=pos_midi_ctx,
                cc_feats=cc_feats,
                midi_roll_feats=midi_roll_feats,
                tech_roll_feats=tech_roll_feats,
                cc_keep=cc_keep,
                return_attn_weights=collect_block_attn,
            )
            if collect_block_attn:
                x, block_attn = out
                all_attn_weights[i] = block_attn
            else:
                x = out  # (N, T, D)

        x = self.final_layer(
            x,
            c,
            cc_feats=cc_feats,
            midi_roll_feats=midi_roll_feats,
            tech_roll_feats=tech_roll_feats,
            cc_keep=cc_keep,
        )                # (N, T, out_channels)
        x = x.transpose(1, 2)                    # (N, out_channels, T)

        if return_attn_weights:
            return x, all_attn_weights
        return x
