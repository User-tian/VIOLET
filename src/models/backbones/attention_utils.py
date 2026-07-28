import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange
from einops_exts import rearrange_many
from functools import partial
from typing import Optional, Tuple
from .layer_utils import LayerNorm
from .operator_utils import l2norm, reshape_for_broadcast
from .utils import exists

def compute_freqs_cis(dim: int, end: int, theta: float = 10000.0,
                      device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.

    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float().to(device) / dim))
    t = torch.arange(end).to(device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs).to(device), freqs)  # complex64
    return freqs_cis

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    cross_attn: bool = False,
    rotary_dim: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.



    """
    head_dim = xq.shape[-1]
    rotary_dim = head_dim if rotary_dim is None else rotary_dim
    rotary_dim = min(rotary_dim, head_dim)
    rotary_dim = rotary_dim - (rotary_dim % 2)
    if rotary_dim <= 0:
        return xq, xk

    xq_rot, xq_pass = xq[..., :rotary_dim], xq[..., rotary_dim:]
    xk_rot, xk_pass = xk[..., :rotary_dim], xk[..., rotary_dim:]

    xq_ = torch.view_as_complex(xq_rot.float().reshape(*xq_rot.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk_rot.float().reshape(*xk_rot.shape[:-1], -1, 2))

    # freqs_cis is computed for full head_dim; slice to match rotary_dim when using partial RoPE
    cis_dim = rotary_dim // 2  # complex dims = rotary_dim / 2
    if freqs_cis.shape[-1] > cis_dim:
        freqs_cis = freqs_cis[..., :cis_dim].contiguous()

    if cross_attn:
        x_shape = list(xq_.shape)
        x_shape[1] = xq_.shape[1] + xk_.shape[1]
        freqs_cis = reshape_for_broadcast(freqs_cis, x_shape)
        xq_out = torch.view_as_real(xq_ * freqs_cis[:, :xq_.shape[1]]).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis[:, xq_.shape[1]:]).flatten(3)
    else:
        freqs_cis = reshape_for_broadcast(freqs_cis, list(xk_.shape))
        xq_out = torch.view_as_real(xq_ * freqs_cis[:, :xq_.shape[1]]).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    xq_out = torch.cat((xq_out.type_as(xq), xq_pass), dim=-1)
    xk_out = torch.cat((xk_out.type_as(xk), xk_pass), dim=-1)
    return xq_out, xk_out


def rope_apply_with_pos(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    pos_ids: torch.Tensor,
    rotary_dim: Optional[int] = None,
) -> torch.Tensor:
    """
    Apply RoPE using explicit per-token absolute positions.
    x: [B, N, H, D], pos_ids: [B, N], freqs_cis: [max_pos, D/2] complex
    """
    head_dim = x.shape[-1]
    rotary_dim = head_dim if rotary_dim is None else rotary_dim
    rotary_dim = min(rotary_dim, head_dim)
    rotary_dim = rotary_dim - (rotary_dim % 2)
    if rotary_dim <= 0:
        return x

    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    x_ = torch.view_as_complex(x_rot.float().reshape(*x_rot.shape[:-1], -1, 2))
    cis_dim = rotary_dim // 2
    if freqs_cis.shape[-1] > cis_dim:
        freqs_cis = freqs_cis[..., :cis_dim].contiguous()
    pos_ids = pos_ids.to(device=x.device, dtype=torch.long)
    pos_ids = pos_ids.clamp(min=0, max=freqs_cis.shape[0] - 1)
    f = freqs_cis[pos_ids]  # [B, N, D/2]
    f = f.unsqueeze(2)  # [B, N, 1, D/2]
    out = torch.view_as_real(x_ * f).flatten(3).type_as(x)
    return torch.cat((out, x_pass), dim=-1)


def apply_rotary_emb_with_pos(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    pos_q: torch.Tensor,
    pos_k: torch.Tensor,
    rotary_dim: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_out = rope_apply_with_pos(xq, freqs_cis, pos_q, rotary_dim=rotary_dim)
    xk_out = rope_apply_with_pos(xk, freqs_cis, pos_k, rotary_dim=rotary_dim)
    return xq_out, xk_out

# attention
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        context_dim = None,
        use_self_text_cond = True,
        use_qk_l2norm = False,
        use_rope = True,
        use_rope_in_cross_attn: bool = True,
        use_absolute_timing_rope_cross_attn: bool = False,
        rope_max_seq_len: Optional[int] = None,
        self_rope_rotary_frac: float = 1.0,
        cross_rope_rotary_frac: float = 1.0,
        rope_cross_attn_apply_to_v: bool = False,
        out_drop: float = 0.,
    ):
        super().__init__()

        self.heads = heads
        assert dim % heads == 0, "Embedding dimension must be 0 modulo number of heads."
        self.head_dim = dim // heads

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, dim * 2, bias=False)
        self.to_context = nn.Linear(context_dim, dim * 2, bias=False) if exists(context_dim) else None
        self.use_self_text_cond = use_self_text_cond
        self.use_rope = use_rope # use rotary position encoding
        self.use_rope_in_cross_attn = use_rope_in_cross_attn
        self.use_absolute_timing_rope_cross_attn = use_absolute_timing_rope_cross_attn
        self.rope_max_seq_len = rope_max_seq_len
        self.self_rope_rotary_frac = self_rope_rotary_frac
        self.cross_rope_rotary_frac = cross_rope_rotary_frac
        self.rope_cross_attn_apply_to_v = rope_cross_attn_apply_to_v
        self.freqs_cis_dict = {}
        self.register_buffer("_rope_freqs_cis", None, persistent=False)

        self.use_qk_l2norm = use_qk_l2norm
        if self.use_qk_l2norm:
            self.q_scale = nn.Parameter(torch.ones(self.head_dim))
            self.k_scale = nn.Parameter(torch.ones(self.head_dim))
            self.scale = self.head_dim ** 0.5
        else:
            self.scale = self.head_dim ** -0.5

        self.to_out = nn.Linear(dim, dim, bias=False)
        self.to_out_drop = nn.Dropout(out_drop)

    def _get_freqs_cis(self, cat_seq_len: int, device: torch.device) -> torch.Tensor:
        rope_max_seq_len = getattr(self, 'rope_max_seq_len', None)
        if rope_max_seq_len is None:
            key = str(cat_seq_len)
            if key not in self.freqs_cis_dict:
                self.freqs_cis_dict[key] = compute_freqs_cis(
                    self.head_dim, cat_seq_len, device=device
                )
            return self.freqs_cis_dict[key]

        if cat_seq_len > rope_max_seq_len:
            raise RuntimeError(
                f"cat_seq_len ({cat_seq_len}) exceeds rope_max_seq_len ({rope_max_seq_len}). "
                "Increase `rope_max_seq_len` for this IMF model."
            )

        if (
            self._rope_freqs_cis is None
            or self._rope_freqs_cis.device != device
            or self._rope_freqs_cis.shape[0] < rope_max_seq_len
        ):
            self._rope_freqs_cis = compute_freqs_cis(
                self.head_dim, rope_max_seq_len, device=device
            )

        return self._rope_freqs_cis[:cat_seq_len]

    def _rotary_dim_from_frac(self, frac: float) -> int:
        frac = float(frac)
        if frac <= 0:
            return 0
        if frac >= 1:
            return self.head_dim
        rotary_dim = int(self.head_dim * frac)
        rotary_dim = rotary_dim - (rotary_dim % 2)
        return max(0, rotary_dim)

    def forward(self, x,
                context=None,
                context_mask=None,
                pos_audio: Optional[torch.Tensor] = None,
                pos_midi: Optional[torch.Tensor] = None,
                return_attn_weights: bool = False):

        q = self.to_q(x)

        # add text conditioning, if present
        if self.use_self_text_cond and exists(context):

            assert exists(self.to_context)
            k, v = self.to_kv(x).chunk(2, dim = -1)

            ck, cv = self.to_context(context).chunk(2, dim = -1)
            k = torch.cat((k, ck), dim = -2)
            v = torch.cat((v, cv), dim = -2)

            if self.use_rope:
                # rope after concat
                q, k = rearrange_many((q, k), "b n (h d) -> b n h d", h=self.heads)
                cat_seq_len = k.shape[1]
                freqs_cis = self._get_freqs_cis(cat_seq_len, k.device)
                q, k = apply_rotary_emb(
                    q,
                    k,
                    freqs_cis=freqs_cis,
                    rotary_dim=self._rotary_dim_from_frac(self.self_rope_rotary_frac),
                )
                q, k = rearrange_many((q, k), "b n h d -> b n (h d)")

            if context_mask is not None:
                x_mask_pad = torch.ones(x.shape[0], x.shape[-2]).to(context_mask.device)
                context_mask = torch.cat((x_mask_pad, context_mask), 1)

        else:
            if exists(context):

                k, v = self.to_context(context).chunk(2, dim = -1)

                if self.use_rope and self.use_rope_in_cross_attn:
                    # rope on cross attention
                    q, k = rearrange_many((q, k), "b n (h d) -> b n h d", h=self.heads)
                    rotary_dim = self._rotary_dim_from_frac(self.cross_rope_rotary_frac)
                    v_rope = None
                    if self.rope_cross_attn_apply_to_v:
                        v_rope = rearrange(v, "b n (h d) -> b n h d", h=self.heads)
                    if (
                        self.use_absolute_timing_rope_cross_attn
                        and exists(pos_audio)
                        and exists(pos_midi)
                    ):
                        max_pos = int(torch.max(torch.cat((pos_audio, pos_midi), dim=1)).item()) + 1
                        freqs_cis = self._get_freqs_cis(max_pos, k.device)
                        q, k = apply_rotary_emb_with_pos(
                            q,
                            k,
                            freqs_cis=freqs_cis,
                            pos_q=pos_audio,
                            pos_k=pos_midi,
                            rotary_dim=rotary_dim,
                        )
                        if v_rope is not None:
                            v_rope = rope_apply_with_pos(
                                v_rope,
                                freqs_cis=freqs_cis,
                                pos_ids=pos_midi,
                                rotary_dim=rotary_dim,
                            )
                    else:
                        cat_seq_len = k.shape[1] + q.shape[1]
                        freqs_cis = self._get_freqs_cis(cat_seq_len, k.device)
                        q, k = apply_rotary_emb(
                            q,
                            k,
                            freqs_cis=freqs_cis,
                            cross_attn=True,
                            rotary_dim=rotary_dim,
                        )
                        if v_rope is not None:
                            _, v_rope = apply_rotary_emb(
                                q,
                                v_rope,
                                freqs_cis=freqs_cis,
                                cross_attn=True,
                                rotary_dim=rotary_dim,
                            )
                    q, k = rearrange_many((q, k), "b n h d -> b n (h d)")
                    if v_rope is not None:
                        v = rearrange(v_rope, "b n h d -> b n (h d)")

            else:
                k, v = self.to_kv(x).chunk(2, dim = -1)
                if self.use_rope:
                    # RoPE on pure self-attention (no external context).
                    q, k = rearrange_many((q, k), "b n (h d) -> b n h d", h=self.heads)
                    seq_len = k.shape[1]
                    freqs_cis = self._get_freqs_cis(seq_len, k.device)
                    q, k = apply_rotary_emb(
                        q,
                        k,
                        freqs_cis=freqs_cis,
                        rotary_dim=self._rotary_dim_from_frac(self.self_rope_rotary_frac),
                    )
                    q, k = rearrange_many((q, k), "b n h d -> b n (h d)")

        # reshape for multi-head attention
        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=self.heads)

        if self.use_qk_l2norm:
            # qk l2norm
            q, k = map(l2norm, (q, k))
            q = q * self.q_scale
            k = k * self.k_scale

        # calculate query / key similarities
        sim = einsum("... n d, ... m d -> ... n m", q, k) * self.scale

        # masking
        if exists(context_mask):
            max_neg_value = torch.finfo(sim.dtype).min
            context_mask = rearrange(context_mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(context_mask==0, max_neg_value)

        # attention
        attn = sim.softmax(dim = -1, dtype = torch.float32)
        attn = attn.to(sim.dtype)

        # aggregate values
        out = einsum("... n j, ... j d -> ... n d", attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)').contiguous()
        out = self.to_out(out).contiguous()
        if return_attn_weights:
            return out, attn  # attn: (b, h, n, m)
        return out


def FeedForward(dim, mult = 2):
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, hidden_dim, bias = False),
        nn.GELU(),
        LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, dim, bias = False)
    )

ChanLayerNorm = partial(LayerNorm, dim = -3)

def ChanFeedForward(dim, mult = 2):  # in paper, it seems for self attention layers they did feedforwards with twice channel width
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        ChanLayerNorm(dim),
        nn.Conv2d(dim, hidden_dim, 1, bias=False),
        nn.GELU(),
        ChanLayerNorm(hidden_dim),
        nn.Conv2d(hidden_dim, dim, 1, bias=False)
    )

class LinearAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dropout = 0.05,
        context_dim = None,
    ):
        super().__init__()

        self.heads = heads
        assert dim % heads == 0, "Embedding dimension must be 0 modulo number of heads."
        head_dim = dim // heads
        inner_dim = head_dim * heads
        self.norm = ChanLayerNorm(dim)

        self.scale = head_dim ** -0.5

        self.nonlin = nn.SiLU()

        self.to_q = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(dim, inner_dim, 1, bias=False),
            nn.Conv2d(inner_dim, inner_dim, 3, bias=False, padding = 1, groups = inner_dim)
        )

        self.to_k = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(dim, inner_dim, 1, bias = False),
            nn.Conv2d(inner_dim, inner_dim, 3, bias=False,
                      padding = 1, groups = inner_dim)
        )

        self.to_v = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(dim, inner_dim, 1, bias = False),
            nn.Conv2d(inner_dim, inner_dim, 3, bias=False,
                      padding = 1, groups = inner_dim)
        )

        self.to_context = nn.Linear(context_dim, inner_dim * 2, bias=False) if exists(context_dim) else None

        self.to_out = nn.Conv2d(inner_dim, dim, 1, bias = False)

    def forward(self, fmap, context = None):
        h, x, y = self.heads, *fmap.shape[-2:]

        fmap = self.norm(fmap)
        q, k, v = map(lambda fn: fn(fmap), (self.to_q, self.to_k, self.to_v))
        q, k, v = rearrange_many((q, k, v), 'b (h c) x y -> (b h) (x y) c', h = h)

        if exists(context):
            assert exists(self.to_context)
            ck, cv = self.to_context(context).chunk(2, dim = -1)
            ck, cv = rearrange_many((ck, cv), 'b n (h d) -> (b h) n d', h = h)
            k = torch.cat((k, ck), dim = -2)
            v = torch.cat((v, cv), dim = -2)

        q = q.softmax(dim = -1)
        k = k.softmax(dim = -2)

        q = q * self.scale

        context = einsum('b n d, b n e -> b d e', k, v)
        out = einsum('b n d, b d e -> b n e', q, context)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h = h, x = x, y = y)

        out = self.nonlin(out)
        return self.to_out(out)
