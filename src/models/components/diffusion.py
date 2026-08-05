""" Rectified Flow training objective for VIOLET. """

from typing import Callable, Optional
import torch
import torch.nn as nn
from torch import Tensor
from .utils import extend_dim, to_batch


def _has_any_condition(kwargs: dict, *keys: str) -> bool:
    return any((key in kwargs) and (kwargs[key] is not None) for key in keys)


def _apply_cfg(
    predict_fn: Callable[..., Tensor],
    *,
    cond_scale: Optional[float],
    w_tech: Optional[float],
    w_cc: Optional[float],
    kwargs: dict,
) -> Tensor:
    """
    Apply either:
      1) MIDI-anchored *compositional* CFG (when cond_scale is None), or
      2) Classic CFG between FULL and MIDI-only (when cond_scale is not None).

    Compositional guidance (cond_scale is None):
      eps_M
        + w_tech * (eps_MT - eps_M)
        + w_cc   * (eps_FULL - eps_MT)
    where:
      - M  = MIDI only
      - MT = MIDI + Technique
      - FULL = MIDI + Technique + CC

    Classic CFG (cond_scale is not None):
      eps = eps_M + cond_scale * (eps_FULL - eps_M)
    where eps_M is MIDI-only and eps_FULL is MIDI+Technique+CC.
    """
    w_tech = 0.0 if w_tech is None else float(w_tech)
    w_cc = 0.0 if w_cc is None else float(w_cc)

    has_midi = _has_any_condition(kwargs, "midi_roll")
    has_tech = _has_any_condition(kwargs, "tech_roll")
    has_cc = _has_any_condition(kwargs, "cc_tokens")

    # ------------------------------------------------------------------
    # Branch 1: Classic CFG (FULL vs MIDI-only), used when cond_scale
    #           is explicitly set in the config (not null).
    #           Here "unconditional" = MIDI-only.
    # ------------------------------------------------------------------
    if cond_scale is not None:
        cond_scale_value = float(cond_scale)

        # If we don't have any extra modalities beyond MIDI, there is
        # nothing left to guide compositionally or classically, so just
        # use the MIDI-only prediction directly.
        if not (has_tech or has_cc):
            return predict_fn(
                cond_drop_prob=0.0,
                midi_cond_drop_prob=0.0,
            )

        if not has_midi:
            raise ValueError("Classic MIDI-anchored CFG requires MIDI conditioning.")

        full_kwargs = dict(cond_drop_prob=0.0)
        midi_kwargs = dict(cond_drop_prob=0.0)

        # MIDI is always kept in both branches.
        full_kwargs["midi_cond_drop_prob"] = 0.0
        midi_kwargs["midi_cond_drop_prob"] = 0.0

        # Technique / CC present: FULL keeps them, MIDI-only drops them.
        if has_tech:
            full_kwargs["tech_cond_drop_prob"] = 0.0
            midi_kwargs["tech_cond_drop_prob"] = 1.0
        if has_cc:
            full_kwargs["cc_cond_drop_prob"] = 0.0
            midi_kwargs["cc_cond_drop_prob"] = 1.0

        pred_m = predict_fn(**midi_kwargs)
        pred_full = predict_fn(**full_kwargs)

        if cond_scale_value == 1.0:
            return pred_m + (pred_full - pred_m)
        return pred_m + (pred_full - pred_m) * cond_scale_value

    # ------------------------------------------------------------------
    # Branch 2: Compositional CFG (cond_scale is None).
    #           w_tech / w_cc can be zero; in that case we simply use
    #           the MIDI-only prediction eps_M.
    # ------------------------------------------------------------------
    if not has_midi:
        raise ValueError("Compositional CFG requires MIDI conditioning.")

    base_kwargs = dict(
        cond_drop_prob=0.0,
        midi_cond_drop_prob=0.0,
    )

    # MIDI-only baseline: drop technique / CC if they exist.
    tech_drop_all = 1.0 if has_tech else 1.0
    cc_drop_all = 1.0 if has_cc else 1.0

    pred_m = predict_fn(
        **base_kwargs,
        tech_cond_drop_prob=tech_drop_all,
        cc_cond_drop_prob=cc_drop_all,
    )
    pred = pred_m
    pred_mt = pred_m

    # Add technique component if requested and available.
    if w_tech != 0.0 and has_tech:
        pred_mt = predict_fn(
            **base_kwargs,
            tech_cond_drop_prob=0.0,
            cc_cond_drop_prob=cc_drop_all,
        )
        pred = pred + w_tech * (pred_mt - pred_m)

    # Add CC as a technique-aware residual on top of MIDI+Technique,
    # so CC scaling refines expression without fighting the technique branch.
    if w_cc != 0.0 and has_cc:
        pred_full = predict_fn(
            **base_kwargs,
            tech_cond_drop_prob=0.0 if has_tech else tech_drop_all,
            cc_cond_drop_prob=0.0,
        )
        pred = pred + w_cc * (pred_full - pred_mt)

    return pred


class ReFlow(nn.Module):
    # Rectified flow training
    # Reference:
    #   https://github.com/cloneofsimo/minRF/blob/main/advanced/main_t2i.py

    def __init__(
        self,
        for_edm: bool = False,
    ):
        super().__init__()
        self.for_edm = for_edm

    def sigma_to_t(self, t):
        return t / (t + 1)

    def v_to_x0(self, x_noisy: Tensor, v_pred: Tensor, sigmas: Tensor) -> Tensor:
        return x_noisy - v_pred * sigmas

    def v_to_eps(self, x_noisy: Tensor, v_pred: Tensor, sigmas: Tensor) -> Tensor:
        return x_noisy + v_pred * (1 - sigmas)

    def denoise_fn(self, x_noisy: Tensor,
        net: nn.Module = None,
        inference: bool = False,
        cond_scale: float = 1.0,
        w_tech: Optional[float] = None,
        w_cc: Optional[float] = None,
        sigmas: Optional[Tensor] = None,
        sigma: Optional[float] = None,
        **kwargs) -> Tensor:
        # denoise means an EDM wrapper for ReFlow sampling when for_edm is True

        batch_size, device = x_noisy.shape[0], x_noisy.device
        sigmas = to_batch(x=sigma, xs=sigmas, batch_size=batch_size, device=device)

        if self.for_edm:
            sigmas = self.sigma_to_t(sigmas)
            x_noisy = x_noisy * (1 - sigmas)

        # cfg interpolation during inference, skip during training
        if inference:
            def _predict(**guidance_kwargs) -> Tensor:
                return net(x_noisy, sigmas, **guidance_kwargs, **kwargs)

            x_pred = _apply_cfg(
                _predict,
                cond_scale=cond_scale,
                w_tech=w_tech,
                w_cc=w_cc,
                kwargs=kwargs,
            )
        else:
            x_pred = net(x_noisy, sigmas, **kwargs)

        if self.for_edm:  # output x0 prediction
            x_pred = self.v_to_x0(x_noisy, x_pred, sigmas)

        return x_pred

    def forward(self, x: Tensor,
                net: nn.Module,
                sigmas: Tensor,
                inference: bool = False,
                cond_scale: float = 1.0,
                **kwargs):

        # EDM wrapper for ReFlow training
        t = sigmas
        t_padded = extend_dim(sigmas, dim=x.ndim)
        z1 = torch.randn_like(x)
        zt = (1 - t_padded) * x + t_padded * z1

        # make t, zt into same dtype as x
        zt, t = zt.to(x.dtype), t.to(x.dtype)

        vtheta = self.denoise_fn(zt, net, sigmas=t,
                                 inference=inference,
                                 cond_scale=cond_scale,
                                 **kwargs)

        losses = ((z1 - x - vtheta) ** 2).mean(dim=list(range(1, len(x.shape))))
        return losses
