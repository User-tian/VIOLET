from __future__ import annotations

from typing import List, Optional, Union

import torch
from dacvae import DACVAE


class FTDACVAE(DACVAE):
    """DACVAE with optional posterior mean encoding."""

    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 4, 8, 8],
        latent_dim: Optional[int] = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 4, 2],
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, List[int]] = 8,
        quantizer_dropout: bool = False,
        sample_rate: int = 44100,
        posterior_mode: str = "sample",
    ):
        super().__init__(
            encoder_dim=encoder_dim,
            encoder_rates=encoder_rates,
            latent_dim=latent_dim,
            decoder_dim=decoder_dim,
            decoder_rates=decoder_rates,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=quantizer_dropout,
            sample_rate=sample_rate,
        )
        self.posterior_mode = posterior_mode

    @classmethod
    def load(cls, path: str, posterior_mode: str = "sample") -> "FTDACVAE":
        model = super().load(path)
        if not isinstance(model, cls):
            model.__class__ = cls
        model.posterior_mode = posterior_mode
        return model

    @classmethod
    def load_with_finetuned_weights(
        cls,
        base_ckpt: str,
        finetuned_ckpt: Optional[str] = None,
        use_finetuned: bool = False,
        posterior_mode: str = "sample",
    ) -> "FTDACVAE":
        model = cls.load(base_ckpt, posterior_mode=posterior_mode)
        if use_finetuned and finetuned_ckpt:
            ft_data = torch.load(finetuned_ckpt, map_location="cpu")
            if isinstance(ft_data, dict) and "state_dict" in ft_data:
                state_dict = ft_data["state_dict"]
            else:
                state_dict = ft_data
            model.load_state_dict(state_dict)
        return model

    def encode(
        self,
        audio_data: torch.Tensor,
        n_quantizers: Optional[int] = None,
        return_tuple: bool = False,
    ):
        z = self.encoder(self._pad(audio_data))
        mean, scale = self.quantizer.in_proj(z).chunk(2, dim=1)

        mode = self.posterior_mode
        if mode == "mean":
            stdev = torch.nn.functional.softplus(scale) + 1e-4
            var = stdev * stdev
            logvar = torch.log(var)
            kl = (mean * mean + var - logvar - 1).sum(1).mean()
            z_q = mean
        elif mode == "sample":
            z_q, kl = self.quantizer._vae_sample(mean, scale)
        else:
            raise ValueError(f"Unsupported posterior_mode: {mode}")

        if return_tuple:
            return z_q, torch.zeros_like(z_q), z_q, kl, self.quantizer.dummy_codebook_loss

        return z_q
