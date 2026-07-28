import torch
from torch import Tensor
import math

class Distribution:
    def __call__(self, num_samples: int, device: torch.device):
        raise NotImplementedError()

class LogNormalDistribution(Distribution):
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    def __call__(self, num_samples: int, device: torch.device = torch.device("cpu")) -> Tensor:
        normal = self.mean + self.std * torch.randn((num_samples,), device=device)
        return normal.exp()

class UniformDistribution(Distribution):
    def __init__(self, vmin: float = 0.0, vmax: float = 1.0):
        super().__init__()
        self.vmin, self.vmax = vmin, vmax

    def __call__(self, num_samples: int, device: torch.device = torch.device("cpu")):
        vmax, vmin = self.vmax, self.vmin
        return (vmin - vmax) * torch.rand(num_samples, device=device) + vmax

class LogUniformDistribution(Distribution):

    # For plotting noise versus loss shown in Fig.5(a) in the paper

    def __init__(self, sigma_min: float = 0.001, sigma_max: float = 100):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, num_samples: int, device: torch.device = torch.device("cpu")):

        rnd_uniform = torch.rand(num_samples, device=device)

        return self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)

class LogitDistribution(Distribution):
    def __init__(self, logit_mean: float = 0.0, logit_std: float = 1.0, ln_scale=False, stratified=False):
        super().__init__()
        self.logit_std, self.logit_mean = logit_std, logit_mean
        self.ln_scale = ln_scale
        self.stratified = stratified

    def __call__(self, num_samples: int, device: torch.device = torch.device("cpu")):

        if self.ln_scale:
            if self.stratified:
                # stratified sampling of normals
                # first stratified sample from uniform
                quantiles = torch.linspace(0, 1, num_samples + 1).to(device)
                z = quantiles[:-1] + torch.rand(num_samples, device=device) / num_samples
                # now transform to normal
                z = torch.erfinv(2 * z - 1) * math.sqrt(2)
                z = z * self.logit_std + self.logit_mean
                t = torch.sigmoid(z)
            else:
                nt = torch.randn(num_samples, device=device) * self.logit_std + self.logit_mean
                t = torch.sigmoid(nt)
        else:
            # uniform [0, 1]
            t = torch.rand(num_samples, device=device)

        return t