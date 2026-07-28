from typing import Callable, Optional
import torch
import torch.nn as nn
from torch import Tensor

_UNSET = object()


class ReflowEulerSampler(nn.Module):
    """
    Reflow Euler sampler: Euler sampler with a fixed step size

    """

    def __init__(
        self,
        num_steps: int = 200,
        cond_scale: Optional[float] = 1.0,
        w_tech: float = 0.0,
        w_cc: float = 0.0,
        use_heun: bool = True,
        clip_output: Optional[float] = 1.0,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.cond_scale = None if cond_scale is None else float(cond_scale)
        self.w_tech = float(w_tech)
        self.w_cc = float(w_cc)
        self.use_heun = use_heun
        self.clip_output = clip_output

    def _clip_if_needed(self, x: Tensor) -> Tensor:
        if self.clip_output is None:
            return x
        clip_value = float(self.clip_output)
        return x.clamp(-clip_value, clip_value)

    def step(self, x: Tensor,
             fn: Callable,
             net: nn.Module,
             sigma: float,
             sigma_next: float,
             cond_scale=_UNSET,
             w_tech=_UNSET,
             w_cc=_UNSET,
             **kwargs) -> Tensor:

        """One step of RF Euler sampler"""

        cond_scale_value = self.cond_scale if cond_scale is _UNSET else cond_scale
        w_tech_value = self.w_tech if w_tech is _UNSET else float(w_tech)
        w_cc_value = self.w_cc if w_cc is _UNSET else float(w_cc)

        # Evaluate ∂x/∂sigma at sigma_hat
        vc =  fn(x, net=net,
                 sigma=sigma,
                 inference=True,
                 cond_scale=cond_scale_value,
                 w_tech=w_tech_value,
                 w_cc=w_cc_value,
                 **kwargs)

        # Take euler step from sigma_hat to sigma_next
        x_next = x + (sigma_next - sigma) * vc

        # Second order correction
        if sigma_next != 0 and self.use_heun:
            vc_next = fn(x_next, net=net,
                         sigma=sigma_next,
                         inference=True,
                         cond_scale=cond_scale_value,
                         w_tech=w_tech_value,
                         w_cc=w_cc_value,
                         **kwargs)
            x_next = x + 0.5 * (sigma_next - sigma) * (vc + vc_next)

        return x_next

    def forward(self, noise: Tensor,
                fn: Callable,
                net: nn.Module,
                sigmas: Tensor,
                **kwargs) -> Tensor:

        cond_scale = kwargs.pop("cond_scale", _UNSET)
        w_tech = kwargs.pop("w_tech", _UNSET)
        w_cc = kwargs.pop("w_cc", _UNSET)

        # pay attention to this step
        x = sigmas[0] * noise

        # Denoise to sample
        for i in range(len(sigmas) - 1):
            x = self.step(x, fn=fn, net=net,
                          sigma=sigmas[i],
                          sigma_next=sigmas[i+1],
                          cond_scale=cond_scale,
                          w_tech=w_tech,
                          w_cc=w_cc,
                          **kwargs)

        return self._clip_if_needed(x)

class DPM2MSANASampler(nn.Module):

    """
    An implementation based on crowsonkb and hallatore from:
    https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/8457

    a.k.a DPM-Solver++(2M) Karras.

    Deterministic sampler
    """

    def __init__(self, num_steps: int = 50, cond_scale: float=1.0, time_shift=1.0):
        super().__init__()
        self.num_steps = num_steps
        self.cond_scale = cond_scale
        self.time_shift = time_shift

    def step(self, x: Tensor,
             fn: Callable,
             net: nn.Module,
             sigma_last: float,
             sigma: float,
             sigma_next: float,
             old_denoised: Tensor,
             **kwargs):

        t_fn = lambda sigma: sigma.log().neg()

        h = t_fn(sigma_next) - t_fn(sigma)

        v_pred = fn(x, net=net,
                      sigma=sigma,
                      inference=True,
                      cond_scale=self.cond_scale,
                      **kwargs)

        # x_0 prediction
        denoised = (x - v_pred * sigma)

        if old_denoised is None or sigma_next == 0:
            x = (sigma_next / sigma) * x - (-h).expm1() * denoised
        else:
            h_last = t_fn(sigma) - t_fn(sigma_last)
            r = h_last / h

            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (sigma_next / sigma) * x - (-h).expm1() * denoised_d

        return x, denoised

    def forward(self, noise: Tensor,
                fn: Callable,  # denoising function
                net: nn.Module,
                sigmas: Tensor, **kwargs) -> Tensor:

        x = sigmas[0] * noise

        # SANA rescaling
        sigmas = sigmas * self.time_shift /(1 + (self.time_shift - 1) * sigmas)

        # Denoise to sample
        old_denoised = None
        for i in range(self.num_steps):

            x, denoised = self.step(x, fn=fn,
                                    net=net,
                                    sigma_last=sigmas[i-1], # it's not used in the first step
                                    sigma=sigmas[i],
                                    sigma_next=sigmas[i+1],
                                    old_denoised=old_denoised,
                                    **kwargs)
            old_denoised = denoised

        return x.clamp(-1.0, 1.0)

class DPMSampler(nn.Module):

    """
    guided sampling with small guidance scale
    singlestep DPM-Solver ("DPM-Solver-fast" in the paper) with `order = 3`

    Deterministic sampler

    In EDM framework, corresponding variables are:
    alpha = 1.0
    sigma = sigma
    lambd = log(alpha/sigma) = -log(sigma)

    single step: NFE <= num_steps // order + 1
    multistep: NFE = num_steps + 1

    """

    def __init__(self, cond_scale,
                 order=1,
                 num_steps=10,
                 multisteps=False):
        super().__init__()

        self.order = order
        self.num_steps = num_steps
        self.cond_scale = cond_scale
        self.multisteps = multisteps

    def lambd(self, sigma):
        return -sigma.log()

    def sigma(self, lambd):
        return lambd

    def inv_lambd(self, lambd):
        return lambd.neg().exp()

    def model_fn(self, x, fn, net, lambd, **kwargs):

        model_t = fn(x, net=net,
                    sigma=self.sigma(lambd),
                    inference=True,
                    cond_scale=self.cond_scale,
                    **kwargs)
        model_t = (x - model_t * self.sigma(lambd))

        return model_t

    def eps(self, eps_cache, key, x, lambd, fn,
            net, **kwargs):

        if key in eps_cache:
            return eps_cache[key], eps_cache

        eps = self.model_fn(x, fn, net, lambd, **kwargs)

        return eps, {key: eps, **eps_cache}

    def dpm_solver_1_step(self, x, lambd_cur, lambd_next, fn, net, eps_cache=None, **kwargs):

        h = self.lambd(lambd_next) - self.lambd(lambd_cur)
        eps_cache = {} if eps_cache is None else eps_cache
        eps, eps_cache = self.eps(eps_cache, 'eps', x, lambd_cur, fn, net, **kwargs)

        x_1 = self.sigma(lambd_next) / self.sigma(lambd_cur) * x - torch.expm1(-h) * eps

        return x_1, eps_cache

    def dpm_solver_2_step(self, x, lambd_cur, lambd_next, fn, net, r1=1 / 2, eps_cache=None, **kwargs):

        h = self.lambd(lambd_next) - self.lambd(lambd_cur)
        s1 = lambd_cur + r1 * h
        s1 = self.inv_lambd(s1)
        eps_cache = {} if eps_cache is None else eps_cache
        eps, eps_cache = self.eps(eps_cache, 'eps', x, lambd_cur, fn, net, **kwargs)

        u1 = self.sigma(s1) / self.sigma(lambd_cur) * x - torch.expm1(-r1 * h) * eps
        eps_r1, eps_cache = self.eps(eps_cache, 'eps_r1', u1, s1, fn, net)
        x_2 = self.sigma(lambd_next) / self.sigma(lambd_cur) * x - torch.expm1(-h) * eps - 1 / (2 * r1) * torch.expm1(-h) * (eps_r1 - eps)

        return x_2, eps_cache

    def dpm_solver_3_step(self, x, lambd_cur, lambd_next, fn, net, r1=1/3, r2=2/3, eps_cache=None, **kwargs):
        eps_cache = {} if eps_cache is None else eps_cache
        h = self.lambd(lambd_next) - self.lambd(lambd_cur)
        eps, eps_cache = self.eps(eps_cache, 'eps', x, lambd_cur, fn, net, **kwargs)
        s1 = lambd_cur + r1 * h
        s1 = self.inv_lambd(s1)
        s2 = lambd_cur + r2 * h
        s2 = self.inv_lambd(s2)

        u1 = self.sigma(s1) / self.sigma(lambd_cur) * x - (-r1 * h).expm1() * eps
        eps_r1, eps_cache = self.eps(eps_cache, 'eps_r1', u1, s1, fn, net)
        u2 = self.sigma(s2) / self.sigma(lambd_cur) * x - (-r2 * h).expm1() * eps + (r2 / r1) * ((-r2 * h).expm1() / (r2 * h) + 1) * (eps_r1 - eps)
        eps_r2, eps_cache = self.eps(eps_cache, 'eps_r2', u2, s2, fn, net)
        x_3 = self.sigma(lambd_next) / self.sigma(lambd_cur) * x - torch.expm1(-h) * eps + 1 / r2 * (torch.expm1(-h) / h + 1) * (eps_r2 - eps)

        return x_3, eps_cache

    def multistep_dpm_solver_1_step(self, x, lambd_prev, lambd_cur,
                                    fn, net, model_s=None, **kwargs):
        h = self.lambd(lambd_cur) - self.lambd(lambd_prev)

        phi_1 = torch.expm1(-h)
        if model_s is None:
            model_s = fn(x, net=net, inference=True,
                        sigma=self.sigma(lambd_prev),
                        cond_scale=self.cond_scale, **kwargs)
            model_s = (x - model_s * self.sigma(lambd_prev))
        x_t =  self.sigma(lambd_cur) / self.sigma(lambd_prev) * x - phi_1 * model_s

        return x_t

    def multistep_dpm_solver_2_step(self, x, model_prev_list, lambd_prev_list, lambd_cur):

        lambd_prev_1, lambd_prev_0 = lambd_prev_list[-2], lambd_prev_list[-1]
        model_prev_1, model_prev_0 = model_prev_list[-2], model_prev_list[-1]
        h_1 = self.lambd(lambd_prev_0) - self.lambd(lambd_prev_1)
        h = self.lambd(lambd_cur) - self.lambd(lambd_prev_0)

        r0 = h_1 / h
        D1_0 = (1. / r0) * (model_prev_0 - model_prev_1)

        phi_1 = torch.expm1(-h)
        x_t = (self.sigma(lambd_cur) / self.sigma(lambd_prev_0) * x - phi_1 * model_prev_0 - 0.5 * phi_1 * D1_0)
        return x_t

    def multistep_dpm_solver_3_step(self, x, model_prev_list, lambd_prev_list, lambd_cur):

        lambd_prev_2, lambd_prev_1, lambd_prev_0 = lambd_prev_list
        model_prev_2, model_prev_1, model_prev_0 = model_prev_list

        h_1 = self.lambd(lambd_prev_1) - self.lambd(lambd_prev_2)
        h_0 = self.lambd(lambd_prev_0) - self.lambd(lambd_prev_1)
        h = self.lambd(lambd_cur) - self.lambd(lambd_prev_0)
        r0, r1 = h_0 / h, h_1 / h

        D1_0 = (1. / r0) * (model_prev_0 - model_prev_1)
        D1_1 = (1. / r1) * (model_prev_1 - model_prev_2)
        D1 = D1_0 + (r0 / (r0 + r1)) * (D1_0 - D1_1)
        D2 = (1. / (r0 + r1)) * (D1_0 - D1_1)

        phi_1 = torch.expm1(-h)
        phi_2 = phi_1 / h + 1.
        phi_3 = phi_2 / h - 0.5
        x_t = self.sigma(lambd_cur) / self.sigma(lambd_prev_0) * x - phi_1 * model_prev_0 + phi_2 * D1 - phi_3 * D2

        return x_t

    def forward(self, noise: Tensor,
                fn: Callable,  # denoising function
                net: nn.Module,
                sigmas: Tensor,
                **kwargs):

        x = sigmas[0] * noise

        lambds = sigmas

        if self.multisteps:
            assert self.num_steps >= self.order

            # Init the initial values.
            step = 0
            lambd = lambds[step]
            model_t = self.model_fn(x, fn, net, lambd, **kwargs)
            model_prev_list = [model_t]
            lambd_prev_list = [lambd]

            for step in range(1, self.order):

                lambd_prev, lambd_cur = lambd_prev_list[-1], lambds[step]

                if step == 1:
                    x = self.multistep_dpm_solver_1_step(x, lambd_prev, lambd_cur, fn, net,
                                                         model_s=model_prev_list[-1], **kwargs)
                elif step == 2:
                    x = self.multistep_dpm_solver_2_step(x, model_prev_list, lambd_prev_list, lambd_cur)
                elif step == 3:
                    x = self.multistep_dpm_solver_3_step(x, model_prev_list, lambd_prev_list, lambd_cur)

                lambd_prev_list.append(lambd_cur)
                model_t = self.model_fn(x, fn, net, lambd_cur, **kwargs)
                model_prev_list.append(model_t)

            # Compute the remaining values by `order`-th order multistep DPM-Solver.
            for step in range(self.order, self.num_steps + 1):

                lambd_prev, lambd_cur = lambd_prev_list[-1], lambds[step]

                step_order = min(self.order, self.num_steps + 1 - step) # We only use lower order for steps < 10

                if step_order == 1:
                    x = self.multistep_dpm_solver_1_step(x, lambd_prev, lambd_cur, fn, net,
                                                         model_s=model_prev_list[-1], **kwargs)
                elif step_order == 2:
                    x = self.multistep_dpm_solver_2_step(x, model_prev_list, lambd_prev_list, lambd_cur)
                elif step_order == 3:
                    x = self.multistep_dpm_solver_3_step(x, model_prev_list, lambd_prev_list, lambd_cur)

                for i in range(self.order - 1):
                    lambd_prev_list[i] = lambd_prev_list[i + 1]
                    model_prev_list[i] = model_prev_list[i + 1]
                lambd_prev_list[-1] = lambd_cur
                # We do not need to evaluate the final model value.
                if step < self.num_steps:
                    model_t = self.model_fn(x, fn, net, lambd_cur, **kwargs)
                    model_prev_list[-1] = model_t

        else:
            if self.order == 3:
                K = self.num_steps // 3 + 1
                if self.num_steps % 3 == 0:
                    orders = [3,] * (K - 2) + [2, 1]
                else:
                    orders = [3,] * (K - 1) + [self.num_steps % 3]

            elif self.order == 2:
                if self.num_steps % 2 == 0:
                    K = self.num_steps // 2
                    orders = [2,] * K
                else:
                    K = self.num_steps // 2 + 1
                    orders = [2,] * (K - 1) + [1]
            elif self.order == 1:
                K = self.num_steps
                orders = [1,] * self.num_steps
            else:
                raise ValueError("'order' must be '1' or '2' or '3'.")

            for i in range(len(orders)):
                eps_cache = {}
                lambd_cur, lambd_next = lambds[i], lambds[i + 1]
                _, eps_cache = self.eps(eps_cache, 'eps', x, lambd_cur, fn, net, **kwargs)

                if orders[i] == 1:
                    x, eps_cache = self.dpm_solver_1_step(x, lambd_cur, lambd_next, fn, net, eps_cache=eps_cache)
                elif orders[i] == 2:
                    x, eps_cache = self.dpm_solver_2_step(x, lambd_cur, lambd_next, fn, net, eps_cache=eps_cache)
                else:
                    x, eps_cache = self.dpm_solver_3_step(x, lambd_cur, lambd_next, fn, net, eps_cache=eps_cache)

        return x.clamp(-1.0, 1.0)

class UniPCSampler(nn.Module):

    """
    Uni-PC sampler, built upon multistep DPM
    we use x0 prediction method since it achieves better performance with EDM scheduler.

    NFE = num_steps

    reference: https://github.com/thu-ml/DPM-Solver-v3/blob/main/codebases/edm/samplers/uni_pc.py
    https://github.com/wl-zhao/UniPC
    """

    def __init__(self, num_steps: int = 20,
                 order: int = 2,
                 cond_scale: float = 1.0):

        super().__init__()
        self.num_steps = num_steps
        self.order = order
        self.cond_scale = cond_scale

    def model_fn(self, x, lambd, fn, net, **kwargs):

        model_t = fn(x, net=net,
                    sigma=self.lambda_to_sigma(lambd),
                    inference=True,
                    cond_scale=self.cond_scale, **kwargs)

        model_t = (x - model_t * self.lambda_to_sigma(lambd))
        return model_t

    def sigma_to_lambd(self, sigma):
        return -sigma.log()

    def lambda_to_sigma(self, lambd):
        return lambd

    def multistep_uni_pc_update(self, x, model_prev_list,
                                lambd_prev_list, lambd_cur, order,
                                fn, net, x_t=None, use_corrector=True,
                                variant='bh2', **kwargs):

        if len(lambd_cur.shape) == 0:
            lambd_cur = lambd_cur.view(-1)

        assert order <= len(model_prev_list)

        # first compute rks
        lambd_prev_0 = lambd_prev_list[-1]
        model_prev_0 = model_prev_list[-1]
        h = self.sigma_to_lambd(lambd_cur) - self.sigma_to_lambd(lambd_prev_0)

        rks = []
        D1s = []
        for i in range(1, order):

            lambd_prev_i = lambd_prev_list[-(i + 1)]
            rk = (self.sigma_to_lambd(lambd_prev_i)  - self.sigma_to_lambd(lambd_prev_0)) / h
            rks.append(rk)

            model_prev_i = model_prev_list[-(i + 1)]
            D1s.append((model_prev_i - model_prev_0) / rk)

        rks.append(1.)
        rks = torch.tensor(rks, device=x.device)

        R = []
        b = []

        hh = -h
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1

        factorial_i = 1

        if variant == 'bh1':
            B_h = hh
        elif variant == 'bh2':
            B_h = torch.expm1(hh)
        else:
            raise NotImplementedError()

        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= (i + 1)
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = torch.stack(R)
        b = torch.cat(b)

        # now predictor
        use_predictor = len(D1s) > 0 and x_t is None
        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1) # (B, K)
            if x_t is None:
                # for order 2, we use a simplified version
                if order == 2:
                    rhos_p = torch.tensor([0.5], device=b.device)
                else:
                    rhos_p = torch.linalg.solve(R[:-1, :-1], b[:-1])
        else:
            D1s = None

        if use_corrector:
            # for order 1, we use a simplified version
            if order == 1:
                rhos_c = torch.tensor([0.5], device=b.device)
            else:
                rhos_c = torch.linalg.solve(R, b)

        model_t = None
        x_t_ = self.lambda_to_sigma(lambd_cur) / self.lambda_to_sigma(lambd_prev_0) * x - h_phi_1 * model_prev_0

        if x_t is None:
            if use_predictor:
                pred_res = torch.einsum('k,bkchw->bchw', rhos_p, D1s)
            else:
                pred_res = 0
            x_t = x_t_ - B_h * pred_res

        if use_corrector:
            model_t = self.model_fn(x_t, lambd_cur[0], fn, net, **kwargs)

            if D1s is not None:
                corr_res = torch.einsum('k,bkchw->bchw', rhos_c[:-1], D1s)
            else:
                corr_res = 0
            D1_t = (model_t - model_prev_0)
            x_t = x_t_ - B_h * (corr_res + rhos_c[-1] * D1_t)

        return x_t, model_t

    @torch.no_grad()
    def forward(self, noise: Tensor,
                fn: Callable,  # denoising function
                net: nn.Module,
                sigmas: Tensor,
                **kwargs) -> Tensor:

        assert self.num_steps >= self.order
        x = sigmas[0] * noise

        lambd_start = sigmas[0]
        lambd_end = sigmas[-1]
        lambds = torch.linspace(lambd_start, lambd_end, self.num_steps + 1, device=x.device)

        # Init the initial values.
        step = 0
        lambd = lambds[step]
        model_t = self.model_fn(x, lambd, fn, net, **kwargs)

        model_prev_list = [model_t]
        lambd_prev_list = [lambd]

        for step in range(1, self.order):
            lambd_cur = lambds[step]
            x, model_x = self.multistep_uni_pc_update(x, model_prev_list,
                                                      lambd_prev_list, lambd_cur, step, fn,
                                                      net, use_corrector=True)
            if model_x is None:
                model_t = self.model_fn(x, lambd_cur, fn, net, **kwargs)

            lambd_prev_list.append(lambd_cur)
            model_prev_list.append(model_x)

        # Compute the remaining values by `order`-th order multistep DPM-Solver.
        for step in range(self.order, self.num_steps + 1):

            lambd_cur = lambds[step]

            step_order = min(self.order, self.num_steps + 1 - step)

            if step == self.num_steps:
                use_corrector = False
            else:
                use_corrector = True

            x, model_x = self.multistep_uni_pc_update(x, model_prev_list,
                                                      lambd_prev_list,
                                                      lambd_cur,
                                                      step_order, fn, net,
                                                      use_corrector=use_corrector)

            for i in range(self.order - 1):
                lambd_prev_list[i] = lambd_prev_list[i + 1]
                model_prev_list[i] = model_prev_list[i + 1]
            lambd_prev_list[-1] = lambd_cur
            # We do not need to evaluate the final model value.
            if step < self.num_steps:
                if model_x is None:
                    model_x = self.model_fn(x, lambd_cur, fn, net, **kwargs)

                model_prev_list[-1] = model_x

        return x.clamp(-1.0, 1.0)
