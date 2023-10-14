import functools
import numpy as np

import torch
import torch.nn as nn

from sgmse.util.registry import Registry


BackboneRegistry = Registry("Backbone")


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=16, complex_valued=False):
        super().__init__()
        self.complex_valued = complex_valued
        if not complex_valued:
            # If the output is real-valued, we concatenate sin+cos of the features to avoid ambiguities.
            # Therefore, in this case the effective embed_dim is cut in half. For the complex-valued case,
            # we use complex numbers which each represent sin+cos directly, so the ambiguity is avoided directly,
            # and this halving is not necessary.
            embed_dim = embed_dim // 2
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim) * scale, requires_grad=False)

    def forward(self, t):
        t_proj = t[:, None] * self.W[None, :] * 2*np.pi
        if self.complex_valued:
            return torch.exp(1j * t_proj)
        else:
            return torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)


class DiffusionStepEmbedding(nn.Module):
    """Diffusion-Step embedding as in DiffWave / Vaswani et al. 2017."""

    def __init__(self, embed_dim, complex_valued=False):
        super().__init__()
        self.complex_valued = complex_valued
        if not complex_valued:
            # If the output is real-valued, we concatenate sin+cos of the features to avoid ambiguities.
            # Therefore, in this case the effective embed_dim is cut in half. For the complex-valued case,
            # we use complex numbers which each represent sin+cos directly, so the ambiguity is avoided directly,
            # and this halving is not necessary.
            embed_dim = embed_dim // 2
        self.embed_dim = embed_dim

    def forward(self, t):
        fac = 10**(4*torch.arange(self.embed_dim, device=t.device) / (self.embed_dim-1))
        inner = t[:, None] * fac[None, :]
        if self.complex_valued:
            return torch.exp(1j * inner)
        else:
            return torch.cat([torch.sin(inner), torch.cos(inner)], dim=-1)


class ComplexLinear(nn.Module):
    """A potentially complex-valued linear layer. Reduces to a regular linear layer if `complex_valued=False`."""
    def __init__(self, input_dim, output_dim, complex_valued):
        super().__init__()
        self.complex_valued = complex_valued
        if self.complex_valued:
            self.re = nn.Linear(input_dim, output_dim)
            self.im = nn.Linear(input_dim, output_dim)
        else:
            self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        if self.complex_valued:
            return (self.re(x.real) - self.im(x.imag)) + 1j*(self.re(x.imag) + self.im(x.real))
        else:
            return self.lin(x)


class FeatureMapDense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""

    def __init__(self, input_dim, output_dim, complex_valued=False):
        super().__init__()
        self.complex_valued = complex_valued
        self.dense = ComplexLinear(input_dim, output_dim, complex_valued=complex_valued)

    def forward(self, x):
        return self.dense(x)[..., None, None]


def torch_complex_from_reim(re, im):
    return torch.view_as_complex(torch.stack([re, im], dim=-1))


class ArgsComplexMultiplicationWrapper(nn.Module):
    """Adapted from `asteroid`'s `complex_nn.py`, allowing args/kwargs to be passed through forward().

    Make a complex-valued module `F` from a real-valued module `f` by applying
    complex multiplication rules:

    F(a + i b) = f1(a) - f1(b) + i (f2(b) + f2(a))

    where `f1`, `f2` are instances of `f` that do *not* share weights.

    Args:
        module_cls (callable): A class or function that returns a Torch module/functional.
            Constructor of `f` in the formula above.  Called 2x with `*args`, `**kwargs`,
            to construct the real and imaginary component modules.
    """

    def __init__(self, module_cls, *args, **kwargs):
        super().__init__()
        self.re_module = module_cls(*args, **kwargs)
        self.im_module = module_cls(*args, **kwargs)

    def forward(self, x, *args, **kwargs):
        return torch_complex_from_reim(
            self.re_module(x.real, *args, **kwargs) - self.im_module(x.imag, *args, **kwargs),
            self.re_module(x.imag, *args, **kwargs) + self.im_module(x.real, *args, **kwargs),
        )


ComplexConv2d = functools.partial(ArgsComplexMultiplicationWrapper, nn.Conv2d)
ComplexConvTranspose2d = functools.partial(ArgsComplexMultiplicationWrapper, nn.ConvTranspose2d)


## ####################My
import math
from torch.optim.lr_scheduler import _LRScheduler
class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
def logs_model_option(my_args): #dict
    print("\n================ Added Option List ================")
    for args in my_args:
        print(args, my_args[args])
    print("===================================================")