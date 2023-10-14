# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Layers for defining NCSN++.
"""
from . import layers
from . import up_or_down_sampling
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import pdb

conv1x1 = layers.ddpm_conv1x1
conv3x3 = layers.ddpm_conv3x3
NIN = layers.NIN
default_init = layers.default_init

class MSA(nn.Module):
    def __init__(self, dim=192, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
      
      
class GaussianFourierProjection(nn.Module):
  """Gaussian Fourier embeddings for noise levels."""

  def __init__(self, embedding_size=256, scale=1.0):
    super().__init__()
    self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Combine(nn.Module):
  """Combine information from skip connections."""

  def __init__(self, dim1, dim2, method='cat'):
    super().__init__()
    self.Conv_0 = conv1x1(dim1, dim2)
    self.method = method

  def forward(self, x, y):
    h = self.Conv_0(x)
    if self.method == 'cat':
      return torch.cat([h, y], dim=1)
    elif self.method == 'sum':
      padding_size = y.shape[-1] - h.shape[-1] # zero padding 크기 계산
      if padding_size > 0:
        # print("패딩1", y.shape, h.shape)
        h = torch.cat([h, torch.zeros(h.shape[0], h.shape[1], h.shape[2], padding_size).to(h.device)], dim=-1)
      elif padding_size < 0:
        # print("패딩2", y.shape, h.shape)
        y = torch.cat([y, torch.zeros(y.shape[0], y.shape[1], y.shape[2], abs(padding_size)).to(y.device)], dim=-1)
    
      return h + y
    else:
      raise ValueError(f'Method {self.method} not recognized.')

################### Attention Blocks (AttnBlockpp, CrossAttnBlockpp) ######################
class AttnBlockpp(nn.Module):
  """Channel-wise self-attention block. Modified from DDPM."""

  def __init__(self, channels, skip_rescale=False, init_scale=0.):
    super().__init__()
    self.GroupNorm_0 = nn.GroupNorm(num_groups=min(channels // 4, 32), 
                                    num_channels=channels, 
                                    eps=1e-6)
    self.NIN_0 = NIN(channels, channels)
    self.NIN_1 = NIN(channels, channels)
    self.NIN_2 = NIN(channels, channels)
    self.NIN_3 = NIN(channels, channels, init_scale=init_scale)
    
    self.skip_rescale = skip_rescale

  def forward(self, x):
    B, C, H, W = x.shape
    h = self.GroupNorm_0(x)
    q = self.NIN_0(h)
    k = self.NIN_1(h)
    v = self.NIN_2(h)

    w = torch.einsum('bchw,bcij->bhwij', q, k) * (int(C) ** (-0.5))
    w = torch.reshape(w, (B, H, W, H * W))
    w = F.softmax(w, dim=-1)
    w = torch.reshape(w, (B, H, W, H, W))
    h = torch.einsum('bhwij,bcij->bchw', w, v)
    h = self.NIN_3(h)
    
    if not self.skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.)

class CrossAttnBlockpp(nn.Module):
  def __init__(self, channels, skip_rescale=False, init_scale=0., diag_value=1e-12):
    super().__init__()

    # 4채널 입력의 경우 group 설정
    groups_num = channels // 4
    if channels == 2:
      groups_num = 1
    
    # 여기 채널 2개 변경함
    self.GroupNorm_cros_attn_x = nn.GroupNorm(num_groups=min(groups_num, 32), 
                                              num_channels=channels, 
                                              eps=1e-6)
    self.GroupNorm_cros_attn_n = nn.GroupNorm(num_groups=min(groups_num, 32), 
                                              num_channels=channels, 
                                              eps=1e-6)
    # self.BatchNorm_cros_attn_x = nn.BatchNorm2d(channels, eps=1e-05)
    # self.BatchNorm_cros_attn_n = nn.BatchNorm2d(channels, eps=1e-05)
    
    self.NIN_x_q = NIN(channels, channels)
    self.NIN_n_k = NIN(channels, channels)
    self.NIN_x_v = NIN(channels, channels)
    self.NIN_n_v = NIN(channels, channels)
    
    # self.H_in = nn.Linear(256, 128)
    # self.H_out = nn.Linear(128, 256)
    # self.W_in = nn.Linear(256, 128)
    # self.W_out = nn.Linear(128, 256)
    # self.ReLU = nn.ReLU()
    
    ##########################################################
    ## Divide value out weight
    self.NIN_x_out = NIN(channels, channels, init_scale=init_scale)
    self.NIN_n_out = NIN(channels, channels, init_scale=init_scale)
    
    ## Sharing value out weight
    # self.NIN_out = NIN(channels, channels, init_scale=init_scale)
    ##########################################################
            
    self.skip_rescale = skip_rescale
    
    self.diag_value = diag_value
    

  def forward(self, x):
    
    # pdb.set_trace()
    ## Linear 통과하여 shape 256->128로 만듬
    # x = self.W_in(x)
    # x = self.ReLU(x)
    # x = x.permute(0, 1, 3, 2)
    # x = self.H_in(x)
    # x = self.ReLU(x)
    # x = x.permute(0, 1, 3, 2)

    B, o_C, H, W = x.shape
    C = int(o_C / 2)
    
    # concat된 입력 나눔
    n = x[:, C:o_C, :, :]
    x = x[:, :C, :, :]
    
    hx = self.GroupNorm_cros_attn_x(x)
    hn = self.GroupNorm_cros_attn_n(n)    
    # hx = self.BatchNorm_cros_attn_x(x)
    # hn = self.BatchNorm_cros_attn_n(n)
    
    q = self.NIN_x_q(hx)
    k = self.NIN_n_k(hn)
    x_v = self.NIN_x_v(hx)
    n_v = self.NIN_n_v(hn)
    
    w = torch.einsum('bchw,bcij->bhwij', q, k) * (int(C) ** (-0.5))
    w = torch.reshape(w, (B, H, W, H * W))
    w = F.softmax(w, dim=-1)
    w = torch.reshape(w, (B, H, W, H, W))
    
    x_v = torch.einsum('bhwij,bcij->bchw', w, x_v)
    n_v = torch.einsum('bhwij,bcij->bchw', w, n_v)
    
    x_v = self.NIN_x_out(x_v)
    n_v = self.NIN_n_out(n_v)

    # decorrelation을 위한 행렬 생성
    _, _, H, W = x_v.shape
    diag0 = torch.ones(H, W).fill_diagonal_(self.diag_value)\
        .unsqueeze(0).unsqueeze(0)
    diag_zero = torch.ones(H, W).fill_diagonal_(0)\
        .unsqueeze(0).unsqueeze(0)
    
    # 대각행렬 1e-12로 채움 (decorrelation)
    x_v = x_v * diag0.to(x_v.device)
    n_v = n_v * diag0.to(n_v.device)
    
    decor_mat = torch.cat((x_v, n_v), dim=1)
    D_loss = torch.sum(abs(decor_mat) * diag_zero.to(decor_mat.device))
    
    result = torch.cat(((x + x_v), (n + n_v)), dim=1)
    
    if not self.skip_rescale:
      result = result / np.sqrt(2.)
    
    return result, D_loss      
############################################################################################

############################## Channel Decorrelation Block ##############################
class CDblock(nn.Module):
    """
    Calculate the Channel Decorrelation between reference channel and the auxiliary channel.
    """    
    def __init__(self):
        super().__init__()
        self.EPS = 1e-8
        
    def l2norm(self, mat, keepdim=True):
        return torch.norm(mat, dim=-1, keepdim=keepdim)
         
    def forward(self, x):
        B, o_C, H, W = x.shape
        C = int(o_C / 2)
        
        # concat된 입력 나눔 # B, C, N, T
        ref = x[:, :C, :, :].reshape(B, C, -1) # B C H W -> B C(1) HW
        aux = x[:, C:o_C, :, :].reshape(B, C, -1) # B C H W -> B C(1) HW
            
        assert aux.shape == ref.shape
        assert aux.dim() == 3
    
        ref_zm = ref - torch.mean(ref, -1, keepdim=True)
        aux_zm = aux - torch.mean(aux, -1, keepdim=True)
        cos_dis = torch.sum(ref_zm*aux_zm, -1, keepdim=True) / (self.l2norm(ref_zm)*self.l2norm(aux_zm)+self.EPS) 
        ones_dis = torch.ones_like(cos_dis) 
        cda = F.softmax(torch.stack([ones_dis, cos_dis]), 0)[0]

        cd_ref = cda * ref
        cd_ref = cd_ref.reshape(B, C, H, W)
        aux = aux.reshape(B, C, H, W)
        result = torch.concat((cd_ref, aux), dim=1)
        return result
############################################################################################

class Upsample(nn.Module):
  def __init__(self, in_ch=None, out_ch=None, with_conv=False, fir=False,
               fir_kernel=(1, 3, 3, 1)):
    super().__init__()
    out_ch = out_ch if out_ch else in_ch
    if not fir:
      if with_conv:
        self.Conv_0 = conv3x3(in_ch, out_ch)
    else:
      if with_conv:
        self.Conv2d_0 = up_or_down_sampling.Conv2d(in_ch, out_ch,
                                                 kernel=3, up=True,
                                                 resample_kernel=fir_kernel,
                                                 use_bias=True,
                                                 kernel_init=default_init())
    self.fir = fir
    self.with_conv = with_conv
    self.fir_kernel = fir_kernel
    self.out_ch = out_ch

  def forward(self, x):
    B, C, H, W = x.shape
    if not self.fir:
      h = F.interpolate(x, (H * 2, W * 2), 'nearest')
      if self.with_conv:
        h = self.Conv_0(h)
    else:
      if not self.with_conv:
        h = up_or_down_sampling.upsample_2d(x, self.fir_kernel, factor=2)
      else:
        h = self.Conv2d_0(x)

    return h


class Downsample(nn.Module):
  def __init__(self, in_ch=None, out_ch=None, with_conv=False, fir=False,
               fir_kernel=(1, 3, 3, 1)):
    super().__init__()
    out_ch = out_ch if out_ch else in_ch
    if not fir:
      if with_conv:
        self.Conv_0 = conv3x3(in_ch, out_ch, stride=2, padding=0)
    else:
      if with_conv:
        self.Conv2d_0 = up_or_down_sampling.Conv2d(in_ch, out_ch,
                                                 kernel=3, down=True,
                                                 resample_kernel=fir_kernel,
                                                 use_bias=True,
                                                 kernel_init=default_init())
    self.fir = fir
    self.fir_kernel = fir_kernel
    self.with_conv = with_conv
    self.out_ch = out_ch

  def forward(self, x):
    B, C, H, W = x.shape
    if not self.fir:
      if self.with_conv:
        x = F.pad(x, (0, 1, 0, 1))
        x = self.Conv_0(x)
      else:
        x = F.avg_pool2d(x, 2, stride=2)
    else:
      if not self.with_conv:
        x = up_or_down_sampling.downsample_2d(x, self.fir_kernel, factor=2)
      else:
        x = self.Conv2d_0(x)

    return x


class ResnetBlockDDPMpp(nn.Module):
  """ResBlock adapted from DDPM."""

  def __init__(self, act, in_ch, out_ch=None, temb_dim=None, conv_shortcut=False,
               dropout=0.1, skip_rescale=False, init_scale=0.):
    super().__init__()
    out_ch = out_ch if out_ch else in_ch
    self.GroupNorm_0 = nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6)
    self.Conv_0 = conv3x3(in_ch, out_ch)
    if temb_dim is not None:
      self.Dense_0 = nn.Linear(temb_dim, out_ch)
      self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
      nn.init.zeros_(self.Dense_0.bias)
    self.GroupNorm_1 = nn.GroupNorm(num_groups=min(out_ch // 4, 32), num_channels=out_ch, eps=1e-6)
    self.Dropout_0 = nn.Dropout(dropout)
    self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
    if in_ch != out_ch:
      if conv_shortcut:
        self.Conv_2 = conv3x3(in_ch, out_ch)
      else:
        self.NIN_0 = NIN(in_ch, out_ch)

    self.skip_rescale = skip_rescale
    self.act = act
    self.out_ch = out_ch
    self.conv_shortcut = conv_shortcut

  def forward(self, x, temb=None):
    h = self.act(self.GroupNorm_0(x))
    h = self.Conv_0(h)
    if temb is not None:
      h += self.Dense_0(self.act(temb))[:, :, None, None]
    h = self.act(self.GroupNorm_1(h))
    h = self.Dropout_0(h)
    h = self.Conv_1(h)
    if x.shape[1] != self.out_ch:
      if self.conv_shortcut:
        x = self.Conv_2(x)
      else:
        x = self.NIN_0(x)
    if not self.skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.)


class ResnetBlockBigGANpp(nn.Module):
  def __init__(self, act, in_ch, out_ch=None, temb_dim=None, up=False, down=False,
               dropout=0.1, fir=False, fir_kernel=(1, 3, 3, 1),
               skip_rescale=True, init_scale=0.):
    super().__init__()
    
    # 4채널 입력의 경우 group 설정
    groups_num = in_ch // 4
    if in_ch == 2:
      groups_num = 1
      
    out_ch = out_ch if out_ch else in_ch
    self.GroupNorm_0 = nn.GroupNorm(num_groups=groups_num, num_channels=in_ch, eps=1e-6)
    self.up = up
    self.down = down
    self.fir = fir
    self.fir_kernel = fir_kernel

    self.Conv_0 = conv3x3(in_ch, out_ch)
    if temb_dim is not None:
      self.Dense_0 = nn.Linear(temb_dim, out_ch)
      self.Dense_0.weight.data = default_init()(self.Dense_0.weight.shape)
      nn.init.zeros_(self.Dense_0.bias)

    self.GroupNorm_1 = nn.GroupNorm(num_groups=min(out_ch // 4, 32), num_channels=out_ch, eps=1e-6)
    self.Dropout_0 = nn.Dropout(dropout)
    self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
    if in_ch != out_ch or up or down:
      self.Conv_2 = conv1x1(in_ch, out_ch)

    self.skip_rescale = skip_rescale
    self.act = act
    self.in_ch = in_ch
    self.out_ch = out_ch

  def forward(self, x, temb=None):
    h = self.act(self.GroupNorm_0(x))

    if self.up:
      if self.fir:
        h = up_or_down_sampling.upsample_2d(h, self.fir_kernel, factor=2)
        x = up_or_down_sampling.upsample_2d(x, self.fir_kernel, factor=2)
      else:
        h = up_or_down_sampling.naive_upsample_2d(h, factor=2)
        x = up_or_down_sampling.naive_upsample_2d(x, factor=2)
    elif self.down:
      if self.fir:
        h = up_or_down_sampling.downsample_2d(h, self.fir_kernel, factor=2)
        x = up_or_down_sampling.downsample_2d(x, self.fir_kernel, factor=2)
      else:
        h = up_or_down_sampling.naive_downsample_2d(h, factor=2)
        x = up_or_down_sampling.naive_downsample_2d(x, factor=2)

    h = self.Conv_0(h)
    # Add bias to each feature map conditioned on the time embedding
    if temb is not None:
      h += self.Dense_0(self.act(temb))[:, :, None, None]
    h = self.act(self.GroupNorm_1(h))
    h = self.Dropout_0(h)
    h = self.Conv_1(h)

    if self.in_ch != self.out_ch or self.up or self.down:
      x = self.Conv_2(x)

    if not self.skip_rescale:
      return x + h
    else:
      return (x + h) / np.sqrt(2.)
