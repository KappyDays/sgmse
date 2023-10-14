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

import pdb

from .ncsnpp_utils import layers, layerspp, normalization
import torch.nn as nn
import functools
import torch
import numpy as np

from .shared import BackboneRegistry, logs_model_option

ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp
Combine = layerspp.Combine
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init
CDblock = layerspp.CDblock

@BackboneRegistry.register("ncsnpp_flcattn")
class NCSNpp_FlCAttn(nn.Module):
    """NCSN++ model, adapted from https://github.com/yang-song/score_sde repository"""

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--ch_mult",type=int, nargs='+', default=[1,1,2,2,2,2,2])
        parser.add_argument("--num_res_blocks", type=int, default=2)
        parser.add_argument("--attn_resolutions", type=int, nargs='+', default=[16])
        parser.add_argument("--no-centered", dest="centered", action="store_false", help="The data is not centered [-1, 1]")
        parser.add_argument("--centered", dest="centered", action="store_true", help="The data is centered [-1, 1]")
        parser.set_defaults(centered=True)
        
        parser.add_argument("--prog_attn_resolutions", type=int, nargs='+', default=[0])
        parser.add_argument("--front_cross_attn", type=str, default="False")
        parser.add_argument("--bottleneck_cd_block", type=str, default="False")
        parser.add_argument("--cd_block_resolutions", type=int, nargs='+', default=[0])
        parser.add_argument("--nf", type=int, default=128)
        parser.add_argument("--image_size", type=int, default=256)
        return parser

    def __init__(self,
        scale_by_sigma = True,
        nonlinearity = 'swish',
        nf = 128, #128,
        ch_mult = (1, 1, 2, 2, 2, 2, 2),
        num_res_blocks = 2,
        attn_resolutions = (16,),
        resamp_with_conv = True,
        conditional = True,
        fir = True,
        fir_kernel = [1, 3, 3, 1],
        skip_rescale = True,
        resblock_type = 'biggan',
        progressive = 'output_skip',
        progressive_input = 'input_skip',
        progressive_combine = 'sum',
        init_scale = 0.,
        fourier_scale = 16,
        image_size = 256,
        embedding_type = 'fourier',
        dropout = .0,
        centered = True,
        
        prog_attn_resolutions = (0),
        front_cross_attn = 'False',
        bottleneck_cd_block = 'False',
        cd_block_resolutions = (0),
        **unused_kwargs
    ):
        super().__init__()
        self.act = act = get_act(nonlinearity)

        self.nf = nf = nf
        ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions = attn_resolutions
        dropout = dropout
        resamp_with_conv = resamp_with_conv
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = all_resolutions = [image_size // (2 ** i) for i in range(num_resolutions)]
                
        self.conditional = conditional = conditional # noise-conditional
        self.centered = centered
        self.scale_by_sigma = scale_by_sigma
        
        ############################# My Args
        self.my_args = {}
        self.my_args['prog_attn_resolutions'] = self.prog_attn_resolutions = prog_attn_resolutions
        self.my_args['front_cross_attn'] = self.front_cross_attn = front_cross_attn
        self.my_args['bottleneck_cd_block'] = self.bottleneck_cd_block = bottleneck_cd_block
        self.my_args['cd_block_resolutions'] = self.cd_block_resolutions = cd_block_resolutions
        self.my_args['image_size'] = self.image_size = image_size
        self.DeCAttn_loss = False
        self.ProgDeCAttn_loss = False
        ##############################
        
        fir = fir
        fir_kernel = fir_kernel
        self.skip_rescale = skip_rescale = skip_rescale
        self.resblock_type = resblock_type = resblock_type.lower()
        self.progressive = progressive = progressive.lower()
        self.progressive_input = progressive_input = progressive_input.lower()
        self.embedding_type = embedding_type = embedding_type.lower()
        init_scale = init_scale
        assert progressive in ['none', 'output_skip', 'residual']
        assert progressive_input in ['none', 'input_skip', 'residual']
        assert embedding_type in ['fourier', 'positional']
        combine_method = progressive_combine.lower()
        combiner = functools.partial(Combine, method=combine_method)

        num_channels = 4  # x.real, x.imag, y.real, y.imag
        self.output_layer = nn.Conv2d(num_channels, 2, 1)

        modules = []
        # timestep/noise_level embedding
        if embedding_type == 'fourier':
            # Gaussian Fourier features embeddings.
            modules.append(layerspp.GaussianFourierProjection(
                embedding_size=nf, scale=fourier_scale
            ))
            embed_dim = 2 * nf
        elif embedding_type == 'positional':
            embed_dim = nf
        else:
            raise ValueError(f'embedding type {embedding_type} unknown.')

        if conditional:
            modules.append(nn.Linear(embed_dim, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)
            modules.append(nn.Linear(nf * 4, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)

        AttnBlock = functools.partial(layerspp.AttnBlockpp,
                                      init_scale=init_scale, skip_rescale=skip_rescale)
        CrossAttnBlock = functools.partial(layerspp.CrossAttnBlockpp,
                                      init_scale=init_scale, skip_rescale=skip_rescale)
        Upsample = functools.partial(layerspp.Upsample,
            with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

        if progressive == 'output_skip':
            self.pyramid_upsample = layerspp.Upsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
        elif progressive == 'residual':
            pyramid_upsample = functools.partial(layerspp.Upsample, fir=fir,
                fir_kernel=fir_kernel, with_conv=True)

        Downsample = functools.partial(layerspp.Downsample, with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

        if progressive_input == 'input_skip':
            self.pyramid_downsample = layerspp.Downsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
        elif progressive_input == 'residual':
            pyramid_downsample = functools.partial(layerspp.Downsample,
                fir=fir, fir_kernel=fir_kernel, with_conv=True)

        if resblock_type == 'ddpm':
            ResnetBlock = functools.partial(ResnetBlockDDPM, act=act,
                dropout=dropout, init_scale=init_scale,
                skip_rescale=skip_rescale, temb_dim=nf * 4)

        elif resblock_type == 'biggan':
            ResnetBlock = functools.partial(ResnetBlockBigGAN, act=act,
                dropout=dropout, fir=fir, fir_kernel=fir_kernel,
                init_scale=init_scale, skip_rescale=skip_rescale, temb_dim=nf * 4)

        else:
            raise ValueError(f'resblock type {resblock_type} unrecognized.')
        
        
        # modules.append(AttnBlock(channels=2, loss=True))
        
        # Downsampling block
        channels = num_channels
        if progressive_input != 'none':
            input_pyramid_ch = channels

        ################## unet의 cross attention을 사용할 경우,
        # 입력(4 x 256 x 256)에 Cross Attention Block 추가
        if self.front_cross_attn == 'True':
            # print("UNet Cross Attn 추가요")
            modules.append(CrossAttnBlock(channels=num_channels//2, loss=True))
        ##########################################################
        
        # 채널 수 늘리는 부분
        modules.append(conv3x3(channels, nf))
        hs_c = [nf]

        in_ch = nf
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch

                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                    # modules.append(CrossAttnBlock(channels=in_ch//2))
                hs_c.append(in_ch)
            ############### CD block 추가 ############### 
            if all_resolutions[i_level] in cd_block_resolutions:
                print("Down path CD block 추가: ", all_resolutions[i_level])
                modules.append(CDblock())
            #############################################
            
            # Progressive blocks
            if i_level != num_resolutions - 1:
                if all_resolutions[i_level] in prog_attn_resolutions:
                    print("prog_attn_resol 추가요: ", all_resolutions[i_level])
                    modules.append(CrossAttnBlock(channels=num_channels//2, loss=True))
                # if all_resolutions[i_level] in [4]:
                #     print("CD block 추가염")
                #     modules.append(CDblock())
                if resblock_type == 'ddpm':
                    modules.append(Downsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(down=True, in_ch=in_ch))

                if progressive_input == 'input_skip':
                    modules.append(combiner(dim1=input_pyramid_ch, dim2=in_ch))
                    if combine_method == 'cat':
                        in_ch *= 2

                elif progressive_input == 'residual':
                    modules.append(pyramid_downsample(in_ch=input_pyramid_ch, out_ch=in_ch))
                    input_pyramid_ch = in_ch

                hs_c.append(in_ch)
            # elif i_level == num_resolutions - 1: # progressive 마지막 layer
            #     modules.append(CDblock())

        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))
        ## CD_block 사용할 경우만 작동
        if self.bottleneck_cd_block == "True":
            modules.append(CDblock())

        pyramid_ch = 0
        # Upsampling block
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):  # +1 blocks in upsampling because of skip connection from combiner (after downsampling)
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
                in_ch = out_ch

            if all_resolutions[i_level] in attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))
            
            ################# Up path CD block 추가 #################
            if all_resolutions[i_level] in cd_block_resolutions:
                print("Up path CD block 추가: ", all_resolutions[i_level])
                modules.append(CDblock())
            ###################################################
            
            if progressive != 'none':
                if i_level == num_resolutions - 1:
                    if progressive == 'output_skip':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                            num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, channels, init_scale=init_scale))
                        pyramid_ch = channels
                    elif progressive == 'residual':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, in_ch, bias=True))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f'{progressive} is not a valid name.')
                else:
                    if progressive == 'output_skip':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                            num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, channels, bias=True, init_scale=init_scale))
                        pyramid_ch = channels
                    elif progressive == 'residual':
                        modules.append(pyramid_upsample(in_ch=pyramid_ch, out_ch=in_ch))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f'{progressive} is not a valid name')
                    
            if all_resolutions[i_level] in prog_attn_resolutions:
                print("Prog_Attn_Resolutions 추가욤: ", all_resolutions[i_level])
                modules.append(CrossAttnBlock(channels=num_channels//2, loss=True))    
            if i_level != 0:
                if resblock_type == 'ddpm':
                    modules.append(Upsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(in_ch=in_ch, up=True))

        assert not hs_c

        if progressive != 'output_skip':
            modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                                                    num_channels=in_ch, eps=1e-6))
            modules.append(conv3x3(in_ch, channels, init_scale=init_scale))
        # 여기까지 NCSNpp(Noise Conditional Score Network++) init
        self.all_modules = nn.ModuleList(modules)
    # def cross_attn_layer(self, cross_attn, h_s, h_n):
    #     CrossAttn = cross_attn
    #     cross_attn_in = torch.cat((h_s, h_n), dim=1)
    #     h_s, h_n = CrossAttn(cross_attn_in)
    #     return h_s, h_n
        self.log_count = 0
    def forward(self, x, time_cond):
        if self.log_count == 0:
            self.my_args['DeCAttn_loss'] = self.DeCAttn_loss
            self.my_args['ProgDeCAttn_loss'] = self.ProgDeCAttn_loss      
            logs_model_option(self.my_args)
            self.log_count += 1
            
        # timestep/noise_level embedding; only for continuous training
        modules_s = self.all_modules
        m_idx = 0
        D_loss = 0
        
        # Convert real and imaginary parts of (x,y) into four channel dimensions
        x = torch.cat((x[:,[0],:,:].real, x[:,[0],:,:].imag,
                      x[:,[1],:,:].real, x[:,[1],:,:].imag), dim=1)
        
        if self.embedding_type == 'fourier':
            # Gaussian Fourier features embeddings.
            used_sigmas = time_cond
            temb = modules_s[m_idx](torch.log(used_sigmas))
            m_idx += 1
        
        elif self.embedding_type == 'positional':
            # Sinusoidal positional embeddings.
            timesteps = time_cond
            used_sigmas = self.sigmas[time_cond.long()]
            temb = layers.get_timestep_embedding(timesteps, self.nf)
        
        else:
            raise ValueError(f'embedding type {self.embedding_type} unknown.')
        
        if self.conditional:
            temb = modules_s[m_idx](temb)
            m_idx += 1
            temb = modules_s[m_idx](self.act(temb))
            m_idx += 1
        else:
            temb = None

        if not self.centered:
            # If input data is in [0, 1]
            x = 2 * x - 1.

        # Downsampling block
        input_pyramid_s = None
        if self.progressive_input != 'none':
            input_pyramid_s = x

        ################## unet의 cross attention을 사용할 경우,
        # 입력(4 x 256 x 256)에 Cross Attention
        # pdb.set_trace()
        if self.front_cross_attn == 'True':
            # x = modules_s[m_idx](x)
            x, temp = modules_s[m_idx](x)
            D_loss += temp
            m_idx += 1
        ##################
                    
        # Input layer: Conv2d: 4ch -> 128ch
        hs = [modules_s[m_idx](x)]
        m_idx += 1

        # Down path in U-Net
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                h_s = modules_s[m_idx](hs[-1], temb)
                m_idx += 1
                # Attention layer (optional)
                if h_s.shape[-2] in self.attn_resolutions: # edit: check H dim (-2) not W dim (-1)
                    h_s = modules_s[m_idx](h_s)
                    m_idx += 1
                hs.append(h_s)
                
            ############### CD block (Down path) ###############
            if h_s.shape[-2] in self.cd_block_resolutions:
                # print("Down path: ", h_s.shape[-2])
                h_s = modules_s[m_idx](h_s)
                m_idx += 1
            ########################################
            
            # Downsampling
            if i_level != self.num_resolutions - 1:
                if self.ProgDeCAttn_loss == 'True':
                    if h_s.shape[-2] in self.prog_attn_resolutions: # edit: check H dim (-2) not W dim (-1)
                        modules_s[m_idx].loss = True
                        input_pyramid_s, temp = modules_s[m_idx](input_pyramid_s)
                        D_loss += temp
                        m_idx += 1
                else:
                    if h_s.shape[-2] in self.prog_attn_resolutions: # edit: check H dim (-2) not W dim (-1)
                        modules_s[m_idx].loss = False
                        input_pyramid_s = modules_s[m_idx](input_pyramid_s)
                        m_idx += 1
                if self.resblock_type == 'ddpm':
                    h_s = modules_s[m_idx](hs[-1])
                    m_idx += 1
                else: # ResidualBigGAN 사용, 채널 감소됨
                    # pdb.set_trace()
                    h_s = modules_s[m_idx](hs[-1], temb)
                    m_idx += 1

                if self.progressive_input == 'input_skip':   # Combine h with x
                    input_pyramid_s = self.pyramid_downsample(input_pyramid_s)
                    h_s = modules_s[m_idx](input_pyramid_s, h_s)
                    m_idx += 1

                elif self.progressive_input == 'residual':
                    input_pyramid_s = modules_s[m_idx](input_pyramid_s)
                    m_idx += 1
                    if self.skip_rescale:
                        input_pyramid_s = (input_pyramid_s + h_s) / np.sqrt(2.)
                    else:
                        input_pyramid_s = input_pyramid_s + h_s
                    h_s = input_pyramid_s
                    
                hs.append(h_s)
            # else:
            #     ######## CD block (Bottle Neck) ########
            #     if h_s.shape[-2] in [4]: # edit: check H dim (-2) not W dim (-1)
            #         CD_value = modules_s[m_idx](input_pyramid_s)
            #         m_idx += 1  
            #     ########################                      
        h_s = hs[-1] # actualy equal to: h = h
        h_s = modules_s[m_idx](h_s, temb)  # ResNet block
        m_idx += 1
        # h_s = modules_s[m_idx](h_s)  # Attention block
        
        if self.DeCAttn_loss == 'True':
            modules_s[m_idx].loss = True
            h_s, temp = modules_s[m_idx](h_s)  # Attention block
            D_loss += temp
        else:
            modules_s[m_idx].loss = False
            h_s = modules_s[m_idx](h_s)  # Attention block
        m_idx += 1
        h_s = modules_s[m_idx](h_s, temb)  # ResNet block
        m_idx += 1
        ## CDblock ###
        if self.bottleneck_cd_block == "True":
            h_s = modules_s[m_idx](h_s) # CD block
            m_idx += 1
        ##############

        pyramid_s = None
        ProgAttn_idx = []
        # Upsampling block
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h_s = modules_s[m_idx](torch.cat([h_s, hs.pop()], dim=1), temb)
                m_idx += 1

            # edit: from -1 to -2
            if h_s.shape[-2] in self.attn_resolutions:
                h_s = modules_s[m_idx](h_s)
                m_idx += 1
                
            ############### CD block (Up path) ###############
            if h_s.shape[-2] in self.cd_block_resolutions:
                # print("Up path: ", h_s.shape[-2])
                h_s = modules_s[m_idx](h_s)
                m_idx += 1
            ########################################
            
            if self.progressive != 'none':
                if i_level == self.num_resolutions - 1: # 첫 layer인 경우 최초 progressive input 설정
                    if self.progressive == 'output_skip':
                        pyramid_s = self.act(modules_s[m_idx](h_s))  # GroupNorm
                        m_idx += 1
                        pyramid_s = modules_s[m_idx](pyramid_s)  # Conv2D: 256 -> 4
                        m_idx += 1
                    elif self.progressive == 'residual':
                        pyramid_s = self.act(modules_s[m_idx](h_s))
                        m_idx += 1
                        pyramid_s = modules_s[m_idx](pyramid_s)
                        m_idx += 1
                    else:
                        raise ValueError(f'{self.progressive} is not a valid name.')
                else:
                    # ###S### ProgDeCAttn_loss 계산
                    # pdb.set_trace()
                    # if self.ProgDeCAttn_loss == True:
                    #     ProgAttn_idx = m_idx
                    #     if pyramid_s.shape[-2] in self.prog_attn_resolutions: # edit: check H dim (-2) not W dim (-1)
                    #         pyramid_s, temp = modules_s[m_idx](pyramid_s)
                    #         D_loss += temp
                    #         m_idx += 1
                    # else:
                    #     if pyramid_s.shape[-2] in self.prog_attn_resolutions: # edit: check H dim (-2) not W dim (-1)
                    #         pyramid_s = modules_s[m_idx](pyramid_s)
                    #         m_idx += 1         
                    # ###E### ProgDeCAttn_loss 계산
                                               
                    if self.progressive == 'output_skip':
                        pyramid_s = self.pyramid_upsample(pyramid_s)  # Upsample
                        pyramid_hs = self.act(modules_s[m_idx](h_s))  # GroupNorm
                        m_idx += 1
                        pyramid_hs = modules_s[m_idx](pyramid_hs)
                        m_idx += 1
                        pyramid_s = pyramid_s + pyramid_hs
                    elif self.progressive == 'residual':
                        pyramid_s = modules_s[m_idx](pyramid_s)
                        m_idx += 1
                        if self.skip_rescale:
                            pyramid_s = (pyramid_s + h_s) / np.sqrt(2.)
                        else:
                            pyramid_s = pyramid_s + h_s
                        h_s = pyramid_s
                    else:
                        raise ValueError(f'{self.progressive} is not a valid name')

            ###S### ProgDeCAttn_loss 계산
            if self.ProgDeCAttn_loss == 'True':
                ProgAttn_idx.append(m_idx)
                if pyramid_s.shape[-2] in self.prog_attn_resolutions: # edit: check H dim (-2) not W dim (-1)
                    modules_s[m_idx].loss = True
                    pyramid_s, temp = modules_s[m_idx](pyramid_s)
                    D_loss += temp
                    m_idx += 1
            else:
                if pyramid_s.shape[-2] in self.prog_attn_resolutions: # edit: check H dim (-2) not W dim (-1)
                    modules_s[m_idx].loss = False
                    pyramid_s = modules_s[m_idx](pyramid_s)
                    m_idx += 1                    
            ###E### ProgDeCAttn_loss 계산
                                        
            # Upsampling Layer
            if i_level != 0:
                if self.resblock_type == 'ddpm':
                    h_s = modules_s[m_idx](h_s)
                    m_idx += 1
                else:
                    h_s = modules_s[m_idx](h_s, temb)  # Upspampling
                    m_idx += 1

        assert not hs

        if self.progressive == 'output_skip':
            h_s = pyramid_s
        else:
            h_s = self.act(modules_s[m_idx](h_s))
            m_idx += 1
            h_s = modules_s[m_idx](h_s)
            m_idx += 1

        assert m_idx == len(modules_s), "Implementation error s"
        if self.scale_by_sigma:
            used_sigmas = used_sigmas.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
            h_s = h_s / used_sigmas

        # Convert back to complex number
        h = h_s
        h = self.output_layer(h) 
        h = torch.permute(h, (0, 2, 3, 1)).contiguous()
        h = torch.view_as_complex(h)[:,None, :, :]
        
        # Attn loss 또는 ProgAttn loss를 사용하면 D_loss 반환
        if self.ProgDeCAttn_loss == 'True':
        # D_loss != 0: # D_loss 계산을 했으면 반환
            # print("ProgDeCAttn_loss 사용")
            return h, D_loss
        else:
            return h
        # if modules_s[Attn_idx].loss == True or modules_s[ProgAttn_idx[0]].loss == True:
        #     return h, D_loss
        # else:
        #     return h