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

from .ncsnpp_utils import layers, layerspp, normalization
import torch.nn as nn
import functools
import torch
import numpy as np

from .shared import BackboneRegistry, logs_model_option

import pdb

ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp
Combine = layerspp.Combine
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init


@BackboneRegistry.register("ncsnpp_deca")
class NCSNpp_DeCA(nn.Module):
    """NCSN++ model, adapted from https://github.com/yang-song/score_sde repository"""

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--ch_mult",type=int, nargs='+', default=[1,1,2,2,2,2,2])
        parser.add_argument("--num_res_blocks", type=int, default=2)
        parser.add_argument("--attn_resolutions", type=int, nargs='+', default=[16])
        parser.add_argument("--no-centered", dest="centered", action="store_false", help="The data is not centered [-1, 1]")
        parser.add_argument("--centered", dest="centered", action="store_true", help="The data is centered [-1, 1]")
        parser.set_defaults(centered=True)
        
        ### My Args
        parser.add_argument("--image_size", type=int, default=256)
        parser.add_argument("--deca_resolutions", type=int, nargs='+', default=[0])
        parser.add_argument("--input_deca", type=str, default="True")
        parser.add_argument("--input_split", type=str, default="False")
        parser.add_argument("--dropout", type=float, default=0)
        
        
        return parser

    def __init__(self,
        scale_by_sigma = True,
        nonlinearity = 'swish',
        nf = 128,
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
        
        deca_resolutions = (0),
        input_deca = 'True',
        input_split = 'False',
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

        self.conditional = conditional = conditional  # noise-conditional
        self.centered = centered
        self.scale_by_sigma = scale_by_sigma

        fir = fir
        fir_kernel = fir_kernel
        self.skip_rescale = skip_rescale = skip_rescale
        self.resblock_type = resblock_type = resblock_type.lower()
        self.progressive = progressive = progressive.lower()
        self.progressive_input = progressive_input = progressive_input.lower()
        self.embedding_type = embedding_type = embedding_type.lower()
        init_scale = init_scale
        
        ###################### My Args ######################
        logs = [] #
        self.my_args = {}
        self.my_args['deca_resolutions'] = deca_resolutions
        self.my_args['input_deca'] = input_deca
        self.my_args['input_split'] = input_split
        self.pc_trigger = False
        #####################################################
        
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

        DeCAttnBlock = functools.partial(layerspp.CrossAttnBlockpp,
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

        ResnetBlock_firNo_tembNo = functools.partial(ResnetBlockBigGAN, act=act,
                dropout=dropout, fir=None, fir_kernel=fir_kernel,
                init_scale=init_scale, skip_rescale=skip_rescale, temb_dim=None)
        ResnetBlock_firYes_tembNo = functools.partial(ResnetBlockBigGAN, act=act,
                dropout=dropout, fir=fir, fir_kernel=fir_kernel,
                init_scale=init_scale, skip_rescale=skip_rescale, temb_dim=None)
        ResnetBlock_firNo_tembYes = functools.partial(ResnetBlockBigGAN, act=act,
                dropout=dropout, fir=None, fir_kernel=fir_kernel,
                init_scale=init_scale, skip_rescale=skip_rescale, temb_dim=nf * 4)
        ResnetBlock_firYes_tembYes = functools.partial(ResnetBlockBigGAN, act=act,
                dropout=dropout, fir=fir, fir_kernel=fir_kernel,
                init_scale=init_scale, skip_rescale=skip_rescale, temb_dim=nf * 4)
        
        logs.append(f'####################################################')
        # Downsampling block

        channels = num_channels
        if progressive_input != 'none':
            input_pyramid_ch = channels

        ########## Front Layer DeCorrelated Attention Block, input: [4x256x256] ##########
        if self.my_args['input_deca'] == 'True1':
            modules.append(nn.Sequential(nn.Linear(256, 128),
                                         nn.LeakyReLU(),
                                         nn.Linear(128, 64),
                                         nn.LeakyReLU()))
            # modules.append(nn.Sequential(nn.Linear(128, 64),
            #                              nn.LeakyReLU()))
            logs.append(f'Input_DeCorrelated Attention, Type: FC 추가\n')
            modules.append(DeCAttnBlock(channels=channels//2))
            modules.append(nn.Sequential(nn.Linear(64, 128),
                                         nn.LeakyReLU(),
                                         nn.Linear(128, 256),
                                         nn.LeakyReLU()))
            # modules.append(nn.Sequential(nn.Linear(128, 256),
            #                              nn.LeakyReLU()))        
        ##########################################################################
        ######### ResnetBlock 사용한 down sampling ##############
        if self.my_args['input_deca'] == 'True2':
            logs.append(f'Input_DeCorrelated Attention, Type: CNN_ori 추가\n')
            modules.append(ResnetBlock(down=True, in_ch=4))
            modules.append(DeCAttnBlock(channels=channels//2))
            modules.append(ResnetBlock(in_ch=4, up=True))
        ##########################################################################
        ######### ResnetBlock 사용한 down sampling 2번 ###########################
        if self.my_args['input_deca'] == 'CNN1_128':
            logs.append(f'Input_DeCorrelated Attention, Type: CNN1_128 추가\n')
            modules.append(ResnetBlock_firNo_tembNo(down=True, in_ch=4))
            modules.append(DeCAttnBlock(channels=channels//2))
            modules.append(ResnetBlock_firNo_tembNo(in_ch=4, up=True))
            modules.append(ResnetBlock(in_ch=8, out_ch=4))
        ##########################################################################
        ######### ResnetBlock 사용한 down sampling 3번 ###########################
        if self.my_args['input_deca'] == 'CNN2_128':
            logs.append(f'Input_DeCorrelated Attention, Type: CNN2_128 추가\n')
            modules.append(ResnetBlock_firYes_tembNo(down=True, in_ch=4))
            modules.append(DeCAttnBlock(channels=channels//2))
            modules.append(ResnetBlock_firYes_tembNo(in_ch=4, up=True))
        ##########################################################################
        ######### ResnetBlock 사용한 down sampling 4번 ###########################
        if self.my_args['input_deca'] == 'CNN3_128':
            logs.append(f'Input_DeCorrelated Attention, Type: CNN3_128 추가\n')
            modules.append(ResnetBlock_firNo_tembYes(down=True, in_ch=4))
            modules.append(DeCAttnBlock(channels=channels//2))
            modules.append(ResnetBlock_firNo_tembYes(in_ch=4, up=True))
        ##########################################################################                
        ######### ResnetBlock 사용한 down sampling 4번 ###########################
        if self.my_args['input_deca'] == 'CNN4_128':
            logs.append(f'Input_DeCorrelated Attention, Type: CNN4_128 추가\n')
            modules.append(ResnetBlock_firYes_tembYes(down=True, in_ch=4))
            modules.append(DeCAttnBlock(channels=channels//2))
            modules.append(ResnetBlock_firYes_tembYes(in_ch=4, up=True))
            modules.append(ResnetBlock(in_ch=8, out_ch=4))
        ##########################################################################  


        ######### ResnetBlock 사용한 down sampling 2번 ###########################
        if self.my_args['input_deca'] == 'CNN1_64':
            logs.append(f'Input_DeCorrelated Attention, Type: CNN1_64 추가\n')
            modules.append(ResnetBlock_firNo_tembNo(down=True, in_ch=4))
            modules.append(ResnetBlock_firNo_tembNo(down=True, in_ch=4))
            modules.append(DeCAttnBlock(channels=channels//2))
            modules.append(ResnetBlock_firNo_tembNo(in_ch=4, up=True))
            modules.append(ResnetBlock_firNo_tembNo(in_ch=4, up=True))
            modules.append(ResnetBlock(in_ch=8, out_ch=4))
        ##########################################################################
        ######### ResnetBlock 사용한 down sampling 3번 ###########################
        if self.my_args['input_deca'] == 'CNN2_64':
            logs.append(f'Input_DeCorrelated Attention, Type: CNN2_64 추가\n')
            modules.append(ResnetBlock_firYes_tembNo(down=True, in_ch=4))
            modules.append(ResnetBlock_firYes_tembNo(down=True, in_ch=4))
            modules.append(DeCAttnBlock(channels=channels//2))
            modules.append(ResnetBlock_firYes_tembNo(in_ch=4, up=True))
            modules.append(ResnetBlock_firYes_tembNo(in_ch=4, up=True))
        ##########################################################################
        ######### ResnetBlock 사용한 down sampling 4번 ###########################
        if self.my_args['input_deca'] == 'CNN3_64':
            logs.append(f'Input_DeCorrelated Attention, Type: CNN3_64 추가\n')
            modules.append(ResnetBlock_firNo_tembYes(down=True, in_ch=4))
            modules.append(ResnetBlock_firNo_tembYes(down=True, in_ch=4))
            modules.append(DeCAttnBlock(channels=channels//2))
            modules.append(ResnetBlock_firNo_tembYes(in_ch=4, up=True))
            modules.append(ResnetBlock_firNo_tembYes(in_ch=4, up=True))
        ##########################################################################                
        ######### ResnetBlock 사용한 down sampling 4번 ###########################
        if self.my_args['input_deca'] == 'CNN4_64':
            logs.append(f'Input_DeCorrelated Attention, Type: CNN4_64 추가\n')
            modules.append(ResnetBlock_firYes_tembYes(down=True, in_ch=4))
            modules.append(ResnetBlock_firYes_tembYes(down=True, in_ch=4))
            modules.append(DeCAttnBlock(channels=channels//2))
            modules.append(ResnetBlock_firYes_tembYes(in_ch=4, up=True))
            modules.append(ResnetBlock_firYes_tembYes(in_ch=4, up=True))
            modules.append(ResnetBlock(in_ch=8, out_ch=4))
        ##########################################################################  
        ##########################################################################  
        if self.my_args['input_deca'] == 'CNN3_64_new':
            logs.append(f'Input_DeCorrelated Attention, Type: CNN3_64_new 추가\n')
            modules.append(nn.Sequential(ResnetBlock_firNo_tembYes(down=True, in_ch=4),
                                         ResnetBlock_firNo_tembYes(down=True, in_ch=4)))
            modules.append(DeCAttnBlock(channels=channels//2))
            modules.append(nn.Sequential(ResnetBlock_firNo_tembYes(in_ch=4, up=True),
                                         ResnetBlock_firNo_tembYes(in_ch=4, up=True)))
        ##########################################################################  
        
        if self.my_args['input_deca'] == 'Spine_CNN_64':
            logs.append(f'Input_DeCorrelated Attention, Type: Spine_CNN_64 추가\n')
            modules.append(ResnetBlock(down=True, in_ch=4))
            modules.append(ResnetBlock(down=True, in_ch=4))
            modules.append(DeCAttnBlock(channels=channels//2))
            # all_resolutions = [int(x/4) for x in all_resolutions]
            # self.attn_resolutions = [int(x/4) for x in self.attn_resolutions]
            # attn_resolutions = [int(x/4) for x in attn_resolutions]
            
            modules.append(ResnetBlock(in_ch=4, up=True))
            modules.append(ResnetBlock(in_ch=4, up=True))
        ##########################################################################          
                        
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
                hs_c.append(in_ch)
            if all_resolutions[i_level] in self.my_args['deca_resolutions']:
                logs.append(f'DeCAttnBlock {all_resolutions[i_level]} 추가\n')
                modules.append(ResnetBlock(down=True, in_ch=4))
                modules.append(ResnetBlock(down=True, in_ch=4))
                modules.append(DeCAttnBlock(channels=4//2))
                modules.append(ResnetBlock(in_ch=4, up=True))
                modules.append(ResnetBlock(in_ch=4, up=True))     
            # if all_resolutions[i_level] in self.my_args['deca_resolutions']:
            #     logs.append(f'{all_resolutions[i_level]} {channels}에 deca_resolutions [{self.my_args["deca_resolutions"]}] 추가')
            #     modules.append(DeCAttnBlock(channels=channels//2))
                    
            if i_level != num_resolutions - 1:
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

        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))

        pyramid_ch = 0
        # Upsampling block
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):  # +1 blocks in upsampling because of skip connection from combiner (after downsampling)
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
                in_ch = out_ch

            if all_resolutions[i_level] in attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))

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

        self.all_modules = nn.ModuleList(modules)
        
        logs.append(f'####################################################')
        #######################
        logs_model_option(self.my_args)
        for log in logs:
            print(log, end='')
        
        
    def forward(self, x, time_cond):
        # timestep/noise_level embedding; only for continuous training
        modules = self.all_modules
        m_idx = 0
        deca_loss = 0
        
        # Convert real and imaginary parts of (x,y) into four channel dimensions
        _, C, _, _ = x.shape
        if C == 2:
            x = torch.cat((x[:,[0],:,:].real, x[:,[0],:,:].imag,
                    x[:,[1],:,:].real, x[:,[1],:,:].imag), dim=1)
        else: # if C == 1, in inference with only network
            x = torch.cat((x[:,[0],:,:].real, x[:,[0],:,:].imag), dim=1)
            
        if self.embedding_type == 'fourier':
            # Gaussian Fourier features embeddings.
            used_sigmas = time_cond
            temb = modules[m_idx](torch.log(used_sigmas))
            m_idx += 1

        elif self.embedding_type == 'positional':
            # Sinusoidal positional embeddings.
            timesteps = time_cond
            used_sigmas = self.sigmas[time_cond.long()]
            temb = layers.get_timestep_embedding(timesteps, self.nf)

        else:
            raise ValueError(f'embedding type {self.embedding_type} unknown.')

        if self.conditional:
            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](self.act(temb))
            m_idx += 1
        else:
            temb = None

        if not self.centered:
            # If input data is in [0, 1]
            x = 2 * x - 1.


        # Downsampling block
        
        input_pyramid = None
        
        if self.my_args['input_split'] == 'False':
            if self.progressive_input != 'none':
                input_pyramid = x

        ####################### linear를 이용한 downsampling ######################
        if self.my_args['input_deca'] == 'True1':
            # x_residual = x
            x = x.permute(0, 1, 3, 2).contiguous()
            x = modules[m_idx](x)
            m_idx +=1            
            x = x.permute(0, 1, 3, 2).contiguous()
            
            x, loss = modules[m_idx](x)
            deca_loss += loss
            m_idx += 1
            
            x = x.permute(0, 1, 3, 2).contiguous()
            x = modules[m_idx](x)
            x = x.permute(0, 1, 3, 2).contiguous()
            # x += x_residual
            m_idx +=1        
        ############################################################################
        ###################### ResnetBlock 이용한 downsampling ######################
        # pdb.set_trace()
        if self.my_args['input_deca'] == 'True2':
            x_residual = x
            x = modules[m_idx](x, temb)
            m_idx += 1
            
            x, loss = modules[m_idx](x)
            m_idx += 1
            deca_loss += loss
            
            x = modules[m_idx](x, temb)
            x += x_residual # residual (skip connection)
            m_idx += 1
        #############################################################################
        ###################### ResnetBlock 이용한 downsampling ######################
        # pdb.set_trace()
        if self.my_args['input_deca'] == 'CNN2':
            x_residual = x
            
            x = modules[m_idx](x, temb)
            m_idx += 1
            x = modules[m_idx](x, temb)
            m_idx += 1
                        
            x, loss = modules[m_idx](x)
            m_idx += 1
            deca_loss += loss
            
            x = modules[m_idx](x, temb)
            m_idx += 1
            x = modules[m_idx](x, temb)
            m_idx += 1    
                    
            x += x_residual # residual (skip connection)
        #############################################################################
        
      ######### ResnetBlock 사용한 down sampling 2번 ###########################
        if self.my_args['input_deca'] == 'CNN1_128':
            x_residual = x
            x = modules[m_idx](x)
            m_idx += 1
            
            x, loss = modules[m_idx](x)
            m_idx += 1
            deca_loss += loss
            
            x = modules[m_idx](x)
            m_idx += 1
            
            x = modules[m_idx](torch.cat([x, x_residual], dim=1), temb)
            m_idx +=1        
        ##########################################################################
        ######### ResnetBlock 사용한 down sampling 3번 ###########################
        if self.my_args['input_deca'] == 'CNN2_128':
            x_residual = x
            x = modules[m_idx](x)
            m_idx += 1
            
            x, loss = modules[m_idx](x)
            m_idx += 1
            deca_loss += loss
            
            x = modules[m_idx](x)
            m_idx += 1                     
            x += x_residual
        ##########################################################################
        ######### ResnetBlock 사용한 down sampling 4번 ###########################
        if self.my_args['input_deca'] == 'CNN3_128':
            x_residual = x
            x = modules[m_idx](x, temb)
            m_idx += 1
            
            x, loss = modules[m_idx](x)
            m_idx += 1
            deca_loss += loss            
            
            x = modules[m_idx](x, temb)
            m_idx += 1                     
            x += x_residual
        ##########################################################################                
        ######### ResnetBlock 사용한 down sampling 4번 ###########################
        if self.my_args['input_deca'] == 'CNN4_128':
            x_residual = x
            x = modules[m_idx](x, temb)
            m_idx += 1
            
            x, loss = modules[m_idx](x)
            m_idx += 1
            deca_loss += loss            
            
            x = modules[m_idx](x, temb)
            m_idx += 1                     
            x = modules[m_idx](torch.cat([x, x_residual], dim=1), temb)
            m_idx +=1        
        ##########################################################################  


        ######### ResnetBlock 사용한 down sampling 2번 ###########################
        if self.my_args['input_deca'] == 'CNN1_64':
            x_residual = x
            x = modules[m_idx](x)
            m_idx += 1
            
            x = modules[m_idx](x)
            m_idx += 1            
            
            x, loss = modules[m_idx](x)
            m_idx += 1
            deca_loss += loss            
            
            x = modules[m_idx](x)
            m_idx += 1            
            
            x = modules[m_idx](x)
            m_idx += 1                     
            x = modules[m_idx](torch.cat([x, x_residual], dim=1), temb)
            m_idx +=1        
        ##########################################################################
        ######### ResnetBlock 사용한 down sampling 3번 ###########################
        if self.my_args['input_deca'] == 'CNN2_64':
            x_residual = x
            x = modules[m_idx](x)
            m_idx += 1
            
            x = modules[m_idx](x)
            m_idx += 1            
            
            x, loss = modules[m_idx](x)
            m_idx += 1
            deca_loss += loss            
            
            x = modules[m_idx](x)
            m_idx += 1            
            
            x = modules[m_idx](x)
            m_idx += 1                           
            x += x_residual
        ##########################################################################
        ######### ResnetBlock 사용한 down sampling 4번 ###########################
        if self.my_args['input_deca'] == 'CNN3_64':
            x_residual = x
            x = modules[m_idx](x, temb)
            m_idx += 1
            
            x = modules[m_idx](x, temb)
            m_idx += 1            
            
            x, loss = modules[m_idx](x)
            m_idx += 1
            deca_loss += loss            
            
            x = modules[m_idx](x, temb)
            m_idx += 1            
            
            x = modules[m_idx](x, temb)
            m_idx += 1                     
            x += x_residual
        ##########################################################################                
        ######### ResnetBlock 사용한 down sampling 4번 ###########################
        if self.my_args['input_deca'] == 'CNN4_64':
            x_residual = x
            x = modules[m_idx](x, temb)
            m_idx += 1
            
            x = modules[m_idx](x, temb)
            m_idx += 1            
            
            x, loss = modules[m_idx](x)
            m_idx += 1
            deca_loss += loss            
            
            x = modules[m_idx](x, temb)
            m_idx += 1            
            
            x = modules[m_idx](x, temb)
            m_idx += 1                     
            x = modules[m_idx](torch.cat([x, x_residual], dim=1), temb)
            m_idx +=1        
        ##########################################################################  
        if self.my_args['input_deca'] == 'CNN3_64_new':
            x_residual = x
            x = modules[m_idx](x)
            m_idx += 1
            
            x, loss = modules[m_idx](x)
            m_idx += 1
            deca_loss += loss
            
            x = modules[m_idx](x, temb)
            m_idx += 1            
            x += x_residual
        ##########################################################################
        if self.my_args['input_deca'] == 'Spine_CNN_64':
            # x_residual = x
            
            x = modules[m_idx](x, temb)
            m_idx += 1
            
            x = modules[m_idx](x, temb)
            m_idx += 1
            
            x, loss = modules[m_idx](x)
            m_idx += 1
            deca_loss += loss

            x = modules[m_idx](x, temb)
            m_idx += 1
            
            x = modules[m_idx](x, temb)
            m_idx += 1                        
            # x += x_residual
        ##########################################################################  
            
        if self.my_args['input_split'] == 'True':
            if self.progressive_input != 'none':
                input_pyramid = x
            
        # Input layer: Conv2d: 4ch -> 128ch
        hs = [modules[m_idx](x)]
        m_idx += 1

        # Down path in U-Net
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], temb)
                m_idx += 1
                # Attention layer (optional)
                if h.shape[-2] in self.attn_resolutions: # edit: check H dim (-2) not W dim (-1)
                    h = modules[m_idx](h)
                    m_idx += 1
                hs.append(h)
            if input_pyramid.shape[-2] in self.my_args['deca_resolutions']:
                input_pyramid = modules[m_idx](input_pyramid, temb)
                m_idx += 1
                input_pyramid = modules[m_idx](input_pyramid, temb)
                m_idx += 1                        
                input_pyramid, loss = modules[m_idx](input_pyramid)
                m_idx += 1
                deca_loss += loss
                input_pyramid = modules[m_idx](input_pyramid, temb)
                m_idx += 1
                input_pyramid = modules[m_idx](input_pyramid, temb)
                m_idx += 1                
                             
            # Downsampling
            if i_level != self.num_resolutions - 1:
                if self.resblock_type == 'ddpm':
                    h = modules[m_idx](hs[-1])
                    m_idx += 1
                else:
                    h = modules[m_idx](hs[-1], temb)
                    m_idx += 1
                if self.progressive_input == 'input_skip':   # Combine h with x Default
                    # print(input_pyramid.shape)
                    input_pyramid = self.pyramid_downsample(input_pyramid)
                    h = modules[m_idx](input_pyramid, h)
                    m_idx += 1

                elif self.progressive_input == 'residual':
                    input_pyramid = modules[m_idx](input_pyramid)
                    m_idx += 1
                    if self.skip_rescale:
                        input_pyramid = (input_pyramid + h) / np.sqrt(2.)
                    else:
                        input_pyramid = input_pyramid + h
                    h = input_pyramid
                hs.append(h)

        h = hs[-1] # actualy equal to: h = h
        h = modules[m_idx](h, temb)  # ResNet block
        m_idx += 1
        h = modules[m_idx](h)  # Attention block
        m_idx += 1
        h = modules[m_idx](h, temb)  # ResNet block
        m_idx += 1

        pyramid = None

        # Upsampling block
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
                m_idx += 1

            # edit: from -1 to -2
            if h.shape[-2] in self.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1

            if self.progressive != 'none':
                if i_level == self.num_resolutions - 1:
                    if self.progressive == 'output_skip':
                        pyramid = self.act(modules[m_idx](h))  # GroupNorm
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)  # Conv2D: 256 -> 4
                        m_idx += 1
                    elif self.progressive == 'residual':
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    else:
                        raise ValueError(f'{self.progressive} is not a valid name.')
                else:
                    if self.progressive == 'output_skip':
                        pyramid = self.pyramid_upsample(pyramid)  # Upsample
                        pyramid_h = self.act(modules[m_idx](h))  # GroupNorm
                        m_idx += 1
                        pyramid_h = modules[m_idx](pyramid_h)
                        m_idx += 1
                        pyramid = pyramid + pyramid_h
                    elif self.progressive == 'residual':
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                        if self.skip_rescale:
                            pyramid = (pyramid + h) / np.sqrt(2.)
                        else:
                            pyramid = pyramid + h
                        h = pyramid
                    else:
                        raise ValueError(f'{self.progressive} is not a valid name')

            # Upsampling Layer
            if i_level != 0:
                if self.resblock_type == 'ddpm':
                    h = modules[m_idx](h)
                    m_idx += 1
                else:
                    h = modules[m_idx](h, temb)  # Upspampling
                    m_idx += 1

        assert not hs

        if self.progressive == 'output_skip':
            h = pyramid
        else:
            h = self.act(modules[m_idx](h))
            m_idx += 1
            h = modules[m_idx](h)
            m_idx += 1

        assert m_idx == len(modules), "Implementation error"
        if self.scale_by_sigma:
            used_sigmas = used_sigmas.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
            h = h / used_sigmas

        # Convert back to complex number
        h = self.output_layer(h)
        h = torch.permute(h, (0, 2, 3, 1)).contiguous()
        h = torch.view_as_complex(h)[:,None, :, :]
        
        if self.pc_trigger == True: # pc_sampler에서는 deca_loss 안씀
            return h
        else: # pc_sampler 아니면 두 개 반환
            return h, deca_loss