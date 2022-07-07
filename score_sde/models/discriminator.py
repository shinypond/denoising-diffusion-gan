# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import numpy as np

from . import up_or_down_sampling
from . import dense_layer
from . import layers

dense = dense_layer.dense
conv2d = dense_layer.conv2d
get_sinusoidal_positional_embedding = layers.get_timestep_embedding


class TimestepEmbedding(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, act=nn.LeakyReLU(0.2)):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.main = nn.Sequential(
            dense(embedding_dim, hidden_dim),
            act,
            dense(hidden_dim, output_dim),
        )

    def forward(self, temp):
        temb = get_sinusoidal_positional_embedding(temp, self.embedding_dim)
        temb = self.main(temb)
        return temb


class DownConvBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
        padding=1,
        t_emb_dim = 128,
        downsample=False,
        act = nn.LeakyReLU(0.2),
        fir_kernel=(1, 3, 3, 1),
        spectral_norm=False,
    ):
        super().__init__()
        self.fir_kernel = fir_kernel
        self.downsample = downsample
        self.conv1 = conv2d(in_channel, out_channel, kernel_size, padding=padding)
        self.conv2 = conv2d(out_channel, out_channel, kernel_size, padding=padding,init_scale=0.)
        self.dense_t1 = dense(t_emb_dim, out_channel)
        self.act = act
        self.skip = conv2d(in_channel, out_channel, 1, padding=0, bias=False)
        
        if spectral_norm:
            self.conv1 = nn.utils.spectral_norm(self.conv1)
            self.conv2 = nn.utils.spectral_norm(self.conv2)
            self.dense_t1 = nn.utils.spectral_norm(self.dense_t1)
            self.skip = nn.utils.spectral_norm(self.skip)

    def forward(self, input, t_emb):
        out = self.act(input)
        out = self.conv1(out)
        out += self.dense_t1(t_emb)[..., None, None]
        out = self.act(out)
        if self.downsample:
            out = up_or_down_sampling.downsample_2d(out, self.fir_kernel, factor=2)
            input = up_or_down_sampling.downsample_2d(input, self.fir_kernel, factor=2)
        out = self.conv2(out)
        skip = self.skip(input)
        out = (out + skip) / np.sqrt(2)
        return out
    

class MarginalDiscriminator(nn.Module):
    """A time-dependent discriminator for small images (CIFAR10, StackMNIST)."""
    def __init__(self, nc=3, ngf=64, ch_mult=[2,2,4,8,8], downsamples=[0,1,1,1], t_emb_dim=128, act=nn.LeakyReLU(0.2), spectral_norm=False):
        super().__init__()
        assert len(ch_mult) == len(downsamples) + 1
        
        # Gaussian random feature embedding layer for time
        self.act = act
        
        self.t_embed = TimestepEmbedding(
            embedding_dim=t_emb_dim,
            hidden_dim=t_emb_dim,
            output_dim=t_emb_dim,
            act=act,
        )
        
        # Encoding layers where the resolution decreases
        self.start_conv = conv2d(nc, ngf*ch_mult[0], 1, padding=0)
        
        
        modules = []
        for k in range(len(ch_mult)-1):
            in_ch, out_ch = ngf*ch_mult[k], ngf*ch_mult[k+1]
            modules.append(DownConvBlock(in_ch, out_ch, t_emb_dim=t_emb_dim, downsample=downsamples[k], act=act, spectral_norm=spectral_norm))
                    
        self.main = nn.ModuleList(modules)
        
        self.final_conv = conv2d(out_ch+1, out_ch, 3,padding=1, init_scale=0.)
        self.end_linear = dense(out_ch, 1)
        
        self.stddev_group = 4
        self.stddev_feat = 1

        if spectral_norm:
            self.start_conv = nn.utils.spectral_norm(self.start_conv)
            self.final_conv = nn.utils.spectral_norm(self.final_conv)
            # self.end_linear = nn.utils.spectral_norm(self.end_linear)
            
    def forward(self, x, t, x_t):
        t_embed = self.act(self.t_embed(t))  
        #input_x = torch.cat((x, x_t), dim = 1)
        input_x = x
        
        out = self.start_conv(input_x)
        for module in self.main:
            out = module(out, t_embed)
        
        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(group, -1, self.stddev_feat, channel//self.stddev_feat, height, width)
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)
        
        out = self.final_conv(out)
        out = self.act(out)
    
        out = out.view(out.shape[0], out.shape[1], -1).sum(2)
        out = self.end_linear(out)
        
        return out


class ConditionalDiscriminator(nn.Module):
    """A time-dependent discriminator for small images (CIFAR10, StackMNIST)."""
    def __init__(self, nc=3, ngf=64, ch_mult=[2,2,4,8,8], downsamples=[0,1,1,1], t_emb_dim=128, act=nn.LeakyReLU(0.2), spectral_norm=False):
        super().__init__()
        assert len(ch_mult) == len(downsamples) + 1
        
        # Gaussian random feature embedding layer for time
        self.act = act
        
        self.t_embed = TimestepEmbedding(
            embedding_dim=t_emb_dim,
            hidden_dim=t_emb_dim,
            output_dim=t_emb_dim,
            act=act,
        )
        
        # Encoding layers where the resolution decreases
        self.start_conv = conv2d(nc, ngf*ch_mult[0], 1, padding=0)
        
        
        modules = []
        for k in range(len(ch_mult)-1):
            in_ch, out_ch = ngf*ch_mult[k], ngf*ch_mult[k+1]
            modules.append(DownConvBlock(in_ch, out_ch, t_emb_dim=t_emb_dim, downsample=downsamples[k], act=act, spectral_norm=spectral_norm))
                    
        self.main = nn.ModuleList(modules)
        
        self.final_conv = conv2d(out_ch+1, out_ch, 3,padding=1, init_scale=0.)
        self.end_linear = dense(out_ch, 1)
        
        self.stddev_group = 4
        self.stddev_feat = 1

        if spectral_norm:
            self.start_conv = nn.utils.spectral_norm(self.start_conv)
            self.final_conv = nn.utils.spectral_norm(self.final_conv)
            # self.end_linear = nn.utils.spectral_norm(self.end_linear)
            
    def forward(self, x, t, x_t):
        t_embed = self.act(self.t_embed(t))  
        input_x = torch.cat((x, x_t), dim = 1)
        # input_x = x
        
        out = self.start_conv(input_x)
        for module in self.main:
            out = module(out, t_embed)
        
        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(group, -1, self.stddev_feat, channel//self.stddev_feat, height, width)
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)
        
        out = self.final_conv(out)
        out = self.act(out)
    
        out = out.view(out.shape[0], out.shape[1], -1).sum(2)
        out = self.end_linear(out)
        
        return out