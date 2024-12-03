# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
from scipy import ndimage
from . import nn_configs as configs

from mamba_ssm import Mamba


CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-L_16': configs.get_l16_config(),
    'I2IMamba-B_16': configs.get_i2i_mamba_b16_config(),
    'I2IMamba-L_16': configs.get_i2i_mamba_l16_config(),
}

logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = (
            self.num_attention_heads * self.attention_head_size
        )

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = torch.nn.functional.gelu
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings."""

    def __init__(self, config, img_size, in_channels=3, input_dim=3, old=1):
        super(Embeddings, self).__init__()
        self.config = config
        img_size = _pair(img_size)
        grid_size = config.patches["grid"]
        patch_size = (
            img_size[0] // 16 // grid_size[0],
            img_size[1] // 16 // grid_size[1],
        )
        patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
        n_patches = (img_size[0] // patch_size_real[0]) * (
            img_size[1] // patch_size_real[1]
        )
        in_channels = 1024
        # Learnable patch embeddings
        self.patch_embeddings = Conv2d(
            in_channels=in_channels,
            out_channels=config.hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )
        # learnable positional encodings
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, n_patches, config.hidden_size)
        )
        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        embeddings = x + self.positional_encoding
        embeddings = self.dropout(embeddings)
        return embeddings


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = (
                np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")])
                .view(self.hidden_size, self.hidden_size)
                .t()
            )
            key_weight = (
                np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")])
                .view(self.hidden_size, self.hidden_size)
                .t()
            )
            value_weight = (
                np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")])
                .view(self.hidden_size, self.hidden_size)
                .t()
            )
            out_weight = (
                np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")])
                .view(self.hidden_size, self.hidden_size)
                .t()
            )

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(
                np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            
            self.attention_norm.bias.copy_(
                np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis, in_channels=3, old=1):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(
            config, img_size=img_size, input_dim=in_channels, old=old
        )
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, features


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, dim2=None):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, use_dropout, use_bias
        )

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0

        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True),
        ]

        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class BottleneckCNN(nn.Module):
    def __init__(self, config):
        super(BottleneckCNN, self).__init__()
        self.config = config
        use_bias = False
        norm_layer = nn.BatchNorm2d
        padding_type = "reflect"
        
        # Residual CNN
        model = [
            ResnetBlock(
                256,
                padding_type=padding_type,
                norm_layer=norm_layer,
                use_dropout=False,
                use_bias=use_bias,
            )
        ]
        setattr(self, "residual_cnn", nn.Sequential(*model))

    def forward(self, x):
        x = self.residual_cnn(x)
        return x


class ChannelCompression(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Args:
          in_channels (int):  Number of input channels.
          out_channels (int): Number of output channels.
          stride (int):       Controls the stride.
        """
        super(ChannelCompression, self).__init__()

        self.skip = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = None

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        out = self.block(x)
        out += x if self.skip is None else self.skip(x)
        out = F.relu(out)
        return out
    

class MambaLayerOnlyspiral(nn.Module):
    """ Mamba layer for state-space sequence modeling

    Args:
        dim (int): Model dimension.
        d_state (int): SSM state expansion factor.
        d_conv (int): Local convolution width.
        expand (int): Block expansion factor.
    
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba1 = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.mamba2 = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        
        self.conv1d = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.spiral_eye = torch.tensor(np.load("spiral_eye.npy"), dtype=torch.float)
        self.despiral_eye = torch.tensor(np.load("despiral_eye.npy"), dtype=torch.float)
        self.despiral_r_eye = torch.tensor(np.load("despiral_r_eye.npy"), dtype=torch.float)

    def forward(self, x):

        device = next(self.parameters()).device
        B, C, H, W = x.shape

        # Check model dimension
        assert C == self.dim

        x1 = x.view(B, C, -1)
        device = next(self.parameters()).device
        x1 = x1.to(device)
        self.spiral_eye = self.spiral_eye.to(device)
        x1 = torch.einsum('ij,klj->kli',self.spiral_eye,x1).permute(0, 2, 1)
        x2 = torch.flip(x1, dims=[1])



        # Pass three scans through mamba
        norm_out1 = self.norm(x1)
        mamba_out1 = self.mamba1(norm_out1)
        norm_out2 = self.norm(x2)
        mamba_out2 = self.mamba2(norm_out2)

        self.despiral_eye = self.despiral_eye.to(device)
        self.despiral_r_eye = self.despiral_r_eye.to(device)
        out1 = torch.einsum('ij,klj->kli',self.despiral_eye,mamba_out1.permute(0, 2, 1)).view(B, C, H, W)
        out2 = torch.einsum('ij,klj->kli',self.despiral_r_eye,mamba_out2.permute(0, 2, 1)).view(B, C, H, W)


        out1 = out1.to(device)
        out2 = out2.to(device)

        concatenated = torch.cat((out1, out2), dim=1)
        output = self.conv1d(concatenated) 
        mamba_out = output.view(-1, 256, 64, 64)
        

        return mamba_out



class cmMambaWithCNN(nn.Module):
    """ Channel-mixed Mamba (cmMamba) block with residual CNN block

    Args:
        config (dict): Model configuration.
        in_channels (int): Number of input channels.
        d_state (int): SSM state expansion factor.
        d_conv (int): Local convolution width.
        expand (int): Block expansion factor.
        ngf (int): Number of generator filters.
        norm_layer (nn.Module): Normalization layer.
        use_dropout (bool): Use dropout.
        use_bias (bool): Use bias.
        img_size (int): Image size.
    
    """
    def __init__(
        self,
        config,
        in_channels,
        d_state=16,
        d_conv=4,
        expand=2,
        ngf=64,
        norm_layer=nn.BatchNorm2d,
        use_bias=False
    ):
        super().__init__()
        # Mamba block
        self.mamba_layer = MambaLayerOnlyspiral(
            dim=in_channels, d_state=d_state, d_conv=d_conv, expand=expand
        )

        self.config = config
        ngf = 64
        padding_type = "reflect"
        use_bias = False
        norm_layer = nn.BatchNorm2d

        # Channel compression block
        self.cc = ChannelCompression(ngf*8, ngf*4)

        # Residual CNN block
        model = [
            ResnetBlock(
                256,
                padding_type=padding_type,
                norm_layer=norm_layer,
                use_dropout=False,
                use_bias=use_bias,
            )
        ]
        setattr(self, "residual_cnn", nn.Sequential(*model))

    def forward(self, x):
        # Pass input through Mamba block
        mamba_out = self.mamba_layer(x)
        x = torch.cat([x, mamba_out], dim=1)

        # Pass Mamba block output through channel compression
        x = self.cc(x)
        
        # Pass channel compressed output through residual CNN block
        x = self.residual_cnn(x)

        return x


class I2IMamba(nn.Module):
    def __init__(self, config, input_dim, img_size=224, output_dim=3, vis=False):
        super(I2IMamba, self).__init__()
        self.transformer_encoder = Encoder(config, vis)
        self.config = config
        output_nc = output_dim
        ngf = 64
        use_bias = False
        norm_layer = nn.BatchNorm2d
        padding_type = "reflect"
        mult = 4

        # Layer1-Encoder1
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_dim, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
        ]
      
        setattr(self, "encoder_1", nn.Sequential(*model))

        # Layer2-Encoder2
        n_downsampling = 2
        model = []
        i = 0
        mult = 2**i
        model = [
            nn.Conv2d(
                ngf * mult,
                ngf * mult * 2,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=use_bias,
            ),
            norm_layer(ngf * mult * 2),
            nn.ReLU(True),
        ]

        setattr(self, "encoder_2", nn.Sequential(*model))

        # Layer3-Encoder3
        model = []
        i = 1
        mult = 2**i
        model = [
            nn.Conv2d(
                ngf * mult,
                ngf * mult * 2,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=use_bias,
            ),
            norm_layer(ngf * mult * 2),
            nn.ReLU(True),
        ]
        
        setattr(self, "encoder_3", nn.Sequential(*model))
        
        mult = 4
        img_size = 256 
        input_dim = 256

        # Episodic bottleneck
        # cmMamba block with residual CNN block
        self.bottleneck_1 = cmMambaWithCNN(self.config, img_size)

        self.bottleneck_2 = BottleneckCNN(self.config)
        self.bottleneck_3 = BottleneckCNN(self.config)
        self.bottleneck_4 = BottleneckCNN(self.config)

        # cmMamba block with residual CNN block
        self.bottleneck_5 = cmMambaWithCNN(self.config, img_size)

        self.bottleneck_6 = BottleneckCNN(self.config)
        self.bottleneck_7 = BottleneckCNN(self.config)
        self.bottleneck_8 = BottleneckCNN(self.config)

        # cmMamba block with residual CNN block
        self.bottleneck_9 = cmMambaWithCNN(self.config, img_size)

        # Layer13-Decoder1
        n_downsampling = 2
        i = 0
        mult = 2 ** (n_downsampling - i)
        model = []
        model = [
            nn.ConvTranspose2d(
                2 * ngf * mult,
                int(ngf * mult / 2),
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=use_bias,
            ),
            norm_layer(int(ngf * mult / 2)),
            nn.ReLU(True),
        ]

        setattr(self, "decoder_1", nn.Sequential(*model))

        # Layer14-Decoder2
        i = 1
        mult = 2 ** (n_downsampling - i)
        model = []
        model = [
            nn.ConvTranspose2d(
                2 * ngf * mult,
                int(ngf * mult / 2),
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=use_bias,
            ),
            norm_layer(int(ngf * mult / 2)),
            nn.ReLU(True),
        ]

        setattr(self, "decoder_2", nn.Sequential(*model))

        # Layer15-Decoder3
        model = []
        model = [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(2 * ngf, output_dim, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        setattr(self, "decoder_3", nn.Sequential(*model))

    def forward(self, x):
        # Encoder
        x1 = self.encoder_1(x)
        x2 = self.encoder_2(x1)
        x3 = self.encoder_3(x2)

        # Episodic bottleneck
        x = self.bottleneck_1(x3)
        x = self.bottleneck_2(x)
        x = self.bottleneck_3(x)
        x = self.bottleneck_4(x)
        x = self.bottleneck_5(x)
        x = self.bottleneck_6(x)
        x = self.bottleneck_7(x)
        x = self.bottleneck_8(x)
        x = self.bottleneck_9(x)

        # Decoder
        x = self.decoder_1(torch.cat([x, x3], dim=1))
        x = self.decoder_2(torch.cat([x, x2], dim=1))
        x = self.decoder_3(torch.cat([x, x1], dim=1))
        return x

    def load_from(self, weights):
        with torch.no_grad():

            if self.config.name == "b16":
                self.bottleneck_1.embeddings.patch_embeddings.weight.copy_(
                    np2th(weights["embedding/kernel"], conv=True)
                )
                self.bottleneck_1.embeddings.patch_embeddings.bias.copy_(
                    np2th(weights["embedding/bias"])
                )

                self.bottleneck_6.embeddings.patch_embeddings.weight.copy_(
                    np2th(weights["embedding/kernel"], conv=True)
                )
                self.bottleneck_6.embeddings.patch_embeddings.bias.copy_(
                    np2th(weights["embedding/bias"])
                )

            self.transformer_encoder.encoder_norm.weight.copy_(
                np2th(weights["Transformer/encoder_norm/scale"])
            )
            self.transformer_encoder.encoder_norm.bias.copy_(
                np2th(weights["Transformer/encoder_norm/bias"])
            )

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.bottleneck_1.embeddings.positional_encoding
            if posemb.size() == posemb_new.size():
                self.bottleneck_1.embeddings.positional_encoding.copy_(posemb)
            elif posemb.size()[1] - 1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.bottleneck_1.embeddings.positional_encoding1.copy_(posemb)
            else:
                logger.info(
                    "load_pretrained: resized variant: %s to %s"
                    % (posemb.size(), posemb_new.size())
                )
                ntok_new = posemb_new.size(1)
                _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print("load_pretrained: grid-size from %s to %s" % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.bottleneck_1.embeddings.positional_encoding.copy_(np2th(posemb))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.bottleneck_6.embeddings.positional_encoding
           
            if posemb.size() == posemb_new.size():
                self.bottleneck_6.embeddings.positional_encoding.copy_(posemb)
            elif posemb.size()[1] - 1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.bottleneck_6.embeddings.positional_encoding.copy_(posemb)
            else:
                logger.info(
                    "load_pretrained: resized variant: %s to %s"
                    % (posemb.size(), posemb_new.size())
                )
                ntok_new = posemb_new.size(1)
                _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print("load_pretrained: grid-size from %s to %s" % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.bottleneck_6.embeddings.positional_encoding.copy_(np2th(posemb))

            # Encoder
            for bname, block in self.transformer_encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)


class ResCNN(nn.Module):
    def __init__(self, config, input_dim, img_size=224, output_dim=3, vis=False):
        super(ResCNN, self).__init__()
        self.config = config
        output_nc = output_dim
        ngf = 64
        use_bias = False
        norm_layer = nn.BatchNorm2d
        padding_type = "reflect"
        mult = 4

        # Layer1-Encoder1
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_dim, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
        ]
       
        setattr(self, "encoder_1", nn.Sequential(*model))

        # Layer2-Encoder2
        n_downsampling = 2
        model = []
        i = 0
        mult = 2**i
        model = [
            nn.Conv2d(
                ngf * mult,
                ngf * mult * 2,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=use_bias,
            ),
            norm_layer(ngf * mult * 2),
            nn.ReLU(True),
        ]
       
        setattr(self, "encoder_2", nn.Sequential(*model))

        # Layer3-Encoder3
        model = []
        i = 1
        mult = 2**i
        model = [
            nn.Conv2d(
                ngf * mult,
                ngf * mult * 2,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=use_bias,
            ),
            norm_layer(ngf * mult * 2),
            nn.ReLU(True),
        ]
      
        setattr(self, "encoder_3", nn.Sequential(*model))

        mult = 4
        self.bottleneck_1 = BottleneckCNN(self.config)
        self.bottleneck_2 = BottleneckCNN(self.config)
        self.bottleneck_3 = BottleneckCNN(self.config)
        self.bottleneck_4 = BottleneckCNN(self.config)
        self.bottleneck_5 = BottleneckCNN(self.config)
        self.bottleneck_6 = BottleneckCNN(self.config)
        self.bottleneck_7 = BottleneckCNN(self.config)
        self.bottleneck_8 = BottleneckCNN(self.config)
        self.bottleneck_9 = BottleneckCNN(self.config)

        # Layer13-Decoder1
        n_downsampling = 2
        i = 0
        mult = 2 ** (n_downsampling - i)
        model = []
        model = [
            nn.ConvTranspose2d(
                ngf * mult,
                int(ngf * mult / 2),
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=use_bias,
            ),
            norm_layer(int(ngf * mult / 2)),
            nn.ReLU(True),
        ]
        
        setattr(self, "decoder_1", nn.Sequential(*model))

        # Layer14-Decoder2
        i = 1
        mult = 2 ** (n_downsampling - i)
        model = []
        model = [
            nn.ConvTranspose2d(
                ngf * mult,
                int(ngf * mult / 2),
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=use_bias,
            ),
            norm_layer(int(ngf * mult / 2)),
            nn.ReLU(True),
        ]
       
        setattr(self, "decoder_2", nn.Sequential(*model))

        # Layer15-Decoder3
        model = []
        model = [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_dim, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
      
        setattr(self, "decoder_3", nn.Sequential(*model))

    def forward(self, x):
        # Encoder
        x = self.encoder_1(x)
        x = self.encoder_2(x)
        x = self.encoder_3(x)

        # Information bottleneck
        x = self.bottleneck_1(x)
        x = self.bottleneck_2(x)
        x = self.bottleneck_3(x)
        x = self.bottleneck_4(x)
        x = self.bottleneck_5(x)
        x = self.bottleneck_6(x)
        x = self.bottleneck_7(x)
        x = self.bottleneck_8(x)
        x = self.bottleneck_9(x)
        
        # Decoder
        x = self.decoder_1(x)
        x = self.decoder_2(x)
        x = self.decoder_3(x)
        
        return x
