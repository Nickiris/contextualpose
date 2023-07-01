
from models.layers import  _sigmoid, DropPath
from models.loss import FocalLoss, ContrastiveLoss
from collections import defaultdict

from torch import Tensor
from typing import Tuple, List, Callable, Optional

import os
import math
import torch
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def build_backbone(cfg, is_train):
    model = HRNet(cfg)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED, verbose=cfg.VERBOSE)
    else:
        model.init_weights()

    return model


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1,
                 downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,
                 downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HRModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HRModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
                self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion,
                               momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        nn.BatchNorm2d(num_inchannels[i]),
                        nn.Upsample(scale_factor=2 ** (j - i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3),
                                nn.ReLU(True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck,
}


class HRNet(nn.Module):
    def __init__(self, cfg):
        super(HRNet, self).__init__()

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 64, 4)

        # build stage
        self.spec = cfg.MODEL.BACKBONE
        self.stages_spec = self.spec.STAGES
        self.num_stages = self.spec.STAGES.NUM_STAGES
        num_channels_last = [256]
        for i in range(self.num_stages):
            num_channels = self.stages_spec.NUM_CHANNELS[i]
            transition_layer = \
                self._make_transition_layer(num_channels_last, num_channels)
            setattr(self, 'transition{}'.format(i + 1), transition_layer)

            stage, num_channels_last = self._make_stage(
                self.stages_spec, i, num_channels, True
            )
            setattr(self, 'stage{}'.format(i + 2), stage)

        self.pretrained_layers = self.spec.PRETRAINED_LAYERS
        self.out_channels = num_channels_last

    def _make_head(self, pre_stage_channels):
        head_block = Bottleneck
        head_channels = [32, 64, 128, 256]

        # Increasing the #channels on each resolution
        # from C, 2C, 4C, 8C to 128, 256, 512, 1024
        incre_modules = []
        for i, channels in enumerate(pre_stage_channels):
            incre_module = self._make_layer(head_block,
                                            channels,
                                            head_channels[i],
                                            1,
                                            stride=1)
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)

        # downsampling modules
        downsamp_modules = []
        for i in range(len(pre_stage_channels) - 1):
            in_channels = head_channels[i] * head_block.expansion
            out_channels = head_channels[i + 1] * head_block.expansion

            downsamp_module = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=3,
                          stride=2,
                          padding=1),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            )

            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)

        final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=head_channels[3] * head_block.expansion,
                out_channels=2048,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.BatchNorm2d(2048, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )

        return incre_modules, downsamp_modules, final_layer

    def _make_layer(
            self, block, inplanes, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes,
                            stride, downsample, dilation=dilation))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        nn.BatchNorm2d(num_channels_cur_layer[i]),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outchannels),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_stage(self, stages_spec, stage_index, num_inchannels,
                    multi_scale_output=True):
        num_modules = stages_spec.NUM_MODULES[stage_index]
        num_branches = stages_spec.NUM_BRANCHES[stage_index]
        num_blocks = stages_spec.NUM_BLOCKS[stage_index]
        num_channels = stages_spec.NUM_CHANNELS[stage_index]
        block = blocks_dict[stages_spec['BLOCK'][stage_index]]
        fuse_method = stages_spec.FUSE_METHOD[stage_index]

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HRModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        y_list = [x]
        for i in range(self.num_stages):
            x_list = []
            transition = getattr(self, 'transition{}'.format(i + 1))
            for j in range(self.stages_spec['NUM_BRANCHES'][i]):
                if transition[j]:
                    x_list.append(transition[j](y_list[-1]))
                else:
                    x_list.append(y_list[j])
            y_list = getattr(self, 'stage{}'.format(i + 2))(x_list)

        x0_h, x0_w = y_list[0].size(2), y_list[0].size(3)

        # out = y_list[3]
        # x = torch.cat([y_list[0], \
        #                F.interpolate(y_list[1], size=(x0_h, x0_w), mode='bilinear', align_corners=False), \
        #                F.interpolate(y_list[2], size=(x0_h, x0_w), mode='bilinear', align_corners=False), \
        #                F.interpolate(y_list[3], size=(x0_h, x0_w), mode='bilinear', align_corners=False)], 1)

        return y_list

    def init_weights(self, pretrained='', verbose=True):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if hasattr(m, 'transform_matrix_conv'):
                nn.init.constant_(m.transform_matrix_conv.weight, 0)
                if hasattr(m, 'bias'):
                    nn.init.constant_(m.transform_matrix_conv.bias, 0)
            if hasattr(m, 'translation_conv'):
                nn.init.constant_(m.translation_conv.weight, 0)
                if hasattr(m, 'bias'):
                    nn.init.constant_(m.translation_conv.bias, 0)

        parameters_names = set()
        for name, _ in self.named_parameters():
            parameters_names.add(name)

        buffers_names = set()
        for name, _ in self.named_buffers():
            buffers_names.add(name)

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained,
                                               map_location=lambda storage, loc: storage)
            logger.info('=> loading pretrained model {}'.format(pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                        or self.pretrained_layers[0] is '*':
                    if name in parameters_names or name in buffers_names:
                        if verbose:
                            logger.info(
                                '=> init {} from {}'.format(name, pretrained)
                            )
                        need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)


class PositionalEncodingFourier(nn.Module):
    """
    Positional encoding relying on a fourier kernel matching the one used in the "Attention is all you Need" paper.
    Based on the official XCiT code
        - https://github.com/facebookresearch/xcit/blob/master/xcit.py
    """

    def __init__(self, hidden_dim=32, dim=768, temperature=10000):
        super().__init__()
        self.token_projection = nn.Conv2d(hidden_dim * 2, dim, kernel_size=1)
        self.scale = 2 * math.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.dim = dim
        self.eps = 1e-6

    def forward(self, B: int, H: int, W: int):
        device = self.token_projection.weight.device
        y_embed = torch.arange(1, H+ 1, dtype=torch.float32, device=device).unsqueeze(1).repeat(1, 1, W)
        x_embed = torch.arange(1, W + 1, dtype=torch.float32, device=device).repeat(1, H, 1)
        y_embed = y_embed / (y_embed[:, -1:, :] + self.eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + self.eps) * self.scale
        dim_t = torch.arange(self.hidden_dim, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2) / self.hidden_dim)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack([pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()], dim=4).flatten(3)
        pos_y = torch.stack([pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()], dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = self.token_projection(pos)
        return pos.repeat(B, 1, 1, 1)  # (B, C, H, W)


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
            self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C


class Attention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class CrossAttention(nn.Module):
    def __init__(
            self,
            dim,
            dim_ratio=1.,  # make sure (L1 + L2) * C / 8 > L1 * L2
            num_heads=8,
            qkv_bias=False,
            attn_drop=0.,
            proj_drop=0.,
    ):
        super().__init__()
        self.num_heads = num_heads
        hidden_dim = int(dim * dim_ratio)
        self.hidden_dim = hidden_dim
        head_dim = hidden_dim // num_heads
        self.scale = head_dim ** -0.5

        self.wq = nn.Linear(dim, hidden_dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, hidden_dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape
        # B1C -> B1H(C/H) -> BH1(C/H)
        q = self.wq(q).reshape(B, N, self.num_heads, self.hidden_dim // self.num_heads).permute(0, 2, 1, 3)
        # BNC -> BNH(C/H) -> BHN(C/H)
        k = self.wk(k).reshape(B, -1, self.num_heads, self.hidden_dim // self.num_heads).permute(0, 2, 1, 3)
        # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(v).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class SpatialDecoder(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            hidden_dim=768,
            mlp_ratio=1.,
            qkv_bias=False,
            qk_norm=False,
            drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=hidden_dim,  # int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class SpatialCrossDecoder(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            hidden_dim=512,
            mlp_ratio=1.,
            qkv_bias=False,
            qk_norm=False,
            drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.self_attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.cross_attn = CrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm3 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=hidden_dim,  # int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.ls3 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, q, k, v):
        q = q + self.drop_path1(self.ls1(self.self_attn(self.norm1(q))))
        q = q + self.drop_path2(self.ls2(self.cross_attn(self.norm2(q), k, v)))
        q = q + self.drop_path3(self.ls3(self.mlp(self.norm3(q))))
        return q




class ContextualEncoder(nn.Module):

    def __init__(self, cfg, in_channels, hidden_channels):
        super(ContextualEncoder, self).__init__()

        self.num_joints = cfg.DATASET.NUM_KEYPOINTS
        self.in_channels = in_channels
        self.out_channels = self.num_joints
        self.hidden_channels = hidden_channels
        self.prior_prob = cfg.MODEL.BIAS_PROB
        self.num_center = cfg.DATASET.NUM_CENTER
        self.kernel = cfg.MODEL.KSIZE
        # Down sample the feature map, two conv down for decoupling.
        # Fuse space and channel information..
        self.heat_head = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)

        self.center_head = nn.Conv2d(self.in_channels, self.num_center, 1, 1, 0)

        self.deformable_offset = nn.Sequential(
            nn.Linear(self.in_channels, self.in_channels),
            nn.ReLU(inplace=True),
            # nn.GELU(),
        )
        # self.deformable_head = nn.Linear(self.in_channels, self.num_joints*2)
        self.deformable_head = nn.Linear(self.in_channels, self.kernel * self.kernel * 2)
        self.linear_embed = nn.Linear(self.in_channels, self.hidden_channels)
        self.point_encoding = PositionEmbeddingRandom(num_pos_feats=self.in_channels // 2)
        torch.nn.init.normal_(self.heat_head.weight, std=0.001)
        torch.nn.init.normal_(self.center_head.weight, std=0.001)
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        torch.nn.init.constant_(self.heat_head.bias, bias_value)
        torch.nn.init.constant_(self.center_head.bias, bias_value)

        self.heatmap_loss = FocalLoss()
        self.contrastive_loss = ContrastiveLoss()

        # inference
        self.flip_test = cfg.TEST.FLIP_TEST
        self.max_proposals = cfg.TEST.MAX_PROPOSALS
        self.keypoint_thre = cfg.TEST.KEYPOINT_THRESHOLD
        self.center_pool_kernel = cfg.TEST.CENTER_POOL_KERNEL
        self.pool_thre1 = cfg.TEST.POOL_THRESHOLD1
        self.pool_thre2 = cfg.TEST.POOL_THRESHOLD2

    def forward(self, features, targets=None):
        # features' output size is heatmap size...
        # Decoupling center and heatmap for ONNX...
        multi_heatmap = _sigmoid(self.heat_head(features))
        center_heatmap = _sigmoid(self.center_head(features))
        C, H, W = features.size()[-3:]
        if self.training:
            pred_multi_heatmap = torch.cat([multi_heatmap, center_heatmap], dim=1)
            gt_multi_heatmap = [x['multi_heatmap'].unsqueeze(0).to(features.device) for x in targets]
            gt_multi_heatmap = torch.cat(gt_multi_heatmap, dim=0)
            gt_multi_mask = [x['multi_mask'].unsqueeze(0).to(features.device) for x in targets]
            gt_multi_mask = torch.cat(gt_multi_mask, dim=0)

            multi_heatmap_loss = self.heatmap_loss(pred_multi_heatmap, gt_multi_heatmap, gt_multi_mask)

            contrastive_loss = 0
            total_instances = 0
            instances = defaultdict(list)
            for i in range(features.size(0)):
                if 'instance_coord' not in targets[i]: continue
                # (num_person, 2)
                # location information.
                instance_coord = targets[i]['instance_coord'].to(features.device)
                instance_heatmap = targets[i]['instance_heatmap'].to(features.device)
                instance_mask = targets[i]['instance_mask'].to(features.device)
                instance_imgid = i * torch.ones(instance_coord.size(0), dtype=torch.long).to(features.device)
                # local contextual information
                local_feats, points = self._deformable_sampling(features[i], instance_coord)
                # get center embed
                position_embed = self.point_encoding.forward_with_coords(points, (H, W))
                local_embed = torch.mean(local_feats, dim=1)
                local_feats = self.linear_embed(local_feats + position_embed)
                # local_embed = torch.mean(local_feats, dim=1)
                # Contrastive loss is not stable. This is the cause that convergence is not good for cluster methods...
                contrastive_loss += self.contrastive_loss(local_embed)
                total_instances += instance_coord.size(0)
                instances['instance_imgid'].append(instance_imgid)
                instances['instance_feats'].append(local_feats)
                instances['instance_heatmap'].append(instance_heatmap)
                instances['instance_mask'].append(instance_mask)

            for k, v in instances.items():
                instances[k] = torch.cat(v, dim=0)
            # return multi_heatmap_loss, instances
            return multi_heatmap_loss, contrastive_loss / total_instances, instances
        else:
            instances = {}
            if self.flip_test:
                center_heatmap = center_heatmap.mean(dim=0)

            center_pool = F.avg_pool2d(center_heatmap, self.center_pool_kernel, 1, (self.center_pool_kernel - 1) // 2)
            center_heatmap = (center_heatmap + center_pool) / 2.0
            maxm = self.hierarchical_pool(center_heatmap)
            maxm = torch.eq(maxm, center_heatmap).float()
            center_heatmap = center_heatmap * maxm
            scores = center_heatmap.view(-1)  # 3D ---> 1D
            scores, pos_ind = scores.topk(self.max_proposals, dim=-1)
            select_ind = (scores > (self.keypoint_thre)).nonzero()
            if len(select_ind) > 0:
                scores = scores[select_ind].squeeze(1)
                pos_ind = pos_ind[select_ind].squeeze(1)
                pos_ind = pos_ind % (W * H)
                x = pos_ind % W
                y = (pos_ind / W).long()
                instance_coord = torch.stack((y, x), dim=1)
                local_feats, points = self._deformable_sampling(features[0], instance_coord)
                position_embed = self.point_encoding.forward_with_coords(points, (H, W))
                local_feats = self.linear_embed(local_feats + position_embed)
                instance_imgid = torch.zeros(instance_coord.size(0), dtype=torch.long).to(features.device)
                if self.flip_test:
                    local_feats_flip, points_flip = self._deformable_sampling(features[1], instance_coord)
                    position_embed_flip = self.point_encoding.forward_with_coords(points_flip, (H, W))
                    local_feats_flip = self.linear_embed(local_feats_flip + position_embed_flip)
                    instance_imgid_flip = torch.ones(instance_coord.size(0), dtype=torch.long).to(features.device)
                    local_feats = torch.cat((local_feats, local_feats_flip), dim=0)
                    instance_imgid = torch.cat((instance_imgid, instance_imgid_flip), dim=0)
                instances['instance_imgid'] = instance_imgid
                instances['instance_feats'] = local_feats
                instances['instance_score'] = scores

            return instances

    def _sample_feats(self, features, pos_ind):

        _, height, width = features.shape
        py, px = pos_ind[:, ..., 0], pos_ind[:, ..., 1]
        shape = px.shape
        px = px.reshape(-1)
        py = py.reshape(-1)
        px0 = px.floor().clamp(min=0, max=width - 1)
        py0 = py.floor().clamp(min=0, max=height - 1)
        px1 = (px0 + 1).clamp(min=0, max=width - 1)
        py1 = (py0 + 1).clamp(min=0, max=height - 1)
        px0l, py0l, px1l, py1l = px0.long(), py0.long(), px1.long(), py1.long()
        delta_x0 = (px1 - px).clamp(min=0, max=1.0)
        delta_y0 = (py1 - py).clamp(min=0, max=1.0)
        delta_x1 = (px - px0).clamp(min=0, max=1.0)
        delta_y1 = (py - py0).clamp(min=0, max=1.0)
        features_00 = features[:, py0l, px0l] * delta_y0[None] * delta_x0[None]
        features_01 = features[:, py0l, px1l] * delta_y0[None] * delta_x1[None]
        features_10 = features[:, py1l, px0l] * delta_y1[None] * delta_x0[None]
        features_11 = features[:, py1l, px1l] * delta_y1[None] * delta_x1[None]

        out = features_00 + features_01 + features_10 + features_11
        out = out.permute((1, 0)).contiguous()
        out = out.reshape(*shape, -1)

        return out

    def _deformable_sampling(self, features, points):
        # points: (B, 2), y, x
        # fetures: (C*4, H, W)
        B = len(points)
        h, w = features.size()[-2:]
        max_offset = max(h, w) / 8.
        point_features = self._sample_feats(features, points.float())
        offset_features = self.deformable_offset(point_features)  # (B, C) ---> (B, S*S*2)
        delta_yx = self.deformable_head(offset_features).clamp(-max_offset, max_offset)
        keypoints = points[:, None, :] + delta_yx.reshape(B, -1, 2)
        sampling_feats = self._sample_feats(features, keypoints)
        return sampling_feats, keypoints

    def hierarchical_pool(self, heatmap):
        map_size = (heatmap.shape[1] + heatmap.shape[2]) / 2.0
        if map_size > self.pool_thre1:
            maxm = F.max_pool2d(heatmap, 7, 1, 3)
        elif map_size > self.pool_thre2:
            maxm = F.max_pool2d(heatmap, 5, 1, 2)
        else:
            maxm = F.max_pool2d(heatmap, 3, 1, 1)
        return maxm




class ClusterDecoder(nn.Module):
    def __init__(self, cfg, in_channels, hidden_channels):
        super().__init__()

        self.num_joints = cfg.DATASET.NUM_KEYPOINTS
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.prior_prob = cfg.MODEL.BIAS_PROB
        self.ksize = cfg.MODEL.KSIZE

        self.pose_token = nn.Parameter(torch.zeros(1, self.num_joints, self.hidden_channels))
        self.conv_down = nn.Conv2d(self.in_channels, self.hidden_channels, 1)
        self.heatmap_down = nn.Conv2d(self.in_channels, self.num_joints, 1)
        self.conv_norm = nn.LayerNorm(self.hidden_channels)
        self.position_encoding = PositionEmbeddingRandom(num_pos_feats=self.hidden_channels // 2)
        self.heatmap_conv = nn.Conv2d(self.num_joints, self.num_joints, 7, 1, 3)
        self.norm_feat = nn.LayerNorm(self.hidden_channels)
        self.query_norm = nn.LayerNorm(self.hidden_channels)

        self.heatmap_loss = FocalLoss()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        self.heatmap_conv.bias.data.fill_(bias_value)
        self.contextual_decoder = self._make_decoder(cfg, hidden_channels)

    def _make_decoder(self, cfg, in_channels):
        layers = []
        for i in range(cfg.MODEL.DECODER.NUM_LAYERS):
            layers.append(SpatialCrossDecoder(in_channels, num_heads=cfg.MODEL.DECODER.NUM_HEADS,
                                              mlp_ratio=cfg.MODEL.DECODER.MLP_RATIO,
                                              drop=cfg.MODEL.DECODER.DROPOUT, drop_path=cfg.MODEL.DECODER.DROPOUT))
        return nn.ModuleList(layers)

    def forward(self, features, pixel_tokens, instances):

        contextual_queries = instances['instance_feats']
        contextual_queries = torch.cat((self.pose_token.repeat(len(contextual_queries), 1, 1), contextual_queries),
                                       dim=1)  # N, 17+L, C
        contextual_keys = contextual_values = self.norm_feat(pixel_tokens[instances['instance_imgid']])
        for i, contextual_decoder in enumerate(self.contextual_decoder):
            # contextual_keys = contextual_values = self.norm_feat[i](pixel_tokens[instances['instance_imgid']])
            contextual_queries = contextual_decoder(contextual_queries, contextual_keys, contextual_values)
        # The higher dimensions, the lower results...
        instance_heatmap = self.heatmap_down(features)[instances['instance_imgid']]
        features = self.conv_down(features)
        B, C, H, W = features.size()
        position_embed = self.position_encoding((H, W))
        instance_features = F.normalize(self.conv_norm((features + position_embed).flatten(2).transpose(-1, -2)),
                                        dim=-1, p=2)
        instance_features = instance_features[instances['instance_imgid']]
        pred_instance_heatmaps = _sigmoid(self.heatmap_conv(
            F.relu(instance_heatmap * (F.normalize(self.query_norm(contextual_queries[:, :self.num_joints, :]), dim=-1,
                                                   p=2) @ instance_features.transpose(1, 2)).reshape(-1,
                                                                                                     self.num_joints, H,
                                                                                                     W))))
        if self.training:
            gt_instance_heatmaps = instances['instance_heatmap']
            gt_instance_masks = instances['instance_mask']
            single_heatmap_loss = self.heatmap_loss(pred_instance_heatmaps, gt_instance_heatmaps, gt_instance_masks)
            # Total batch instances use contrastive loss..
            return single_heatmap_loss
        else:
            return pred_instance_heatmaps




class ContextualPose(nn.Module):

    def __init__(self, cfg):
        super(ContextualPose, self).__init__()

        self.num_joints = cfg.DATASET.NUM_KEYPOINTS
        self.max_instances = cfg.DATASET.MAX_INSTANCES
        # inference
        self.flip_test = cfg.TEST.FLIP_TEST
        self.flip_index = cfg.DATASET.FLIP_INDEX
        self.max_proposals = cfg.TEST.MAX_PROPOSALS
        self.keypoint_thre = cfg.TEST.KEYPOINT_THRESHOLD
        self.center_pool_kernel = cfg.TEST.CENTER_POOL_KERNEL
        self.pool_thre1 = cfg.TEST.POOL_THRESHOLD1
        self.pool_thre2 = cfg.TEST.POOL_THRESHOLD2
        self.prior_prob = cfg.MODEL.BIAS_PROB
        self.multi_heatmap_loss_weight = cfg.LOSS.MULTI_HEATMAP_LOSS_WEIGHT
        self.contrastive_loss_weight = cfg.LOSS.CONTRASTIVE_LOSS_WEIGHT
        self.single_heatmap_loss_weight = cfg.LOSS.SINGLE_HEATMAP_LOSS_WEIGHT

        self.backbone = build_backbone(cfg, is_train=True)
        self.level_channels = self.backbone.out_channels
        self.out_channels = 256
        self.position_encoding = PositionEmbeddingRandom(num_pos_feats=self.out_channels // 2)
        self.patch_embed = nn.Identity()
        self.encoder = ContextualEncoder(cfg, sum(self.level_channels), self.out_channels)
        self.decoder = ClusterDecoder(cfg, sum(self.level_channels), self.out_channels)


    def forward(self, images, targets=None):

        # backbone
        features = self.backbone(images)
        h, w = features[0].size()[-2:]
        pixel_features = self.patch_embed(features[-1])
        b, _, h_f, w_f = pixel_features.size()
        # pos_embedding = self.position_encoding(b, h_f, w_f)
        pos_embedding = self.position_encoding((h_f, w_f))
        pixel_features = (pixel_features + pos_embedding).flatten(2).transpose(-1, -2)
        features = torch.cat([features[0], \
                              F.interpolate(features[1], size=(h, w), mode='bilinear', align_corners=False), \
                              F.interpolate(features[2], size=(h, w), mode='bilinear', align_corners=False), \
                              F.interpolate(features[3], size=(h, w), mode='bilinear', align_corners=False)], 1)
        if self.training:
            multi_heatmap_loss, contrastive_loss, instances = self.encoder(features, targets)

            # limit max instances in training
            if 0 <= self.max_instances < instances['instance_feats'].size(0):
                # random choose: the number is max_instance.
                inds = torch.randperm(instances['instance_feats'].size(0), device=features.device).long()
                for k, v in instances.items():
                    instances[k] = v[inds[:self.max_instances]]
            single_heatmap_loss = self.decoder(features, pixel_features, instances)

            losses = {}
            losses.update({'multi_heatmap_loss': multi_heatmap_loss * self.multi_heatmap_loss_weight})
            losses.update({'single_loss': single_heatmap_loss * self.single_heatmap_loss_weight})
            losses.update({'contrastive_loss': contrastive_loss * self.contrastive_loss_weight})
            return losses
        else:
            results = {}
            if self.flip_test:
                features[1, :, :, :] = features[1, :, :, :].flip([2])

            instances = self.encoder(features)
            if len(instances) == 0: return results

            instance_heatmaps = self.decoder(features, pixel_features, instances)
            if self.flip_test:
                instance_heatmaps, instance_heatmaps_flip = torch.chunk(instance_heatmaps, 2, dim=0)
                instance_heatmaps_flip = instance_heatmaps_flip[:, self.flip_index, :, :]
                instance_heatmaps = (instance_heatmaps + instance_heatmaps_flip) / 2.0

            instance_scores = instances['instance_score']
            num_people, num_keypoints, h, w = instance_heatmaps.size()
            center_pool = F.avg_pool2d(instance_heatmaps, self.center_pool_kernel, 1,
                                       (self.center_pool_kernel - 1) // 2)
            instance_heatmaps = (instance_heatmaps + center_pool) / 2.0
            nms_instance_heatmaps = instance_heatmaps.view(num_people, num_keypoints, -1)
            vals, inds = torch.max(nms_instance_heatmaps, dim=2)
            x, y = inds % w, (inds / w).long()
            # shift coords by 0.25
            x, y = self.adjust(x, y, instance_heatmaps)

            vals = vals * instance_scores.unsqueeze(1)
            poses = torch.stack((x, y, vals), dim=2)

            poses[:, :, :2] = poses[:, :, :2] * 4 + 2
            scores = torch.mean(poses[:, :, 2], dim=1)

            results.update({'poses': poses})
            results.update({'scores': scores})

            return results

    def adjust(self, res_x, res_y, heatmaps):
        n, k, h, w = heatmaps.size()  # [2:]

        x_l, x_r = (res_x - 1).clamp(0, w - 1), (res_x + 1).clamp(0, w - 1)
        y_t, y_b = (res_y + 1).clamp(0, h - 1), (res_y - 1).clamp(0, h - 1)
        n_inds = torch.arange(n)[:, None].to(heatmaps.device)
        k_inds = torch.arange(k)[None].to(heatmaps.device)

        px = torch.sign(heatmaps[n_inds, k_inds, res_y, x_r] - heatmaps[n_inds, k_inds, res_y, x_l]) * 0.25
        py = torch.sign(heatmaps[n_inds, k_inds, y_t, res_x] - heatmaps[n_inds, k_inds, y_b, res_x]) * 0.25

        res_x, res_y = res_x.float(), res_y.float()
        x_l, x_r = x_l.float(), x_r.float()
        y_b, y_t = y_b.float(), y_t.float()
        px = px * torch.sign(res_x - x_l) * torch.sign(x_r - res_x)
        py = py * torch.sign(res_y - y_b) * torch.sign(y_t - res_y)

        res_x = res_x.float() + px
        res_y = res_y.float() + py

        return res_x, res_y


if __name__ == "__main__":

    from config.default import _C
    from torchinfo import summary
    import time
    import torchvision

    model = ContextualPose(_C)
    model.eval()
    x = torch.randn(1, 3, 512, 512)
    if _C.TEST.FLIP_TEST:
        flip_image = torch.flip(x, [2])
        x = torch.cat([x, flip_image], dim=0)
    t1 = time.perf_counter()
    z = model(x)
    print(z['poses'].shape, z['scores'].shape)
    t2 = time.perf_counter()
    summary(model, input_size=(len(x), 3, 512, 512))
    print(1000 * (t2 - t1))