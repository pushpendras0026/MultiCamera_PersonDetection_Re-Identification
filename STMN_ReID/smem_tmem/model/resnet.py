"""
model/resnet.py – ResNet-50 backbone for STMN with three parallel layer4 branches.

Architecture:
  - Shared stem + layer1/2/3 (from pretrained ResNet-50)
  - Three parallel layer4 branches:
      layer4_val   → value features  (fed to SMM / avgpool)
      layer4_key_s → spatial key     (fed to SMM)
      layer4_key_t → temporal key    (fed to TMM after avgpool)

Fixes applied vs. original:
  - `models.resnet50(pretrained=True)` → `weights=ResNet50_Weights.IMAGENET1K_V1`
    (suppresses FutureWarning in torchvision ≥ 0.13, required in ≥ 0.17)
"""

import os
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights


# ──────────────────────────────────────────────────────────────────────────────
# Basic building blocks
# ──────────────────────────────────────────────────────────────────────────────

class Bottleneck(nn.Module):
    """Standard ResNet Bottleneck block (includes ReLU after residual add)."""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes * 4)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class Bottleneck_key(nn.Module):
    """
    Final block used in the key branches (layer4_key_s, layer4_key_t).
    No ReLU after residual add – this is the STMN design choice so the key
    features can encode negative activations for attention.
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes * 4)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        # ← intentionally NO relu here (key branch)
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Three-branch ResNet
# ──────────────────────────────────────────────────────────────────────────────

class ResNet(nn.Module):
    def __init__(self, last_stride=1,
                 block=Bottleneck, last_block=Bottleneck_key,
                 layers=(3, 4, 6, 3)):
        self.inplanes = 64
        super().__init__()
        self.conv1   = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1     = nn.BatchNorm2d(64)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1  = self._make_layer(block, 64,  layers[0])
        self.layer2  = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3  = self._make_layer(block, 256, layers[2], stride=2)
        # Three parallel layer4 branches
        self.layer4_val   = self._make_layer(block, 512, layers[3], stride=last_stride)
        self.inplanes = 1024
        self.layer4_key_s = self._make_layer_key(block, last_block, 512, layers[3], stride=last_stride)
        self.inplanes = 1024
        self.layer4_key_t = self._make_layer_key(block, last_block, 512, layers[3], stride=last_stride)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _make_layer_key(self, block, last_block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks - 1):
            layers.append(block(self.inplanes, planes))
        layers.append(last_block(self.inplanes, planes, stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        val   = self.layer4_val(x)
        key_s = self.layer4_key_s(x)
        key_t = self.layer4_key_t(x)
        return val, key_s, key_t


# ──────────────────────────────────────────────────────────────────────────────
# Wrapper: load ImageNet weights into the three-branch ResNet
# ──────────────────────────────────────────────────────────────────────────────

class Resnet50(nn.Module):
    def __init__(self, pooling=True, stride=1):
        super().__init__()
        # ← FIX: use Weights API instead of deprecated pretrained=True
        original = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).state_dict()
        self.backbone = ResNet(last_stride=stride)

        # Copy weights from the single-branch pretrained model into all three branches
        cnt = 0
        layer4_val_keys   = self._get_branch_keys('layer4_val')
        layer4_key_s_keys = self._get_branch_keys('layer4_key_s')
        layer4_key_t_keys = self._get_branch_keys('layer4_key_t')

        for key in original:
            if 'fc' in key:
                continue
            if 'layer4' in key:
                self.backbone.state_dict()[layer4_val_keys[cnt]].copy_(original[key])
                self.backbone.state_dict()[layer4_key_s_keys[cnt]].copy_(original[key])
                self.backbone.state_dict()[layer4_key_t_keys[cnt]].copy_(original[key])
                cnt += 1
            else:
                self.backbone.state_dict()[key].copy_(original[key])
        del original

        self.avgpool = nn.AdaptiveAvgPool2d(1) if pooling else None
        self.out_dim = 2048

    def _get_branch_keys(self, branch_name):
        return [k for k in self.backbone.state_dict() if branch_name in k]

    def forward(self, x):
        val, key_s, key_t = self.backbone(x)   # [BS, 2048, H, W]
        if self.avgpool is not None:
            key_t = self.avgpool(key_t)         # [BS, 2048, 1, 1]
            key_t = key_t.view(key_t.shape[0], -1)  # [BS, 2048]
        return val, key_s, key_t
