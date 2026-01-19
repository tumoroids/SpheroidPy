"""Utilities for HRNet model loading and weight management."""

import logging
import os
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("SpheroidPy.utils.hrnet_utils")


# --------------------------
# Download Progress Bar
# --------------------------
class DownloadProgressBar:
    """Progress bar for downloading files."""
    
    def __init__(self):
        try:
            from tqdm import tqdm
            self.tqdm = tqdm
            self.pbar = None
        except ImportError:
            self.tqdm = None
            self.pbar = None
    
    def __enter__(self):
        if self.tqdm:
            self.pbar = self.tqdm(
                unit="B", 
                unit_scale=True, 
                miniters=1, 
                desc="Downloading HRNet model weights"
            )
        return self
    
    def __exit__(self, *args):
        if self.pbar:
            self.pbar.close()
    
    def update_to(self, b=1, bsize=1, tsize=None):
        """Update progress bar.
        
        Args:
            b: Number of blocks transferred
            bsize: Block size
            tsize: Total size (if known)
        """
        if self.pbar:
            if tsize is not None:
                self.pbar.total = tsize
            self.pbar.update(b * bsize - self.pbar.n)


# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = None
    F = None
    torch = None


# --------------------------
# HRNet Components (based on https://github.com/HRNet)
# --------------------------
BN_MOMENTUM = 0.01


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module if TORCH_AVAILABLE else object):
    """Basic residual block for HRNet."""
    
    expansion = 1

    def __init__(self, inplanes, planes, norm_layer, stride=1, downsample=None):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for HRNet segmentation.")
        super().__init__()
        self.norm_layer = norm_layer

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = self.norm_layer(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = self.norm_layer(planes, momentum=BN_MOMENTUM)
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


class Bottleneck(nn.Module if TORCH_AVAILABLE else object):
    """Bottleneck block for HRNet."""
    
    expansion = 4

    def __init__(self, inplanes, planes, norm_layer, stride=1, downsample=None):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for HRNet segmentation.")
        super().__init__()

        self.norm_layer = norm_layer

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = self.norm_layer(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = self.norm_layer(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = self.norm_layer(planes * self.expansion,
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


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


# --------------------------
# HRNet High-Resolution Module
# --------------------------
class HighResolutionModule(nn.Module if TORCH_AVAILABLE else object):
    """High-Resolution Module for multi-scale feature fusion."""
    
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, norm_layer, multi_scale_output=True):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for HRNet segmentation.")
        super().__init__()

        self.norm_layer = norm_layer

        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

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
                self.norm_layer(num_channels[branch_index] * block.expansion,
                                momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], self.norm_layer, stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index], self.norm_layer))

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
                        self.norm_layer(num_inchannels[i], momentum=BN_MOMENTUM)))
                elif j == i:
                    fuse_layer.append(nn.Identity())
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                self.norm_layer(num_outchannels_conv3x3, 
                                                momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                self.norm_layer(num_outchannels_conv3x3,
                                                momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i, branch in enumerate(self.branches):
            x[i] = branch(x[i])

        x_fuse = []
        for i, fuse_layer in enumerate(self.fuse_layers):
            y = x[0] if i == 0 else fuse_layer[0](x[0])
            for j, fuse_sub_layer in enumerate(fuse_layer):
                if j == 0 or j > self.num_branches:
                    pass
                else:
                    if i == j:
                        y = y + x[j]
                    elif j > i:
                        width_output = x[i].shape[-1]
                        height_output = x[i].shape[-2]
                        y = y + F.interpolate(
                            fuse_sub_layer(x[j]),
                            size=[height_output, width_output],
                            mode='bilinear')
                    else:
                        y = y + fuse_sub_layer(x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


# --------------------------
# HRNet-W30 Backbone
# --------------------------
class HighResolutionNet(nn.Module if TORCH_AVAILABLE else object):
    """HRNet-W30 backbone network (based on https://github.com/HRNet)."""

    def __init__(self, config, norm_layer, **kwargs):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for HRNet segmentation.")
        super().__init__()

        self.norm_layer = norm_layer
        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = self.norm_layer(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = self.norm_layer(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        
        self.stage1_cfg = config['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion*num_channels

        self.stage2_cfg = config['STAGE2']

        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = config['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = config['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)
        
        self.last_inp_channels = int(np.sum(pre_stage_channels))

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
                        self.norm_layer(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(nn.Identity())
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        self.norm_layer(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self.norm_layer(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, self.norm_layer, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, self.norm_layer))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(HighResolutionModule(num_branches,block,num_blocks,num_inchannels,num_channels,fuse_method,self.norm_layer,reset_multi_scale_output))
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.ModuleList(modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for aux in self.transition1:
            if not isinstance(aux, nn.Identity):
                x_list.append(aux(x))
            else:
                x_list.append(x)

        for aux in self.stage2:
            x_list = aux(x_list)
        y_list = x_list

        x_list = []
        for i, aux in enumerate(self.transition2):
            if not isinstance(aux, nn.Identity):
                x_list.append(aux(y_list[-1]))
            else:
                x_list.append(y_list[i])
        for aux in self.stage3:
            x_list = aux(x_list)
        y_list = x_list

        x_list = []
        for i, aux in enumerate(self.transition3):
            if not isinstance(aux, nn.Identity):
                x_list.append(aux(y_list[-1]))
            else:
                x_list.append(y_list[i])
        for aux in self.stage4:
            x_list = aux(x_list)
        x = x_list
        # Upsampling (match original: mode='bilinear' without align_corners)
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear')
        x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear')
        x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear')

        x = torch.cat([x[0], x1, x2, x3], 1)

        return x


# HRNet-W30 Configuration
HRNET_W30_CONFIG = {
    "STAGE1": {
        "NUM_MODULES": 1,
        "NUM_BRANCHES": 1,
        "BLOCK": "BOTTLENECK",
        "NUM_BLOCKS": [4],
        "NUM_CHANNELS": [64],
        "FUSE_METHOD": "SUM"
    },
    "STAGE2": {
        "NUM_MODULES": 1,
        "NUM_BRANCHES": 2,
        "BLOCK": "BASIC",
        "NUM_BLOCKS": [4, 4],
        "NUM_CHANNELS": [30, 60],
        "FUSE_METHOD": "SUM"
    },
    "STAGE3": {
        "NUM_MODULES": 4,
        "NUM_BRANCHES": 3,
        "BLOCK": "BASIC",
        "NUM_BLOCKS": [4, 4, 4],
        "NUM_CHANNELS": [30, 60, 120],
        "FUSE_METHOD": "SUM"
    },
    "STAGE4": {
        "NUM_MODULES": 3,
        "NUM_BRANCHES": 4,
        "BLOCK": "BASIC",
        "NUM_BLOCKS": [4, 4, 4, 4],
        "NUM_CHANNELS": [30, 60, 120, 240],
        "FUSE_METHOD": "SUM"
    }
}


# --------------------------
# HRNet-W30 Segmentation Model
# --------------------------
# Based on SpheroidPy/semtorch/models/archs/hrnet_seg.py
# Pure PyTorch implementation without PIL/fastai dependencies
class HRNetSeg(nn.Module if TORCH_AVAILABLE else object):
    """HRNet-W30 segmentation model for spheroid detection.
    
    This implementation matches the architecture in hrnet_seg.py:
    - Uses HighResolutionNet backbone with W30 configuration
    - Segmentation head with BatchNorm and ReLU
    - Final interpolation to original input size
    
    The model expects 3-channel RGB input (converted from grayscale if needed).
    """
    
    def __init__(self, num_classes=2, backbone_name="hrnet_w30", pretrained=False):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for HRNet segmentation.")
        super().__init__()

        self.backbone_name = backbone_name
        self.nclass = num_classes
        
        # Create backbone (HRNet-W30)
        if backbone_name == "hrnet_w30":
            config = HRNET_W30_CONFIG
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}. Only 'hrnet_w30' is supported.")
        
        self.backbone = HighResolutionNet(config=config, norm_layer=nn.BatchNorm2d)
        
        # Segmentation head (matches hrnet_seg.py)
        self.head = nn.Sequential(
            nn.Conv2d(
                in_channels=self.backbone.last_inp_channels,
                out_channels=self.backbone.last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.BatchNorm2d(self.backbone.last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=self.backbone.last_inp_channels,
                out_channels=num_classes,
                kernel_size=1,
                stride=1,
                padding=0)
        )

    def forward(self, x):
        ori_height, ori_width = x.shape[2], x.shape[3]
        x = self.backbone(x)
        x = self.head(x)
        # Match original implementation: mode='bilinear' without align_corners parameter
        x = F.interpolate(x, size=(ori_height, ori_width), mode='bilinear') 
        return x


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """Preprocess image for HRNet inference.
    
    Normalizes image to [0, 1] range and converts to RGB (3 channels).
    Compatible with the original Deep-Tumour-Spheroid preprocessing.
    
    Args:
        img: Input image as numpy array (H, W) or (H, W, C)
        
    Returns:
        Preprocessed image with shape (3, H, W) normalized to [0, 1]
        (3 channels for RGB compatibility with the original model)
    """
    # Convert to float32
    img = img.astype(np.float32)
    
    # Handle multi-channel images
    if img.ndim == 3:
        if img.shape[2] == 3:
            # Already RGB, use as is
            pass
        elif img.shape[2] > 3:
            # More than 3 channels, take first 3
            img = img[:, :, :3]
        else:
            # Single channel, convert to grayscale first
            img = img.mean(axis=2)
    elif img.ndim == 2:
        # Already grayscale
        pass
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")
    
    # Convert grayscale to RGB by duplicating channels (like color.grey2rgb)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=2)  # (H, W) -> (H, W, 3)
    
    # Normalize to [0, 1] range
    img_min = img.min()
    img_max = img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        # Handle edge case where image is constant
        img = np.zeros_like(img)
    
    # Transpose to (C, H, W) format: (H, W, 3) -> (3, H, W)
    return img.transpose(2, 0, 1)


def preprocess_image_torch(img: np.ndarray | torch.Tensor, device: str = "cpu", already_rescaled: bool = False, use_imagenet_norm: bool = True) -> torch.Tensor:
    """Optimized torch-native preprocessing for HRNet inference.
    
    Performs all preprocessing operations in PyTorch, avoiding numpy-CPU-torch
    conversions. This is faster and allows GPU acceleration.
    
    Matches the original Deep-Tumour-Spheroid preprocessing:
    - Grayscale → RGB (duplicate channels)
    - Intensity rescaling (applied before this function if needed)
    - Normalize to [0, 1] range (ToTensor equivalent)
    - ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    Args:
        img: Input image as numpy array or torch.Tensor (H, W) or (H, W, C)
             Should already have intensity rescaling applied if needed (for fluorescence)
        device: Target device ('cpu' or 'cuda')
        already_rescaled: If True, assumes image is already in [0, 255] range and divides by 255
                         (like ToTensor). If False, uses min/max normalization.
        use_imagenet_norm: If True, applies ImageNet normalization. If False, only normalizes to [0, 1].
        
    Returns:
        Preprocessed image as torch.Tensor with shape (1, 3, H, W) normalized with ImageNet stats
        (batch dimension included, ready for model input)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for torch-native preprocessing.")
    
    # Convert numpy to torch tensor immediately (single conversion)
    if isinstance(img, np.ndarray):
        img_t = torch.from_numpy(img).to(device=device, dtype=torch.float32)
    elif isinstance(img, torch.Tensor):
        img_t = img.to(device=device, dtype=torch.float32)
    else:
        img_t = torch.tensor(img, device=device, dtype=torch.float32)
    
    # Handle multi-channel images
    if img_t.ndim == 3:
        if img_t.shape[2] == 3:
            # Already RGB, permute to (H, W, 3) -> (3, H, W)
            img_t = img_t.permute(2, 0, 1)
        elif img_t.shape[2] > 3:
            # More than 3 channels, take first 3
            img_t = img_t[:, :, :3].permute(2, 0, 1)
        else:
            # Single channel, convert to grayscale first
            img_t = img_t.mean(dim=2)
            # Convert to RGB by duplicating (like color.grey2rgb)
            img_t = img_t.unsqueeze(0).repeat(3, 1, 1)  # (H, W) -> (3, H, W)
    elif img_t.ndim == 2:
        # Already grayscale, convert to RGB by duplicating channels (like color.grey2rgb)
        img_t = img_t.unsqueeze(0).repeat(3, 1, 1)  # (H, W) -> (3, H, W)
    else:
        raise ValueError(f"Unsupported image shape: {img_t.shape}")
    
    # Normalize to [0, 1] range (ToTensor equivalent)
    if already_rescaled:
        # If already rescaled to [0, 255], divide by 255 (like ToTensor)
        # This matches the original: exposure.rescale_intensity → ToTensor (divide by 255)
        img_t = img_t / 255.0
    else:
        # For raw images, use min/max normalization
        img_min = img_t.min()
        img_max = img_t.max()
        if img_max > img_min:
            img_t = (img_t - img_min) / (img_max - img_min + 1e-12)  # [0, 1]
        else:
            # Handle edge case where image is constant
            img_t = torch.zeros_like(img_t)
    
    # Apply ImageNet normalization (matches Deep-Tumour-Spheroid predict.py)
    # Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
    # Note: For fluorescence images, ImageNet normalization might not be appropriate
    if use_imagenet_norm:
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
        img_t = (img_t - mean) / std
    
    # Add batch dimension: (3, H, W) -> (1, 3, H, W)
    img_t = img_t.unsqueeze(0)
    
    return img_t


def rescale_intensity_torch(img_t: torch.Tensor, out_range: tuple[float, float] = (0.0, 255.0)) -> torch.Tensor:
    """Torch-native intensity rescaling (replaces skimage.exposure.rescale_intensity).
    
    Rescales tensor values from their current range to the specified output range.
    Equivalent to skimage.exposure.rescale_intensity but operates entirely on torch tensors.
    
    Args:
        img_t: Input tensor (any shape)
        out_range: Target range (min, max) for output values
        
    Returns:
        Rescaled tensor with values in [out_range[0], out_range[1]]
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for torch-native rescaling.")
    
    img_min = img_t.min()
    img_max = img_t.max()
    out_min, out_max = out_range
    
    if img_max > img_min:
        # Linear rescaling: out = (img - img_min) / (img_max - img_min) * (out_max - out_min) + out_min
        img_t = (img_t - img_min) / (img_max - img_min + 1e-12) * (out_max - out_min) + out_min
    else:
        # Constant image: set to out_min
        img_t = torch.full_like(img_t, out_min)
    
    return img_t


def download_hrnet_weights(
    weights_dir: Path, 
    weight_file: str = "HRNet Seg.pth"
) -> Path:
    """Download HRNet model weights if they don't exist.
    
    Args:
        weights_dir: Directory where weights should be stored
        weight_file: Name of the weight file
        
    Returns:
        Path to the downloaded weight file
        
    Raises:
        FileNotFoundError: If download fails
    """
    weight_path = weights_dir / weight_file
    
    # If weights already exist, return path
    if weight_path.exists():
        logger.info(f"HRNet model weights found at: {weight_path}")
        return weight_path
    
    # URL for downloading weights
    url = "https://dl.dropboxusercontent.com/s/b7ssl9a3wxezahx/HRNet%20Seg.pth?dl=0"
    
    # Create weights directory if it doesn't exist
    os.makedirs(weights_dir, exist_ok=True)
    
    # Download weights file
    logger.info(f"Downloading HRNet model weights from: {url}")
    try:
        with DownloadProgressBar() as progress:
            urllib.request.urlretrieve(
                url, 
                filename=str(weight_path), 
                reporthook=progress.update_to
            )
        logger.info(f"Downloaded HRNet weights to: {weight_path}")
    except Exception as e:
        raise FileNotFoundError(f"Failed to download HRNet weights from {url}: {e}")
    
    if not weight_path.exists():
        raise FileNotFoundError(f"Weight file not found after download: {weight_path}")
    
    return weight_path


def get_hrnet_weights_path(weights_dir: Optional[Path] = None, 
                           weight_file: str = "HRNet Seg.pth") -> Path:
    """Get path to HRNet model weights.
    
    Automatically downloads weights if they don't exist.
    
    Args:
        weights_dir: Optional directory where weights should be stored.
                    If None, uses package-relative path.
        weight_file: Name of the weight file
        
    Returns:
        Path to the weight file
        
    Raises:
        FileNotFoundError: If weights file doesn't exist and download fails
    """
    if weights_dir is None:
        # Use package-relative path: SpheroidPy/weights/
        weights_dir = Path(__file__).parent.parent / 'weights'
    
    weight_path = weights_dir / weight_file
    
    # Automatically download if weights don't exist
    if not weight_path.exists():
        try:
            weight_path = download_hrnet_weights(weights_dir, weight_file)
        except Exception as e:
            raise FileNotFoundError(
                f"HRNet model weights not found at: {weight_path}\n"
                f"Automatic download failed: {e}\n"
                f"Please ensure the weights file '{weight_file}' is placed in: {weights_dir}"
            )
    
    return weight_path


def load_hrnet_model(weights_path: Optional[Path | str] = None,
                     weights_dir: Optional[Path] = None,
                     weight_file: str = "HRNet Seg.pth",
                     force_state_dict: bool = False):
    """Load HRNet model from weights file.
    
    Supports both TorchScript (.pth with torch.jit) and state_dict formats.
    Automatically downloads weights if they don't exist.
    
    Args:
        weights_path: Direct path to weights file (takes precedence)
        weights_dir: Directory where weights should be stored (if weights_path not provided)
        weight_file: Name of the weight file (if weights_path not provided)
        force_state_dict: If True, skip TorchScript loading and directly load as state_dict
        
    Returns:
        Loaded HRNet model in eval mode (either TorchScript or HRNetSeg instance)
        
    Raises:
        ImportError: If PyTorch is not installed
        FileNotFoundError: If weights file doesn't exist and download fails
        RuntimeError: If model loading fails
    """
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is not installed. Please install PyTorch to use HRNet segmentation."
        )
    
    # Determine weights path
    if weights_path is None:
        # get_hrnet_weights_path will automatically download if needed
        weights_path = get_hrnet_weights_path(weights_dir, weight_file)
    else:
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"HRNet weights file not found: {weights_path}")
    
    # If force_state_dict is True, skip TorchScript and load directly as state_dict
    if force_state_dict:
        logger.info(f"Loading HRNet weights as state_dict (force_state_dict=True) from: {weights_path}")
        state_dict = torch.load(str(weights_path), map_location="cpu")
        
        if not isinstance(state_dict, dict):
            raise RuntimeError(
                    f"Expected state_dict (dict), but got {type(state_dict)}. "
                    f"Set force_state_dict=False to try TorchScript loading."
                )
        
        model = HRNetSeg(num_classes=2, backbone_name="hrnet_w30")
        model_keys = set(model.state_dict().keys())
        state_dict_keys = set(state_dict.keys())
        
        missing_keys = model_keys - state_dict_keys
        unexpected_keys = state_dict_keys - model_keys
        
        if missing_keys:
            logger.warning(f"Missing keys in state_dict: {list(missing_keys)[:10]}...")
        if unexpected_keys:
            logger.warning(f"Unexpected keys in state_dict: {list(unexpected_keys)[:10]}...")
        
        try:
            model.load_state_dict(state_dict, strict=True)
            logger.info(f"HRNet-W30 model (state_dict) loaded successfully with strict=True")
        except RuntimeError as e:
            logger.warning(f"Strict loading failed: {e}. Trying with strict=False...")
            filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
            model.load_state_dict(filtered_state_dict, strict=False)
            logger.info(f"HRNet-W30 model (state_dict) loaded with strict=False")
            logger.warning(f"Loaded {len(filtered_state_dict)}/{len(state_dict)} keys from state_dict")
        
        model.eval()
        return model
    
    # Try to load as TorchScript first (most common format for HRNet-W30 weights)
    # Note: The weights in weights/hrnet_seg.pth are for HRNet-W30 and are typically saved as TorchScript
    try:
        model = torch.jit.load(str(weights_path), map_location="cpu")
        model.eval()
        logger.info(f"HRNet TorchScript model (W30) loaded from: {weights_path}")
        return model
    except Exception as jit_error:
        # If TorchScript loading fails, try loading as state_dict
        logger.debug(f"TorchScript loading failed, trying state_dict: {jit_error}")
        try:
            # Load state_dict first to check format
            state_dict = torch.load(str(weights_path), map_location="cpu")
            
            if not isinstance(state_dict, dict):
                raise RuntimeError(
                    f"Unexpected format in weights file. Expected TorchScript or state_dict, "
                    f"got {type(state_dict)}"
                )
            
            # Create model architecture (HRNet-W30)
            # The architecture must match the state_dict keys exactly
            model = HRNetSeg(num_classes=2, backbone_name="hrnet_w30")
            
            # Check if state_dict keys match model keys
            model_keys = set(model.state_dict().keys())
            state_dict_keys = set(state_dict.keys())
            
            missing_keys = model_keys - state_dict_keys
            unexpected_keys = state_dict_keys - model_keys
            
            if missing_keys:
                logger.warning(f"Missing keys in state_dict: {list(missing_keys)[:10]}...")
            if unexpected_keys:
                logger.warning(f"Unexpected keys in state_dict: {list(unexpected_keys)[:10]}...")
            
            # Try to load with strict=True first
            try:
                model.load_state_dict(state_dict, strict=True)
                logger.info(f"HRNet-W30 model (state_dict) loaded successfully with strict=True from: {weights_path}")
            except RuntimeError as e:
                logger.warning(f"Strict loading failed: {e}. Trying with strict=False...")
                # Filter state_dict to only include keys that exist in model
                filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
                model.load_state_dict(filtered_state_dict, strict=False)
                logger.info(f"HRNet-W30 model (state_dict) loaded with strict=False from: {weights_path}")
                logger.warning(f"Loaded {len(filtered_state_dict)}/{len(state_dict)} keys from state_dict")
            
            model.eval()
            return model
            
        except Exception as state_dict_error:
            raise RuntimeError(
                f"Failed to load HRNet-W30 weights from {weights_path}.\n"
                f"Tried TorchScript: {jit_error}\n"
                f"Tried state_dict: {state_dict_error}\n"
                f"Note: If you're using state_dict weights, ensure they match the HRNet-W30 architecture.\n"
                f"The state_dict should contain keys like 'backbone.conv1.weight', 'head.0.weight', etc."
            )


def get_hrnet_predictor(weights_path: Optional[Path | str] = None,
                        weights_dir: Optional[Path] = None,
                        weight_file: str = "HRNet Seg.pth") -> HRNetSeg:
    """Get HRNet predictor (alias for load_hrnet_model for consistency with detectron_utils).
    
    Args:
        weights_path: Direct path to weights file (takes precedence)
        weights_dir: Directory where weights should be stored (if weights_path not provided)
        weight_file: Name of the weight file (if weights_path not provided)
        
    Returns:
        Loaded HRNetSeg model in eval mode
    """
    return load_hrnet_model(weights_path, weights_dir, weight_file)

