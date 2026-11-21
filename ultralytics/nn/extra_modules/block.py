# Weed-YOLOv12 🚀, AGPL-3.0 license
"""Extra blocks used by Weed-YOLOv12 (ported from DEAL-YOLO, ultralytics/nn/extra_modules/block.py)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.conv import Conv

__all__ = ("ScalSeq", "Add")

######################################## Scale Sequence Feature Fusion start ########################################


class ScalSeq(nn.Module):
    """
    Scale Sequence Feature Fusion (SSFF) from ASF-YOLO, as used in DEAL-YOLO.

    Takes three feature maps, projects them to `channel` with 1x1 Convs, resizes them to the size of the FIRST input,
    stacks them along a new scale axis and fuses them with a 1x1x1 Conv3d + BN + LeakyReLU followed by a max-pool over
    the three scales. The output has the spatial size of x[0] and `channel` channels, so it can be summed (`Add`) with
    the feature that goes into the detection head of that level.

    Weed-YOLOv12 change: DEAL-YOLO only ever upsamples the 2nd/3rd input (they are always coarser than x[0]).
    Weed-YOLOv12 also anchors SSFF at P3 and P4, where one input is FINER than x[0]; such inputs are pooled down with
    adaptive max + avg pooling (as in ASF's Zoom_cat) before their 1x1 Conv. For coarser inputs the behaviour is
    exactly DEAL-YOLO's (1x1 Conv, then nearest upsampling).

    Args:
        inc (list[int]): Channels of the three inputs; inc[0] is the target (head) level.
        channel (int): Output channels (must equal the channels of the head feature it is added to).
    """

    def __init__(self, inc, channel):
        """Initialize SSFF with the input channel list and the fused output channel."""
        super().__init__()
        if channel != inc[0]:
            self.conv0 = Conv(inc[0], channel, 1)
        self.conv1 = Conv(inc[1], channel, 1)
        self.conv2 = Conv(inc[2], channel, 1)
        self.conv3d = nn.Conv3d(channel, channel, kernel_size=(1, 1, 1))
        self.bn = nn.BatchNorm3d(channel)
        self.act = nn.LeakyReLU(0.1)
        self.pool_3d = nn.MaxPool3d(kernel_size=(3, 1, 1))

    @staticmethod
    def _align(conv, x, size):
        """1x1 Conv `x` and resize it to `size`: finer maps are pooled before the Conv, coarser maps upsampled after."""
        if x.shape[2] > size[0]:  # finer level (e.g. P2 -> P3)
            return conv(F.adaptive_max_pool2d(x, size) + F.adaptive_avg_pool2d(x, size))
        return F.interpolate(conv(x), size, mode="nearest")  # coarser level (DEAL-YOLO path)

    def forward(self, x):
        """Fuse the three scales into one feature map with the size of x[0]."""
        p3, p4, p5 = x[0], x[1], x[2]  # p3 = target level, p4/p5 = the two other levels
        if hasattr(self, "conv0"):
            p3 = self.conv0(p3)
        p4_2 = self._align(self.conv1, p4, p3.size()[2:])
        p5_2 = self._align(self.conv2, p5, p3.size()[2:])
        p3_3d = torch.unsqueeze(p3, -3)
        p4_3d = torch.unsqueeze(p4_2, -3)
        p5_3d = torch.unsqueeze(p5_2, -3)
        combine = torch.cat([p3_3d, p4_3d, p5_3d], dim=2)
        conv_3d = self.conv3d(combine)
        bn = self.bn(conv_3d)
        act = self.act(bn)
        x = self.pool_3d(act)
        x = torch.squeeze(x, 2)
        return x


class Add(nn.Module):
    """Element-wise sum of a list of tensors with identical shapes (combines SSFF output with the head feature)."""

    def __init__(self):
        """Initialize Add."""
        super().__init__()

    def forward(self, x):
        """Sum the input tensors."""
        return torch.sum(torch.stack(x, dim=0), dim=0)


######################################## Scale Sequence Feature Fusion end ########################################
