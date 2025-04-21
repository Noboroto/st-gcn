import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """Temporal attention module for ST-GCN.

    This module computes attention weights along the temporal dimension
    to focus on the most relevant frames in a sequence.

    Args:
        in_channels (int): Number of input channels
        reduction (int): Channel reduction factor for the attention mechanism
    """

    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(
            (None, 1))  # Pool across node dimension
        mid_channels = max(in_channels // reduction, 8)

        self.fc = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=1),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (N, C, T, V)
        batch_size, channels, time_len, num_joints = x.size()
        y = self.avg_pool(x)  # (N, C, T, 1)
        y = y.squeeze(-1)  # (N, C, T)
        y = self.fc(y)  # (N, C, T)
        y = y.unsqueeze(-1)  # (N, C, T, 1)

        # Apply attention weights
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """Spatial attention module for ST-GCN.

    This module computes attention weights for each joint node
    to focus on the most relevant joints for a given action.

    Args:
        in_channels (int): Number of input channels
        reduction (int): Channel reduction factor for the attention mechanism
    """

    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(
            (1, None))  # Pool across temporal dimension
        mid_channels = max(in_channels // reduction, 8)

        self.fc = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=1),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (N, C, T, V)
        batch_size, channels, time_len, num_joints = x.size()
        y = self.avg_pool(x)  # (N, C, 1, V)
        y = y.squeeze(2)  # (N, C, V)
        y = self.fc(y)  # (N, C, V)
        y = y.unsqueeze(2)  # (N, C, 1, V)

        # Apply attention weights
        return x * y.expand_as(x)


class STAttention(nn.Module):
    """Combined Spatial-Temporal attention module.

    This module applies both spatial and temporal attention in sequence.

    Args:
        in_channels (int): Number of input channels
        reduction (int): Channel reduction factor for the attention mechanism
    """

    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.spatial_attn = SpatialAttention(in_channels, reduction)
        self.temporal_attn = TemporalAttention(in_channels, reduction)

    def forward(self, x):
        x = self.spatial_attn(x)
        x = self.temporal_attn(x)
        return x
