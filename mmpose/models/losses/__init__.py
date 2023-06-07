# Copyright (c) OpenMMLab. All rights reserved.
from .classfication_loss import BCELoss
from .heatmap_loss import AdaptiveWingLoss, FocalHeatmapLoss
from .mesh_loss import GANLoss, MeshLoss
from .mse_loss import JointsMSELoss, JointsOHKMMSELoss, HuberLoss
from .multi_loss_factory import AELoss, HeatmapLoss, MultiLossFactory
from .regression_loss import (BoneLoss, L1Loss, MPJPELoss, MSELoss, RLELoss,
                              SemiSupervisionLoss, SmoothL1Loss,
                              SoftWeightSmoothL1Loss, SoftWingLoss, WingLoss, WingLoss_tril, SoftWingLoss_tril)

__all__ = [
    'JointsMSELoss', 'JointsOHKMMSELoss', 'HeatmapLoss', 'AELoss',
    'MultiLossFactory', 'MeshLoss', 'GANLoss', 'SmoothL1Loss', 'WingLoss',
    'MPJPELoss', 'MSELoss', 'L1Loss', 'BCELoss', 'BoneLoss',
    'SemiSupervisionLoss', 'SoftWingLoss', 'AdaptiveWingLoss', 'RLELoss',
    'SoftWeightSmoothL1Loss', 'FocalHeatmapLoss', 'WingLoss_tril', 'HuberLoss', 'SoftWingLoss_tril'
]
