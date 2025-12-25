import torch.nn as nn
import torch.nn.functional as F

from mmengine.model import BaseModule
from mmdet3d.registry import MODELS

@MODELS.register_module()
class DepthHead(BaseModule):
    def __init__(self, in_channels, feat_channels, depth_num):
        super().__init__()
        self.depth_num = depth_num
        self.feat_channels = feat_channels
        
        self.total_out = feat_channels + depth_num + 1
        
        self.head = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.total_out,
            kernel_size=3,
            padding=1,
            stride=1
        )
        
    def forward(self, x):
        out = self.head(x)
        
        features = out[:, :self.feat_channels, ...]
        
        depth_logits = out[:, self.feat_channels : self.feat_channels + self.depth_num, ...]
        depth = depth_logits.softmax(dim=1) 
        
        opacity = out[:, -1:, ...].sigmoid()
        
        return features, depth, opacity