import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor
import torch.nn as nn 
import torch
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
import torch.utils.checkpoint as cp
class EfficientNetV2s(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.efficientnet_v2_s(weights='DEFAULT')
        return_nodes = {
            'features.2.3.add': 'p2',
            'features.3.3.add': 'p3',
            'features.5.8.add': 'p4',
            'features.7':       'p5',
        }
        self.feature_extractor = create_feature_extractor(self.model, return_nodes=return_nodes)

    def extract_feature(self, x):
        return self.feature_extractor(x)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, lateral_channels=256, out_channels=128):
        super().__init__()
        self.lateral_conv = nn.Conv2d(in_channels, lateral_channels, kernel_size=1)
        
        self.refine_conv = nn.Sequential(
            nn.Conv2d(lateral_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, top_down_feature=None):
        lateral = self.lateral_conv(x)
        
        if top_down_feature is not None:
            top_down_feature = F.interpolate(
                top_down_feature, size=lateral.shape[-2:], mode='nearest'
            )
            lateral = lateral + top_down_feature
        
        out = self.refine_conv(lateral)
        

        return out, lateral

@MODELS.register_module()
class EfficientNetV2sEncoder(BaseModule):
    def __init__(self, init_cfg=None, with_cp=False):
        super().__init__(init_cfg=init_cfg)
        self.effnet = EfficientNetV2s()
        self.with_cp = with_cp
        self.b5 = BasicBlock(in_channels=1280, lateral_channels=256, out_channels=128)
        self.b4 = BasicBlock(in_channels=160,  lateral_channels=256, out_channels=128)
        self.b3 = BasicBlock(in_channels=64,   lateral_channels=256, out_channels=128)
        self.b2 = BasicBlock(in_channels=48,   lateral_channels=256, out_channels=128)
    
        self.downsample = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
    
    def _forward_impl(self, x):
        features = self.effnet.extract_feature(x)
        
        p2 = features['p2']
        p3 = features['p3']
        p4 = features['p4']
        p5 = features['p5']
        
        out5, lateral5 = self.b5(p5) 
        out4, lateral4 = self.b4(p4, lateral5)
        out3, lateral3 = self.b3(p3, lateral4)
        out2, _ = self.b2(p2, lateral3) 
        
        target_h, target_w = out2.shape[-2:]
        
        out5 = F.interpolate(out5, size=(target_h, target_w), mode='bilinear', align_corners=False)
        out4 = F.interpolate(out4, size=(target_h, target_w), mode='bilinear', align_corners=False)
        out3 = F.interpolate(out3, size=(target_h, target_w), mode='bilinear', align_corners=False)

        final_out = torch.cat([out2, out3, out4, out5], dim=1)
        final_out = self.downsample(final_out)
        return final_out

    def forward(self, x):
        if self.with_cp:
            return cp.checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            return self._forward_impl(x)