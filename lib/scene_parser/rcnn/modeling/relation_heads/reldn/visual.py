import torch
import torch.nn as nn

class VisualFeature(nn.Module):
    def __init__(self, dim):
        self.subj_branch = nn.Sequential(nn.Linear())

    def forward(self, subj_feat, obj_feat, rel_feat):
        pass

def build_visual_feature(cfg, in_channels):
    return VisualFeature(cfg, in_channels)
