# Scene Graph Generation with baseline (vanilla) model
# Reimnplemetned by Jianwei Yang (jw2yang@gatech.edu)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
from ..roi_relation_feature_extractors import make_roi_relation_feature_extractor
from ..roi_relation_predictors import make_roi_relation_predictor

class Baseline(nn.Module):
    def __init__(self, cfg, in_channels):
        super(Baseline, self).__init__()
        self.cfg = cfg
        self.feature_extractor = make_roi_relation_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_relation_predictor(cfg, self.feature_extractor.out_channels)

    def forward(self, features, proposals, proposal_pairs):
        obj_class_logits = None # no need to predict object class again
        if self.training:
            x = self.feature_extractor(features, proposal_pairs)
            rel_class_logits = self.predictor(x)
        else:
            with torch.no_grad():
                x = self.feature_extractor(features, proposal_pairs)
                rel_class_logits = self.predictor(x)
        return x, obj_class_logits, rel_class_logits

def build_baseline_model(cfg, in_channels):
    return Baseline(cfg, in_channels)
