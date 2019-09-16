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
        self.pred_feature_extractor = make_roi_relation_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_relation_predictor(cfg, self.pred_feature_extractor.out_channels)

    def forward(self, features, proposals, proposal_pairs):
        obj_class_logits = None # no need to predict object class again
        if self.training:
            x, rel_inds = self.pred_feature_extractor(features, proposals, proposal_pairs)
            rel_class_logits = self.predictor(x)
        else:
            with torch.no_grad():
                x, rel_inds = self.pred_feature_extractor(features, proposals, proposal_pairs)
                rel_class_logits = self.predictor(x)

        if obj_class_logits is None:
            logits = torch.cat([proposal.get_field("logits") for proposal in proposals], 0)
            obj_class_labels = logits[:, 1:].max(1)[1] + 1
        else:
            obj_class_labels = obj_class_logits[:, 1:].max(1)[1] + 1

        return x, obj_class_logits, rel_class_logits, obj_class_labels, rel_inds

def build_baseline_model(cfg, in_channels):
    return Baseline(cfg, in_channels)
