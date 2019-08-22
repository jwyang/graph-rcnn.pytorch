import torch
import torch.nn as nn
import numpy as np
from lib.scene_parser.rcnn.utils.boxes import bbox_transform_inv, boxes_union

class SpatialFeature(nn.Module):
    def __init__(self, cfg, dim):
        super(SpatialFeature, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28, 64), nn.LeakyReLU(0.1),
            nn.Linear(64, 64), nn.LeakyReLU(0.1))

    def _get_pair_feature(self, boxes1, boxes2):
        delta_1 = bbox_transform_inv(boxes1, boxes2)
        delta_2 = bbox_transform_inv(boxes2, boxes1)
        spt_feat = np.hstack((delta_1, delta_2[:, :2]))
        return spt_feat

    def _get_box_feature(self, boxes, width, height):
        f1 = boxes[:, 0] / width
        f2 = boxes[:, 1] / height
        f3 = boxes[:, 2] / width
        f4 = boxes[:, 3] / height
        f5 = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1) / (width * height)
        return np.vstack((f1, f2, f3, f4, f5)).transpose()

    def _get_spt_features(self, boxes1, boxes2, width, height):
        boxes_u = boxes_union(boxes1, boxes2)
        spt_feat_1 = self._get_box_feature(boxes1, width, height)
        spt_feat_2 = self._get_box_feature(boxes2, width, height)
        spt_feat_12 = self._get_pair_feature(boxes1, boxes2)
        spt_feat_1u = self._get_pair_feature(boxes1, boxes_u)
        spt_feat_u2 = self._get_pair_feature(boxes_u, boxes2)
        return np.hstack((spt_feat_12, spt_feat_1u, spt_feat_u2, spt_feat_1, spt_feat_2))

    def forward(self, proposal_pairs):
        spt_feats = []
        for proposal_pair in proposal_pairs:
            boxes_subj = proposal_pair.bbox[:, :4]
            boxes_obj = proposal_pair.bbox[:, 4:]
            spt_feat = self._get_spt_features(boxes_subj.cpu().numpy(), boxes_obj.cpu().numpy(), proposal_pair.size[0], proposal_pair.size[1])
            spt_feat = torch.from_numpy(spt_feat).to(boxes_subj.device)
            spt_feats.append(spt_feat)
        spt_feats = torch.cat(spt_feats, 0).float()
        spt_feats = self.model(spt_feats)
        return spt_feats

def build_spatial_feature(cfg, dim=0):
    return SpatialFeature(cfg, dim)
