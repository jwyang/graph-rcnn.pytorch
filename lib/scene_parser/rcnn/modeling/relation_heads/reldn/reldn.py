# Scene Graph Generation by Iterative Message Passing
# Reimnplemetned by Jianwei Yang (jw2yang@gatech.edu)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from ..roi_relation_feature_extractors import make_roi_relation_feature_extractor
from ..roi_relation_box_feature_extractors import make_roi_relation_box_feature_extractor
from ..roi_relation_box_predictors import make_roi_relation_box_predictor
from ..roi_relation_predictors import make_roi_relation_predictor

from .spatial import build_spatial_feature
# from .semantic import build_semantic_feature
# from .visual import build_visual_feature

class RelDN(nn.Module):
	# def __init__(self, fea_size, dropout=False, gate_width=1, use_kernel_function=False):
    def __init__(self, cfg, in_channels, eps=1e-10):
        super(RelDN, self).__init__()
        self.cfg = cfg
        self.dim = 512
        self.update_step = cfg.MODEL.ROI_RELATION_HEAD.IMP_FEATURE_UPDATE_STEP
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.obj_feature_extractor = make_roi_relation_box_feature_extractor(cfg, in_channels)
        self.pred_feature_extractor = make_roi_relation_feature_extractor(cfg, in_channels)

        num_classes = cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        self.obj_embedding = nn.Sequential(
            nn.Linear(self.pred_feature_extractor.out_channels, self.dim),
            nn.ReLU(True),
            nn.Linear(self.dim, self.dim),
        )

        self.pred_embedding = nn.Sequential(
            nn.Linear(self.pred_feature_extractor.out_channels, self.dim),
            nn.ReLU(True),
            nn.Linear(self.dim, self.dim),
        )

        self.rel_embedding = nn.Sequential(
            nn.Linear(3 * self.dim, self.dim),
            nn.ReLU(True),
            nn.Linear(self.dim, self.dim),
            nn.ReLU(True)
        )

        self.rel_spatial_feat = build_spatial_feature(cfg, self.dim)

        self.rel_subj_predictor = make_roi_relation_predictor(cfg, 512)
        self.rel_obj_predictor = make_roi_relation_predictor(cfg, 512)
        self.rel_pred_predictor = make_roi_relation_predictor(cfg, 512)

        self.rel_spt_predictor = nn.Linear(64, num_classes)


        self.freq_dist = torch.from_numpy(np.load("freq_prior.npy"))
        self.pred_dist = 10 * self.freq_dist # np.log(self.freq_dist + eps)
        self.num_objs = self.pred_dist.shape[0]
        self.pred_dist = torch.FloatTensor(self.pred_dist).view(-1, self.pred_dist.shape[2]).cuda()
        # self.rel_sem_predictor = nn.Embedding(self.pred_dist.size(0), self.pred_dist.size(1))
        # self.rel_sem_predictor.weight.data = self.pred_dist

    def _get_map_idxs(self, proposals, proposal_pairs):
        rel_inds = []
        offset = 0
        for proposal, proposal_pair in zip(proposals, proposal_pairs):
            rel_ind_i = proposal_pair.get_field("idx_pairs").detach().clone()
            rel_ind_i += offset
            offset += len(proposal)
            rel_inds.append(rel_ind_i)

        rel_inds = torch.cat(rel_inds, 0)

        subj_pred_map = rel_inds.new(sum([len(proposal) for proposal in proposals]), rel_inds.shape[0]).fill_(0).float().detach()
        obj_pred_map = rel_inds.new(sum([len(proposal) for proposal in proposals]), rel_inds.shape[0]).fill_(0).float().detach()

        subj_pred_map.scatter_(0, (rel_inds[:, 0].contiguous().view(1, -1)), 1)
        obj_pred_map.scatter_(0, (rel_inds[:, 1].contiguous().view(1, -1)), 1)

        return rel_inds, subj_pred_map, obj_pred_map

    def forward(self, features, proposals, proposal_pairs):
        obj_class_logits = None

        rel_inds, subj_pred_map, obj_pred_map = self._get_map_idxs(proposals, proposal_pairs)
        x_obj = torch.cat([proposal.get_field("features").detach() for proposal in proposals], 0)
        # features = [feature.detach() for feature in features]
        # x_obj = self.avgpool(self.obj_feature_extractor(features, proposals))
        x_pred, _ = self.pred_feature_extractor(features, proposals, proposal_pairs)
        x_pred = self.avgpool(x_pred)
        x_obj = x_obj.view(x_obj.size(0), -1); x_pred = x_pred.view(x_pred.size(0), -1)
        x_obj = self.obj_embedding(x_obj); x_pred = self.pred_embedding(x_pred)

        sub_vert = x_obj[rel_inds[:, 0]]  #
        obj_vert = x_obj[rel_inds[:, 1]]

        '''compute visual scores'''
        rel_subj_class_logits = self.rel_subj_predictor(sub_vert.unsqueeze(2).unsqueeze(3))
        rel_obj_class_logits = self.rel_obj_predictor(obj_vert.unsqueeze(2).unsqueeze(3))

        x_rel = torch.cat([sub_vert, obj_vert, x_pred], 1)
        x_rel = self.rel_embedding(x_rel)
        rel_pred_class_logits = self.rel_pred_predictor(x_rel.unsqueeze(2).unsqueeze(3))
        rel_vis_class_logits = rel_pred_class_logits + rel_subj_class_logits + rel_obj_class_logits
        # rel_vis_class_logits = rel_pred_class_logits # + rel_subj_class_logits + rel_obj_class_logits

        '''compute spatial scores'''
        edge_spt_feats = self.rel_spatial_feat(proposal_pairs)
        rel_spt_class_logits = self.rel_spt_predictor(edge_spt_feats)

        '''compute semantic scores'''
        rel_sem_class_logits = []
        for proposal_per_image, proposal_pairs_per_image in zip(proposals, proposal_pairs):
            obj_labels = proposal_per_image.get_field("labels").detach()
            rel_ind_i = proposal_pairs_per_image.get_field("idx_pairs").detach()
            subj_vert_labels = obj_labels[rel_ind_i[:, 0]]
            obj_vert_labels = obj_labels[rel_ind_i[:, 1]]

            # class_logits_per_image = self.freq_dist[subj_vert_labels, :][:, obj_vert_labels].view(-1, self.freq_dist.size(-1))
            # class_logits_per_image = self.rel_sem_predictor(subj_vert_labels * self.num_objs + obj_vert_labels)
            class_logits_per_image = self.pred_dist[subj_vert_labels * self.num_objs + obj_vert_labels]

            # rmeove duplicate index
            # non_duplicate_idx = (torch.eye(obj_labels.shape[0]).view(-1) == 0).nonzero().view(-1).to(class_logits_per_image.device)
            # class_logits_per_image = class_logits_per_image[non_duplicate_idx]
            rel_sem_class_logits.append(class_logits_per_image)
        rel_sem_class_logits = torch.cat(rel_sem_class_logits, 0)
        rel_class_logits = rel_vis_class_logits + rel_sem_class_logits + rel_spt_class_logits #

        if obj_class_logits is None:
            logits = torch.cat([proposal.get_field("logits") for proposal in proposals], 0)
            obj_class_labels = logits[:, 1:].max(1)[1] + 1
        else:
            obj_class_labels = obj_class_logits[:, 1:].max(1)[1] + 1

        return (x_obj, x_pred), obj_class_logits, rel_class_logits, obj_class_labels, rel_inds

def build_reldn_model(cfg, in_channels):
    return RelDN(cfg, in_channels)
