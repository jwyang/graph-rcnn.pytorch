# Scene Graph Generation by Iterative Message Passing
# Reimplemented by Jianwei Yang (jw2yang@gatech.edu)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from ..roi_relation_feature_extractors import make_roi_relation_feature_extractor
from ..roi_relation_box_feature_extractors import make_roi_relation_box_feature_extractor
from ..roi_relation_box_predictors import make_roi_relation_box_predictor
from ..roi_relation_predictors import make_roi_relation_predictor

class IMP(nn.Module):
	# def __init__(self, fea_size, dropout=False, gate_width=1, use_kernel_function=False):
    def __init__(self, cfg, in_channels):
        super(IMP, self).__init__()
        self.cfg = cfg
        self.dim = 512
        self.update_step = cfg.MODEL.ROI_RELATION_HEAD.IMP_FEATURE_UPDATE_STEP
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.obj_feature_extractor = make_roi_relation_box_feature_extractor(cfg, in_channels)
        self.pred_feature_extractor = make_roi_relation_feature_extractor(cfg, in_channels)

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

        if self.update_step > 0:
            self.edge_gru = nn.GRUCell(input_size=self.dim, hidden_size=self.dim)
            self.node_gru = nn.GRUCell(input_size=self.dim, hidden_size=self.dim)

            self.subj_node_gate = nn.Sequential(nn.Linear(self.dim * 2, 1), nn.Sigmoid())
            self.obj_node_gate = nn.Sequential(nn.Linear(self.dim * 2, 1), nn.Sigmoid())

            self.subj_edge_gate = nn.Sequential(nn.Linear(self.dim * 2, 1), nn.Sigmoid())
            self.obj_edge_gate = nn.Sequential(nn.Linear(self.dim * 2, 1), nn.Sigmoid())

        self.obj_predictor = make_roi_relation_box_predictor(cfg, 512)
        self.pred_predictor = make_roi_relation_predictor(cfg, 512)

    def _get_map_idxs(self, proposals, proposal_pairs):
        rel_inds = []
        offset = 0
        for proposal, proposal_pair in zip(proposals, proposal_pairs):
            rel_ind_i = proposal_pair.get_field("idx_pairs").detach()
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
        rel_inds, subj_pred_map, obj_pred_map = self._get_map_idxs(proposals, proposal_pairs)
        x_obj = torch.cat([proposal.get_field("features") for proposal in proposals], 0)
        # x_obj = self.avgpool(self.obj_feature_extractor(features, proposals))
        x_pred, _ = self.pred_feature_extractor(features, proposals, proposal_pairs)
        x_pred = self.avgpool(x_pred)
        x_obj = x_obj.view(x_obj.size(0), -1); x_pred = x_pred.view(x_pred.size(0), -1)
        x_obj = self.obj_embedding(x_obj); x_pred = self.pred_embedding(x_pred)

        # hx_obj = x_obj.clone().fill_(0).detach()
        # hx_pred = x_pred.clone().fill_(0).detach()
        # hx_obj = [self.node_gru(x_obj, hx_obj)]
        # hx_edge = [self.edge_gru(x_pred, hx_pred)]

        hx_obj = [x_obj]
        hx_edge = [x_pred]

        for t in range(self.update_step):
            sub_vert = hx_obj[t][rel_inds[:, 0]]  #
            obj_vert = hx_obj[t][rel_inds[:, 1]]

            '''update object features'''
            message_pred_to_subj = self.subj_node_gate(torch.cat([sub_vert, hx_edge[t]], 1)) * hx_edge[t]  # nrel x d
            message_pred_to_obj = self.obj_node_gate(torch.cat([obj_vert, hx_edge[t]], 1)) * hx_edge[t]    # nrel x d
            node_message = (torch.mm(subj_pred_map, message_pred_to_subj) / (subj_pred_map.sum(1, keepdim=True) + 1e-5) \
                          + torch.mm(obj_pred_map, message_pred_to_obj) / (obj_pred_map.sum(1, keepdim=True) + 1e-5)) / 2.
            hx_obj.append(self.node_gru(node_message, hx_obj[t]))
            # hx_obj.append(F.relu(node_message + hx_obj[t]))

            '''update predicat features'''
            message_subj_to_pred = self.subj_edge_gate(torch.cat([sub_vert, hx_edge[t]], 1)) * sub_vert  # nrel x d
            message_obj_to_pred = self.obj_edge_gate(torch.cat([obj_vert, hx_edge[t]], 1)) * obj_vert    # nrel x d
            edge_message = (message_subj_to_pred + message_obj_to_pred) / 2.
            hx_edge.append(self.edge_gru(edge_message, hx_edge[t]))
            # hx_edge.append(F.relu(edge_message + hx_edge[t]))

        '''compute results and losses'''
        # final classifier that converts the features into predictions
        # for object prediction, we do not do bbox regression again
        obj_class_logits = self.obj_predictor(hx_obj[-1].unsqueeze(2).unsqueeze(3))
        pred_class_logits = self.pred_predictor(hx_edge[-1].unsqueeze(2).unsqueeze(3))

        if obj_class_logits is None:
            logits = torch.cat([proposal.get_field("logits") for proposal in proposals], 0)
            obj_class_labels = logits[:, 1:].max(1)[1] + 1
        else:
            obj_class_labels = obj_class_logits[:, 1:].max(1)[1] + 1

        return (hx_obj[-1], hx_edge[-1]), obj_class_logits, pred_class_logits, obj_class_labels, rel_inds

def build_imp_model(cfg, in_channels):
    return IMP(cfg, in_channels)
