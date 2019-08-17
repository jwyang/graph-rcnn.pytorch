# Scene Graph Generation by Iterative Message Passing
# Reimnplemetned by Jianwei Yang (jw2yang@gatech.edu)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
from lib.scene_parser.rcnn.modeling.roi_heads.box_head.box_head import build_roi_box_head
from lib.scene_parser.rcnn.modeling.roi_heads.relation_head.relation_head import build_roi_relation_head
from .imp_base import IMP_BASE

class IMP(nn.Module):
	# def __init__(self, fea_size, dropout=False, gate_width=1, use_kernel_function=False):
    def __init__(self, cfg, in_channels):
        super(IMP, self).__init__()
        self.cfg = cfg
        self.box = build_roi_box_head(cfg, in_channels)
        self.relation = build_roi_relation_head(cfg, in_channels)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        assert self.box.feature_extractor.out_channels == self.relation.feature_extractor.out_channels
        self.imp_base = IMP_BASE(self.box.feature_extractor.out_channels, False, 1, use_kernel_function=False)

    def _get_map_idxs(self, proposal_pairs):

        pair_idxs_all = []
        for i, proposal_pairs_per_image in enumerate(proposal_pairs):
            pair_idxs = proposal_pairs_per_image.get_field("idx_pairs")
            pair_idxs += i * pair_idxs.size(0)
            pair_idxs_all.append(pair_idxs)
        pair_idxs_all = torch.cat(pair_idxs_all, 0)

        map_sobj_rel = pair_idxs_all.new(pair_idxs_all.size(0), pair_idxs_all.size(0)).zero_()
        map_oobj_rel = pair_idxs_all.new(pair_idxs_all.size(0), pair_idxs_all.size(0)).zero_()

        map_sobj_rel.scatter_(0, (pair_idxs_all[:, 0].contiguous().view(1, -1)), 1)
        map_oobj_rel.scatter_(0, (pair_idxs_all[:, 1].contiguous().view(1, -1)), 1)

        map_obj_rel = torch.stack((map_sobj_rel, map_oobj_rel), 1)
        map_subj_obj = pair_idxs_all
        return map_obj_rel, map_subj_obj

    def forward(self, features, proposals, targets=None):
        # feature_obj, feature_phrase, mps_object, mps_phrase
        # mps_object [object_batchsize, 2, n_phrase] : the 2 channel means inward(object) and outward(subject) list
        # mps_phrase [phrase_batchsize, 2]
        if self.training:
            with torch.no_grad():
                proposals = self.box.loss_evaluator.subsample(proposals, targets)
                proposal_pairs = self.relation.loss_evaluator.subsample(proposals, targets)

        map_obj_rel, map_subj_obj = self._get_map_idxs(proposal_pairs)

        x_obj = self.avgpool(self.box.feature_extractor(features, proposals))
        x_pred = self.avgpool(self.relation.feature_extractor(features, proposal_pairs))
        x_obj = x_obj.view(x_obj.size(0), -1); x_pred = x_pred.view(x_pred.size(0), -1)

        for t in range(self.cfg.MODEL.SG_HEAD.FEATURE_UPDATE_STEP):
            '''update object features'''
            x_subj_message = self.imp_base.prepare_message(x_obj, x_pred, map_obj_rel[:, 0, :], self.imp_base.gate_edge2vert)
            x_obj_message = self.imp_base.prepare_message(x_obj, x_pred, map_obj_rel[:, 1, :], self.imp_base.gate_edge2vert)
            GRU_input_feature_object = x_subj_message + x_obj_message
            x_obj_updated = self.imp_base.vert_rnn(GRU_input_feature_object, x_obj)

            '''update predicate features'''
            idxs_subj = map_subj_obj[:, 0]; idxs_obj = map_subj_obj[:, 1]
            fea_sub2pred = torch.index_select(x_obj, 0, idxs_subj)
            fea_obj2pred = torch.index_select(x_obj, 0, idxs_obj)
            pred_subj = self.imp_base.gate_vert2edge(x_pred, fea_sub2pred)
            pred_obj = self.imp_base.gate_vert2edge(x_pred,  fea_obj2pred)
            GRU_input_feature_phrase =  pred_subj + pred_obj
            x_pred_updated = self.imp_base.vert_rnn(GRU_input_feature_phrase, x_pred)

            x_obj, x_pred = x_obj_updated, x_pred_updated

        '''compute results and losses'''
        losses = {}
        # final classifier that converts the features into predictions
        class_logits, box_regression = self.box.predictor(x_obj.unsqueeze(2).unsqueeze(3))

        if not self.training:
            result = self.box.post_processor((class_logits, box_regression), proposals)
            return x, result, {}

        loss_classifier, loss_box_reg = self.box.loss_evaluator(
            [class_logits], [box_regression]
        )
        losses.update(dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg))

        class_logits = self.relation.predictor(x_pred.unsqueeze(2).unsqueeze(3))

        if not self.training:
            result = self.relation.post_processor((class_logits), proposal_pairs)
            return x, result, {}

        loss_pred_classifier = self.relation.loss_evaluator([class_logits])
        losses.update(dict(loss_pred_classifier=loss_pred_classifier))
        return (x_obj, x_pred), (proposals, proposal_pairs), losses
