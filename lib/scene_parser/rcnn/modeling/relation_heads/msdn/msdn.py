# MSDN for scene graph generation
# Reimnplemetned by Jianwei Yang (jw2yang@gatech.edu)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
from .msdn_base import MSDN_BASE
from ..roi_relation_feature_extractors import make_roi_relation_feature_extractor
from ..roi_relation_box_predictors import make_roi_relation_box_predictor
from ..roi_relation_predictors import make_roi_relation_predictor

class MSDN(MSDN_BASE):
	def __init__(self, cfg, in_channels, dim=1024, dropout=False, gate_width=128, use_kernel_function=False):
		super(MSDN, self).__init__(dim, dropout, gate_width, use_region=True, use_kernel_function=use_kernel_function)
		self.cfg = cfg
		self.dim = dim
		self.update_step = cfg.MODEL.ROI_RELATION_HEAD.MSDN_FEATURE_UPDATE_STEP
		self.avgpool = nn.AdaptiveAvgPool2d(1)
		self.pred_feature_extractor = make_roi_relation_feature_extractor(cfg, in_channels)

		self.obj_embedding = nn.Sequential(
			nn.Linear(self.pred_feature_extractor.out_channels, self.dim),
			nn.ReLU(True),
			nn.Linear(self.dim, self.dim),
        )
		self.rel_embedding = nn.Sequential(
			nn.Linear(self.pred_feature_extractor.out_channels, self.dim),
			nn.ReLU(True),
			nn.Linear(self.dim, self.dim),
        )

		self.obj_predictor = make_roi_relation_box_predictor(cfg, dim)
		self.pred_predictor = make_roi_relation_predictor(cfg, dim)

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
		x_obj = torch.cat([proposal.get_field("features").detach() for proposal in proposals], 0)
		x_pred, _ = self.pred_feature_extractor(features, proposals, proposal_pairs)
		x_pred = self.avgpool(x_pred)
		x_obj = x_obj.view(x_obj.size(0), -1); x_pred = x_pred.view(x_pred.size(0), -1)
		x_obj = self.obj_embedding(x_obj); x_pred = self.rel_embedding(x_pred)

		x_obj = [x_obj]
		x_pred = [x_pred]

		for t in range(self.update_step):
			'''update object features'''
			object_sub = self.prepare_message(x_obj[t], x_pred[t], subj_pred_map, self.gate_pred2sub)
			object_obj = self.prepare_message(x_obj[t], x_pred[t], obj_pred_map, self.gate_pred2obj)
			GRU_input_feature_object = (object_sub + object_obj) / 2.
			x_obj.append(x_obj[t] + self.GRU_object(GRU_input_feature_object, x_obj[t]))

			'''update predicate features'''
			indices_sub = rel_inds[:, 0]
			indices_obj = rel_inds[:, 1]
			feat_sub2pred = torch.index_select(x_obj[t], 0, indices_sub)
			feat_obj2pred = torch.index_select(x_obj[t], 0, indices_obj)
			phrase_sub = self.gate_sub2pred(x_pred[t], feat_sub2pred)
			phrase_obj = self.gate_obj2pred(x_pred[t],  feat_obj2pred)
			GRU_input_feature_phrase =  phrase_sub / 2. + phrase_obj / 2.
			x_pred.append(x_pred[t] + self.GRU_pred(GRU_input_feature_phrase, x_pred[t]))

		'''compute results and losses'''
		# final classifier that converts the features into predictions
		# for object prediction, we do not do bbox regression again
		obj_class_logits = self.obj_predictor(x_obj[-1].unsqueeze(2).unsqueeze(3))
		pred_class_logits = self.pred_predictor(x_pred[-1].unsqueeze(2).unsqueeze(3))

		if obj_class_logits is None:
			logits = torch.cat([proposal.get_field("logits") for proposal in proposals], 0)
			obj_class_labels = logits[:, 1:].max(1)[1] + 1
		else:
			obj_class_labels = obj_class_logits[:, 1:].max(1)[1] + 1

		return (x_obj[-1], x_pred[-1]), obj_class_logits, pred_class_logits, obj_class_labels, rel_inds

def build_msdn_model(cfg,in_channels):
	return MSDN(cfg, in_channels)
