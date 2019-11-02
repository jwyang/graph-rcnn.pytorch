# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Relation head for predicting relationship between object pairs.
# Written by Jianwei Yang (jw2yang@gatech.edu).
import numpy as np
import torch
from torch import nn
from lib.scene_parser.rcnn.structures.bounding_box_pair import BoxPairList
from lib.scene_parser.rcnn.structures.boxlist_ops import boxlist_iou

from .inference import make_roi_relation_post_processor
from .loss import make_roi_relation_loss_evaluator
from .sparse_targets import FrequencyBias, _get_tensor_from_boxlist, _get_rel_inds

from .relpn.relpn import make_relation_proposal_network
from .baseline.baseline import build_baseline_model
from .imp.imp import build_imp_model
from .msdn.msdn import build_msdn_model
from .grcnn.grcnn import build_grcnn_model
from .reldn.reldn import build_reldn_model

class ROIRelationHead(torch.nn.Module):
    """
    Generic Relation Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIRelationHead, self).__init__()
        self.cfg = cfg

        if cfg.MODEL.ALGORITHM == "sg_baseline":
            self.rel_predictor = build_baseline_model(cfg, in_channels)
        elif cfg.MODEL.ALGORITHM == "sg_imp":
            self.rel_predictor = build_imp_model(cfg, in_channels)
        elif cfg.MODEL.ALGORITHM == "sg_msdn":
            self.rel_predictor = build_msdn_model(cfg, in_channels)
        elif cfg.MODEL.ALGORITHM == "sg_grcnn":
            self.rel_predictor = build_grcnn_model(cfg, in_channels)
        elif cfg.MODEL.ALGORITHM == "sg_reldn":
            self.rel_predictor = build_reldn_model(cfg, in_channels)

        self.post_processor = make_roi_relation_post_processor(cfg)
        self.loss_evaluator = make_roi_relation_loss_evaluator(cfg)

        if self.cfg.MODEL.USE_RELPN:
            self.relpn = make_relation_proposal_network(cfg)

        self.freq_dist = None
        self.use_bias = self.cfg.MODEL.ROI_RELATION_HEAD.USE_BIAS
        if self.cfg.MODEL.USE_FREQ_PRIOR or self.cfg.MODEL.ROI_RELATION_HEAD.USE_BIAS:
            # print("Using frequency bias: ", cfg.MODEL.FREQ_PRIOR)
            # self.freq_dist_file = op.join(cfg.DATA_DIR, cfg.MODEL.FREQ_PRIOR)
            self.freq_dist_file = "freq_prior.npy"
            self.freq_dist = np.load(self.freq_dist_file)
            if self.cfg.MODEL.USE_FREQ_PRIOR:
                # never predict __no_relation__ for frequency prior
                self.freq_dist[:, :, 0] = 0
                # we use probability directly
                self.freq_bias = FrequencyBias(self.freq_dist)
            else:
                self.freq_dist = np.log(self.freq_dist + 1e-3)
                self.freq_bias = FrequencyBias(self.freq_dist)

        # if self.cfg.MODEL.USE_FREQ_PRIOR:
        #     self.freq_dist = torch.from_numpy(np.load("freq_prior.npy"))
        #     self.freq_dist[:, :, 0] = 0

    def _get_proposal_pairs(self, proposals):
        proposal_pairs = []
        for i, proposals_per_image in enumerate(proposals):
            box_subj = proposals_per_image.bbox
            box_obj = proposals_per_image.bbox

            box_subj = box_subj.unsqueeze(1).repeat(1, box_subj.shape[0], 1)
            box_obj = box_obj.unsqueeze(0).repeat(box_obj.shape[0], 1, 1)
            proposal_box_pairs = torch.cat((box_subj.view(-1, 4), box_obj.view(-1, 4)), 1)

            idx_subj = torch.arange(box_subj.shape[0]).view(-1, 1, 1).repeat(1, box_obj.shape[0], 1).to(proposals_per_image.bbox.device)
            idx_obj = torch.arange(box_obj.shape[0]).view(1, -1, 1).repeat(box_subj.shape[0], 1, 1).to(proposals_per_image.bbox.device)
            proposal_idx_pairs = torch.cat((idx_subj.view(-1, 1), idx_obj.view(-1, 1)), 1)

            keep_idx = (proposal_idx_pairs[:, 0] != proposal_idx_pairs[:, 1]).nonzero().view(-1)

            # if we filter non overlap bounding boxes
            if self.cfg.MODEL.ROI_RELATION_HEAD.FILTER_NON_OVERLAP:
                ious = boxlist_iou(proposals_per_image, proposals_per_image).view(-1)
                ious = ious[keep_idx]
                keep_idx = keep_idx[(ious > 0).nonzero().view(-1)]
            proposal_idx_pairs = proposal_idx_pairs[keep_idx]
            proposal_box_pairs = proposal_box_pairs[keep_idx]
            proposal_pairs_per_image = BoxPairList(proposal_box_pairs, proposals_per_image.size, proposals_per_image.mode)
            proposal_pairs_per_image.add_field("idx_pairs", proposal_idx_pairs)

            proposal_pairs.append(proposal_pairs_per_image)
        return proposal_pairs

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            if self.cfg.MODEL.USE_RELPN:
                proposal_pairs = self.relpn(proposals, targets)
            else:
                with torch.no_grad():
                    proposal_pairs = self.loss_evaluator.subsample(proposals, targets)
        else:
            if self.cfg.MODEL.USE_RELPN:
                proposal_pairs = self.relpn(proposals, targets)
            else:
                proposal_pairs = self._get_proposal_pairs(proposals)

        if self.cfg.MODEL.USE_FREQ_PRIOR:
            """
            if use frequency prior, we directly use the statistics
            """
            x = None
            obj_class_logits = None
            _, obj_labels, im_inds = _get_tensor_from_boxlist(proposals, 'labels')
            _, proposal_idx_pairs, im_inds_pairs = _get_tensor_from_boxlist(proposal_pairs, 'idx_pairs')
            rel_inds = _get_rel_inds(im_inds, im_inds_pairs, proposal_idx_pairs)
            pred_class_logits = self.freq_bias.index_with_labels(
                torch.stack((obj_labels[rel_inds[:, 0]],obj_labels[rel_inds[:, 1]],), 1))
        else:
            # extract features that will be fed to the final classifier. The
            # feature_extractor generally corresponds to the pooler + heads
            x, obj_class_logits, pred_class_logits, obj_class_labels, rel_inds = \
                self.rel_predictor(features, proposals, proposal_pairs)

            if self.use_bias:
                pred_class_logits = pred_class_logits + self.freq_bias.index_with_labels(
                    torch.stack((
                        obj_class_labels[rel_inds[:, 0]],
                        obj_class_labels[rel_inds[:, 1]],
                    ), 1))

        if not self.training:
            # NOTE: if we have updated object class logits, then we need to update proposals as well!!!
            # if obj_class_logits is not None:
            #     boxes_per_image = [len(proposal) for proposal in proposals]
            #     obj_logits = obj_class_logits
            #     obj_scores, obj_labels = obj_class_logits[:, 1:].max(1)
            #     obj_labels = obj_labels + 1
            #     obj_logits = obj_logits.split(boxes_per_image, dim=0)
            #     obj_scores = obj_scores.split(boxes_per_image, dim=0)
            #     obj_labels = obj_labels.split(boxes_per_image, dim=0)
            #     for proposal, obj_logit, obj_score, obj_label in \
            #         zip(proposals, obj_logits, obj_scores, obj_labels):
            #         proposal.add_field("logits", obj_logit)
            #         proposal.add_field("scores", obj_score)
            #         proposal.add_field("labels", obj_label)
            result = self.post_processor((pred_class_logits), proposal_pairs, use_freq_prior=self.cfg.MODEL.USE_FREQ_PRIOR)
            return x, result, {}

        loss_obj_classifier = 0
        if obj_class_logits is not None:
            loss_obj_classifier = self.loss_evaluator.obj_classification_loss(proposals, [obj_class_logits])
        loss_pred_classifier = self.loss_evaluator([pred_class_logits])

        return (
            x,
            proposal_pairs,
            dict(loss_obj_classifier=loss_obj_classifier, loss_pred_classifier=loss_pred_classifier),
        )

def build_roi_relation_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIRelationHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIRelationHead(cfg, in_channels)
