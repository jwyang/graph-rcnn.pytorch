# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
import torch
from torch import nn

from .roi_relation_feature_extractors import make_roi_relation_feature_extractor
from .roi_relation_predictors import make_roi_relation_predictor
from .inference import make_roi_relation_post_processor
from .loss import make_roi_relation_loss_evaluator
from lib.scene_parser.rcnn.structures.bounding_box_pair import BoxPairList

class ROIRelationHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIRelationHead, self).__init__()
        self.cfg = cfg
        self.feature_extractor = make_roi_relation_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_relation_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_relation_post_processor(cfg)
        self.loss_evaluator = make_roi_relation_loss_evaluator(cfg)

        if self.cfg.MODEL.USE_FREQ_PRIOR:
            freq_prior = torch.from_numpy(np.load("freq_prior.npy"))
            freq_prior[:, :, 0] += 1
            self.freq_dist = freq_prior / (freq_prior.sum(2).unsqueeze(2) + 1e-5)
            self.freq_dist[:, :, 0] = 0

    def _get_proposal_pairs(self, proposals):
        proposal_pairs = []
        for i, proposals_per_image in enumerate(proposals):
            box_subj = proposals_per_image.bbox
            box_obj = proposals_per_image.bbox
            box_subj = box_subj.unsqueeze(1).repeat(1, box_subj.shape[0], 1)
            box_obj = box_obj.unsqueeze(0).repeat(box_obj.shape[0], 1, 1)
            proposal_box_pairs = torch.cat((box_subj.view(-1, 4), box_obj.view(-1, 4)), 1)
            proposal_pairs_per_image = BoxPairList(proposal_box_pairs, proposals_per_image.size, proposals_per_image.mode)

            idx_subj = torch.arange(box_subj.shape[0]).view(-1, 1, 1).repeat(1, box_obj.shape[0], 1).to(proposals_per_image.bbox.device)
            idx_obj = torch.arange(box_obj.shape[0]).view(1, -1, 1).repeat(box_subj.shape[0], 1, 1).to(proposals_per_image.bbox.device)
            proposal_idx_pairs = torch.cat((idx_subj.view(-1, 1), idx_obj.view(-1, 1)), 1)
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
            with torch.no_grad():
                proposal_pairs = self.loss_evaluator.subsample(proposals, targets)
        else:
            # proposals = [proposal[:32] for proposal in proposals]
            proposal_pairs = self._get_proposal_pairs(proposals)

        if self.cfg.MODEL.USE_FREQ_PRIOR:
            """
            if use frequency prior, we directly use the statistics
            """
            x = None
            class_logits = []
            for proposal_per_image in proposals:
                obj_labels = proposal_per_image.get_field("labels")
                for subj_label in obj_labels:
                    for obj_label in obj_labels:
                        class_logit = self.freq_dist[subj_label.item(), obj_label.item()]
                        class_logits.append(class_logit)
            class_logits = torch.stack(class_logits, 0)
        else:
            # extract features that will be fed to the final classifier. The
            # feature_extractor generally corresponds to the pooler + heads
            x = self.feature_extractor(features, proposal_pairs)
            # final classifier that converts the features into predictions
            class_logits = self.predictor(x)

        if not self.training:
            result = self.post_processor((class_logits), proposal_pairs, use_freq_prior=self.cfg.MODEL.USE_FREQ_PRIOR)
            return x, result, {}

        loss_pred_classifier = self.loss_evaluator([class_logits])
        return (
            x,
            proposal_pairs,
            dict(loss_pred_classifier=loss_pred_classifier),
        )


def build_roi_relation_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIRelationHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIRelationHead(cfg, in_channels)
