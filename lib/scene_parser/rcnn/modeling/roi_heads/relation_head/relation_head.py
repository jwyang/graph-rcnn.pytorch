# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from .roi_relation_feature_extractors import make_roi_relation_feature_extractor
from .roi_relation_predictors import make_roi_relation_predictor
from .inference import make_roi_relation_post_processor
from .loss import make_roi_relation_loss_evaluator


class ROIRelationHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIRelationHead, self).__init__()
        self.feature_extractor = make_roi_relation_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_relation_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_relation_post_processor(cfg)
        self.loss_evaluator = make_roi_relation_loss_evaluator(cfg)

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

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposal_pairs)
        # final classifier that converts the features into predictions
        class_logits = self.predictor(x)

        if not self.training:
            result = self.post_processor((class_logits), proposal_pairs)
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
