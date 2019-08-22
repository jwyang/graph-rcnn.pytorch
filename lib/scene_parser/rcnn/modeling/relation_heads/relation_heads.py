# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Relation head for predicting relationship between object pairs.
# Written by Jianwei Yang (jw2yang@gatech.edu).
import numpy as np
import torch
from torch import nn

# from .roi_relation_feature_extractors import make_roi_relation_feature_extractor
# from .roi_relation_predictors import make_roi_relation_predictor
from .inference import make_roi_relation_post_processor
from .loss import make_roi_relation_loss_evaluator
from lib.scene_parser.rcnn.structures.bounding_box_pair import BoxPairList

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

        self.freq_dist = None
        if self.cfg.MODEL.USE_FREQ_PRIOR:
            self.freq_dist = torch.from_numpy(np.load("freq_prior.npy"))
            self.freq_dist[:, :, 0] = 0

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

            non_duplicate_idx = (proposal_idx_pairs[:, 0] != proposal_idx_pairs[:, 1]).nonzero()
            proposal_idx_pairs = proposal_idx_pairs[non_duplicate_idx.view(-1)]
            proposal_box_pairs = proposal_box_pairs[non_duplicate_idx.view(-1)]
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
            obj_class_logits = None
            class_logits = []
            for proposal_per_image in proposals:
                obj_labels = proposal_per_image.get_field("labels")
                class_logits_per_image = self.freq_dist[obj_labels, :][:, obj_labels].view(-1, self.freq_dist.size(-1))
                # rmeove duplicate index
                non_duplicate_idx = (torch.eye(obj_labels.shape[0]).view(-1) == 0).nonzero().view(-1).to(class_logits_per_image.device)
                class_logits_per_image = class_logits_per_image[non_duplicate_idx]
                class_logits.append(class_logits_per_image)
            pred_class_logits = torch.cat(class_logits, 0)
        else:
            # extract features that will be fed to the final classifier. The
            # feature_extractor generally corresponds to the pooler + heads
            x, obj_class_logits, pred_class_logits = self.rel_predictor(features, proposals, proposal_pairs)
        
        if not self.training:
            result = self.post_processor((pred_class_logits), proposal_pairs, use_freq_prior=self.cfg.MODEL.USE_FREQ_PRIOR)
            # boxes_per_image = [len(proposal) for proposal in proposals]
            # obj_labels = obj_class_logits[:, 1:].max(1)[1] + 1
            # obj_labels = obj_labels.split(boxes_per_image, dim=0)
            # for proposal, obj_label in zip(proposals, obj_labels):
            #     proposal.add_field("labels", obj_label)
            return x, result, {}

        if self.cfg.MODEL.ALGORITHM in ["sg_baseline", "sg_reldn"]:
            loss_obj_classifier = 0
        else:
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
