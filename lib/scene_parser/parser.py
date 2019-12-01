"""
Main code of scene parser
"""
import os
import logging
import torch
import torch.nn as nn

from .rcnn.modeling.detector.generalized_rcnn import GeneralizedRCNN
from .rcnn.solver import make_lr_scheduler
from .rcnn.solver import make_optimizer
from .rcnn.utils.checkpoint import SceneParserCheckpointer
from .rcnn.structures.image_list import to_image_list
from .rcnn.utils.comm import synchronize, get_rank
from .rcnn.modeling.relation_heads.relation_heads import build_roi_relation_head

SCENE_PAESER_DICT = ["sg_baseline", "sg_imp", "sg_msdn", "sg_grcnn", "sg_reldn"]

class SceneParser(GeneralizedRCNN):
    "Scene Parser"
    def __init__(self, cfg):
        GeneralizedRCNN.__init__(self, cfg)
        self.cfg = cfg

        self.rel_heads = None
        if cfg.MODEL.RELATION_ON and self.cfg.MODEL.ALGORITHM in SCENE_PAESER_DICT:
            self.rel_heads = build_roi_relation_head(cfg, self.backbone.out_channels)
        self._freeze_components(self.cfg)

    def _freeze_components(self, cfg):
        if cfg.MODEL.BACKBONE.FREEZE_PARAMETER:
            for param in self.backbone.parameters():
                param.requires_grad = False

        if cfg.MODEL.RPN.FREEZE_PARAMETER:
            for param in self.rpn.parameters():
                param.requires_grad = False

        if cfg.MODEL.ROI_BOX_HEAD.FREEZE_PARAMETER:
            for param in self.roi_heads.parameters():
                param.requires_grad = False

    def train(self):
        if self.cfg.MODEL.BACKBONE.FREEZE_PARAMETER:
            self.backbone.eval()
        else:
            self.backbone.train()

        if self.cfg.MODEL.RPN.FREEZE_PARAMETER:
            self.rpn.eval()
        else:
            self.rpn.train()

        if self.cfg.MODEL.ROI_BOX_HEAD.FREEZE_PARAMETER:
            self.roi_heads.eval()
        else:
            self.roi_heads.train()

        if self.rel_heads:
            self.rel_heads.train()

        self.training = True

    def eval(self):
        self.backbone.eval()
        self.rpn.eval()
        self.roi_heads.eval()
        if self.rel_heads:
            self.rel_heads.eval()
        self.training = False

    def _post_processing(self, result):
        """
        Arguments:
            result: (object_predictions, predicate_predictions)

        Returns:
            sort the object-predicate triplets, and output the top
        """
        result_obj, result_pred = result
        result_obj_new, result_pred_new = [], []
        assert len(result_obj) == len(result_pred), "object list must have equal number to predicate list"
        for result_obj_i, result_pred_i in zip(result_obj, result_pred):
            obj_scores = result_obj_i.get_field("scores")
            rel_inds = result_pred_i.get_field("idx_pairs")
            pred_scores = result_pred_i.get_field("scores")
            scores = torch.stack((
                obj_scores[rel_inds[:,0]],
                obj_scores[rel_inds[:,1]],
                pred_scores[:, 1:].max(1)[0]
            ), 1).prod(1)
            scores_sorted, order = scores.sort(0, descending=True)
            result_pred_i = result_pred_i[order[:self.cfg.MODEL.ROI_RELATION_HEAD.TRIPLETS_PER_IMG]]
            result_obj_new.append(result_obj_i)
            result_pred_new.append(result_pred_i)
        return (result_obj_new, result_pred_new)

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        proposals, proposal_losses = self.rpn(images, features, targets)
        scene_parser_losses = {}
        if self.roi_heads:
            x, detections, roi_heads_loss = self.roi_heads(features, proposals, targets)
            result = detections
            scene_parser_losses.update(roi_heads_loss)

            if self.rel_heads:
                relation_features = features
                # optimization: during training, if we share the feature extractor between
                # the box and the relation heads, then we can reuse the features already computed
                if (
                    self.training
                    and self.cfg.MODEL.ROI_RELATION_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
                ):
                    relation_features = x
                # During training, self.box() will return the unaltered proposals as "detections"
                # this makes the API consistent during training and testing
                x_pairs, detection_pairs, rel_heads_loss = self.rel_heads(relation_features, detections, targets)
                scene_parser_losses.update(rel_heads_loss)

                x = (x, x_pairs)
                result = (detections, detection_pairs)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            scene_parser_losses = {}

        if self.training:
            losses = {}
            losses.update(scene_parser_losses)
            losses.update(proposal_losses)
            return losses

        # NOTE: if object scores are updated in rel_heads, we need to ensure detections are updated accordingly
        # result = self._post_processing(result)
        return result

def get_save_dir(cfg):
    train_mode = "joint" if cfg.MODEL.WEIGHT_DET == "" else "step"
    iter_step = max([cfg.MODEL.ROI_RELATION_HEAD.IMP_FEATURE_UPDATE_STEP, \
                     cfg.MODEL.ROI_RELATION_HEAD.MSDN_FEATURE_UPDATE_STEP, \
                     cfg.MODEL.ROI_RELATION_HEAD.GRCNN_FEATURE_UPDATE_STEP])
    train_alg = (cfg.MODEL.ALGORITHM + '_' + train_mode + '_' + str(iter_step)) if "sg" in cfg.MODEL.ALGORITHM else cfg.MODEL.ALGORITHM
    outdir = os.path.join(
        cfg.DATASET.NAME + '_' + cfg.DATASET.MODE + '_' + cfg.DATASET.LOADER,
        cfg.MODEL.BACKBONE.CONV_BODY, train_alg,
        'BatchSize_{}'.format(cfg.DATASET.TRAIN_BATCH_SIZE),
        'Base_LR_{}'.format(cfg.SOLVER.BASE_LR)
        )
    if not os.path.exists(os.path.join("checkpoints", outdir)):
        os.makedirs(os.path.join("checkpoints", outdir))
    return os.path.join("checkpoints", outdir)

def build_scene_parser(cfg):
    return SceneParser(cfg)

def build_scene_parser_optimizer(cfg, model, local_rank=0, distributed=False):
    save_to_disk = True
    save_dir = get_save_dir(cfg)
    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )
    save_to_disk = get_rank() == 0
    checkpointer = SceneParserCheckpointer(cfg, model, optimizer, scheduler, save_dir, save_to_disk,
        logger=logging.getLogger("scene_graph_generation.checkpointer"))
    model_weight = cfg.MODEL.WEIGHT_DET if cfg.MODEL.WEIGHT_DET != "" else cfg.MODEL.WEIGHT_IMG
    extra_checkpoint_data = checkpointer.load(model_weight, resume=cfg.resume)
    return optimizer, scheduler, checkpointer, extra_checkpoint_data
