"""
Main code of scene parser
"""
import os
import logging
import torch
import torch.nn as nn

from .mask_rcnn.modeling.detector.generalized_rcnn import GeneralizedRCNN
from .mask_rcnn.solver import make_lr_scheduler
from .mask_rcnn.solver import make_optimizer
from .mask_rcnn.utils.checkpoint import SceneParserCheckpointer
from .mask_rcnn.structures.image_list import to_image_list

class SceneParser(GeneralizedRCNN):
    "Scene Parser"
    def __init__(self, opt):
        GeneralizedRCNN.__init__(self, opt)
        self.opt = opt

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
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result

def get_save_dir(cfg):
    outdir = os.path.join(
        cfg.DATASET.NAME + '_' + cfg.DATASET.MODE + '_' + cfg.DATASET.LOADER,
        cfg.MODEL.BACKBONE.CONV_BODY,
        cfg.MODEL.ALGORITHM,
        'BatchSize_{}'.format(cfg.DATASET.TRAIN_BATCH_SIZE),
        'Base_LR_{}'.format(cfg.SOLVER.BASE_LR)
        )
    if not os.path.exists(os.path.join("checkpoints", outdir)):
        os.makedirs(os.path.join("checkpoints", outdir))
    return os.path.join("checkpoints", outdir)

def build_scene_parser(cfg):
    return SceneParser(cfg)

def build_scene_parser_optimizer(cfg, model):
    save_to_disk = True
    save_dir = get_save_dir(cfg)
    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)
    checkpointer = SceneParserCheckpointer(cfg, model, optimizer, scheduler, save_dir, save_to_disk,
        logger=logging.getLogger("scene_graph_generation.checkpointer"))
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT, resume=cfg.resume)
    return optimizer, scheduler, checkpointer, extra_checkpoint_data
