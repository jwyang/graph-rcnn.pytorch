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
from .imp.imp import IMP
# from .msdn.msdn import MSDN

SCENE_PAESER_DICT = {"imp": IMP} #, "msdn": MSDN}

class SceneParser(GeneralizedRCNN):
    "Scene Parser"
    def __init__(self, cfg):
        GeneralizedRCNN.__init__(self, cfg)
        self.cfg = cfg

        self.sg_heads = None
        if self.cfg.MODEL.ALGORITHM in SCENE_PAESER_DICT:
            self.sg_heads = SCENE_PAESER_DICT[self.cfg.MODEL.ALGORITHM](cfg, self.backbone.out_channels)

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
        
        if self.sg_heads:
            x, result, scene_parser_losses = self.sg_heads(features, proposals, targets)
        if self.roi_heads:
            x, result, scene_parser_losses = self.roi_heads(features, proposals, targets)
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
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT, resume=cfg.resume)
    return optimizer, scheduler, checkpointer, extra_checkpoint_data
