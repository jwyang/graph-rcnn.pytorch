# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F

from lib.scene_parser.rcnn.modeling import registry
from lib.scene_parser.rcnn.modeling.backbone import resnet
from lib.scene_parser.rcnn.modeling.poolers import Pooler
from lib.scene_parser.rcnn.modeling.make_layers import group_norm
from lib.scene_parser.rcnn.modeling.make_layers import make_fc
from .sparse_targets import _get_tensor_from_boxlist, _get_rel_inds

@registry.ROI_RELATION_FEATURE_EXTRACTORS.register("ResNet50Conv5ROIRelationFeatureExtractor")
class ResNet50Conv5ROIFeatureExtractor(nn.Module):
    def __init__(self, config, in_channels):
        super(ResNet50Conv5ROIFeatureExtractor, self).__init__()

        resolution = config.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = config.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = config.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )

        stage = resnet.StageSpec(index=4, block_count=3, return_features=False)
        head = resnet.ResNetHead(
            block_module=config.MODEL.RESNETS.TRANS_FUNC,
            stages=(stage,),
            num_groups=config.MODEL.RESNETS.NUM_GROUPS,
            width_per_group=config.MODEL.RESNETS.WIDTH_PER_GROUP,
            stride_in_1x1=config.MODEL.RESNETS.STRIDE_IN_1X1,
            stride_init=None,
            res2_out_channels=config.MODEL.RESNETS.RES2_OUT_CHANNELS,
            dilation=config.MODEL.RESNETS.RES5_DILATION
        )

        self.pooler = pooler
        self.head = head
        self.out_channels = head.out_channels

    def _union_box_feats(self, x, proposal_pairs):
        proposals_union = [proposal_pair.copy_with_union() for proposal_pair in proposal_pairs]
        x_union = self.pooler(x, proposals_union)
        x = self.head(x_union)
        return x

    def forward(self, x, proposals, proposal_pairs):

        # acquire tensor format per batch data
        # bboxes, cls_prob (N, k)
        # im_inds: (N,1), img ind for each roi in the batch
        obj_box_priors, obj_labels, im_inds \
            = _get_tensor_from_boxlist(proposals, 'labels')
        # get index in the proposal pairs
        _, proposal_idx_pairs, im_inds_pairs = _get_tensor_from_boxlist(
            proposal_pairs, 'idx_pairs')

        rel_inds = _get_rel_inds(im_inds, im_inds_pairs, proposal_idx_pairs)
        x = self._union_box_feats(x, proposal_pairs)
        return x, rel_inds


@registry.ROI_RELATION_FEATURE_EXTRACTORS.register("FPN2MLPRelationFeatureExtractor")
class FPN2MLPFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg, in_channels):
        super(FPN2MLPFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_RELATION_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_RELATION_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_RELATION_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        input_size = in_channels * resolution ** 2
        representation_size = cfg.MODEL.ROI_RELATION_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_RELATION_HEAD.USE_GN
        self.pooler = pooler
        self.fc6 = make_fc(input_size, representation_size, use_gn)
        self.fc7 = make_fc(representation_size, representation_size, use_gn)
        self.out_channels = representation_size

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


@registry.ROI_RELATION_FEATURE_EXTRACTORS.register("FPNXconv1fcRelationFeatureExtractor")
class FPNXconv1fcFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg, in_channels):
        super(FPNXconv1fcFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_RELATION_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_RELATION_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_RELATION_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        self.pooler = pooler

        use_gn = cfg.MODEL.ROI_RELATION_HEAD.USE_GN
        conv_head_dim = cfg.MODEL.ROI_RELATION_HEAD.CONV_HEAD_DIM
        num_stacked_convs = cfg.MODEL.ROI_RELATION_HEAD.NUM_STACKED_CONVS
        dilation = cfg.MODEL.ROI_RELATION_HEAD.DILATION

        xconvs = []
        for ix in range(num_stacked_convs):
            xconvs.append(
                nn.Conv2d(
                    in_channels,
                    conv_head_dim,
                    kernel_size=3,
                    stride=1,
                    padding=dilation,
                    dilation=dilation,
                    bias=False if use_gn else True
                )
            )
            in_channels = conv_head_dim
            if use_gn:
                xconvs.append(group_norm(in_channels))
            xconvs.append(nn.ReLU(inplace=True))

        self.add_module("xconvs", nn.Sequential(*xconvs))
        for modules in [self.xconvs,]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if not use_gn:
                        torch.nn.init.constant_(l.bias, 0)

        input_size = conv_head_dim * resolution ** 2
        representation_size = cfg.MODEL.ROI_RELATION_HEAD.MLP_HEAD_DIM
        self.fc6 = make_fc(input_size, representation_size, use_gn=False)
        self.out_channels = representation_size

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.xconvs(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        return x


def make_roi_relation_feature_extractor(cfg, in_channels):
    func = registry.ROI_RELATION_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_RELATION_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg, in_channels)
