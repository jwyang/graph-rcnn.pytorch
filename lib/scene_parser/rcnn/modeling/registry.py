# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from lib.scene_parser.rcnn.utils.registry import Registry

BACKBONES = Registry()
RPN_HEADS = Registry()
ROI_BOX_FEATURE_EXTRACTORS = Registry()
ROI_BOX_PREDICTOR = Registry()
ROI_KEYPOINT_FEATURE_EXTRACTORS = Registry()
ROI_KEYPOINT_PREDICTOR = Registry()
ROI_MASK_FEATURE_EXTRACTORS = Registry()
ROI_MASK_PREDICTOR = Registry()
ROI_RELATION_FEATURE_EXTRACTORS = Registry()
ROI_RELATION_BOX_FEATURE_EXTRACTORS = Registry()
ROI_RELATION_PREDICTOR = Registry()
ROI_RELATION_BOX_PREDICTOR = Registry()
