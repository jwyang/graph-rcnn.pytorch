# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import numpy.random as npr

from model.utils.config import cfg
from generate_anchors import generate_anchors
from bbox_transform import co_bbox_overlaps_batch, bbox_overlaps, bbox_overlaps_batch, bbox_transform_batch

import pdb

DEBUG = False

class _AnchorTargetLayer(nn.Module):
    """
        Assign anchors to ground-truth targets. Produces anchor classification
        labels and bounding-box regression targets.
    """
    def __init__(self, feat_stride, scales):
        super(_AnchorTargetLayer, self).__init__()

        self._feat_stride = feat_stride
        self._scales = scales
        anchor_scales = scales
        self._anchors = torch.from_numpy(generate_anchors(scales=np.array(anchor_scales))).float()
        self._num_anchors = self._anchors.size(0)

        if DEBUG:
            print 'anchors:'
            print self._anchors
            print 'anchor shapes:'
            print np.hstack((
                self._anchors[:, 2::4] - self._anchors[:, 0::4],
                self._anchors[:, 3::4] - self._anchors[:, 1::4],
            ))
            self._counts = cfg.EPS
            self._sums = np.zeros((1, 4))
            self._squared_sums = np.zeros((1, 4))
            self._fg_sum = 0
            self._bg_sum = 0
            self._count = 0

        # allow boxes to sit over the edge by a small amount
        self._allowed_border = 0  # default is 0

    def forward(self, input):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate 9 anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the 9 anchors
        # filter out-of-image anchors
        rois = input[0]
        relpn_cls_score = input[1]
        gt_boxes = input[2]
        im_info = input[3]
        num_gt_boxes = input[4]

        # map of shape (..., H, W)
        batch_size = relpn_cls_score.size(0)
        num_rel_pairs = relpn_cls_score.size(1)

        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = gt_boxes.new(batch_size, num_rel_pairs).fill_(-1)

        # compute overlap between gt rel pairs and all roi pairs  
        gt_pairs = []
        gt_boxes_subject = []
        gt_boxes_object = []
        max_num_rels = 0     

        gt_box_pairs = rois.new(batch_size, cfg.MAX_ROI_PAIR_NUMBER, 8).zero_()

        for i in range(batch_size):
            if (gt_boxes[i, :, 21:] > 0).sum() == 0: # no relation
                continue
            gt_pairs_i = (gt_boxes[i, :, 21:] > 0).nonzero()
            n_rel = min(gt_box_pairs[i].size(0), gt_pairs_i.size(0))
            gt_box_pairs[i][:n_rel, :4] = gt_boxes[i][gt_pairs_i[:n_rel, 0]][:, :4]
            gt_box_pairs[i][:n_rel, 4:] = gt_boxes[i][gt_pairs_i[:n_rel, 1]][:, :4]

        map_x = np.arange(0, rois.size(1))
        map_y = np.arange(0, rois.size(1))
        map_x_g, map_y_g = np.meshgrid(map_x, map_y)
        map_yx = torch.from_numpy(np.vstack((map_y_g.ravel(), map_x_g.ravel())).transpose()).cuda()

        all_box_pairs = []
        for i in range(batch_size):
            all_box_pairs.append(torch.cat((rois[i][map_yx[:, 0]][:, 1:], rois[i][map_yx[:, 1]][:, 1:]), 1))

        all_box_pairs = torch.stack(all_box_pairs, 0)

        overlaps = co_bbox_overlaps_batch(all_box_pairs, gt_box_pairs)
        
        max_overlaps, argmax_overlaps = torch.max(overlaps, 2)
        gt_max_overlaps, _ = torch.max(overlaps, 1)

        if not cfg.TRAIN.RELPN_CLOBBER_POSITIVES:
            labels[max_overlaps < cfg.TRAIN.RELPN_NEGATIVE_OVERLAP] = 0

        gt_max_overlaps[gt_max_overlaps==0] = 1e-5
        keep = torch.sum(overlaps.eq(gt_max_overlaps.view(batch_size,1,-1).expand_as(overlaps)), 2)

        if torch.sum(keep) > 0:
            labels[keep>0] = 1

        # fg label: above threshold IOU
        labels[max_overlaps >= cfg.TRAIN.RELPN_POSITIVE_OVERLAP] = 1

        if cfg.TRAIN.RELPN_CLOBBER_POSITIVES:
            labels[max_overlaps < cfg.TRAIN.RELPN_NEGATIVE_OVERLAP] = 0

        num_fg = int(cfg.TRAIN.RELPN_FG_FRACTION * cfg.TRAIN.RELPN_BATCHSIZE)

        sum_fg = torch.sum((labels == 1).int(), 1)
        sum_bg = torch.sum((labels == 0).int(), 1)

        for i in range(batch_size):
            # subsample positive labels if we have too many
            if sum_fg[i] > num_fg:
                # fg_inds = torch.nonzero(labels[i] == 1).view(-1)
                # rand_num = torch.randperm(fg_inds.size(0)).type_as(gt_boxes).long()
                # disable_inds = fg_inds[rand_num[:fg_inds.size(0)-num_fg]]
                # labels[i][disable_inds] = -1
                fg_inds = torch.nonzero(labels[i] == 1).view(-1)
                rand_num = torch.from_numpy(np.random.permutation(fg_inds.size(0))).type_as(gt_boxes).long()
                disable_inds = fg_inds[rand_num[:fg_inds.size(0)-num_fg]]
                labels[i][disable_inds] = -1

            num_bg = cfg.TRAIN.RELPN_BATCHSIZE - sum_fg[i]

            # subsample negative labels if we have too many
            if sum_bg[i] > num_bg:
                # bg_inds = torch.nonzero(labels[i] == 0).view(-1)
                # rand_num = torch.randperm(bg_inds.size(0)).type_as(gt_boxes).long()
                # disable_inds = bg_inds[rand_num[:bg_inds.size(0)-num_bg]]
                # labels[i][disable_inds] = -1
                bg_inds = torch.nonzero(labels[i] == 0).view(-1)
                rand_num = torch.from_numpy(np.random.permutation(bg_inds.size(0))).type_as(gt_boxes).long()
                disable_inds = bg_inds[rand_num[:bg_inds.size(0)-num_bg]]
                labels[i][disable_inds] = -1

        return labels

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
