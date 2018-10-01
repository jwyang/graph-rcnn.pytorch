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
from ..utils.config import cfg
from bbox_transform import bbox_transform, bbox_overlaps, co_bbox_overlaps_batch2, bbox_transform_batch2, bbox_overlaps_batch2
import pdb

DEBUG = False

class _RelProposalTargetLayer(nn.Module):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def __init__(self, nclasses_rel):
        super(_RelProposalTargetLayer, self).__init__()
        self._num_classes_rel = nclasses_rel
        self.BBOX_NORMALIZE_MEANS = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
        self.BBOX_NORMALIZE_STDS = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS)
        self.BBOX_INSIDE_WEIGHTS = torch.FloatTensor(cfg.TRAIN.BBOX_INSIDE_WEIGHTS)

    def forward(self, roi_pairs, gt_boxes, num_boxes):

        batch_size = gt_boxes.size(0)

        # compute overlap between gt rel pairs and all roi pairs
        gt_box_pairs = roi_pairs.new(batch_size, cfg.MAX_ROI_PAIR_NUMBER, 9).zero_()

        for i in range(batch_size):
            if (gt_boxes[i, :, 21:] > 0).sum() == 0: # no relation
                continue
            gt_pairs_i = (gt_boxes[i, :, 21:] > 0).nonzero()
            n_rel = min(gt_box_pairs[i].size(0), gt_pairs_i.size(0))
            gt_box_pairs[i][:n_rel, 0:4] = gt_boxes[i][gt_pairs_i[:n_rel, 0]][:, :4]
            gt_box_pairs[i][:n_rel, 4:8] = gt_boxes[i][gt_pairs_i[:n_rel, 1]][:, :4]
            gt_box_pairs[i][:n_rel, 8] = gt_boxes[i][gt_pairs_i[:n_rel, 0], 21 + gt_pairs_i[:n_rel, 1]]
        
        # Include ground-truth boxes in the set of candidate rois
        gt_box_pairs_append = roi_pairs.new(batch_size, gt_box_pairs.size(1), roi_pairs.size(2)).zero_()
        gt_box_pairs_append[:,:,1:9] = gt_box_pairs[:,:,:8]
        for i in range(batch_size):
            gt_box_pairs_append[i, :, 0] = i
        
        # roi_pairs = torch.cat([roi_pairs, gt_box_pairs_append], 1)
        roi_pairs = roi_pairs.contiguous()

        num_images = 1
        rois_per_image = int(cfg.TRAIN.BATCH_SIZE / num_images)
        fg_rois_per_image = int(np.round(cfg.TRAIN.FG_FRACTION * rois_per_image))

        labels, rois, keeps = self._sample_roi_pairs_pytorch(roi_pairs, gt_box_pairs, fg_rois_per_image,
                                                      rois_per_image, self._num_classes_rel)

        return rois, labels, keeps

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def _sample_roi_pairs_pytorch(self, all_roi_pairs, gt_box_pairs, fg_rois_per_image, rois_per_image, num_classes):
        """Generate a random sample of RoIs comprising foreground and background
        examples.
        """
        # overlaps: (rois x gt_boxes)

        overlaps = co_bbox_overlaps_batch2(all_roi_pairs[:,:,1:].contiguous(),
                                           gt_box_pairs[:,:,:8].contiguous())

        max_overlaps, gt_assignment = torch.max(overlaps, 2)

        batch_size = overlaps.size(0)
        num_proposal = overlaps.size(1)
        num_boxes_per_img = overlaps.size(2)

        offset = torch.arange(0, batch_size) * gt_box_pairs.size(1)
        offset = offset.view(-1, 1).type_as(gt_assignment) + gt_assignment

        labels = gt_box_pairs[:,:,8].contiguous().view(-1).index(offset.view(-1))\
                                                            .view(batch_size, -1)

        fg_mask = max_overlaps >= cfg.TRAIN.RELPN_FG_THRESH

        keep_inds_batch = labels.new(batch_size, rois_per_image).zero_()

        labels_rel_batch = labels.new(batch_size, rois_per_image).zero_()

        roi_pairs_batch  = all_roi_pairs.new(batch_size, rois_per_image, 9).zero_()
        # Guard against the case when an image has fewer than max_fg_rois_per_image
        # foreground RoIs
        for i in range(batch_size):

            fg_inds = torch.nonzero(max_overlaps[i] >= cfg.TRAIN.RELPN_FG_THRESH).view(-1)
            fg_num_rois = fg_inds.numel()

            # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
            bg_inds = torch.nonzero((max_overlaps[i] < cfg.TRAIN.RELPN_BG_THRESH_HI) &
                                    (max_overlaps[i] >= cfg.TRAIN.RELPN_BG_THRESH_LO)).view(-1)
            bg_num_rois = bg_inds.numel()

            # print(fg_num_rois, bg_num_rois)

            # pdb.set_trace()

            if fg_num_rois > 0 and bg_num_rois > 0:
                # sampling fg
                fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)
                # rand_num = torch.randperm(fg_num_rois).long().cuda()
                rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).long().cuda()
                fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]

                # sampling bg
                bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image

                # Seems torch.rand has a bug, it will generate very large number and make an error.
                # We use numpy rand instead.
                #rand_num = (torch.rand(bg_rois_per_this_image) * bg_num_rois).long().cuda()
                rand_num = np.floor(np.random.rand(bg_rois_per_this_image) * bg_num_rois)
                rand_num = torch.from_numpy(rand_num).long().cuda()
                bg_inds = bg_inds[rand_num]

            elif fg_num_rois > 0 and bg_num_rois == 0:
                # sampling fg
                #rand_num = torch.floor(torch.rand(rois_per_image) * fg_num_rois).long().cuda()
                rand_num = np.floor(np.random.rand(rois_per_image) * fg_num_rois)
                rand_num = torch.from_numpy(rand_num).long().cuda()
                fg_inds = fg_inds[rand_num]
                fg_rois_per_this_image = rois_per_image
                bg_rois_per_this_image = 0
            elif bg_num_rois > 0 and fg_num_rois == 0:
                # sampling bg
                #rand_num = torch.floor(torch.rand(rois_per_image) * bg_num_rois).long().cuda()
                rand_num = np.floor(np.random.rand(rois_per_image) * bg_num_rois)
                rand_num = torch.from_numpy(rand_num).long().cuda()

                bg_inds = bg_inds[rand_num]
                bg_rois_per_this_image = rois_per_image
                fg_rois_per_this_image = 0
            else:
                print("relpn: bg_num_rois = 0 and fg_num_rois = 0, this should not happen!")

            # The indices that we're selecting (both fg and bg)
            keep_inds = torch.cat([fg_inds, bg_inds], 0)

            keep_inds_batch[i].copy_(keep_inds)

            # Select sampled values from various arrays:
            labels_rel_batch[i].copy_(labels[i][keep_inds])

            # Clamp relation labels for the background RoIs to 0
            labels_rel_batch[i][fg_rois_per_this_image:] = 0

            roi_pairs_batch[i].copy_(all_roi_pairs[i][keep_inds])
            roi_pairs_batch[i,:,0] = i

        return labels_rel_batch, roi_pairs_batch, keep_inds_batch
