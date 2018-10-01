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
import math
import yaml
from model.utils.config import cfg
from generate_anchors import generate_anchors
from bbox_transform import bbox_transform_inv, clip_boxes, clip_boxes_batch
from model.co_nms.co_nms_wrapper import co_nms

import pdb

DEBUG = False

class _ProposalLayer(nn.Module):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def __init__(self, feat_stride, scales):
        super(_ProposalLayer, self).__init__()

    def forward(self, input, use_gt_boxes=False):

        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)

        rois = input[0]
        scores = input[1]
        im_info = input[2]
        cfg_key = input[3]

        batch_size = rois.size(0)

        pre_nms_topN  = cfg[cfg_key].RELPN_PRE_NMS_TOP_N
        post_nms_topN = cfg[cfg_key].RELPN_POST_NMS_TOP_N
        nms_thresh    = cfg[cfg_key].RELPN_NMS_THRESH
        min_size      = cfg[cfg_key].RELPN_MIN_SIZE

        if DEBUG:
            print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
            print 'scale: {}'.format(im_info[2])
            print 'score map size: {}'.format(scores.shape)

        map_x = np.arange(0, rois.size(1))
        map_y = np.arange(0, rois.size(1))
        map_x_g, map_y_g = np.meshgrid(map_x, map_y)
        map_yx = torch.from_numpy(np.vstack((map_y_g.ravel(), map_x_g.ravel())).transpose()).cuda()
        proposals = map_yx.expand(batch_size, rois.size(1) * rois.size(1), 2) # B x (N * N) x 2

        # filter diagnal entries
        keep = self._filter_diag(proposals)

        scores_keep = scores.view(-1)[keep].contiguous().view(batch_size, -1).contiguous()
        proposals_keep = proposals.contiguous().view(-1, 2)[keep.nonzero().squeeze(), :].contiguous().view(batch_size, -1, 2).contiguous()
        
        # if use_gt_boxes:
        #     num_proposal = proposals_keep.size(1)
        #     output = scores.new(batch_size, num_proposal, 9).zero_()
        #     output[0,:,0] = 0
        #     output[0,:,1:5] = rois[0][proposals_keep[0][:, 0], :][:, 1:5]
        #     output[0,:,5:] = rois[0][proposals_keep[0][:, 1], :][:, 1:5]
        
        #     return output, proposals_keep, scores_keep.view(1, num_proposal, 1)

        _, order = torch.sort(scores_keep, 1, True)

        if use_gt_boxes:
            post_nms_topN = proposals_keep.size(1)

        output = scores.new(batch_size, post_nms_topN, 9).zero_()
        output_score = scores.new(batch_size, post_nms_topN, 1).zero_()
        output_proposals = proposals.new(batch_size, post_nms_topN, 2).zero_()


        for i in range(batch_size):
            # # 3. remove predicted boxes with either height or width < threshold
            # # (NOTE: convert min_size to input image scale stored in im_info[2])
            proposals_i = proposals_keep[i]
            scores_i = scores_keep[i]

            # # 4. sort all (proposal, score) pairs by score from highest to lowest
            # # 5. take top pre_nms_topN (e.g. 6000)
            order_i = order[i]

            if pre_nms_topN > 0 and pre_nms_topN < scores_keep.numel():
                order_single = order_i[:pre_nms_topN]
            else:
                order_single = order_i

            proposals_single = proposals_i[order_single, :]
            scores_single = scores_i[order_single].view(-1,1)

            # 6. apply nms (e.g. threshold = 0.7)
            # 7. take after_nms_topN (e.g. 300)
            # 8. return the top proposals (-> RoIs top)
            if not use_gt_boxes:
                proposals_subject = rois[i][proposals_single[:, 0], :][:, 1:5]
                proposals_object = rois[i][proposals_single[:, 1], :][:, 1:5]

                keep_idx_i = co_nms(torch.cat((proposals_subject, proposals_object), 1), nms_thresh)
                keep_idx_i = keep_idx_i.long().view(-1)

                if post_nms_topN > 0:
                    keep_idx_i = keep_idx_i[:post_nms_topN]
                proposals_single = proposals_single[keep_idx_i, :]
                scores_single = scores_single[keep_idx_i, :]
            else:
                proposals_single = proposals_single[:post_nms_topN, :]
                scores_single = scores_single[:post_nms_topN, :]

            # padding 0 at the end.
            num_proposal = proposals_single.size(0)
            output[i,:num_proposal,0] = i
            output[i,:num_proposal,1:5] = rois[i][proposals_single[:, 0], :][:, 1:5]
            output[i,:num_proposal,5:] = rois[i][proposals_single[:, 1], :][:, 1:5]
            output_score[i, :num_proposal, 0] = scores_single
            output_proposals[i,:num_proposal, :] = proposals_single
        return output, output_proposals, output_score

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _filter_boxes(self, boxes, min_size):
        """Remove all boxes with any side smaller than min_size."""
        ws = boxes[:, :, 2] - boxes[:, :, 0] + 1
        hs = boxes[:, :, 3] - boxes[:, :, 1] + 1
        keep = ((ws >= min_size.view(-1,1).expand_as(ws)) & (hs >= min_size.view(-1,1).expand_as(hs)))
        return keep

    def _filter_diag(self, roi_pairs):
        """Remove all boxes with any side smaller than min_size."""
        keep = roi_pairs[:, :, 0] != roi_pairs[:, :, 1]
        return keep.view(-1)
