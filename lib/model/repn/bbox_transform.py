# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------

import torch
import numpy as np
import pdb

def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = torch.log(gt_widths / ex_widths)
    targets_dh = torch.log(gt_heights / ex_heights)

    targets = torch.stack(
        (targets_dx, targets_dy, targets_dw, targets_dh),1)

    return targets

def bbox_transform_batch(ex_rois, gt_rois):

    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, :, 2] - gt_rois[:, :, 0] + 1.0
    gt_heights = gt_rois[:, :, 3] - gt_rois[:, :, 1] + 1.0
    gt_ctr_x = gt_rois[:, :, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, :, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x.view(1,-1).expand_as(gt_ctr_x)) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y.view(1,-1).expand_as(gt_ctr_y)) / ex_heights
    targets_dw = torch.log(gt_widths / ex_widths.view(1,-1).expand_as(gt_widths))
    targets_dh = torch.log(gt_heights / ex_heights.view(1,-1).expand_as(gt_heights))

    targets = torch.stack(
        (targets_dx, targets_dy, targets_dw, targets_dh),2)

    return targets

def bbox_transform_batch2(ex_rois, gt_rois):

    ex_widths = ex_rois[:, :, 2] - ex_rois[:, :, 0] + 1.0
    ex_heights = ex_rois[:,:, 3] - ex_rois[:,:, 1] + 1.0
    ex_ctr_x = ex_rois[:, :, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, :, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, :, 2] - gt_rois[:, :, 0] + 1.0
    gt_heights = gt_rois[:, :, 3] - gt_rois[:, :, 1] + 1.0
    gt_ctr_x = gt_rois[:, :, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, :, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = torch.log(gt_widths / ex_widths)
    targets_dh = torch.log(gt_heights / ex_heights)

    targets = torch.stack(
        (targets_dx, targets_dy, targets_dw, targets_dh),2)

    return targets


def bbox_transform_inv(boxes, deltas, batch_size):
    widths = boxes[:, :, 2] - boxes[:, :, 0] + 1.0
    heights = boxes[:, :, 3] - boxes[:, :, 1] + 1.0
    ctr_x = boxes[:, :, 0] + 0.5 * widths
    ctr_y = boxes[:, :, 1] + 0.5 * heights

    dx = deltas[:, :, 0::4]
    dy = deltas[:, :, 1::4]
    dw = deltas[:, :, 2::4]
    dh = deltas[:, :, 3::4]

    pred_ctr_x = dx * widths.unsqueeze(2) + ctr_x.unsqueeze(2)
    pred_ctr_y = dy * heights.unsqueeze(2) + ctr_y.unsqueeze(2)
    pred_w = np.exp(dw) * widths.unsqueeze(2)
    pred_h = np.exp(dh) * heights.unsqueeze(2)

    pred_boxes = deltas.clone()
    # x1
    pred_boxes[:, :, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, :, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, :, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, :, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes



def clip_boxes_batch(boxes, im_shape, batch_size):
    """
    Clip boxes to image boundaries.
    """
    num_rois = boxes.size(1)

    boxes[boxes < 0] = 0
    # batch_x = (im_shape[:,0]-1).view(batch_size, 1).expand(batch_size, num_rois)
    # batch_y = (im_shape[:,1]-1).view(batch_size, 1).expand(batch_size, num_rois)

    batch_x = im_shape[:, 1] - 1
    batch_y = im_shape[:, 0] - 1

    boxes[:,:,0][boxes[:,:,0] > batch_x] = batch_x
    boxes[:,:,1][boxes[:,:,1] > batch_y] = batch_y
    boxes[:,:,2][boxes[:,:,2] > batch_x] = batch_x
    boxes[:,:,3][boxes[:,:,3] > batch_y] = batch_y

    return boxes

def clip_boxes(boxes, im_shape, batch_size):

    for i in range(batch_size):
        boxes[i,:,0::4].clamp_(0, im_shape[i, 1]-1)
        boxes[i,:,1::4].clamp_(0, im_shape[i, 0]-1)
        boxes[i,:,2::4].clamp_(0, im_shape[i, 1]-1)
        boxes[i,:,3::4].clamp_(0, im_shape[i, 0]-1)

    return boxes


def bbox_overlaps(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (K, 4) ndarray of float

    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = anchors.size(0)
    K = gt_boxes.size(0)

    gt_boxes_area = ((gt_boxes[:,2] - gt_boxes[:,0] + 1) *
                (gt_boxes[:,3] - gt_boxes[:,1] + 1)).view(1, K)

    anchors_area = ((anchors[:,2] - anchors[:,0] + 1) *
                (anchors[:,3] - anchors[:,1] + 1)).view(N, 1)

    boxes = anchors.view(N, 1, 4).expand(N, K, 4)
    query_boxes = gt_boxes.view(1, K, 4).expand(N, K, 4)

    iw = (torch.min(boxes[:,:,2], query_boxes[:,:,2]) -
        torch.max(boxes[:,:,0], query_boxes[:,:,0]) + 1)
    iw[iw < 0] = 0

    ih = (torch.min(boxes[:,:,3], query_boxes[:,:,3]) -
        torch.max(boxes[:,:,1], query_boxes[:,:,1]) + 1)
    ih[ih < 0] = 0

    ua = anchors_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih / ua

    return overlaps

def bbox_overlaps_batch(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (b, K, 5) ndarray of float

    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """

    batch_size = gt_boxes.size(0)
    N = anchors.size(0)
    K = gt_boxes.size(1)

    anchors = anchors.view(1, N, 4).expand(batch_size, N, 4).contiguous()
    gt_boxes = gt_boxes[:,:,:4].contiguous()


    gt_boxes_x = (gt_boxes[:,:,2] - gt_boxes[:,:,0] + 1)
    gt_boxes_y = (gt_boxes[:,:,3] - gt_boxes[:,:,1] + 1)
    gt_boxes_area = (gt_boxes_x * gt_boxes_y).view(batch_size, 1, K)

    anchors_boxes_x = (anchors[:,:,2] - anchors[:,:,0] + 1)
    anchors_boxes_y = (anchors[:,:,3] - anchors[:,:,1] + 1)
    anchors_area = (anchors_boxes_x * anchors_boxes_y).view(batch_size, N, 1)

    gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
    anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)

    boxes = anchors.view(batch_size, N, 1, 4).expand(batch_size, N, K, 4)
    query_boxes = gt_boxes.view(batch_size, 1, K, 4).expand(batch_size, N, K, 4)

    iw = (torch.min(boxes[:,:,:,2], query_boxes[:,:,:,2]) -
        torch.max(boxes[:,:,:,0], query_boxes[:,:,:,0]) + 1)
    iw[iw < 0] = 0

    ih = (torch.min(boxes[:,:,:,3], query_boxes[:,:,:,3]) -
        torch.max(boxes[:,:,:,1], query_boxes[:,:,:,1]) + 1)
    ih[ih < 0] = 0
    ua = anchors_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih / ua

    # mask the overlap here.
    overlaps.masked_fill_(gt_area_zero.view(batch_size, 1, K).expand(batch_size, N, K), 0)
    overlaps.masked_fill_(anchors_area_zero.view(batch_size, N, 1).expand(batch_size, N, K), 0)

    return overlaps

def co_bbox_overlaps_batch(all_box_pairs, gt_box_pairs):
    """
    all_box_pairs: (B, N, 8) ndarray of float
    gt_boxes: (B, K, 8) ndarray of float

    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """

    B = gt_box_pairs.size(0)
    N = all_box_pairs.size(1)
    K = gt_box_pairs.size(1)

    # compute areas of subject boxes for gt_boxes
    gt_boxes_x = (gt_box_pairs[:,:,2] - gt_box_pairs[:,:,0] + 1)  # B x K
    gt_boxes_y = (gt_box_pairs[:,:,3] - gt_box_pairs[:,:,1] + 1)  # B x K
    gt_boxes_area_s = (gt_boxes_x * gt_boxes_y).view(B, 1, K) # B x 1 x K

    # compute areas of subject boxes for all_boxes
    all_boxes_x = (all_box_pairs[:,:,2] - all_box_pairs[:,:,0] + 1) # B x N
    all_boxes_y = (all_box_pairs[:,:,3] - all_box_pairs[:,:,1] + 1) # B x N
    all_boxes_area_s = (all_boxes_x * all_boxes_y).view(B, N, 1) # B x N x 1

    gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
    all_area_zero = (all_boxes_x == 1) & (all_boxes_y == 1)

    # compute areas of object boxes for gt_boxes
    gt_boxes_x = (gt_box_pairs[:,:,6] - gt_box_pairs[:,:,4] + 1) # B x K
    gt_boxes_y = (gt_box_pairs[:,:,7] - gt_box_pairs[:,:,5] + 1) # B x K
    gt_boxes_area_o = (gt_boxes_x * gt_boxes_y).view(B, 1, K) # B x 1 x K

    # compute areas of object boxes for all_boxes    
    all_boxes_x = (all_box_pairs[:,:,6] - all_box_pairs[:,:,4] + 1) # B x N
    all_boxes_y = (all_box_pairs[:,:,7] - all_box_pairs[:,:,5] + 1) # B x N
    all_boxes_area_o = (all_boxes_x * all_boxes_y).view(B, N, 1) # B x N x 1

    gt_area_zero &= (gt_boxes_x == 1) & (gt_boxes_y == 1)
    all_area_zero &= (all_boxes_x == 1) & (all_boxes_y == 1)

    bg_boxes = all_box_pairs.view(B, N, 1, 8).expand(B, N, K, 8)
    fg_boxes = gt_box_pairs.view(B, 1, K, 8).expand(B, N, K, 8)

    # compute intersection areas for subject boxes
    iw_s = (torch.min(bg_boxes[:,:,:,2], fg_boxes[:,:,:,2]) -
        torch.max(bg_boxes[:,:,:,0], fg_boxes[:,:,:,0]) + 1)
    iw_s[iw_s < 0] = 0

    ih_s = (torch.min(bg_boxes[:,:,:,3], fg_boxes[:,:,:,3]) -
        torch.max(bg_boxes[:,:,:,1], fg_boxes[:,:,:,1]) + 1)
    ih_s[ih_s < 0] = 0

    # compute union areas for subject boxes
    interS_s = iw_s * ih_s
    unionS_s = all_boxes_area_s + gt_boxes_area_s - interS_s
    IoUs_s = interS_s / unionS_s

    # compute intersection areas for object boxes
    iw_o = (torch.min(bg_boxes[:,:,:,6], fg_boxes[:,:,:,6]) -
        torch.max(bg_boxes[:,:,:,4], fg_boxes[:,:,:,4]) + 1)
    iw_o[iw_o < 0] = 0

    ih_o = (torch.min(bg_boxes[:,:,:,7], fg_boxes[:,:,:,7]) -
        torch.max(bg_boxes[:,:,:,5], fg_boxes[:,:,:,5]) + 1)
    ih_o[ih_o < 0] = 0

    # compute union areas for object boxes
    interS_o = iw_o * ih_o
    unionS_o = all_boxes_area_o + gt_boxes_area_o - interS_o
    IoUs_o = interS_o / unionS_o

    # One way is computing the ratio between sum of interS and sum of unionS_s,
    # however, this might approve the case when subject boxes have full overlap but
    # object boxes have a few overlap, which is not consistant to the metric where 
    # we want both subject and object overlaps are larger than 0.5
    # IoUs = (interS_s + interS_o) / (unionS_s + unionS_o)

    # Hence, we directly compute the product of two IoUs, i.e., IoUs_s * IoUs_o. This way,
    # we encourage the box pairs whose subject and object boxes both have some overlap with
    # the groundtruth.

    IoUs = IoUs_s * IoUs_o

    # mask the overlap here.
    IoUs.masked_fill_(gt_area_zero.view(B, 1, K).expand(B, N, K), 0)
    IoUs.masked_fill_(all_area_zero.view(B, N, 1).expand(B, N, K), 0)

    return IoUs


def bbox_overlaps_batch2(anchors, gt_boxes):
    """
    anchors: (b, N, 5) ndarray of float
    gt_boxes: (b, K, 5) ndarray of float

    overlaps: (b, N, K) ndarray of overlap between boxes and query_boxes
    """

    batch_size = gt_boxes.size(0)
    N = anchors.size(1)
    K = gt_boxes.size(1)

    anchors = anchors.contiguous()
    gt_boxes = gt_boxes.contiguous()

    gt_boxes_x = (gt_boxes[:,:,2] - gt_boxes[:,:,0] + 1)
    gt_boxes_y = (gt_boxes[:,:,3] - gt_boxes[:,:,1] + 1)
    gt_boxes_area = (gt_boxes_x * gt_boxes_y).view(batch_size, 1, K)

    anchors_boxes_x = (anchors[:,:,2] - anchors[:,:,0] + 1)
    anchors_boxes_y = (anchors[:,:,3] - anchors[:,:,1] + 1)
    anchors_area = (anchors_boxes_x * anchors_boxes_y).view(batch_size, N, 1)

    gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
    anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)

    boxes = anchors.view(batch_size, N, 1, 4).expand(batch_size, N, K, 4)
    query_boxes = gt_boxes.view(batch_size, 1, K, 4).expand(batch_size, N, K, 4)

    iw = (torch.min(boxes[:,:,:,2], query_boxes[:,:,:,2]) -
        torch.max(boxes[:,:,:,0], query_boxes[:,:,:,0]) + 1)
    iw[iw < 0] = 0

    ih = (torch.min(boxes[:,:,:,3], query_boxes[:,:,:,3]) -
        torch.max(boxes[:,:,:,1], query_boxes[:,:,:,1]) + 1)
    ih[ih < 0] = 0
    ua = anchors_area + gt_boxes_area - (iw * ih)

    overlaps = iw * ih / ua

    # mask the overlap here.
    # overlaps.masked_fill_(gt_area_zero.view(batch_size, 1, K).expand(batch_size, N, K), 0)
    # overlaps.masked_fill_(anchors_area_zero.view(batch_size, N, 1).expand(batch_size, N, K), -1)

    return overlaps, anchors_area_zero, gt_area_zero

def co_bbox_overlaps_batch2(all_box_pairs, gt_box_pairs):
    """
    all_box_pairs: (B, N, 8) ndarray of float
    gt_boxes: (B, K, 8) ndarray of float

    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    B = gt_box_pairs.size(0)
    N = all_box_pairs.size(1)
    K = gt_box_pairs.size(1)

    # compute areas of subject boxes for gt_boxes
    gt_boxes_x = (gt_box_pairs[:,:,2] - gt_box_pairs[:,:,0] + 1)  # B x K
    gt_boxes_y = (gt_box_pairs[:,:,3] - gt_box_pairs[:,:,1] + 1)  # B x K
    gt_boxes_area_s = (gt_boxes_x * gt_boxes_y).view(B, 1, K) # B x 1 x K

    # compute areas of subject boxes for all_boxes
    all_boxes_x = (all_box_pairs[:,:,2] - all_box_pairs[:,:,0] + 1) # B x N
    all_boxes_y = (all_box_pairs[:,:,3] - all_box_pairs[:,:,1] + 1) # B x N
    all_boxes_area_s = (all_boxes_x * all_boxes_y).view(B, N, 1) # B x N x 1

    gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
    all_area_zero = (all_boxes_x == 1) & (all_boxes_y == 1)

    # compute areas of object boxes for gt_boxes
    gt_boxes_x = (gt_box_pairs[:,:,6] - gt_box_pairs[:,:,4] + 1) # B x K
    gt_boxes_y = (gt_box_pairs[:,:,7] - gt_box_pairs[:,:,5] + 1) # B x K
    gt_boxes_area_o = (gt_boxes_x * gt_boxes_y).view(B, 1, K) # B x 1 x K

    # compute areas of object boxes for all_boxes    
    all_boxes_x = (all_box_pairs[:,:,6] - all_box_pairs[:,:,4] + 1) # B x N
    all_boxes_y = (all_box_pairs[:,:,7] - all_box_pairs[:,:,5] + 1) # B x N
    all_boxes_area_o = (all_boxes_x * all_boxes_y).view(B, N, 1) # B x N x 1

    gt_area_zero &= (gt_boxes_x == 1) & (gt_boxes_y == 1)
    all_area_zero &= (all_boxes_x == 1) & (all_boxes_y == 1)

    bg_boxes = all_box_pairs.view(B, N, 1, 8).expand(B, N, K, 8)
    fg_boxes = gt_box_pairs.view(B, 1, K, 8).expand(B, N, K, 8)

    # compute intersection areas for subject boxes
    iw_s = (torch.min(bg_boxes[:,:,:,2], fg_boxes[:,:,:,2]) -
        torch.max(bg_boxes[:,:,:,0], fg_boxes[:,:,:,0]) + 1)
    iw_s[iw_s < 0] = 0

    ih_s = (torch.min(bg_boxes[:,:,:,3], fg_boxes[:,:,:,3]) -
        torch.max(bg_boxes[:,:,:,1], fg_boxes[:,:,:,1]) + 1)
    ih_s[ih_s < 0] = 0

    # compute union areas for subject boxes
    interS_s = iw_s * ih_s
    unionS_s = all_boxes_area_s + gt_boxes_area_s - interS_s
    IoUs_s = interS_s / unionS_s

    # compute intersection areas for object boxes
    iw_o = (torch.min(bg_boxes[:,:,:,6], fg_boxes[:,:,:,6]) -
        torch.max(bg_boxes[:,:,:,4], fg_boxes[:,:,:,4]) + 1)
    iw_o[iw_o < 0] = 0

    ih_o = (torch.min(bg_boxes[:,:,:,7], fg_boxes[:,:,:,7]) -
        torch.max(bg_boxes[:,:,:,5], fg_boxes[:,:,:,5]) + 1)
    ih_o[ih_o < 0] = 0

    # compute union areas for object boxes
    interS_o = iw_o * ih_o
    unionS_o = all_boxes_area_o + gt_boxes_area_o - interS_o
    IoUs_o = interS_o / unionS_o

    # the reason using product of two IoUs is above
    # IoUs = (interS_s + interS_o) / (unionS_s + unionS_o)
    IoUs = IoUs_s * IoUs_o

    # mask the overlap here.
    IoUs.masked_fill_(gt_area_zero.view(B, 1, K).expand(B, N, K), 0)
    IoUs.masked_fill_(all_area_zero.view(B, N, 1).expand(B, N, K), -1)

    return IoUs

def combine_box_pairs(box_pairs):
    comb_boxes = box_pairs.new(box_pairs.size(0), 5).zero_()
    comb_boxes[:, 0] = box_pairs[:, 0]
    comb_boxes[:, 1] = torch.min(box_pairs[:,1], box_pairs[:,5])
    comb_boxes[:, 2] = torch.min(box_pairs[:,2], box_pairs[:,6])
    comb_boxes[:, 3] = torch.max(box_pairs[:,3], box_pairs[:,7])
    comb_boxes[:, 4] = torch.max(box_pairs[:,4], box_pairs[:,8])

    return comb_boxes
