# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Transform a roidb into a trainable roidb by adding a bunch of metadata."""

import numpy as np
from config import cfg
# from fast_rcnn.bbox_transform import bbox_transform
from model.utils.cython_bbox import bbox_overlaps

def prepare_roidb(roidb_batch):
    """Enrich the imdb's roidb by adding some derived quantities that
    are useful for training. This function precomputes the maximum
    overlap, taken over ground-truth boxes, between each ROI and
    each ground-truth box. The class with maximum overlap is also
    recorded.
    """
    for roidb in roidb_batch:
        # need gt_overlaps as a dense array for argmax
        gt_overlaps = roidb['gt_overlaps'].toarray()
        # max overlap with gt over classes (columns)
        max_overlaps = gt_overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = gt_overlaps.argmax(axis=1)
        roidb['max_classes'] = max_classes
        roidb['max_overlaps'] = max_overlaps
        # sanity checks
        # max overlap of 0 => class should be zero (background)
        zero_inds = np.where(max_overlaps == 0)[0]
        assert all(max_classes[zero_inds] == 0)
        # max overlap > 0 => class should not be zero (must be a fg class)
        nonzero_inds = np.where(max_overlaps > 0)[0]
        assert all(max_classes[nonzero_inds] != 0)

def compute_bbox_target_normalization(roidb):
    num_images = len(roidb)
    # Infer number of classes from the number of columns in gt_overlaps
    num_classes = roidb[0]['gt_overlaps'].shape[1]
    for im_i in xrange(num_images):
        rois = roidb[im_i]['boxes']
        max_overlaps = roidb[im_i]['max_overlaps']
        max_classes = roidb[im_i]['max_classes']
        roidb[im_i]['bbox_targets'], roidb[im_i]['gt_ind_assignments'] = \
                _compute_targets(rois, max_overlaps, max_classes)

    class_counts = np.zeros((num_classes, 1)) + cfg.EPS
    sums = np.zeros((num_classes, 4))
    squared_sums = np.zeros((num_classes, 4))
    for im_i in xrange(num_images):
        targets = roidb[im_i]['bbox_targets']
        image_target_classes = np.unique(targets[:, 0]).astype(int)
        for cls in image_target_classes:
            if(cls > 0):
                cls_inds = np.where(targets[:, 0] == cls)[0]
                class_counts[cls] += cls_inds.size
                sums[cls, :] += targets[cls_inds, 1:].sum(axis=0)
                squared_sums[cls, :] += \
                        (targets[cls_inds, 1:] ** 2).sum(axis=0)

    means = sums / class_counts
    stds = np.sqrt(squared_sums / class_counts - means ** 2)
    np.save(cfg.TRAIN.BBOX_TARGET_NORMALIZATION_FILE, {'means': means, 'stds': stds})

    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS:
        print "Normalizing targets"
        for im_i in xrange(num_images):
            targets = roidb[im_i]['bbox_targets']
            image_target_classes = np.unique(targets[:, 0]).astype(int)
            for cls in image_target_classes:
                if(cls > 0):
                    cls_inds = np.where(targets[:, 0] == cls)[0]
                    roidb[im_i]['bbox_targets'][cls_inds, 1:] -= means[cls, :]
                    roidb[im_i]['bbox_targets'][cls_inds, 1:] /= stds[cls, :]
    else:
        print "NOT normalizing targets"

    return means, stds

def add_bbox_regression_targets(roidb, means=None, stds=None):
    """Add information needed to train bounding-box regressors."""
    num_images = len(roidb)
    # Iner number of classes from the number of columns in gt_overlaps
    num_classes = roidb[0]['gt_overlaps'].shape[1]
    for im_i in xrange(num_images):
        rois = roidb[im_i]['boxes']
        max_overlaps = roidb[im_i]['max_overlaps']
        max_classes = roidb[im_i]['max_classes']
        roidb[im_i]['bbox_targets'], roidb[im_i]['fg_gt_ind_assignments'] = \
                _compute_targets(rois, max_overlaps, max_classes)

    # Normalize targets
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS:
        assert(means is not None and stds is not None)
        for im_i in xrange(num_images):
            targets = roidb[im_i]['bbox_targets']
            for cls in xrange(1, num_classes):
                cls_inds = np.where(targets[:, 0] == cls)[0]
                roidb[im_i]['bbox_targets'][cls_inds, 1:] -= means[cls, :]
                roidb[im_i]['bbox_targets'][cls_inds, 1:] /= stds[cls, :]

def _compute_targets(rois, overlaps, labels):
    """Compute bounding-box regression targets for an image."""

    # We are sampling relations from fg rois, hence each
    # fg box must be assigned to an gt box
    assert(cfg.TRAIN.FG_THRESH >= cfg.TRAIN.BBOX_THRESH)

    # Indices of ground-truth ROIs
    gt_inds = np.where(overlaps == 1)[0]

    if len(gt_inds) == 0:
        # Bail if the image has no ground-truth ROIs
        return np.zeros((rois.shape[0], 5), dtype=np.float32)
    else:
        # sanity check
        assert(gt_inds[0] == 0)
        for i in range(1, len(gt_inds)):
            assert(gt_inds[i] - gt_inds[i-1] == 1)
    # Indices of examples for which we try to make predictions
    ex_inds = np.where(overlaps >= cfg.TRAIN.BBOX_THRESH)[0]


    # Get IoU overlap between each ex ROI and gt ROI
    ex_gt_overlaps = bbox_overlaps(
        np.ascontiguousarray(rois[ex_inds, :], dtype=np.float),
        np.ascontiguousarray(rois[gt_inds, :], dtype=np.float))

    # Find which gt ROI each ex ROI has max overlap with:
    # this will be the ex ROI's gt target
    gt_assignment = ex_gt_overlaps.argmax(axis=1)

    # guarding against the case where a gt box doesn't get assigned to itself
    gt_to_ex_inds = [np.where(ex_inds == g)[0][0] for g in gt_inds]
    for i, g in enumerate(gt_to_ex_inds):
        gt_assignment[g] = gt_inds[i]

    # assign rois
    gt_rois = rois[gt_inds[gt_assignment], :]
    ex_rois = rois[ex_inds, :]

    # record target assignments for all foreground rois
    fg_gt_ind_assignment = {}
    for i, e in enumerate(ex_inds):
        if overlaps[e] >= cfg.TRAIN.FG_THRESH:
            fg_gt_ind_assignment[e] = gt_inds[gt_assignment[i]]

    # check if all gt has been assigned
    for g in gt_inds:
        assert(g in list(fg_gt_ind_assignment.values()))

    targets = np.zeros((rois.shape[0], 5), dtype=np.float32)
    targets[ex_inds, 0] = labels[ex_inds]
    # transfer to center and log
    # targets[ex_inds, 1:] = bbox_transform(ex_rois, gt_rois)
    return targets, fg_gt_ind_assignment
