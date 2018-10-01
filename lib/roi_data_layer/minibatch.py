# --------------------------------------------------------
# Adapted from Faster R-CNN (https://github.com/rbgirshick/py-faster-rcnn)
# Written by Danfei Xu
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""

import numpy as np
import numpy.random as npr
from fast_rcnn.config import cfg
#from lib.utils.blob import prep_im_for_blob, im_list_to_blob
from model.utils.blob import prep_im_for_blob, im_list_to_blob
#from datasets.viz import viz_scene_graph
import data_utils
from IPython import embed
from utils.timer import Timer
import pdb

def get_minibatch(roidb, num_classes):
    """Given a mini batch of roidb, construct a data blob from it."""
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                                    size=num_images)
    assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
        format(num_images, cfg.TRAIN.BATCH_SIZE)
    rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
    fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

    im_timer = Timer()
    im_timer.tic()
    im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)
    im_timer.toc()

    blobs = {'data': im_blob}

    # gt boxes: (x1, y1, x2, y2, cls)
    num_boxes = roidb[0]['boxes'].shape[0]
    gt_boxes = np.empty((num_boxes, 5), dtype=np.float32)
    gt_boxes[:, :4] = roidb[0]['boxes'] * im_scales[0]
    gt_boxes[:, 4] = roidb[0]['gt_classes']
    blobs['gt_boxes'] = gt_boxes

    blobs['gt_relations'] = roidb[0]['gt_relations']

    blobs['im_info'] = np.array(
        [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
        dtype=np.float32)
    blobs['img_id'] = roidb[0]['db_idx']

    return blobs

    # pdb.set_trace()

    # # Now, build the region of interest and label blobs
    # rois_blob = np.zeros((0, 5), dtype=np.float32)
    # labels_blob = np.zeros((0), dtype=np.float32)
    # rels_blob = np.zeros((0, 3), dtype=np.int32)
    # bbox_targets_blob = np.zeros((0, 4 * num_classes), dtype=np.float32)
    # bbox_inside_blob = np.zeros(bbox_targets_blob.shape, dtype=np.float32)
    # all_overlaps = []

    # box_idx_offset = 0

    # d_timer = Timer()
    # d_timer.tic()
    # for im_i in xrange(num_images):
    #     # sample graph
    #     roi_inds, rels = _sample_graph(roidb[im_i],
    #                                     fg_rois_per_image,
    #                                     rois_per_image,
    #                                     num_neg_rels=cfg.TRAIN.NUM_NEG_RELS)
    #     if rels.size == 0:
    #         print('batch skipped')
    #         return None

    #     # gather all samples based on the sampled graph
    #     rels, labels, overlaps, im_rois, bbox_targets, bbox_inside_weights =\
    #         _gather_samples(roidb[im_i], roi_inds, rels, num_classes)

    #     # Add to RoIs blob
    #     rois = _project_im_rois(im_rois, im_scales[im_i])

    #     batch_ind = im_i * np.ones((rois.shape[0], 1)) #im id for roi_pooling
    #     rois_blob_this_image = np.hstack((batch_ind, rois))
    #     rois_blob = np.vstack((rois_blob, rois_blob_this_image))
    #     # Add to labels, bbox targets, and bbox loss blobs
    #     labels_blob = np.hstack((labels_blob, labels))
    #     bbox_targets_blob = np.vstack((bbox_targets_blob, bbox_targets))
    #     bbox_inside_blob = np.vstack((bbox_inside_blob, bbox_inside_weights))
    #     all_overlaps = np.hstack((all_overlaps, overlaps))

    #     # offset the relationship reference idx the number of previously
    #     # added box
    #     rels_offset = rels.copy()
    #     rels_offset[:, :2] += box_idx_offset
    #     rels_blob = np.vstack([rels_blob, rels_offset])
    #     box_idx_offset += rois.shape[0]

    #     #viz_inds = np.where(overlaps == 1)[0] # ground truth
    #     #viz_inds = npr.choice(np.arange(rois.shape[0]), size=50, replace=False) # random sample
    #     #viz_inds = np.where(overlaps > cfg.TRAIN.FG_THRESH)[0]  # foreground
    #     #viz_scene_graph(im_blob[im_i], rois, labels, viz_inds, rels)

    # blobs['rois'] = rois_blob.copy()
    # blobs['labels'] = labels_blob.copy().astype(np.int32)
    # blobs['relations'] = rels_blob[:,:2].copy().astype(np.int32)
    # blobs['predicates'] = rels_blob[:,2].copy().astype(np.int32)
    # blobs['bbox_targets'] = bbox_targets_blob.copy()
    # blobs['bbox_inside_weights'] = bbox_inside_blob.copy()
    # blobs['bbox_outside_weights'] = \
    #     np.array(bbox_inside_blob > 0).astype(np.float32).copy()


    # num_roi = rois_blob.shape[0]
    # num_rel = rels_blob.shape[0]
    # blobs['rel_rois'] = data_utils.compute_rel_rois(num_rel,
    #                                                 rois_blob,
    #                                                 rels_blob)

    # d_timer.toc()
    # graph_dict = data_utils.create_graph_data(num_roi, num_rel, rels_blob[:, :2])

    # for k in graph_dict:
    #     blobs[k] = graph_dict[k]

    # return blobs

def _gather_samples(roidb, roi_inds, rels, num_classes):
    """
    join all samples and produce sampled items
    """
    rois = roidb['boxes']
    labels = roidb['max_classes']
    overlaps = roidb['max_overlaps']

    # decide bg rois
    bg_inds = np.where(overlaps < cfg.TRAIN.FG_THRESH)[0]

    labels = labels.copy()
    labels[bg_inds] = 0
    labels = labels[roi_inds]
    # print('num bg = %i' % np.where(labels==0)[0].shape[0])

    # rois and bbox targets
    overlaps = overlaps[roi_inds]
    rois = rois[roi_inds]

    # convert rel index
    roi_ind_map = {}
    for i, roi_i in enumerate(roi_inds):
        roi_ind_map[roi_i] = i
    for i, rel in enumerate(rels):
        rels[i] = [roi_ind_map[rel[0]], roi_ind_map[rel[1]], rel[2]]

    bbox_targets, bbox_inside_weights = _get_bbox_regression_labels(
        roidb['bbox_targets'][roi_inds, :], num_classes)

    return rels, labels, overlaps, rois, bbox_targets, bbox_inside_weights

def _sample_graph(roidb, num_fg_rois, num_rois, num_neg_rels=128):
    """
    Sample a graph from the foreground rois of an image

    roidb: roidb of an image
    rois_per_image: maximum number of rois per image
    """

    gt_rels = roidb['gt_relations']
    # index of assigned gt box for foreground boxes
    fg_gt_ind_assignments = roidb['fg_gt_ind_assignments']

    # find all fg proposals that are mapped to a gt
    gt_to_fg_roi_inds = {}
    all_fg_roi_inds = []
    for ind, gt_ind in fg_gt_ind_assignments.items():
        if gt_ind not in gt_to_fg_roi_inds:
            gt_to_fg_roi_inds[gt_ind] = []
        gt_to_fg_roi_inds[gt_ind].append(ind)
        all_fg_roi_inds.append(ind)

    # print('gt rois = %i' % np.where(roidb['max_overlaps']==1)[0].shape[0])
    # print('assigned gt = %i' % len(gt_to_fg_roi_inds.keys()))
    # dedup the roi inds
    all_fg_roi_inds = np.array(list(set(all_fg_roi_inds)))

    # find all valid relations in fg objects
    pos_rels = []
    for rel in gt_rels:
        for sub_i in gt_to_fg_roi_inds[rel[0]]:
            for obj_i in gt_to_fg_roi_inds[rel[1]]:
                pos_rels.append([sub_i, obj_i, rel[2]])

    # print('num fg rois = %i' % all_fg_roi_inds.shape[0])

    rels = []
    rels_inds = []
    roi_inds = []

    if len(pos_rels) > 0:
        # de-duplicate the relations
        _, indices = np.unique(["{} {}".format(i, j) for i,j,k in pos_rels], return_index=True)
        pos_rels = np.array(pos_rels)[indices, :]
        # print('num pos rels = %i' % pos_rels.shape[0])

        # construct graph based on valid relations
        for rel in pos_rels:
            roi_inds += rel[:2].tolist()
            roi_inds = list(set(roi_inds)) # keep roi inds unique
            rels.append(rel)
            rels_inds.append(rel[:2].tolist())

            if len(roi_inds) >= num_fg_rois:
                break

    # print('sampled rels = %i' % len(rels))

    roi_candidates = np.setdiff1d(all_fg_roi_inds, roi_inds)
    num_rois_to_sample = int(min(num_fg_rois - len(roi_inds), len(roi_candidates)))
    # if not enough rois, sample fg rois
    if num_rois_to_sample > 0:
        roi_sample = npr.choice(roi_candidates, size=num_rois_to_sample,
                                replace=False)
        roi_inds = np.hstack([roi_inds, roi_sample])

    # sample background relations
    sample_rels = []
    sample_rels_inds = []
    for i in roi_inds:
        for j in roi_inds:
            if i != j and [i, j] not in rels_inds:
                sample_rels.append([i,j,0])
                sample_rels_inds.append([i,j])

    if len(sample_rels) > 0:
        # randomly sample negative edges to prevent no edges
        num_neg_rels = np.minimum(len(sample_rels), num_neg_rels)
        inds = npr.choice(np.arange(len(sample_rels)), size=num_neg_rels, replace=False)
        rels += [sample_rels[i] for i in inds]
        rels_inds += [sample_rels_inds[i] for i in inds]


    # if still not enough rois, sample bg rois
    num_rois_to_sample = num_rois - len(roi_inds)
    if num_rois_to_sample > 0:
        bg_roi_inds = _sample_bg_rois(roidb, num_rois_to_sample)
        roi_inds = np.hstack([roi_inds, bg_roi_inds])

    roi_inds = np.array(roi_inds).astype(np.int64)
    # print('sampled rois = %i' % roi_inds.shape[0])
    return roi_inds.astype(np.int64), np.array(rels).astype(np.int64)

def _sample_bg_rois(roidb, num_bg_rois):
    """
    Sample rois from background
    """
    # label = class RoI has max overlap with
    labels = roidb['max_classes']
    overlaps = roidb['max_overlaps']

    bg_inds = np.where(((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (overlaps >= cfg.TRAIN.BG_THRESH_LO)) |
                       (labels == 0))[0]
    bg_rois_per_this_image = np.minimum(num_bg_rois, bg_inds.size)
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

    return bg_inds

def _get_image_blob(roidb, scale_inds):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    for i in xrange(num_images):
        im = roidb[i]['image']() # use image getter
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                                        cfg.TRAIN.MAX_SIZE)
        im_scales.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_scales

def _project_im_rois(im_rois, im_scale_factor):
    """Project image RoIs into the rescaled training image."""
    rois = im_rois * im_scale_factor
    return rois

def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets are stored in a compact form in the
    roidb.

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets). The loss weights
    are similarly expanded.

    Returns:
        bbox_target_data (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """
    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind].astype(np.int64)
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights
