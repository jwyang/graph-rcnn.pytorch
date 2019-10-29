# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from lib.scene_parser.rcnn.structures.bounding_box import BoxList
from lib.scene_parser.rcnn.structures.boxlist_ops import boxlist_nms
from lib.scene_parser.rcnn.structures.boxlist_ops import cat_boxlist
from lib.scene_parser.rcnn.modeling.box_coder import BoxCoder


class PostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(
        self,
        score_thresh=0.05,
        nms=0.5,
        detections_per_img=100,
        min_detections_per_img=0,
        box_coder=None,
        cls_agnostic_bbox_reg=False,
        bbox_aug_enabled=False,
        relation_on=False
    ):
        """
        Arguments:
            score_thresh (float)
            nms (float)
            detections_per_img (int)
            box_coder (BoxCoder)
        """
        super(PostProcessor, self).__init__()
        self.score_thresh = score_thresh
        self.nms = nms
        self.detections_per_img = detections_per_img
        self.min_detections_per_img = min_detections_per_img
        if box_coder is None:
            box_coder = BoxCoder(weights=(10., 10., 5., 5.))
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg
        self.bbox_aug_enabled = bbox_aug_enabled
        self.relation_on = relation_on

    def forward(self, x, boxes):
        """
        Arguments:
            x (tuple[tensor, tensor]): x contains the class logits
                and the box_regression from the model.
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """
        class_logit, box_regression = x
        class_prob = F.softmax(class_logit, -1)

        # TODO think about a representation of batch of boxes
        image_shapes = [box.size for box in boxes]
        boxes_per_image = [len(box) for box in boxes]
        features = [box.get_field("features") for box in boxes]
        concat_boxes = torch.cat([a.bbox for a in boxes], dim=0)

        if self.cls_agnostic_bbox_reg:
            box_regression = box_regression[:, -4:]
        proposals = self.box_coder.decode(
            box_regression.view(sum(boxes_per_image), -1), concat_boxes
        )
        if self.cls_agnostic_bbox_reg:
            proposals = proposals.repeat(1, class_prob.shape[1])

        num_classes = class_prob.shape[1]

        proposals = proposals.split(boxes_per_image, dim=0)
        class_prob = class_prob.split(boxes_per_image, dim=0)
        class_logit = class_logit.split(boxes_per_image, dim=0)

        results = []
        for prob, logit, boxes_per_img, features_per_img, image_shape in zip(
            class_prob, class_logit, proposals, features, image_shapes
        ):
            boxlist = self.prepare_boxlist(boxes_per_img, features_per_img, prob, logit, image_shape)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            if not self.bbox_aug_enabled:  # If bbox aug is enabled, we will do it later
                if not self.relation_on:
                    boxlist_filtered = self.filter_results(boxlist, num_classes)
                else:
                    # boxlist_pre = self.filter_results(boxlist, num_classes)
                    boxlist_filtered = self.filter_results_nm(boxlist, num_classes)

                    # to enforce minimum number of detections per image
                    # we will do a binary search on the confidence threshold
                    score_thresh = 0.05
                    while len(boxlist_filtered) < self.min_detections_per_img:
                        score_thresh /= 2.0
                        print(("\nNumber of proposals {} is too small, "
                               "retrying filter_results with score thresh"
                               " = {}").format(len(boxlist_filtered), score_thresh))
                        boxlist_filtered = self.filter_results_nm(boxlist, num_classes, thresh=score_thresh)
            else:
                boxlist_filtered = boxlist

            if len(boxlist) == 0:
                raise ValueError("boxlist shoud not be empty!")

            results.append(boxlist_filtered)
        return results

    def prepare_boxlist(self, boxes, features, scores, logits, image_shape):
        """
        Returns BoxList from `boxes` and adds probability scores information
        as an extra field
        `boxes` has shape (#detections, 4 * #classes), where each row represents
        a list of predicted bounding boxes for each of the object classes in the
        dataset (including the background class). The detections in each row
        originate from the same object proposal.
        `scores` has shape (#detection, #classes), where each row represents a list
        of object detection confidence scores for each of the object classes in the
        dataset (including the background class). `scores[i, j]`` corresponds to the
        box at `boxes[i, j * 4:(j + 1) * 4]`.
        """
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        boxlist = BoxList(boxes, image_shape, mode="xyxy")
        boxlist.add_field("scores", scores)
        boxlist.add_field("logits", logits)
        boxlist.add_field("features", features)
        return boxlist

    def filter_results(self, boxlist, num_classes):
        """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS).
        """
        # unwrap the boxlist to avoid additional overhead.
        # if we had multi-class NMS, we could perform this directly on the boxlist
        boxes = boxlist.bbox.reshape(-1, num_classes * 4)
        scores = boxlist.get_field("scores").reshape(-1, num_classes)
        logits = boxlist.get_field("logits").reshape(-1, num_classes)
        features = boxlist.get_field("features")

        device = scores.device
        result = []
        # Apply threshold on detection probabilities and apply NMS
        # Skip j = 0, because it's the background class
        inds_all = scores > self.score_thresh
        for j in range(1, num_classes):
            inds = inds_all[:, j].nonzero().squeeze(1)
            scores_j = scores[inds, j]
            features_j = features[inds]
            boxes_j = boxes[inds, j * 4 : (j + 1) * 4]
            boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
            boxlist_for_class.add_field("scores", scores_j)
            boxlist_for_class.add_field("features", features_j)
            boxlist_for_class = boxlist_nms(
                boxlist_for_class, self.nms
            )
            num_labels = len(boxlist_for_class)
            boxlist_for_class.add_field(
                "labels", torch.full((num_labels,), j, dtype=torch.int64, device=device)
            )
            result.append(boxlist_for_class)

        result = cat_boxlist(result)
        number_of_detections = len(result)

        # Limit to max_per_image detections **over all classes**
        if number_of_detections > self.detections_per_img > 0:
            cls_scores = result.get_field("scores")
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(), number_of_detections - self.detections_per_img + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep).squeeze(1)
            result = result[keep]
        return result

    def filter_results_nm(self, boxlist, num_classes, thresh=0.05):
        """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS). Similar to Neural-Motif Network
        """
        # unwrap the boxlist to avoid additional overhead.
        # if we had multi-class NMS, we could perform this directly on the boxlist
        boxes = boxlist.bbox.reshape(-1, num_classes * 4)
        scores = boxlist.get_field("scores").reshape(-1, num_classes)
        logits = boxlist.get_field("logits").reshape(-1, num_classes)
        features = boxlist.get_field("features")

        valid_cls = (scores[:, 1:].max(0)[0] > thresh).nonzero() + 1

        nms_mask = scores.clone()
        nms_mask.zero_()

        device = scores.device
        result = []
        # Apply threshold on detection probabilities and apply NMS
        # Skip j = 0, because it's the background class
        inds_all = scores > self.score_thresh
        for j in valid_cls.view(-1).cpu():
            scores_j = scores[:, j]
            boxes_j = boxes[:, j * 4 : (j + 1) * 4]
            boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
            boxlist_for_class.add_field("scores", scores_j)
            boxlist_for_class.add_field("idxs", torch.arange(0, scores.shape[0]).long())
            # boxlist_for_class = boxlist_nms(
            #     boxlist_for_class, self.nms
            # )
            boxlist_for_class = boxlist_nms(
                boxlist_for_class, 0.3
            )
            nms_mask[:, j][boxlist_for_class.get_field("idxs")] = 1

            num_labels = len(boxlist_for_class)
            boxlist_for_class.add_field(
                "labels", torch.full((num_labels,), j, dtype=torch.int64, device=device)
            )
            result.append(boxlist_for_class)

        dists_all = nms_mask * scores

        # filter duplicate boxes
        scores_pre, labels_pre = dists_all.max(1)
        inds_all = scores_pre.nonzero()
        assert inds_all.dim() != 0
        inds_all = inds_all.squeeze(1)

        labels_all = labels_pre[inds_all]
        scores_all = scores_pre[inds_all]
        features_all = features[inds_all]
        logits_all = logits[inds_all]

        box_inds_all = inds_all * scores.shape[1] + labels_all
        result = BoxList(boxlist.bbox.view(-1, 4)[box_inds_all], boxlist.size, mode="xyxy")
        result.add_field("labels", labels_all)
        result.add_field("scores", scores_all)
        result.add_field("logits", logits_all)
        result.add_field("features", features_all)
        number_of_detections = len(result)

        vs, idx = torch.sort(scores_all, dim=0, descending=True)
        idx = idx[vs > thresh]
        if self.detections_per_img < idx.size(0):
            idx = idx[:self.detections_per_img]
        result = result[idx]
        return result

def make_roi_box_post_processor(cfg):
    use_fpn = cfg.MODEL.ROI_HEADS.USE_FPN

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH
    nms_thresh = cfg.MODEL.ROI_HEADS.NMS
    detections_per_img = cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG
    min_detections_per_img = cfg.MODEL.ROI_HEADS.MIN_DETECTIONS_PER_IMG
    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG
    bbox_aug_enabled = cfg.TEST.BBOX_AUG.ENABLED

    postprocessor = PostProcessor(
        score_thresh,
        nms_thresh,
        detections_per_img,
        min_detections_per_img,
        box_coder,
        cls_agnostic_bbox_reg,
        bbox_aug_enabled,
        relation_on=cfg.MODEL.RELATION_ON
    )
    return postprocessor
