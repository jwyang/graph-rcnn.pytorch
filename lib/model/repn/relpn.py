import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.utils.config import cfg
from proposal_layer import _ProposalLayer
from anchor_target_layer import _AnchorTargetLayer
from model.utils.network import _smooth_l1_loss

import numpy as np
import math
import pdb
import time

class _RelPN(nn.Module):
    """ region proposal network """
    def __init__(self, dim=512, n_obj_classes=151):
        super(_RelPN, self).__init__()
        self.anchor_scales = cfg.ANCHOR_SCALES
        self.feat_stride = cfg.FEAT_STRIDE[0]

        roi_feat_dim = n_obj_classes

        if cfg.TRAIN.RELPN_WITH_BBOX_INFO:
            roi_feat_dim += 4

        self.RelPN_bilinear_sub = nn.Sequential(
            nn.Linear(roi_feat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            )

        self.RelPN_bilinear_obj = nn.Sequential(
            nn.Linear(roi_feat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            )

        # define proposal layer
        self.RelPN_proposal = _ProposalLayer(self.feat_stride, self.anchor_scales)

        # define anchor target layer
        self.RelPN_anchor_target = _AnchorTargetLayer(self.feat_stride, self.anchor_scales)

        self.relpn_loss_cls = 0
        self.relpn_loss_box = 0

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        # x = x.permute(0, 3, 1, 2)
        # b c w h
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        # x = x.permute(0, 2, 3, 1)
        return x

    def forward(self, rois, roi_feat, im_info, gt_boxes, num_boxes, use_gt_boxes=False):
        # roi_feat: N x D
        assert roi_feat.dim() == 3, "roi_feat must be B x N x D shape"
        B = roi_feat.size(0)
        N = roi_feat.size(1)
        D = roi_feat.size(2)

        ####################################################
        ### Method 1.Use bilinear to compute the scores  ###
        ####################################################

        if cfg.TRAIN.RELPN_WITH_BBOX_INFO:
            rois_nm = rois.new(rois.size(0), rois.size(1), 4)
            rois_nm[:, :, :2] = rois[:, :, 1:3] / im_info[:, 1]
            rois_nm[:, :, 2:] = rois[:, :, 3:5] / im_info[:, 0]
            roi_feat4prop = torch.cat((roi_feat, Variable(rois_nm)), 2)
            D += 4
        else:
            roi_feat4prop = roi_feat

        # pdb.set_trace()
        roi_feat4prop = roi_feat4prop.view(B * N, D)

        # (B*N) x D ==> (B*N) x D'

        # we do not back-prop through this path
        x_sub = self.RelPN_bilinear_sub(roi_feat4prop.detach())
        D = x_sub.size(1)
        x_sub = x_sub.view(B, N, D)

        x_obj = self.RelPN_bilinear_obj(roi_feat4prop.detach())
        x_obj = x_obj.view(B, N, D)
        x_obj = x_obj.permute(0, 2, 1).contiguous()

        # B x N x D, B x D x N ==> B x N x N
        x_bilinear = torch.bmm(x_sub, x_obj)

        x_bilinear = x_bilinear.view(B, N * N) # B x (N * N)

        vis_score = F.sigmoid(x_bilinear) # B x (N * N)

        ### also compute the spatial relations
        # normalize rois with the image size
        # rois_nm = rois.new(rois.size(0), rois.size(1), 4)
        # rois_nm[:, :, :2] = rois[:, :, 1:3] / im_info[:, 1]
        # rois_nm[:, :, 2:] = rois[:, :, 3:5] / im_info[:, 0]
        #
        # map_x = np.arange(0, rois.size(1))
        # map_y = np.arange(0, rois.size(1))
        # map_x_g, map_y_g = np.meshgrid(map_x, map_y)
        # map_yx = torch.from_numpy(np.vstack((map_y_g.ravel(), map_x_g.ravel())).transpose()).cuda()
        #
        # bbox_feat = rois_nm.new(B, N * N, 4 + 4)
        # for i in range(B):
        #     bbox_feat[i] = torch.cat((rois_nm[i][map_yx[:, 0], :], rois_nm[i][map_yx[:, 1], :]), 1)
        #
        # bbox_feat_v = Variable(bbox_feat.view(B * N * N, 8))
        # x_bbox = self.RelPN_box_nonlinear(bbox_feat_v.detach())
        # x_bbox = x_bbox.view(B, N * N)
        # bbox_score = F.sigmoid(x_bbox)

        # compute Delta(box_1, box_2) = box_1 - box_2 = [delta_x1, delta_x2, delta_y1, delta_y2]
        # concatenate box_1, (box_1 - box_2), box_2 as the feature

        relpn_cls_score = vis_score

        ####################################################
        ###  Method 2.Use linear to compute the scores   ###
        ####################################################
        # map_x = np.arange(0, N)
        # map_y = np.arange(0, N)
        # map_x_g, map_y_g = np.meshgrid(map_x, map_y)
        # map_yx = torch.from_numpy(np.vstack((map_y_g.ravel(), map_x_g.ravel())).transpose()).cuda()
        # proposals = map_yx.expand(B, rois.size(1) * rois.size(1), 2) # B x (N * N) x 2


        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'
        
        roi_pairs, roi_proposals, roi_pairs_scores = self.RelPN_proposal((rois, relpn_cls_score.data, im_info, cfg_key), use_gt_boxes)

        self.relpn_loss_cls = 0

        # generating training labels and build the rpn loss
        if self.training:
            assert gt_boxes is not None, "gt_boxes should not be none"

            relpn_label = self.RelPN_anchor_target((rois, relpn_cls_score.data, gt_boxes, im_info, num_boxes))

            # compute classification loss
            relpn_keep = Variable(relpn_label.view(-1).ne(-1).nonzero().view(-1))

            relpn_cls_score = relpn_cls_score.view(-1)[relpn_keep]
            relpn_label = relpn_label.view(-1)[relpn_keep.data]
            relpn_label = Variable(relpn_label)

            self.relpn_loss_cls = F.binary_cross_entropy(relpn_cls_score, relpn_label)

        return roi_pairs, roi_proposals, roi_pairs_scores, self.relpn_loss_cls
