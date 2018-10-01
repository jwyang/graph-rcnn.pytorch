"""The data layer used during training to train a Fast R-CNN network.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
from PIL import Image
import torch

from model.utils.config import cfg
from roi_data_layer.minibatch import get_minibatch
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes

import numpy as np
import random
import time
import pdb

class roibatchLoader(data.Dataset):
  def __init__(self, roidb, batch_size, num_classes, training=True, normalize=None):
    self._roidb = roidb
    self.max_num_box = cfg.MAX_ROI_NUMBER
    self._num_classes = num_classes
    self.training = training
    self.normalize = normalize
    self.batch_size = batch_size
    self._get_aspect_ratio()
    self.data_size = len(self.ratio_list)

    # given the ratio_list, we want to make the ratio same for each batch.
    self.ratio_list_batch = torch.Tensor(self.data_size).zero_()
    num_batch = int(np.ceil(len(self.ratio_index) / batch_size))
    for i in range(num_batch):
        left_idx = i*batch_size
        right_idx = min((i+1)*batch_size-1, self.data_size-1)

        if self.ratio_list[right_idx] < 1:
            # for ratio < 1, we preserve the leftmost in each batch.
            target_ratio = self.ratio_list[left_idx]
        elif self.ratio_list[left_idx] > 1:
            # for ratio > 1, we preserve the rightmost in each batch.
            target_ratio = self.ratio_list[right_idx]
        else:
            # for ratio cross 1, we make it to be 1.
            target_ratio = 1

        self.ratio_list_batch[left_idx:(right_idx+1)] = target_ratio

  def _get_aspect_ratio(self):
    widths = np.array([r['width'] for r in self._roidb])
    heights = np.array([r['height'] for r in self._roidb])
    ratios = widths.astype(float) / heights.astype(float)
    self.need_crop = (ratios > cfg.TRAIN.ASPECT_RATIO_MAX).astype(int) \
            + (ratios < cfg.TRAIN.ASPECT_RATIO_MIN).astype(int)
    ratio_index = np.argsort(ratios)
    self.ratio_list = ratios[ratio_index]
    self.ratio_index = ratio_index

  def __getitem__(self, index):
    if self.training:
        index_ratio = int(self.ratio_index[index])
    else:
        index_ratio = index

    # get the anchor index for current sample index
    # here we set the anchor index to the last one
    # sample in this group
    minibatch_db = [self._roidb[index_ratio]]
    blobs = get_minibatch(minibatch_db, self._num_classes)
    data = torch.from_numpy(blobs['data'])

    Flag = False
    if blobs['gt_boxes'].size == 0:
        gt_boxes = torch.FloatTensor([0,0,10,10,0]).view(1,-1)
        Flag = True
    else:
        #np.random.shuffle(blobs['gt_boxes'])
        gt_boxes = torch.from_numpy(blobs['gt_boxes'])

    gt_boxes = torch.from_numpy(blobs['gt_boxes'])

    if blobs['gt_relations'].size == 0:
        gt_relations = torch.LongTensor([0,0,0]).view(1,-1)
    else:
        gt_relations = torch.from_numpy(blobs['gt_relations']).long()

    # append gt_attributes to gt_boxes
    gt_att_mat = gt_boxes.new(gt_boxes.size(0), 16).zero_()
    gt_boxes = torch.cat((gt_boxes, gt_att_mat), 1)

    # append gt_relations to gt_boxes
    gt_rels_mat = gt_boxes.new(gt_boxes.size(0), gt_boxes.size(0)).zero_()
    gt_rels_mat[gt_relations[:, 0], gt_relations[:, 1]] = gt_relations[:, 2].float()
    gt_boxes = torch.cat((gt_boxes, gt_rels_mat), 1)

    im_info = torch.from_numpy(blobs['im_info'])
    data_height, data_width = data.size(1), data.size(2)

    if self.training:
        # we need to random shuffle the bounding box.
        ratio = self.ratio_list_batch[index]
        # np.random.shuffle(blobs['gt_boxes'])


        ########################################################
        # padding the input image to fixed size for each group #
        ########################################################

        # NOTE: need to cope with vanished gt boxes after cropping

        # get the index range
        # if the image need to crop, crop to the target size.


        if self.need_crop[index_ratio] > 0:
            if ratio < 1:
                # this means that data_width << data_height, we need to crop the
                # data_height
                min_y = int(torch.min(gt_boxes[:,1]))
                max_y = int(torch.max(gt_boxes[:,3]))
                trim_size = int(np.floor(data_width / ratio))
                if trim_size > data_height:
                    trim_size = data_height
                y_s = 0
                box_region = max_y - min_y + 1
                # if min_y == 0:
                #     y_s = 0
                # else:
                #     if (box_region-trim_size) < 0:
                #         y_s_min = max(max_y-trim_size, 0)
                #         y_s_max = min(min_y, data_height-trim_size)
                #         if y_s_min == y_s_max:
                #             y_s = y_s_min
                #         else:
                #             y_s = np.random.choice(range(y_s_min, y_s_max))
                #     else:
                #         y_s_add = int((box_region-trim_size)/2)
                #         if y_s_add == 0:
                #             y_s = min_y
                #         else:
                #             y_s = np.random.choice(range(min_y, min_y+y_s_add))
                # crop the image
                if trim_size <= 0:
                    pdb.set_trace()                
                data = data[:, y_s:(y_s + trim_size), :, :]
        
                # shift y coordiante of gt_boxes
                gt_boxes[:, 1] = gt_boxes[:, 1] - y_s
                gt_boxes[:, 3] = gt_boxes[:, 3] - y_s
        
                # update gt bounding box according the trip
                gt_boxes[:, 1].clamp_(0, trim_size - 1)
                gt_boxes[:, 3].clamp_(0, trim_size - 1)
        
            else:
                # this means that data_width >> data_height, we need to crop the
                # data_width
                min_x = int(torch.min(gt_boxes[:,0]))
                max_x = int(torch.max(gt_boxes[:,2]))
                trim_size = int(np.floor(data_height * ratio))
                if trim_size > data_width:
                    trim_size = data_width
                x_s = 0        
                # box_region = max_x - min_x + 1
                # if min_x == 0:
                #     x_s = 0
                # else:
                #     if (box_region-trim_size) < 0:
                #         x_s_min = max(max_x-trim_size, 0)
                #         x_s_max = min(min_x, data_width-trim_size)
                #         if x_s_min == x_s_max:
                #             x_s = x_s_min
                #         else:
                #             x_s = np.random.choice(range(x_s_min, x_s_max))
                #     else:
                #         x_s_add = int((box_region-trim_size)/2)
                #         if x_s_add == 0:
                #             x_s = min_x
                #         else:
                #             x_s = np.random.choice(range(min_x, min_x+x_s_add))
                # crop the image
                if trim_size <= 0:
                    pdb.set_trace()
                data = data[:, :, x_s:(x_s + trim_size), :]
        
                # shift x coordiante of gt_boxes
                gt_boxes[:, 0] = gt_boxes[:, 0] - x_s
                gt_boxes[:, 2] = gt_boxes[:, 2] - x_s
                # update gt bounding box according the trip
                gt_boxes[:, 0].clamp_(0, trim_size - 1)
                gt_boxes[:, 2].clamp_(0, trim_size - 1)

        # based on the ratio, padding the image.
        if ratio < 1:
            # this means that data_width < data_height
            trim_size = int(np.floor(data_width / ratio))

            padding_data = torch.FloatTensor(int(np.ceil(data_width / ratio)), \
                                             data_width, 3).zero_()
            data_height = data[0].size(0)
            padding_data[:data_height, :, :] = data[0]
            # update im_info
            im_info[0, 0] = padding_data.size(0)
            # print("height %d %d \n" %(index, anchor_idx))
        elif ratio > 1:
            # this means that data_width > data_height
            # if the image need to crop.
            padding_data = torch.FloatTensor(data_height, \
                                             int(np.ceil(data_height * ratio)), 3).zero_()
            data_width = data[0].size(1)
            padding_data[:, :data_width, :] = data[0]
            im_info[0, 1] = padding_data.size(1)
        else:
            trim_size = min(data_height, data_width)
            padding_data = torch.FloatTensor(trim_size, trim_size, 3).zero_()
            padding_data = data[0][:trim_size, :trim_size, :]
            gt_boxes.clamp_(0, trim_size)
            im_info[0, 0] = trim_size
            im_info[0, 1] = trim_size


        if gt_boxes.size(0) > self.max_num_box:
            if not cfg.HAS_RELATIONS:
                gt_boxes = gt_boxes[:self.max_num_box]
            else:
                gt_boxes = gt_boxes[:self.max_num_box, :(self.max_num_box + 21)]

        # check the bounding box:
        not_keep = (gt_boxes[:,0] == gt_boxes[:,2]) | (gt_boxes[:,1] == gt_boxes[:,3])
        keep = torch.nonzero(not_keep == 0).view(-1)

        gt_boxes_padding = torch.FloatTensor(self.max_num_box, self.max_num_box + 21).zero_()
        if keep.numel() != 0:
            gt_boxes = gt_boxes[keep]
            if cfg.HAS_RELATIONS:
                gt_boxes = gt_boxes[:, torch.cat((torch.arange(0, 21).long(), keep + 21), 0)]

            num_boxes = min(gt_boxes.size(0), self.max_num_box)
            gt_boxes_padding[:num_boxes,:gt_boxes.size(1)] = gt_boxes[:num_boxes]
        else:
            num_boxes = 0
        # take the top num_boxes
        # permute trim_data to adapt to downstream processing
        padding_data = padding_data.permute(2, 0, 1).contiguous()
        im_info = im_info.view(3)

        if self.normalize:
            padding_data = padding_data / 255.0
            padding_data = self.normalize(padding_data)
        return padding_data, im_info, gt_boxes_padding, num_boxes
    else:
        data = data.permute(0, 3, 1, 2).contiguous().view(3, data_height, data_width)
        num_boxes = gt_boxes.size(0)
        im_info = im_info.view(3)

        if self.normalize:
            data = data / 255.0
            data = self.normalize(data)

        if Flag:
            num_boxes = 0
        else:
            num_boxes = min(gt_boxes.size(0), self.max_num_box)

        return data, im_info, gt_boxes, num_boxes

  def __len__(self):
    return len(self._roidb)
