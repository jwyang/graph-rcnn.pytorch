import os
import numpy as np
import copy
import pickle
import scipy.sparse
from PIL import Image
import h5py, json
import torch
from torch.utils.data import Dataset
from lib.scene_parser.rcnn.structures.bounding_box import BoxList

class vg_hdf5(Dataset):
    def __init__(self, cfg, split="train", transforms=None, num_im=-1):
        assert split == "train" or split == "test", "split must be one of [train, val, test]"
        assert num_im >= -1, "the number of samples must be >= 0"
        self.transforms = transforms
        self.data_dir = "datasets/vg_bm"
        self.roidb_file = os.path.join(self.data_dir, "roidb.h5")
        # read in dataset from a h5 file and a dict (json) file
        assert os.path.exists(self.data_dir), \
            "cannot find folder {}, please download the visual genome data into this folder".format(self.data_dir)
        self.im_h5 = h5py.File(os.path.join(self.data_dir, "imdb_1024.h5"), 'r')
        self.roi_h5 = h5py.File(os.path.join(self.data_dir, "VG-SGG.h5"), 'r')
        self.info = json.load(open(os.path.join(self.data_dir, "VG-SGG-dicts.json"), 'r'))

        self.im_refs = self.im_h5['images'] # image data reference
        im_scale = self.im_refs.shape[2]

        print('split=' + split)
        data_split = self.roi_h5['split'][:]

        self.split = split
        if split == "train" or split == "test":
            split_label = 0 if split == "train" else 2
            split_mask = data_split == split_label # current split
        else: # -1
            split_mask = data_split >= 0 # all
        # get rid of images that do not have box
        valid_mask = self.roi_h5['img_to_first_box'][:] >= 0
        valid_mask = np.bitwise_and(split_mask, valid_mask)
        self.image_index = np.where(valid_mask)[0] # split index

        if num_im > -1:
            self.image_index = self.image_index[:num_im]

        # override split mask
        split_mask = np.zeros_like(data_split).astype(bool)
        split_mask[self.image_index] = True  # build a split mask
        # if use all images
        self.im_sizes = np.vstack([self.im_h5['image_widths'][split_mask],
                                   self.im_h5['image_heights'][split_mask]]).transpose()

        # h5 file is in 1-based index
        self.im_to_first_box = self.roi_h5['img_to_first_box'][split_mask]
        self.im_to_last_box = self.roi_h5['img_to_last_box'][split_mask]
        self.all_boxes = self.roi_h5['boxes_%i' % im_scale][:]  # will index later
        self.all_boxes[:, :2] = self.all_boxes[:, :2]
        assert(np.all(self.all_boxes[:, :2] >= 0))  # sanity check
        assert(np.all(self.all_boxes[:, 2:] > 0))  # no empty box


        # convert from xc, yc, w, h to x1, y1, x2, y2
        self.all_boxes[:, :2] = self.all_boxes[:, :2] - self.all_boxes[:, 2:]/2
        self.all_boxes[:, 2:] = self.all_boxes[:, :2] + self.all_boxes[:, 2:]
        self.labels = self.roi_h5['labels'][:,0]

        # add background class
        self.info['label_to_idx']['__background__'] = 0
        self.class_to_ind = self.info['label_to_idx']
        self.ind_to_classes = sorted(self.class_to_ind, key=lambda k:
                               self.class_to_ind[k])
        # cfg.ind_to_class = self.ind_to_classes

        # load relation labels
        self.im_to_first_rel = self.roi_h5['img_to_first_rel'][split_mask]
        self.im_to_last_rel = self.roi_h5['img_to_last_rel'][split_mask]
        self._relations = self.roi_h5['relationships'][:]
        self._relation_predicates = self.roi_h5['predicates'][:,0]
        assert(self.im_to_first_rel.shape[0] == self.im_to_last_rel.shape[0])
        assert(self._relations.shape[0] == self._relation_predicates.shape[0]) # sanity check
        self.predicate_to_ind = self.info['predicate_to_idx']
        self.predicate_to_ind['__background__'] = 0
        self.ind_to_predicates = sorted(self.predicate_to_ind, key=lambda k:
                                  self.predicate_to_ind[k])

        # cfg.ind_to_predicate = self.ind_to_predicates

    def _im_getter(self, idx):
        w, h = self.im_sizes[idx, :]
        ridx = self.image_index[idx]
        im = self.im_refs[ridx]
        im = im[:, :h, :w] # crop out
        im = im.transpose((1,2,0)) # c h w -> h w c
        return im

    def __len__(self):
        return len(self.image_index)

    def __getitem__(self, index):
        """
        get dataset item
        """
        i = index; assert(self.im_to_first_box[i] >= 0)
        # get image
        img = Image.fromarray(self._im_getter(i)); width, height = img.size

        # get object bounding boxes, labels and relations
        obj_boxes = self.all_boxes[self.im_to_first_box[i]:self.im_to_last_box[i]+1,:]
        obj_labels = self.labels[self.im_to_first_box[i]:self.im_to_last_box[i]+1]
        obj_relations = np.zeros((obj_boxes.shape[0], obj_boxes.shape[0]))
        if self.im_to_first_rel[i] >= 0: # if image has relations
            predicates = self._relation_predicates[self.im_to_first_rel[i]
                                         :self.im_to_last_rel[i]+1]
            obj_idx = self._relations[self.im_to_first_rel[i]
                                         :self.im_to_last_rel[i]+1]
            obj_idx = obj_idx - self.im_to_first_box[i]
            assert(np.all(obj_idx>=0) and np.all(obj_idx<obj_boxes.shape[0])) # sanity check
            for j, p in enumerate(predicates):
                # gt_relations.append([obj_idx[j][0], obj_idx[j][1], p])
                obj_relations[obj_idx[j][0], obj_idx[j][1]] = p

        target_raw = BoxList(obj_boxes, (width, height), mode="xyxy")
        img, target = self.transforms(img, target_raw)
        target.add_field("labels", torch.from_numpy(obj_labels))
        target.add_field("pred_labels", torch.from_numpy(obj_relations))
        target = target.clip_to_image(remove_empty=False)
        return img, target, index

    def get_img_info(self, img_id):
        w, h = self.im_sizes[img_id, :]
        return {"height": h, "width": w}
