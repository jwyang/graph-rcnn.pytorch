import os
import numpy as np
import copy
import pickle
import scipy.sparse
from PIL import Image
import h5py, json
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from lib.scene_parser.rcnn.structures.bounding_box import BoxList
from lib.utils.box import bbox_overlaps

class vg_hdf5(Dataset):
    def __init__(self, cfg, split="train", transforms=None, num_im=-1):
        assert split == "train" or split == "test", "split must be one of [train, val, test]"
        assert num_im >= -1, "the number of samples must be >= 0"
        self.transforms = transforms
        self.data_dir = "datasets/vg_bm"
        self.roidb_file = os.path.join(self.data_dir, "VG-SGG.h5")
        self.image_file = os.path.join(self.data_dir, "imdb_1024.h5")
        # read in dataset from a h5 file and a dict (json) file
        assert os.path.exists(self.data_dir), \
            "cannot find folder {}, please download the visual genome data into this folder".format(self.data_dir)
        self.im_h5 = h5py.File(self.image_file, 'r')
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


        self.split_mask, self.image_index, self.im_sizes, self.gt_boxes, self.gt_classes, self.relationships = load_graphs(
            self.roidb_file, self.image_file,
            self.split, num_im, num_val_im=5000,
            filter_empty_rels=True,
            filter_non_overlap=False and split == "train",
        )

        self.json_category_id_to_contiguous_id = self.class_to_ind

        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

        # self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

    @property
    def coco(self):
        """
        :return: a Coco-like object that we can use to evaluate detection!
        """
        anns = []
        for i, (cls_array, box_array) in enumerate(zip(self.gt_classes, self.gt_boxes)):
            for cls, box in zip(cls_array.tolist(), box_array.tolist()):
                anns.append({
                    'area': (box[3] - box[1] + 1) * (box[2] - box[0] + 1),
                    'bbox': [box[0], box[1], box[2] - box[0] + 1, box[3] - box[1] + 1],
                    'category_id': cls,
                    'id': len(anns),
                    'image_id': i,
                    'iscrowd': 0,
                })
        fauxcoco = COCO()
        fauxcoco.dataset = {
            'info': {'description': 'ayy lmao'},
            'images': [{'id': i} for i in range(self.__len__())],
            'categories': [{'supercategory': 'person',
                               'id': i, 'name': name} for i, name in enumerate(self.ind_to_classes) if name != '__background__'],
            'annotations': anns,
        }
        fauxcoco.createIndex()
        return fauxcoco

    def _im_getter(self, idx):
        w, h = self.im_sizes[idx, :]
        ridx = self.image_index[idx]
        im = self.im_refs[ridx]
        im = im[:, :h, :w] # crop out
        im = im.transpose((1,2,0)) # c h w -> h w c
        return im

    def __len__(self):
        return len(self.image_index)

    # def __getitem__(self, index):
    #     """
    #     get dataset item
    #     """
    #     i = index; assert(self.im_to_first_box[i] >= 0)
    #     # get image
    #     img = Image.fromarray(self._im_getter(i)); width, height = img.size
    #
    #     # get object bounding boxes, labels and relations
    #     obj_boxes = self.all_boxes[self.im_to_first_box[i]:self.im_to_last_box[i]+1,:]
    #     obj_labels = self.labels[self.im_to_first_box[i]:self.im_to_last_box[i]+1]
    #     obj_relations = np.zeros((obj_boxes.shape[0], obj_boxes.shape[0]))
    #     if self.im_to_first_rel[i] >= 0: # if image has relations
    #         predicates = self._relation_predicates[self.im_to_first_rel[i]
    #                                      :self.im_to_last_rel[i]+1]
    #         obj_idx = self._relations[self.im_to_first_rel[i]
    #                                      :self.im_to_last_rel[i]+1]
    #         obj_idx = obj_idx - self.im_to_first_box[i]
    #         assert(np.all(obj_idx>=0) and np.all(obj_idx<obj_boxes.shape[0])) # sanity check
    #         for j, p in enumerate(predicates):
    #             # gt_relations.append([obj_idx[j][0], obj_idx[j][1], p])
    #             obj_relations[obj_idx[j][0], obj_idx[j][1]] = p
    #
    #     target_raw = BoxList(obj_boxes, (width, height), mode="xyxy")
    #     img, target = self.transforms(img, target_raw)
    #     target.add_field("labels", torch.from_numpy(obj_labels))
    #     target.add_field("pred_labels", torch.from_numpy(obj_relations))
    #     target = target.clip_to_image(remove_empty=False)
    #     return img, target, index

    def __getitem__(self, index):
        """
        get dataset item
        """
        # get image
        img = Image.fromarray(self._im_getter(index)); width, height = img.size

        # get object bounding boxes, labels and relations
        obj_boxes = self.gt_boxes[index].copy()
        obj_labels = self.gt_classes[index].copy()
        obj_relation_triplets = self.relationships[index].copy()

        obj_relations = np.zeros((obj_boxes.shape[0], obj_boxes.shape[0]))

        for i in range(obj_relation_triplets.shape[0]):
            subj_id = obj_relation_triplets[i][0]
            obj_id = obj_relation_triplets[i][1]
            pred = obj_relation_triplets[i][2]
            obj_relations[subj_id, obj_id] = pred

        target_raw = BoxList(obj_boxes, (width, height), mode="xyxy")
        img, target = self.transforms(img, target_raw)
        target.add_field("labels", torch.from_numpy(obj_labels))
        target.add_field("pred_labels", torch.from_numpy(obj_relations))
        target.add_field("relation_labels", torch.from_numpy(obj_relation_triplets))
        target = target.clip_to_image(remove_empty=False)

        return img, target, index

    def get_groundtruth(self, index):
        width, height = self.im_sizes[index, :]
        # get object bounding boxes, labels and relations

        obj_boxes = self.gt_boxes[index].copy()
        obj_labels = self.gt_classes[index].copy()
        obj_relation_triplets = self.relationships[index].copy()


        obj_relations = np.zeros((obj_boxes.shape[0], obj_boxes.shape[0]))

        for i in range(obj_relation_triplets.shape[0]):
            subj_id = obj_relation_triplets[i][0]
            obj_id = obj_relation_triplets[i][1]
            pred = obj_relation_triplets[i][2]
            obj_relations[subj_id, obj_id] = pred

        target = BoxList(obj_boxes, (width, height), mode="xyxy")
        target.add_field("labels", torch.from_numpy(obj_labels))
        target.add_field("pred_labels", torch.from_numpy(obj_relations))
        target.add_field("relation_labels", torch.from_numpy(obj_relation_triplets))
        target.add_field("difficult", torch.from_numpy(obj_labels).clone().fill_(0))
        return target

    def get_img_info(self, img_id):
        w, h = self.im_sizes[img_id, :]
        return {"height": h, "width": w}

    # def get_groundtruth(self, index):
    #     i = index; assert(self.im_to_first_box[i] >= 0)
    #     width, height = self.im_sizes[i, :]
    #     # get object bounding boxes, labels and relations
    #     obj_boxes = self.all_boxes[self.im_to_first_box[i]:self.im_to_last_box[i]+1,:]
    #     obj_labels = self.labels[self.im_to_first_box[i]:self.im_to_last_box[i]+1]
    #     obj_relations = np.zeros((obj_boxes.shape[0], obj_boxes.shape[0]))
    #     obj_relation_triplets = np.zeros((self.im_to_last_rel[i] - self.im_to_first_rel[i] + 1, 3))
    #     if self.im_to_first_rel[i] >= 0: # if image has relations
    #         predicates = self._relation_predicates[self.im_to_first_rel[i]
    #                                      :self.im_to_last_rel[i]+1]
    #         obj_idx = self._relations[self.im_to_first_rel[i]
    #                                      :self.im_to_last_rel[i]+1]
    #         obj_idx = obj_idx - self.im_to_first_box[i]
    #         assert(np.all(obj_idx>=0) and np.all(obj_idx<obj_boxes.shape[0])) # sanity check
    #         for j, p in enumerate(predicates):
    #             # gt_relations.append([obj_idx[j][0], obj_idx[j][1], p])
    #             obj_relations[obj_idx[j][0], obj_idx[j][1]] = p
    #             obj_relation_triplets[j][0] = obj_idx[j][0]
    #             obj_relation_triplets[j][1] = obj_idx[j][1]
    #             obj_relation_triplets[j][2] = p
    #
    #     target = BoxList(obj_boxes, (width, height), mode="xyxy")
    #     target.add_field("labels", torch.from_numpy(obj_labels))
    #     target.add_field("pred_labels", torch.from_numpy(obj_relations))
    #     target.add_field("relation_labels", torch.from_numpy(obj_relation_triplets))
    #     target.add_field("difficult", torch.from_numpy(obj_labels).clone().fill_(0))
    #     return target

    def map_class_id_to_class_name(self, class_id):
        return self.ind_to_classes[class_id]


def load_graphs(graphs_file, images_file, mode='train', num_im=-1, num_val_im=0, filter_empty_rels=True,
                filter_non_overlap=False):
    """
    Load the file containing the GT boxes and relations, as well as the dataset split
    :param graphs_file: HDF5
    :param mode: (train, val, or test)
    :param num_im: Number of images we want
    :param num_val_im: Number of validation images
    :param filter_empty_rels: (will be filtered otherwise.)
    :param filter_non_overlap: If training, filter images that dont overlap.
    :return: image_index: numpy array corresponding to the index of images we're using
             boxes: List where each element is a [num_gt, 4] array of ground
                    truth boxes (x1, y1, x2, y2)
             gt_classes: List where each element is a [num_gt] array of classes
             relationships: List where each element is a [num_r, 3] array of
                    (box_ind_1, box_ind_2, predicate) relationships
    """
    if mode not in ('train', 'val', 'test'):
        raise ValueError('{} invalid'.format(mode))

    roi_h5 = h5py.File(graphs_file, 'r')
    im_h5 = h5py.File(images_file, 'r')

    data_split = roi_h5['split'][:]
    split = 2 if mode == 'test' else 0
    split_mask = data_split == split

    # Filter out images without bounding boxes
    split_mask &= roi_h5['img_to_first_box'][:] >= 0
    if filter_empty_rels:
        split_mask &= roi_h5['img_to_first_rel'][:] >= 0

    image_index = np.where(split_mask)[0]
    if num_im > -1:
        image_index = image_index[:num_im]
    if num_val_im > 0:
        if mode == 'val':
            image_index = image_index[:num_val_im]
        elif mode == 'train':
            image_index = image_index[num_val_im:]


    split_mask = np.zeros_like(data_split).astype(bool)
    split_mask[image_index] = True

    # Get box information
    all_labels = roi_h5['labels'][:, 0]
    all_boxes = roi_h5['boxes_{}'.format(1024)][:]  # will index later
    assert np.all(all_boxes[:, :2] >= 0)  # sanity check
    assert np.all(all_boxes[:, 2:] > 0)  # no empty box

    # convert from xc, yc, w, h to x1, y1, x2, y2
    all_boxes[:, :2] = all_boxes[:, :2] - all_boxes[:, 2:] / 2
    all_boxes[:, 2:] = all_boxes[:, :2] + all_boxes[:, 2:]

    im_to_first_box = roi_h5['img_to_first_box'][split_mask]
    im_to_last_box = roi_h5['img_to_last_box'][split_mask]
    im_to_first_rel = roi_h5['img_to_first_rel'][split_mask]
    im_to_last_rel = roi_h5['img_to_last_rel'][split_mask]

    im_widths = im_h5["image_widths"][split_mask]
    im_heights = im_h5["image_heights"][split_mask]

    # load relation labels
    _relations = roi_h5['relationships'][:]
    _relation_predicates = roi_h5['predicates'][:, 0]
    assert (im_to_first_rel.shape[0] == im_to_last_rel.shape[0])
    assert (_relations.shape[0] == _relation_predicates.shape[0])  # sanity check

    # Get everything by image.
    im_sizes = []
    image_index_valid = []
    boxes = []
    gt_classes = []
    relationships = []
    for i in range(len(image_index)):
        boxes_i = all_boxes[im_to_first_box[i]:im_to_last_box[i] + 1, :]
        gt_classes_i = all_labels[im_to_first_box[i]:im_to_last_box[i] + 1]

        if im_to_first_rel[i] >= 0:
            predicates = _relation_predicates[im_to_first_rel[i]:im_to_last_rel[i] + 1]
            obj_idx = _relations[im_to_first_rel[i]:im_to_last_rel[i] + 1] - im_to_first_box[i]
            assert np.all(obj_idx >= 0)
            assert np.all(obj_idx < boxes_i.shape[0])
            rels = np.column_stack((obj_idx, predicates))
        else:
            assert not filter_empty_rels
            rels = np.zeros((0, 3), dtype=np.int32)

        if filter_non_overlap:
            assert mode == 'train'
            inters = bbox_overlaps(torch.from_numpy(boxes_i), torch.from_numpy(boxes_i)).numpy()
            rel_overs = inters[rels[:, 0], rels[:, 1]]
            inc = np.where(rel_overs > 0.0)[0]

            if inc.size > 0:
                rels = rels[inc]
            else:
                split_mask[image_index[i]] = 0
                continue
        image_index_valid.append(image_index[i])
        im_sizes.append(np.array([im_widths[i], im_heights[i]]))
        boxes.append(boxes_i)
        gt_classes.append(gt_classes_i)
        relationships.append(rels)

    im_sizes = np.stack(im_sizes, 0)
    return split_mask, image_index_valid, im_sizes, boxes, gt_classes, relationships
