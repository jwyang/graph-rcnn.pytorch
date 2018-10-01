import os
import os.path as osp
from datasets.imdb import imdb
import numpy as np
import copy
import scipy.sparse
import h5py, json
from config import cfg
from vg_eval import vg_eval
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import pickle
import cPickle

import pdb

class vg_hdf5(imdb):
    def __init__(self, roidb_file, dict_file, imdb_file, rpndb_file, split, num_im):
        imdb.__init__(self, roidb_file[:-3])
        # read in dataset from a h5 file and a dict (json) file
        self.im_h5 = h5py.File(os.path.join(cfg.VG_DIR, imdb_file), 'r')
        self.roi_h5 = h5py.File(os.path.join(cfg.VG_DIR, roidb_file), 'r')

        self.roidb_file = os.path.join(cfg.VG_DIR, roidb_file)
        self.mode = "train" if split == 0 else "test"
        num_val_im = 0
        filter_empty_rels = True
        self.is_train = True if split == 0 else False
        self.filter_non_overlap = False

        # roidb metadata
        self.info = json.load(open(os.path.join(cfg.VG_DIR,
                                                dict_file), 'r'))
        self.im_refs = self.im_h5['images'] # image data reference
        im_scale = self.im_refs.shape[2]

        self._COCO = COCO()

        print('split==%i' % split)
        data_split = self.roi_h5['split'][:]
        self.split = split
        if split >= 0:
            split_mask = data_split == split # current split
        else: # -1
            split_mask = data_split >= 0 # all
        # get rid of images that do not have box
        valid_mask = self.roi_h5['img_to_first_box'][:] >= 0
        if filter_empty_rels:
            valid_mask &= self.roi_h5['img_to_first_rel'][:] >= 0

        valid_mask = np.bitwise_and(split_mask, valid_mask)
        self._image_index = np.where(valid_mask)[0] # split index

        if num_im > -1:
            self._image_index = self._image_index[:num_im]


        if split == 2:
            self.split_mask, self.gt_boxes, self.gt_classes, self.relationships = load_graphs(
                self.roidb_file, self.mode, num_im, num_val_im=num_val_im,
                filter_empty_rels=filter_empty_rels,
                filter_non_overlap=self.filter_non_overlap and self.is_train,
            )


        # override split mask
        split_mask = np.zeros_like(data_split).astype(bool)
        split_mask[self.image_index] = True  # build a split mask
        # if use all images
        self.im_sizes = np.vstack([self.im_h5['image_widths'][split_mask],
                                   self.im_h5['image_heights'][split_mask]]).transpose()

        # filter rpn roidb with split_mask
        if cfg.TRAIN.USE_RPN_DB:
            self.rpn_h5_fn = os.path.join(cfg.VG_DIR, rpndb_file)
            self.rpn_h5 = h5py.File(os.path.join(cfg.VG_DIR, rpndb_file), 'r')
            self.rpn_rois = self.rpn_h5['rpn_rois']
            self.rpn_scores = self.rpn_h5['rpn_scores']
            self.rpn_im_to_roi_idx = np.array(self.rpn_h5['im_to_roi_idx'][split_mask])
            self.rpn_num_rois = np.array(self.rpn_h5['num_rois'][split_mask])

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
        cfg.ind_to_class = self.ind_to_classes

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

        cfg.ind_to_predicate = self.ind_to_predicates

        # Default to roidb handler
        self._roidb_handler = self.gt_roidb

    def im_getter(self, idx):
        w, h = self.im_sizes[idx, :]
        ridx = self.image_index[idx]
        im = self.im_refs[ridx]
        im = im[:, :h, :w] # crop out
        im = im.transpose((1,2,0)) # c h w -> h w c
        return im

    @property
    def coco(self):
        """
        :return: a Coco-like object that we can use to evaluate detection!
        """
        anns = []
        for i, (cls_array, box_array) in enumerate(zip(self.gt_classes[:self.num_imgs], self.gt_boxes[:self.num_imgs])):
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
            'images': [{'id': i} for i in range(self.num_imgs)],
            'categories': [{'supercategory': 'person',
                               'id': i, 'name': name} for i, name in enumerate(self.ind_to_classes) if name != '__background__'],
            'annotations': anns,
        }
        fauxcoco.createIndex()
        return fauxcoco

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        """
        gt_roidb = []
        for i in range(self.num_images):
            assert(self.im_to_first_box[i] >= 0)
            boxes = self.all_boxes[self.im_to_first_box[i]
                                   :self.im_to_last_box[i]+1,:]

            gt_classes = self.labels[self.im_to_first_box[i]
                                     :self.im_to_last_box[i]+1]

            overlaps = np.zeros((boxes.shape[0], self.num_classes))
            for j, o in enumerate(overlaps): # to one-hot
                #if gt_classes[j] > 0: # consider negative sample
                o[gt_classes[j]] = 1.0
            overlaps = scipy.sparse.csr_matrix(overlaps)

            # make ground-truth relations
            gt_relations = []
            if self.im_to_first_rel[i] >= 0: # if image has relations
                predicates = self._relation_predicates[self.im_to_first_rel[i]
                                             :self.im_to_last_rel[i]+1]
                obj_idx = self._relations[self.im_to_first_rel[i]
                                             :self.im_to_last_rel[i]+1]
                obj_idx = obj_idx - self.im_to_first_box[i]
                assert(np.all(obj_idx>=0) and np.all(obj_idx<boxes.shape[0])) # sanity check
                for j, p in enumerate(predicates):
                    gt_relations.append([obj_idx[j][0], obj_idx[j][1], p])

            gt_relations = np.array(gt_relations)

            seg_areas = np.multiply((boxes[:, 2] - boxes[:, 0] + 1),
                                    (boxes[:, 3] - boxes[:, 1] + 1)) # box areas
            gt_roidb.append({'boxes': boxes,
                             'gt_classes' : gt_classes,
                             'gt_overlaps' : overlaps,
                             'gt_relations': gt_relations,
                             'flipped' : False,
                             'seg_areas' : seg_areas,
                             'db_idx': i,
                             'image': lambda im_i=i: self.im_getter(im_i),
                             'roi_scores': np.ones(boxes.shape[0]),
                             'width': self.im_sizes[i][0],
                             'height': self.im_sizes[i][1]})
        return gt_roidb


    def add_rpn_rois(self, gt_roidb_batch, make_copy=True):
        """
        Load precomputed RPN proposals
        """
        gt_roidb = copy.deepcopy(gt_roidb_batch) if make_copy else gt_roidb_batch
        rpn_roidb = self._load_rpn_roidb(gt_roidb)
        roidb = imdb.merge_gt_rpn_roidb(gt_roidb, rpn_roidb)
        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        # load an precomputed ROIDB to the current gt ROIDB
        box_list = []
        score_list = []
        for entry in gt_roidb:
            i = entry['db_idx']
            im_rois = self.rpn_rois[self.rpn_im_to_roi_idx[i]: self.rpn_im_to_roi_idx[i]+self.rpn_num_rois[i], :].copy()
            roi_scores = self.rpn_scores[self.rpn_im_to_roi_idx[i]: self.rpn_im_to_roi_idx[i]+self.rpn_num_rois[i], 0].copy()
            box_list.append(im_rois)
            score_list.append(roi_scores)
        roidb = self.create_roidb_from_box_list(box_list, gt_roidb)
        for i, rdb in enumerate(roidb):
            rdb['roi_scores'] = score_list[i]
        return roidb

    def _get_widths(self):
        return self.im_sizes[:,0]

    def evaluate_attributes(self, all_boxes, output_dir):
        self._write_voc_results_file(self.attributes, all_boxes, output_dir)
        self._do_python_eval(output_dir, eval_attributes = True)
        # if self.config['cleanup']:
        #     for cls in self._attributes:
        #         if cls == '__no_attribute__':
        #             continue
        #         filename = self._get_vg_results_file_template(output_dir).format(cls)
        #         os.remove(filename)

    def _get_vg_results_file_template(self, output_dir):
        filename = 'detections_' + str(self.split) + '_{:s}.txt'
        path = os.path.join(output_dir, filename)
        return path

    def _coco_results_one_category(self, boxes, cat_id):
        results = []
        for im_ind in range(len(boxes)):
            index = self.image_index[im_ind]

            if boxes[im_ind] == []:
                continue
            dets = boxes[im_ind].astype(np.float)
            scores = dets[:, -1]
            xs = dets[:, 0]
            ys = dets[:, 1]
            ws = dets[:, 2] - xs + 1
            hs = dets[:, 3] - ys + 1
            results.extend(
              [{'image_id': im_ind,
                'category_id': cat_id,
                'bbox': [xs[k], ys[k], ws[k], hs[k]],
                'score': scores[k]} for k in range(dets.shape[0])])
        return results

    def _write_coco_results_file(self, all_boxes, res_file):
    	# [{"image_id": 42,
    	#   "category_id": 18,
    	#   "bbox": [258.15,41.29,348.26,243.78],
    	#   "score": 0.236}, ...]
    	results = []
    	for cls_ind, cls in enumerate(self.classes):
      	    if cls == '__background__':
        	   continue
      	    print('Collecting {} results ({:d}/{:d})'.format(cls, cls_ind,
                                                     self.num_classes - 1))
      	    coco_cat_id = self.class_to_ind[cls]
      	    results.extend(self._coco_results_one_category(all_boxes[cls_ind],
                                                     coco_cat_id))
    	print('Writing results json to {}'.format(res_file))
    	with open(res_file, 'w') as fid:
      	    json.dump(results, fid)

    def _print_detection_eval_metrics(self, coco_eval):
        IoU_lo_thresh = 0.5
        IoU_hi_thresh = 0.95

        def _get_thr_ind(coco_eval, thr):
            ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                       (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
            iou_thr = coco_eval.params.iouThrs[ind]
            assert np.isclose(iou_thr, thr)
            return ind

        ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
        ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)
        # precision has dims (iou, recall, cls, area range, max dets)
        # area range index 0: all area ranges
        # max dets index 2: 100 per image
        precision = \
            coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
        ap_default = np.mean(precision[precision > -1])
        print(('~~~~ Mean and per-category AP @ IoU=[{:.2f},{:.2f}] '
               '~~~~').format(IoU_lo_thresh, IoU_hi_thresh))
        print('{:.1f}'.format(100 * ap_default))
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            # minus 1 because of __background__
            precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, cls_ind - 1, 0, 2]
            ap = np.mean(precision[precision > -1])
            print('{:.1f}'.format(100 * ap))

        print('~~~~ Summary metrics ~~~~')
        coco_eval.summarize()

    def _do_detection_eval(self, res_file, output_dir):
        ann_type = 'bbox'

        coco_dt = self.coco.loadRes(res_file)

        coco_eval = COCOeval(self.coco, coco_dt)

        coco_eval.params.useSegm = (ann_type == 'segm')

        coco_eval.evaluate()

        coco_eval.accumulate()

        self._print_detection_eval_metrics(coco_eval)

        eval_file = osp.join(output_dir, 'detection_results.pkl')

        with open(eval_file, 'wb') as fid:
            pickle.dump(coco_eval, fid, pickle.HIGHEST_PROTOCOL)
        print('Wrote COCO eval results to: {}'.format(eval_file))

    def evaluate_detections_coco(self, all_boxes, output_dir):
    	res_file = osp.join(output_dir, ('detections_' +
        	                             "train" if self.split == 0 else "test" +
                	                     '_results'))

        self.num_imgs = len(all_boxes[0])

    	res_file += '.json'
    	self._write_coco_results_file(all_boxes, res_file)
    	# Only do evaluation on non-test sets
    	# if self._image_set.find('test') == -1:
        self._do_detection_eval(res_file, output_dir)
    	# Optionally cleanup results json file
        os.remove(res_file)

    # evaluate based on voc metric
    def evaluate_detections_voc(self, all_boxes, output_dir):
        self._write_voc_results_file(self.classes, all_boxes, output_dir)
        self._do_python_eval(output_dir)
        # if self.config['cleanup']:
        #     for cls in self._classes:
        #         if cls == '__background__':
        #             continue
        #         filename = self._get_vg_results_file_template(output_dir).format(cls)
        #         os.remove(filename)

    def _write_voc_results_file(self, classes, all_boxes, output_dir):
        for cls_ind, cls in enumerate(classes):
            if cls == '__background__':
                continue
            print 'Writing "{}" vg results file'.format(cls)
            filename = self._get_vg_results_file_template(output_dir).format(cls)
            with open(filename, 'wt') as f:
                # for im_ind, index in enumerate(self.image_index):
                # pdb.set_trace()
                for im_ind in range(len(all_boxes[cls_ind])):
                    index = self.image_index[im_ind]
                    #print(cls_ind)
                    #print(im_ind)
                    #print(len(all_boxes))
                    #print(len(all_boxes[cls_ind]))
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(str(index), dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir, pickle=True, eval_attributes = False):
        # We re-use parts of the pascal voc python code for visual genome
        aps = []
        nposs = []
        thresh = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = False
        print 'VOC07 metric? ' + ('Yes' if use_07_metric else 'No')
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        # Load ground truth
        gt_roidb = self.gt_roidb()
        if eval_attributes:
            classes = self.attributes
        else:
            classes = self.classes
        for i, cls in enumerate(classes):
            if cls == '__background__' or cls == '__no_attribute__':
                continue
            filename = self._get_vg_results_file_template(output_dir).format(cls)
            rec, prec, ap, scores, npos = vg_eval(
                filename, gt_roidb, self.image_index, i, ovthresh=0.5,
                use_07_metric=use_07_metric, eval_attributes=eval_attributes)

            # Determine per class detection thresholds that maximise f score
            if npos > 1:
                f = np.nan_to_num((prec*rec)/(prec+rec))
                thresh += [scores[np.argmax(f)]]
            else:
                thresh += [0]
            aps += [ap]
            nposs += [float(npos)]
            print('AP for {} = {:.4f} (npos={:,})'.format(cls, ap, npos))
            if pickle:
                with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                    cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap,
                        'scores': scores, 'npos':npos}, f)

        # Set thresh to mean for classes with poor results
        thresh = np.array(thresh)
        avg_thresh = np.mean(thresh[thresh!=0])
        thresh[thresh==0] = avg_thresh
        if eval_attributes:
            filename = 'attribute_thresholds_' + "train" if self.split == 0 else "test" + '.txt'
        else:
            filename = 'object_thresholds_' + "train" if self.split == 0 else "test" + '.txt'
        path = os.path.join(output_dir, filename)
        with open(path, 'wt') as f:
            for i, cls in enumerate(classes[1:]):
                f.write('{:s} {:.3f}\n'.format(cls, thresh[i]))

        weights = np.array(nposs)
        weights /= weights.sum()
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        #print('Weighted Mean AP = {:.4f}'.format(np.average(aps, weights=weights)))
        print('Mean Detection Threshold = {:.3f}'.format(avg_thresh))
        print('~~~~~~~~')
        print('Results:')
        for ap,npos in zip(aps,nposs):
            print('{:.3f}\t{:.3f}'.format(ap,npos))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** PASCAL VOC Python eval code.')
        print('--------------------------------------------------------------')

def load_graphs(graphs_file, mode='train', num_im=-1, num_val_im=0, filter_empty_rels=True,
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

    BOX_SCALE = 1024
    # Get box information
    all_labels = roi_h5['labels'][:, 0]
    all_boxes = roi_h5['boxes_{}'.format(BOX_SCALE)][:]  # will index later
    assert np.all(all_boxes[:, :2] >= 0)  # sanity check
    assert np.all(all_boxes[:, 2:] > 0)  # no empty box

    # convert from xc, yc, w, h to x1, y1, x2, y2
    all_boxes[:, :2] = all_boxes[:, :2] - all_boxes[:, 2:] / 2
    all_boxes[:, 2:] = all_boxes[:, :2] + all_boxes[:, 2:]

    im_to_first_box = roi_h5['img_to_first_box'][split_mask]
    im_to_last_box = roi_h5['img_to_last_box'][split_mask]
    im_to_first_rel = roi_h5['img_to_first_rel'][split_mask]
    im_to_last_rel = roi_h5['img_to_last_rel'][split_mask]

    # load relation labels
    _relations = roi_h5['relationships'][:]
    _relation_predicates = roi_h5['predicates'][:, 0]
    assert (im_to_first_rel.shape[0] == im_to_last_rel.shape[0])
    assert (_relations.shape[0] == _relation_predicates.shape[0])  # sanity check

    # Get everything by image.
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
            inters = bbox_overlaps(boxes_i, boxes_i)
            rel_overs = inters[rels[:, 0], rels[:, 1]]
            inc = np.where(rel_overs > 0.0)[0]

            if inc.size > 0:
                rels = rels[inc]
            else:
                split_mask[image_index[i]] = 0
                continue

        boxes.append(boxes_i)
        gt_classes.append(gt_classes_i)
        relationships.append(rels)

    return split_mask, boxes, gt_classes, relationships

if __name__ == '__main__':
    import roi_data_layer.roidb as roidb
    import roi_data_layer.layer as layer
    d = vg_hdf5('VG.h5', 'VG-dicts.json', 'imdb_512.h5', 'proposals.h5', 0)
    roidb.prepare_roidb(d)
    roidb.add_bbox_regression_targets(d.roidb)
    l = layer.RoIDataLayer(d.num_classes)
    l.set_roidb(d.roidb)
    l.next_batch()
    from IPython import embed; embed()
