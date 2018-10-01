"""
A helper class for evaluating scene graph prediction tasks
"""

import numpy as np
from sg_eval import eval_relation_recall

class SceneGraphEvaluator:

    def __init__(self, imdb, mode):
        self.roidb = imdb.roidb
        self.result_dict = {}
        self.mode = mode

        self.result_dict = {}
        self.result_dict[self.mode + '_recall'] = {20:[], 50:[], 100:[]}


    def evaluate_scene_graph_entry(self, sg_entry, im_idx, iou_thresh):
        pred_triplets, triplet_boxes = \
            eval_relation_recall(sg_entry, self.roidb[im_idx],
                                self.result_dict,
                                self.mode,
                                iou_thresh=iou_thresh)
        return pred_triplets, triplet_boxes


    def save(self, fn):
        np.save(fn, self.result_dict)


    def print_stats(self):
        print('======================' + self.mode + '============================')
        for k, v in self.result_dict[self.mode + '_recall'].items():
            print('R@%i: %f' % (k, np.mean(v)))
