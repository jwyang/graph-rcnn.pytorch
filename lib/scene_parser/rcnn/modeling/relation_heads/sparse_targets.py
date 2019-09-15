import torch
import torch.nn as nn


class FrequencyBias(nn.Module):
    """
    The goal of this is to provide a simplified way of computing
    P(predicate | obj1, obj2, img).
    """

    def __init__(self, pred_dist):
        # pred_dist: [num_classes, num_classes, num_preds] numpy array
        super(FrequencyBias, self).__init__()

        self.num_objs = pred_dist.shape[0]
        pred_dist = torch.FloatTensor(pred_dist).view(-1, pred_dist.shape[2])

        self.obj_baseline = nn.Embedding(pred_dist.size(0), pred_dist.size(1))
        self.obj_baseline.weight.data = pred_dist

    def index_with_labels(self, labels):
        """
        :param labels: [batch_size, 2]
        :return:
        """
        return self.obj_baseline(labels[:, 0] * self.num_objs + labels[:, 1])

    def forward(self, obj_cands0, obj_cands1):
        """
        :param obj_cands0: [batch_size, 151] prob distibution over cands.
        :param obj_cands1: [batch_size, 151] prob distibution over cands.
        :return: [batch_size, #predicates] array, which contains potentials for
        each possibility
        """
        # [batch_size, 151, 151] repr of the joint distribution
        joint_cands = obj_cands0[:, :, None] * obj_cands1[:, None]

        # [151, 151, 51] of targets per.
        baseline = joint_cands.view(joint_cands.size(0), -1) @ self.obj_baseline.weight

        return baseline


def _get_tensor_from_boxlist(proposals, field='labels'):
    # helper function for getting
    # tensor data from BoxList structures

    # /*need to specify data field name*/
    assert proposals[0].extra_fields[field] is not None

    for im_ind, prop_per_im in enumerate(proposals):
        if im_ind == 0:
            num_proposals_im = prop_per_im.bbox.size(0)
            # get data
            bbox_batch = prop_per_im.bbox
            output_batch = prop_per_im.extra_fields[field]

            # im_inds
            im_inds = im_ind * torch.ones(num_proposals_im, 1)
        else:
            num_proposals_im = prop_per_im.bbox.size(0)
            bbox_batch = torch.cat((bbox_batch, prop_per_im.bbox),
                                   dim=0)  # N by 4
            output_batch = torch.cat(
                (output_batch, prop_per_im.extra_fields[field]), dim=0)

            im_inds = torch.cat(
                (im_inds, im_ind * torch.ones(num_proposals_im, 1)), dim=0)

    # TODO: support both cpu and gpu
    im_inds_batch = torch.Tensor(im_inds).long().cuda()

    return bbox_batch, output_batch, im_inds_batch


def _get_rel_inds(im_inds, im_inds_pairs, proposal_idx_pairs):
    rel_ind_sub = proposal_idx_pairs[:, 0]
    rel_ind_obj = proposal_idx_pairs[:, 1]

    # idxs in the rel_ind_sub, rel_ind_obj are based on per image index
    # we need to add those inds by a offset [0,0,0... 64, 64, 64...]
    # per image number objects
    # num_obj_im = torch.unique(im_inds, return_counts=True)[1]
    num_obj_im = torch.unique(im_inds)
    # cum sum torch.cumsum. this is the raw value for offsets
    num_obj_im = torch.cumsum(num_obj_im, dim=0)

    # im_inds -1 for offset value
    # then set 0-th image has offset 0
    rel_ind_offset_im = num_obj_im[im_inds_pairs - 1]
    # num_rels_im = torch.unique(im_inds_pairs, return_counts=True)[1]
    num_rels_im = torch.unique(im_inds_pairs)
    rel_ind_offset_im[:num_rels_im[0]] = 0  # first image needs no offset
    rel_ind_offset_im = torch.squeeze(rel_ind_offset_im)

    rel_ind_sub += rel_ind_offset_im
    rel_ind_obj += rel_ind_offset_im
    return torch.cat((rel_ind_sub[:, None], rel_ind_obj[:, None]), 1)
