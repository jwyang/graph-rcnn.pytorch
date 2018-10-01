import numpy as np

def create_graph_data(num_roi, num_rel, relations):
    """
    compute graph structure from relations
    """

    rel_mask = np.zeros((num_roi, num_rel)).astype(np.bool)
    roi_rel_inds = np.ones((num_roi, num_roi)).astype(np.int32) * -1
    for i, rel in enumerate(relations):
        rel_mask[rel[0], i] = True
        rel_mask[rel[1], i] = True
        roi_rel_inds[rel[0], rel[1]] = i

    rel_mask_inds = []
    rel_segment_inds = []
    for i, mask in enumerate(rel_mask):
        mask_inds = np.where(mask)[0].tolist() + [num_rel]
        segment_inds = [i for _ in mask_inds]
        rel_mask_inds += mask_inds
        rel_segment_inds += segment_inds

    # compute relation pair inds
    rel_pair_mask_inds = []  #
    rel_pair_segment_inds = []  # for segment gather
    for i in xrange(num_roi):
        mask_inds = []
        for j in xrange(num_roi):
            out_inds = roi_rel_inds[i,j]
            in_inds = roi_rel_inds[j,i]
            if out_inds >= 0 and in_inds >= 0:
                out_inds = out_inds if out_inds >=0 else num_rel
                in_inds = in_inds if in_inds >=0 else num_rel
                mask_inds.append([out_inds, in_inds])

        mask_inds.append([num_rel, num_rel]) # pad with dummy edge ind
        rel_pair_mask_inds += mask_inds
        rel_pair_segment_inds += [i for _ in mask_inds]

    # sanity check
    for i, inds in enumerate(rel_pair_mask_inds):
        if inds[0] < num_rel:
            assert(relations[inds[0]][0] == rel_pair_segment_inds[i])
        if inds[1] < num_rel:
            assert(relations[inds[1]][1] == rel_pair_segment_inds[i])

    output_dict = {
        'rel_mask_inds': np.array(rel_mask_inds).astype(np.int32),
        'rel_segment_inds': np.array(rel_segment_inds).astype(np.int32),
        'rel_pair_segment_inds': np.array(rel_pair_segment_inds).astype(np.int32),
        'rel_pair_mask_inds': np.array(rel_pair_mask_inds).astype(np.int32),
        'num_roi': num_roi,
        'num_rel': num_rel
    }

    return output_dict


def compute_rel_rois(num_rel, rois, relations):
    """
    union subject boxes and object boxes given a set of rois and relations
    """
    rel_rois = np.zeros([num_rel, 5])
    for i, rel in enumerate(relations):
        sub_im_i = rois[rel[0], 0]
        obj_im_i = rois[rel[1], 0]
        assert(sub_im_i == obj_im_i)
        rel_rois[i, 0] = sub_im_i

        sub_roi = rois[rel[0], 1:]
        obj_roi = rois[rel[1], 1:]
        union_roi = [np.minimum(sub_roi[0], obj_roi[0]),
                    np.minimum(sub_roi[1], obj_roi[1]),
                    np.maximum(sub_roi[2], obj_roi[2]),
                    np.maximum(sub_roi[3], obj_roi[3])]
        rel_rois[i, 1:] = union_roi

    return rel_rois
