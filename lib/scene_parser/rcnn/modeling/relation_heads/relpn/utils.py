import torch

def box_pos_encoder(bboxes, width, height):
    """
    bounding box encoding
    """
    bboxes_enc = bboxes.clone()

    dim0 = bboxes_enc[:, 0] / width
    dim1 = bboxes_enc[:, 1] / height
    dim2 = bboxes_enc[:, 2] / width
    dim3 = bboxes_enc[:, 3] / height
    dim4 = (bboxes_enc[:, 2] - bboxes_enc[:, 0]) * (bboxes_enc[:, 3] - bboxes_enc[:, 1]) / height / width
    dim5 = (bboxes_enc[:, 3] - bboxes_enc[:, 1]) / (bboxes_enc[:, 2] - bboxes_enc[:, 0] + 1)

    return torch.stack((dim0,dim1,dim2,dim3,dim4,dim5), 1)
