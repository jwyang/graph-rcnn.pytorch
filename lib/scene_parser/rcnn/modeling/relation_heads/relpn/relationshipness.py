import torch
import torch.nn as nn
from .utils import box_pos_encoder
from ..auxilary.multi_head_att import MultiHeadAttention

class Relationshipness(nn.Module):
    """
    compute relationshipness between subjects and objects
    """
    def __init__(self, dim, pos_encoding=False):
        super(Relationshipness, self).__init__()

        self.subj_proj = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 64)
        )

        self.obj_prof = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 64)
        )

        self.pos_encoding = False
        if pos_encoding:
            self.pos_encoding = True
            self.sub_pos_encoder = nn.Sequential(
                nn.Linear(6, 64),
                nn.ReLU(True),
                nn.Linear(64, 64)
            )

            self.obj_pos_encoder = nn.Sequential(
                nn.Linear(6, 64),
                nn.ReLU(True),
                nn.Linear(64, 64)
            )

    def forward(self, x, bbox=None, imsize=None):
        x_subj = self.subj_proj(x) # k x 64
        x_obj = self.obj_prof(x)   # k x 64
        scores = torch.mm(x_subj, x_obj.t()) # k x k
        if self.pos_encoding:
            pos = box_pos_encoder(bbox, imsize[0], imsize[1])
            pos_subj = self.sub_pos_encoder(pos)
            pos_obj = self.obj_pos_encoder(pos)
            pos_scores = torch.mm(pos_subj, pos_obj.t()) # k x k
            scores = scores + pos_scores
        relness = torch.sigmoid(scores)      # k x k
        return relness

class Relationshipnessv2(nn.Module):
    """
    compute relationshipness between subjects and objects
    """
    def __init__(self, dim, pos_encoding=False):
        super(Relationshipnessv2, self).__init__()

        self.subj_proj = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 64)
        )

        self.obj_proj = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 64)
        )

        self.pos_encoding = False
        if pos_encoding:
            self.pos_encoding = True
            self.sub_pos_encoder = nn.Sequential(
                nn.Linear(6, 64),
                nn.ReLU(True),
                nn.Linear(64, 64)
            )

            self.obj_pos_encoder = nn.Sequential(
                nn.Linear(6, 64),
                nn.ReLU(True),
                nn.Linear(64, 64)
            )

        # using context to modulate the relationshipness scores
        self.self_att_subj = MultiHeadAttention(8, 64)
        self.self_att_obj = MultiHeadAttention(8, 64)

        self.self_att_pos_subj = MultiHeadAttention(8, 64)
        self.self_att_pos_obj = MultiHeadAttention(8, 64)

    def forward(self, x, bbox=None, imsize=None):
        x_subj = self.subj_proj(x) # k x 64
        x_subj = self.self_att_subj(x_subj, x_subj, x_subj).squeeze(1)

        x_obj = self.obj_proj(x)   # k x 64
        x_obj = self.self_att_obj(x_obj, x_obj, x_obj).squeeze(1)

        scores = torch.mm(x_subj, x_obj.t()) # k x k
        if self.pos_encoding:
            pos = box_pos_encoder(bbox, imsize[0], imsize[1])
            pos_subj = self.sub_pos_encoder(pos)
            pos_subj = self.self_att_pos_subj(pos_subj, pos_subj, pos_subj).squeeze(1)

            pos_obj = self.obj_pos_encoder(pos)
            pos_obj = self.self_att_pos_obj(pos_obj, pos_obj, pos_obj).squeeze(1)

            pos_scores = torch.mm(pos_subj, pos_obj.t()) # k x k
            scores = scores + pos_scores
        relness = torch.sigmoid(scores)      # k x k
        return relness
