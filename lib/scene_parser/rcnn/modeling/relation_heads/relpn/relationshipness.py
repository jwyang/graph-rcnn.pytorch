import torch
import torch.nn as nn
from .utils import box_pos_encoder

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
