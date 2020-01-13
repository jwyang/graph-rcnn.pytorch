import torch
import torch.nn as nn

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

    def forward(self, x, bbox=None):
        x_subj = self.subj_proj(x) # k x 64
        x_obj = self.obj_prof(x)   # k x 64
        scores = torch.mm(x_subj, x_obj.t()) # k x k
        relness = torch.sigmoid(scores)      # k x k
        return relness
