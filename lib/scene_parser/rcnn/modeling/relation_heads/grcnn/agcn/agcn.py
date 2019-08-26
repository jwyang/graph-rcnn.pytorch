import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time

def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()

class _Collection_Unit(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(_Collection_Unit, self).__init__()
        self.fc = nn.Linear(dim_in, dim_out, bias=True)
        normal_init(self.fc, 0, 0.01)

    def forward(self, target, source, attention_base):
        # assert attention_base.size(0) == source.size(0), "source number must be equal to attention number"
        fc_out = F.relu(self.fc(source))
        collect = torch.mm(attention_base, fc_out)  # Nobj x Nrel Nrel x dim
        collect_avg = collect / (attention_base.sum(1).view(collect.size(0), 1) + 1e-7)
        return collect_avg

    # def __init__(self, dim, dim_lr):
    #     super(_Collection_Unit, self).__init__()
    #     dim_lowrank = dim_lr
    #     self.fc_lft = nn.Linear(dim, dim_lowrank, bias=True)
    #     self.fc_rgt = nn.Linear(dim_lowrank, dim, bias=True)
    #     normal_init(self.fc_lft, 0, 0.001, cfg.TRAIN.TRUNCATED)
    #     normal_init(self.fc_rgt, 0, 0.001, cfg.TRAIN.TRUNCATED)
    # def forward(self, target, source, attention_base):
    #     # assert attention_base.size(0) == source.size(0), "source number must be equal to attention number"
    #     fc_left_out = self.fc_lft(source)
    #     fc_right_out = self.fc_rgt(fc_left_out)
    #     collect = torch.mm(attention_base, fc_right_out)
    #     collect_avg = collect / (attention_base.sum(1).view(collect.size(0), 1) + 1e-7)
    #     out = F.relu(collect_avg)
    #     return out

class _Update_Unit(nn.Module):
    def __init__(self, dim):
        super(_Update_Unit, self).__init__()
    def forward(self, target, source):
        assert target.size() == source.size(), "source dimension must be equal to target dimension"
        update = target + source
        return update

class _GraphConvolutionLayer_Collect(nn.Module):
    """ graph convolutional layer """
    """ collect information from neighbors """
    def __init__(self, dim_obj, dim_rel):
        super(_GraphConvolutionLayer_Collect, self).__init__()
        self.collect_units = nn.ModuleList()
        self.collect_units.append(_Collection_Unit(dim_rel, dim_obj)) # obj (subject) from rel
        self.collect_units.append(_Collection_Unit(dim_rel, dim_obj)) # obj (object) from rel
        self.collect_units.append(_Collection_Unit(dim_obj, dim_rel)) # rel from obj (subject)
        self.collect_units.append(_Collection_Unit(dim_obj, dim_rel)) # rel from obj (object)
        self.collect_units.append(_Collection_Unit(dim_obj, dim_obj)) # obj from obj

    def forward(self, target, source, attention, unit_id):
        collection = self.collect_units[unit_id](target, source, attention)
        return collection

class _GraphConvolutionLayer_Update(nn.Module):
    """ graph convolutional layer """
    """ update target nodes """
    def __init__(self, dim_obj, dim_rel):
        super(_GraphConvolutionLayer_Update, self).__init__()
        self.update_units = nn.ModuleList()
        self.update_units.append(_Update_Unit(dim_obj)) # obj from others
        self.update_units.append(_Update_Unit(dim_rel)) # rel from others

    def forward(self, target, source, unit_id):
        update = self.update_units[unit_id](target, source)
        return update
