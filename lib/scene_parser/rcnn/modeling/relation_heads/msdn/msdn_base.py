import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
import pdb


class Message_Passing_Unit_v2(nn.Module):
	def __init__(self, fea_size, filter_size = 128):
		super(Message_Passing_Unit_v2, self).__init__()
		self.w = nn.Linear(fea_size, filter_size, bias=True)
		self.fea_size = fea_size
		self.filter_size = filter_size

	def forward(self, unary_term, pair_term):

		if unary_term.size()[0] == 1 and pair_term.size()[0] > 1:
			unary_term = unary_term.expand(pair_term.size()[0], unary_term.size()[1])
		if unary_term.size()[0] > 1 and pair_term.size()[0] == 1:
			pair_term = pair_term.expand(unary_term.size()[0], pair_term.size()[1])

		# print '[unary_term, pair_term]', [unary_term, pair_term]
		gate = self.w(F.relu(unary_term)) * self.w(F.relu(pair_term))
		gate = torch.sigmoid(gate.sum(1))
		# print 'gate', gate
		output = pair_term * gate.expand(gate.size()[0], pair_term.size()[1])

		return output


class Message_Passing_Unit_v1(nn.Module):
	def __init__(self, fea_size, filter_size = 128):
		super(Message_Passing_Unit_v1, self).__init__()
		self.w = nn.Linear(fea_size * 2, filter_size, bias=True)
		self.fea_size = fea_size
		self.filter_size = filter_size

	def forward(self, unary_term, pair_term):

		if unary_term.size()[0] == 1 and pair_term.size()[0] > 1:
			unary_term = unary_term.expand(pair_term.size()[0], unary_term.size()[1])
		if unary_term.size()[0] > 1 and pair_term.size()[0] == 1:
			pair_term = pair_term.expand(unary_term.size()[0], pair_term.size()[1])

		# print '[unary_term, pair_term]', [unary_term, pair_term]
		gate = torch.cat([unary_term, pair_term], 1)
		gate = F.relu(gate)
		gate = torch.sigmoid(self.w(gate)).mean(1)
		# print 'gate', gate
		output = pair_term * gate.view(-1, 1).expand(gate.size()[0], pair_term.size()[1])

		return output

class Gated_Recurrent_Unit(nn.Module):
	def __init__(self, fea_size, dropout):
		super(Gated_Recurrent_Unit, self).__init__()
		self.wih = nn.Linear(fea_size, fea_size, bias=True)
		self.whh = nn.Linear(fea_size, fea_size, bias=True)
		self.dropout = dropout

	def forward(self, input, hidden):
		output = self.wih(F.relu(input)) + self.whh(F.relu(hidden))
		if self.dropout:
			output = F.dropout(output, training=self.training)
		return output



class MSDN_BASE(nn.Module):
	def __init__(self, fea_size, dropout=False, gate_width=128, use_region=False, use_kernel_function=False):
		super(MSDN_BASE, self).__init__()
		#self.w_object = Parameter()
		if use_kernel_function:
			Message_Passing_Unit = Message_Passing_Unit_v2
		else:
			Message_Passing_Unit = Message_Passing_Unit_v1

		self.gate_sub2pred = Message_Passing_Unit(fea_size, gate_width)
		self.gate_obj2pred = Message_Passing_Unit(fea_size, gate_width)
		self.gate_pred2sub = Message_Passing_Unit(fea_size, gate_width)
		self.gate_pred2obj = Message_Passing_Unit(fea_size, gate_width)

		self.GRU_object = Gated_Recurrent_Unit(fea_size, dropout) # nn.GRUCell(fea_size, fea_size) #
		self.GRU_pred = Gated_Recurrent_Unit(fea_size, dropout)

	def forward(self, feature_obj, feature_phrase, feature_region, mps_object, mps_phrase, mps_region):
		raise Exception('Please implement the forward function')

	# Here, we do all the operations outof loop, the loop is just to combine the features
	# Less kernel evoke frequency improve the speed of the model
	def prepare_message(self, target_features, source_features, select_mat, gate_module):
		feature_data = []

		# transfer_list = np.where(select_mat > 0)

		if select_mat.data.sum() == 0:
			temp = Variable(torch.zeros(target_features.size()[1:]), requires_grad=True).type_as(target_features)
			feature_data.append(temp)
		else:
			transfer_list = (select_mat.data > 0).nonzero()
			source_indices = Variable(transfer_list[:, 1])
			target_indices = Variable(transfer_list[:, 0])
			source_f = torch.index_select(source_features, 0, source_indices)
			target_f = torch.index_select(target_features, 0, target_indices)
			transferred_features = gate_module(target_f, source_f)

			for f_id in range(target_features.size()[0]):
				if select_mat[f_id, :].data.sum() > 0:
					feature_indices = (transfer_list[:, 0] == f_id).nonzero()[0]
					indices = Variable(feature_indices)
					features = torch.index_select(transferred_features, 0, indices).mean(0).view(-1)
					feature_data.append(features)
				else:
					temp = Variable(torch.zeros(target_features.size()[1:]), requires_grad=True).type_as(target_features)
					feature_data.append(temp)

		return torch.stack(feature_data, 0)
