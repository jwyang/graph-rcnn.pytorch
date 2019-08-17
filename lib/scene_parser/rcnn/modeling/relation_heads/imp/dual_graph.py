# --------------------------------------------------------
# Scene Graph Generation by Iterative Message Passing
# Licensed under The MIT License [see LICENSE for details]
# Written by Danfei Xu
# --------------------------------------------------------

from networks.network import Network
import losses
from fast_rcnn.config import cfg
import net_utils as utils

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


class dual_graph_vrd(basenet):
    def __init__(self, data):
        basenet.__init__(self, data)

        self.num_roi = data['num_roi']
        self.num_rel = data['num_rel']
        self.rel_rois = data['rel_rois']
        self.iterable = True

        self.edge_mask_inds = data['rel_mask_inds']
        self.edge_segment_inds = data['rel_segment_inds']

        self.edge_pair_mask_inds = data['rel_pair_mask_inds']
        self.edge_pair_segment_inds = data['rel_pair_segment_inds']

        # number of refine iterations
        self.n_iter = data['n_iter']
        self.relations = data['relations']

        self.vert_state_dim = 512
        self.edge_state_dim = 512

    def setup(self):
        self.layers = dict({'ims': self.ims, 'rois': self.rois, 'rel_rois': self.rel_rois})
        self._vgg_conv()
        self._vgg_fc()
        self._union_rel_vgg_fc()
        self._cells()
        self._iterate()

    def _cells(self):
        """
        construct RNN cells and states
        """
        # intiialize lstms
        # self.vert_rnn = tf.nn.rnn_cell.GRUCell(self.vert_state_dim, activation=tf.tanh)
        # self.edge_rnn = tf.nn.rnn_cell.GRUCell(self.edge_state_dim, activation=tf.tanh)

        self.vert_rnn = torch.nn.GRUCell(self.vert_state_dim, 512)
        self.edge_rnn = torch.nn.GRUCell(self.edge_state_dim, 512)

        # lstm states
        self.vert_state = self.vert_rnn.zero_state(self.num_roi, tf.float32)
        self.edge_state = self.edge_rnn.zero_state(self.num_rel, tf.float32)

    def _iterate(self):
        (self.feed('vgg_out')
             .fc(self.vert_state_dim, relu=False, name='vert_unary'))

        (self.feed('rel_vgg_out')
             .fc(self.edge_state_dim, relu=True, name='edge_unary'))

        vert_unary = self.get_output('vert_unary')
        edge_unary = self.get_output('edge_unary')
        vert_factor = self._vert_rnn_forward(vert_unary, reuse=False)
        edge_factor = self._edge_rnn_forward(edge_unary, reuse=False)

        for i in xrange(self.n_iter):
            reuse = i > 0
            # compute edge states
            edge_ctx = self._compute_edge_context(vert_factor, edge_factor, reuse=reuse)
            edge_factor = self._edge_rnn_forward(edge_ctx, reuse=True)

            # compute vert states
            vert_ctx = self._compute_vert_context(edge_factor, vert_factor, reuse=reuse)
            vert_factor = self._vert_rnn_forward(vert_ctx, reuse=True)

            
            vert_in = vert_factor
            edge_in = edge_factor

            self._update_inference(vert_in, edge_in, i)

    def _compute_edge_context_hard(self, vert_factor, reduction_mode='max'):
        """
        max or average message pooling
        """
        if reduction_mode=='max':
            return tf.reduce_max(tf.gather(vert_factor, self.relations), [1])
        elif reduction_mode=='mean':
            return tf.reduce_mean(tf.gather(vert_factor, self.relations), [1])

    def _compute_vert_context_hard(self, edge_factor, vert_factor, reduction_mode='max'):
        """
        max or average message pooling
        """
        edge_factor_gathered = utils.pad_and_gather(edge_factor, self.edge_mask_inds, None)

        vert_ctx = utils.padded_segment_reduce(edge_factor_gathered, self.edge_segment_inds,
                                               vert_factor.get_shape()[0], reduction_mode)

        return vert_ctx

    def _compute_edge_context_soft(self, vert_factor, edge_factor, reuse=False):
        """
        attention-based edge message pooling
        """
        vert_pairs = utils.gather_vec_pairs(vert_factor, self.relations)

        sub_vert, obj_vert = tf.split(split_dim=1, num_split=2, value=vert_pairs)
        sub_vert_w_input = tf.concat(concat_dim=1, values=[sub_vert, edge_factor])
        obj_vert_w_input = tf.concat(concat_dim=1, values=[obj_vert, edge_factor])

        # compute compatibility scores
        (self.feed(sub_vert_w_input)
             .fc(1, relu=False, reuse=reuse, name='sub_vert_w_fc')
             .sigmoid(name='sub_vert_score'))
        (self.feed(obj_vert_w_input)
             .fc(1, relu=False, reuse=reuse, name='obj_vert_w_fc')
             .sigmoid(name='obj_vert_score'))

        sub_vert_w = self.get_output('sub_vert_score')
        obj_vert_w = self.get_output('obj_vert_score')

        weighted_sub = tf.mul(sub_vert, sub_vert_w)
        weighted_obj = tf.mul(obj_vert, obj_vert_w)
        return weighted_sub + weighted_obj

    def _compute_vert_context_soft(self, edge_factor, vert_factor, reuse=False):
        """
        attention-based vertex(node) message pooling
        """

        out_edge = utils.pad_and_gather(edge_factor, self.edge_pair_mask_inds[:,0])
        in_edge = utils.pad_and_gather(edge_factor, self.edge_pair_mask_inds[:,1])
        # gather correspounding vert factors
        vert_factor_gathered = tf.gather(vert_factor, self.edge_pair_segment_inds)

        # concat outgoing edges and ingoing edges with gathered vert_factors
        out_edge_w_input = tf.concat(concat_dim=1, values=[out_edge, vert_factor_gathered])
        in_edge_w_input = tf.concat(concat_dim=1, values=[in_edge, vert_factor_gathered])

        # compute compatibility scores
        (self.feed(out_edge_w_input)
             .fc(1, relu=False, reuse=reuse, name='out_edge_w_fc')
             .sigmoid(name='out_edge_score'))
        (self.feed(in_edge_w_input)
             .fc(1, relu=False, reuse=reuse, name='in_edge_w_fc')
             .sigmoid(name='in_edge_score'))

        out_edge_w = self.get_output('out_edge_score')
        in_edge_w = self.get_output('in_edge_score')

        # weight the edge factors with computed weigths
        out_edge_weighted = tf.mul(out_edge, out_edge_w)
        in_edge_weighted = tf.mul(in_edge, in_edge_w)


        edge_sum = out_edge_weighted + in_edge_weighted
        vert_ctx = tf.segment_sum(edge_sum, self.edge_pair_segment_inds)
        return vert_ctx

    def _vert_rnn_forward(self, vert_in, reuse=False):
        with tf.variable_scope('vert_rnn'):
            if reuse: tf.get_variable_scope().reuse_variables()
            (vert_out, self.vert_state) = self.vert_rnn(vert_in, self.vert_state)
        return vert_out

    def _edge_rnn_forward(self, edge_in, reuse=False):
        with tf.variable_scope('edge_rnn'):
            if reuse: tf.get_variable_scope().reuse_variables()
            (edge_out, self.edge_state) = self.edge_rnn(edge_in, self.edge_state)
        return edge_out

    def _update_inference(self, vert_factor, edge_factor, iter_i):
        # make predictions
        reuse = iter_i > 0  # reuse variables

        iter_suffix = '_iter%i' % iter_i if iter_i < self.n_iter - 1 else ''
        self._cls_pred(vert_factor, layer_suffix=iter_suffix, reuse=reuse)
        self._bbox_pred(vert_factor, layer_suffix=iter_suffix, reuse=reuse)
        self._rel_pred(edge_factor, layer_suffix=iter_suffix, reuse=reuse)

    def losses(self):
        return self._sg_losses()


class vrd(basenet):
    """
    Baseline: the visual relation detection module proposed by
    Lu et al.
    """

    def __init__(self, data):
        basenet.__init__(self, data)
        self.rel_rois = data['rel_rois']

    def setup(self):
        self.layers = dict({'ims': self.ims, 'rois': self.rois, 'rel_rois': self.rel_rois})
        self._vgg_conv()
        self._vgg_fc()
        self._union_rel_vgg_fc()
        self._cls_pred('vgg_out')
        self._bbox_pred('vgg_out')
        self._rel_pred('rel_vgg_out')

    def losses(self):
        return self._sg_losses()


class dual_graph_vrd_maxpool(dual_graph_vrd):
    """
    Baseline: context-pooling by max pooling
    """
    def _compute_edge_context(self, vert_factor, edge_factor, reuse):
        return self._compute_edge_context_hard(vert_factor, reduction_mode='max')

    def _compute_vert_context(self, edge_factor, vert_factor, reuse):
        return self._compute_vert_context_hard(edge_factor, vert_factor, reduction_mode='max')


class dual_graph_vrd_avgpool(dual_graph_vrd):
    """
    Baseline: context-pooling by avg. pooling
    """
    def _compute_edge_context(self, vert_factor, edge_factor, reuse):
        return self._compute_edge_context_hard(vert_factor, reduction_mode='mean')

    def _compute_vert_context(self, edge_factor, vert_factor, reuse):
        return self._compute_vert_context_hard(edge_factor, vert_factor, reduction_mode='mean')


class dual_graph_vrd_final(dual_graph_vrd):
    """
    Our final model: context-pooling by attention
    """
    def _compute_edge_context(self, vert_factor, edge_factor, reuse):
        return self._compute_edge_context_soft(vert_factor, edge_factor, reuse)

    def _compute_vert_context(self, edge_factor, vert_factor, reuse):
        return self._compute_vert_context_soft(edge_factor, vert_factor, reuse)
