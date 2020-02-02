import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as FN
import numpy as np

dgl.load_backend('pytorch')

from .gcn import init_bias, init_weight


class LinearResnet(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearResnet, self).__init__()
        self.scaling = False
        if input_size != output_size:
            self.scaling = True
            self.iscale = nn.Linear(input_size, output_size)
            init_weight(self.iscale.weight, 'xavier_uniform_', 'linear')
            init_bias(self.iscale.bias)
        self.W1 = nn.Linear(input_size, output_size)
        self.bn1 = torch.nn.BatchNorm1d(output_size)
        self.W2 = nn.Linear(output_size, output_size)
        self.bn2 = torch.nn.BatchNorm1d(output_size)

        init_weight(self.W1.weight, 'xavier_uniform_', 'leaky_relu')
        init_bias(self.W1.bias)
        init_weight(self.W2.weight, 'xavier_uniform_', 'leaky_relu')
        init_bias(self.W2.bias)

    def forward(self, h):
        identity = h
        if self.scaling:
            identity = self.iscale(identity)
        out = F.leaky_relu(self.bn1(self.W1(h)), negative_slope=0.1)
        out = F.leaky_relu(self.bn2(self.W2(out)), negative_slope=0.1)
        out = out + identity
        return out


def mix_embeddings(ndata, projection):
    """Adds external (categorical and numeric) features into node representation G.ndata['h']"""
    h_concat = torch.cat([ndata['h'], ndata['content']], 1)
    ndata['h'] = projection(h_concat)


class GraphSageConvWithSampling(nn.Module):
    def __init__(self, feature_size, width, dropout, conv_depth, activation_last_layer):
        super(GraphSageConvWithSampling, self).__init__()

        self.feature_size = feature_size

        w1 = nn.Linear(feature_size * 2, width * 2)
        init_weight(w1.weight, 'xavier_uniform_', 'leaky_relu')
        init_bias(w1.bias)
        self.w1 = nn.Sequential(w1, nn.LeakyReLU(negative_slope=0.1))

        drop = nn.Dropout(dropout)
        layers = [drop, LinearResnet(width * 2, width)]
        for _ in range(conv_depth - 1):
            drop = nn.Dropout(dropout)
            layers.append(drop)
            layers.append(LinearResnet(width, width))
        W = nn.Linear(width, feature_size)
        layers.append(W)
        if activation_last_layer:
            init_weight(W.weight, 'xavier_uniform_', 'leaky_relu')
            layers.append(nn.LeakyReLU(negative_slope=0.1))
        else:
            init_weight(W.weight, 'xavier_uniform_', 'linear')
        init_bias(W.bias)
        self.layers = nn.Sequential(*layers)

    def forward(self, nodes):
        h_agg = nodes.data['h_agg']
        h = nodes.data['h']
        w = nodes.data['w'][:, None]
        h_agg = (h_agg - h) / (w - 1).clamp(min=1)  # HACK 1

        h_concat = torch.cat([h, h_agg], 1)
        h_concat = self.w1(h_concat)
        h_new = self.layers(h_concat)
        return {'h': h_new / h_new.norm(dim=1, keepdim=True).clamp(min=1e-6)}


class GraphSageWithSampling(nn.Module):
    def __init__(self, n_content_dims, feature_size, width, n_layers, conv_depth, dropout, G, init_node_vectors=None):
        super(GraphSageWithSampling, self).__init__()

        self.feature_size = feature_size
        self.n_layers = n_layers

        convs = []
        for i in range(n_layers):
            if i >= n_layers - 1:
                convs.append(GraphSageConvWithSampling(feature_size, width, dropout, conv_depth, False))
            else:
                convs.append(GraphSageConvWithSampling(feature_size, width, dropout, conv_depth, True))

        self.convs = nn.ModuleList(convs)

        w1 = nn.Linear(n_content_dims + feature_size, width * 2)
        init_weight(w1.weight, 'xavier_uniform_', 'leaky_relu')
        init_bias(w1.bias)
        drop1 = nn.Dropout(dropout)
        self.projection = nn.Sequential(w1, nn.LeakyReLU(negative_slope=0.1), drop1,
                                        LinearResnet(width * 2, feature_size))
        self.G = G

        self.node_emb = nn.Embedding(G.number_of_nodes() + 1, feature_size)
        if init_node_vectors is None:
            nn.init.normal_(self.node_emb.weight, std=1 / self.feature_size)
        else:
            # self.node_emb.weight = nn.Parameter(init_node_vectors)
            self.node_emb = nn.Embedding.from_pretrained(init_node_vectors, freeze=False)

    msg = [FN.copy_src('h', 'h'),
           FN.copy_src('one', 'one')]
    red = [FN.sum('h', 'h_agg'), FN.sum('one', 'w')]

    def forward(self, nf):
        '''
        nf: NodeFlow.
        '''
        nf.copy_from_parent(edge_embed_names=None)
        for i in range(nf.num_layers):
            nf.layers[i].data['h'] = self.node_emb(nf.layer_parent_nid(i) + 1)
            nf.layers[i].data['one'] = torch.ones(nf.layer_size(i))
            mix_embeddings(nf.layers[i].data, self.projection)
        if self.n_layers == 0:
            return nf.layers[i].data['h']
        for i in range(self.n_layers):
            nf.block_compute(i, self.msg, self.red, self.convs[i])

        result = nf.layers[self.n_layers].data['h']
        assert (result != result).sum() == 0
        return result


class ResnetScorer(nn.Module):
    def __init__(self, feature_size, width, depth, dropout, n_content_dims, n_collaborative_dims, batch_size):
        super(ResnetScorer, self).__init__()

        w1 = nn.Linear(14, 64)
        init_weight(w1.weight, 'xavier_uniform_', 'leaky_relu')
        init_bias(w1.bias)
        self.w1 = nn.Sequential(w1, nn.LeakyReLU(negative_slope=0.1))

        inw = 2 * (n_content_dims + n_collaborative_dims) + 4 * feature_size
        w2 = nn.Linear(inw, width * 2)
        init_weight(w2.weight, 'xavier_uniform_', 'leaky_relu')
        init_bias(w2.bias)
        self.w2 = nn.Sequential(w2, nn.LeakyReLU(negative_slope=0.1))

        drop = nn.Dropout(dropout)
        layers = [drop, LinearResnet(width * 2 + 64, width)]
        for _ in range(depth - 1):
            drop = nn.Dropout(dropout)
            layers.append(drop)
            layers.append(LinearResnet(width, width))
        w2 = nn.Linear(width, 1)
        init_weight(w2.weight, 'xavier_uniform_', 'linear')
        init_bias(w2.bias)

        layers.append(w2)
        self.layers = nn.Sequential(*layers)
        self.batch_size = batch_size
        self.feature_size = feature_size
        self.width = width
        self.n_content_dims = n_content_dims
        self.n_collaborative_dims = n_collaborative_dims

    def forward(self, src, dst, mean, node_biases,
                h_dst, s2d, s2dc, s2d_imp,
                h_src, d2s, d2sc, d2s_imp,
                zeroed_indices, user_content_vector, item_content_vector, user_vector, item_vector):
        user_item_vec_dot = (h_src * h_dst).sum(1)
        user_bias = node_biases[src + 1]
        item_bias = node_biases[dst + 1]
        biased_rating = mean + user_item_vec_dot + user_bias + item_bias

        for x in zeroed_indices:
            s2d_imp[s2d == x] = 0.0

        for x in zeroed_indices:
            d2s_imp[d2s == x] = 0.0

        s2d_imp_vec = s2d_imp.mean(1)
        d2s_imp_vec = d2s_imp.mean(1)
        s2d_imp = s2dc * (h_dst * s2d_imp_vec).sum(1)
        d2s_imp = d2sc * (h_src * d2s_imp_vec).sum(1)
        implicit = s2d_imp + d2s_imp
        implicit_rating = biased_rating + implicit

        user_item_content_similarity = (user_content_vector * item_content_vector).sum(1)
        user_item_vec_similarity = (user_vector * item_vector).sum(1)

        uv = torch.cat([user_content_vector, user_vector], 1)
        iv = torch.cat([item_content_vector, item_vector], 1)

        uv_iv_sim = (uv * iv).sum(1)

        meta = torch.cat([user_item_vec_dot.reshape((-1, 1)), user_bias.reshape(-1, 1),
                          mean.expand(user_item_vec_dot.shape).reshape(-1, 1),
                          item_bias.reshape(-1, 1), biased_rating.reshape(-1, 1),
                          uv_iv_sim.reshape(-1, 1),
                          s2d_imp.float().reshape(-1, 1), d2s_imp.float().reshape(-1, 1),
                          s2dc.float().reshape(-1, 1),
                          d2sc.float().reshape(-1, 1), implicit.float().reshape(-1, 1),
                          implicit_rating.float().reshape(-1, 1),
                          user_item_content_similarity.reshape(-1, 1),
                          user_item_vec_similarity.reshape(-1, 1)
                          ], 1)

        vectors = torch.cat([uv, iv, h_src, h_dst, s2d_imp_vec, d2s_imp_vec], 1)

        meta = self.w1(meta)
        vectors = self.w2(vectors)
        h = torch.cat([meta, vectors], 1)
        h = self.layers(h)
        rating = h.flatten() + implicit_rating
        return rating


class GraphSAGERecommenderImplicitResnet(nn.Module):
    def __init__(self, gcn, mu, node_biases, padding_length, zeroed_indices,
                 feature_size, width, depth, dropout, n_content_dims, n_collaborative_dims, batch_size):
        super(GraphSAGERecommenderImplicitResnet, self).__init__()

        self.gcn = gcn

        assert len(node_biases) == gcn.G.number_of_nodes() + 1
        node_biases[zeroed_indices] = 0.0
        self.node_biases = nn.Parameter(torch.FloatTensor(node_biases))

        self.padding_length = padding_length
        self.zeroed_indices = zeroed_indices

        self.mu = nn.Parameter(torch.tensor(mu), requires_grad=True)
        self.scorer = ResnetScorer(feature_size, width, depth, dropout, n_content_dims, n_collaborative_dims,
                                   batch_size)

    def forward(self, nf, src, dst, s2d, s2dc, d2s, d2sc,
                user_content_vector, item_content_vector, user_vector, item_vector):
        h_output = self.gcn(nf)
        h_src = h_output[nf.map_from_parent_nid(-1, src, True)]
        h_dst = h_output[nf.map_from_parent_nid(-1, dst, True)]

        s2d_imp = h_output[nf.map_from_parent_nid(-1, s2d.flatten(), True)]
        s2d_imp = s2d_imp.reshape(tuple(s2d.shape) + (s2d_imp.shape[-1],))
        d2s_imp = h_output[nf.map_from_parent_nid(-1, d2s.flatten(), True)]
        d2s_imp = d2s_imp.reshape(tuple(d2s.shape) + (d2s_imp.shape[-1],))

        score = self.scorer(src, dst, self.mu, self.node_biases,
                            h_dst, s2d, s2dc, s2d_imp,
                            h_src, d2s, d2sc, d2s_imp,
                            self.zeroed_indices,
                            user_content_vector, item_content_vector, user_vector, item_vector)

        return score


class GraphSAGETripletEmbedding(nn.Module):
    def __init__(self, gcn, margin=0.1):
        super(GraphSAGETripletEmbedding, self).__init__()

        self.gcn = gcn
        self.margin = margin

    def forward(self, nf, src, dst, neg):
        h_output = self.gcn(nf)
        h_src = h_output[nf.map_from_parent_nid(-1, src, True)]
        h_dst = h_output[nf.map_from_parent_nid(-1, dst, True)]
        h_neg = h_output[nf.map_from_parent_nid(-1, neg, True)]
        d_a_b = 1.0 - (h_src * h_dst).sum(1)
        d_a_c = 1.0 - (h_src * h_neg).sum(1)
        score = F.relu(d_a_b + self.margin - d_a_c)
        return score
