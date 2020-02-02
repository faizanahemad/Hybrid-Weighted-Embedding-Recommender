import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as FN
import numpy as np
dgl.load_backend('pytorch')


def mix_embeddings(ndata, proj):
    """Adds external (categorical and numeric) features into node representation G.ndata['h']"""
    ndata['h'] = ndata['h'] + proj(ndata['content'])


def init_weight(param, initializer, nonlinearity):
    initializer = getattr(nn.init, initializer)
    if nonlinearity is not None:
        initializer(param)
    else:
        initializer(param, nn.init.calculate_gain(nonlinearity))


def init_bias(param):
    nn.init.constant_(param, 0)


class GraphSageConvWithSampling(nn.Module):
    def __init__(self, feature_size, dropout, activation):
        super(GraphSageConvWithSampling, self).__init__()

        self.feature_size = feature_size
        self.W = nn.Linear(feature_size * 2, feature_size)
        self.drop = nn.Dropout(dropout)
        self.activation = activation

        if self.activation is not None:
            init_weight(self.W.weight, 'xavier_uniform_', 'leaky_relu')
        else:
            init_weight(self.W.weight, 'xavier_uniform_', 'linear')
        init_bias(self.W.bias)

    def forward(self, nodes):
        h_agg = nodes.data['h_agg']
        h = nodes.data['h']
        w = nodes.data['w'][:, None]
        h_agg = (h_agg - h) / (w - 1).clamp(min=1)  # HACK 1
        h_concat = torch.cat([h, h_agg], 1)
        h_concat = self.drop(h_concat)
        h_new = self.W(h_concat)
        if self.activation is not None:
            h_new = self.activation(h_new, negative_slope=0.1)
        return {'h': h_new / h_new.norm(dim=1, keepdim=True).clamp(min=1e-6)}


class GraphSageWithSampling(nn.Module):
    def __init__(self, n_content_dims, feature_size, n_layers, dropout, G, init_node_vectors=None):
        super(GraphSageWithSampling, self).__init__()

        self.feature_size = feature_size
        self.n_layers = n_layers

        convs = []
        for i in range(n_layers):
            if i >= n_layers - 1:
                convs.append(GraphSageConvWithSampling(feature_size, dropout, None))
            else:
                convs.append(GraphSageConvWithSampling(feature_size, dropout, F.leaky_relu))

        self.convs = nn.ModuleList(convs)
        w = nn.Linear(n_content_dims, feature_size)
        init_weight(w.weight, 'xavier_uniform_', 'leaky_relu')
        init_bias(w.bias)

        self.proj = nn.Sequential(w, nn.LeakyReLU(negative_slope=0.1))

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
            mix_embeddings(nf.layers[i].data, self.proj)
        if self.n_layers == 0:
            return nf.layers[i].data['h']
        for i in range(self.n_layers):
            nf.block_compute(i, self.msg, self.red, self.convs[i])

        result = nf.layers[self.n_layers].data['h']
        assert (result != result).sum() == 0
        return result


def get_score(src, dst, mean, node_biases,
              h_dst, h_src):

    score = mean + (h_src * h_dst).sum(1) + node_biases[src + 1] + node_biases[dst + 1]
    return score


class GraphSAGERecommenderImplicit(nn.Module):
    def __init__(self, gcn, mu, node_biases, padding_length, zeroed_indices):
        super(GraphSAGERecommenderImplicit, self).__init__()

        self.gcn = gcn
        if node_biases is not None:
            assert len(node_biases) == gcn.G.number_of_nodes() + 1
            node_biases[zeroed_indices] = 0.0
            self.node_biases = nn.Parameter(torch.FloatTensor(node_biases))
        else:
            self.node_biases = nn.Parameter(torch.zeros(gcn.G.number_of_nodes() + 1))
        self.padding_length = padding_length
        self.zeroed_indices = zeroed_indices
        self.mu = nn.Parameter(torch.tensor(mu), requires_grad=True)

    def forward(self, nf, src, dst):
        h_output = self.gcn(nf)
        h_src = h_output[nf.map_from_parent_nid(-1, src, True)]
        h_dst = h_output[nf.map_from_parent_nid(-1, dst, True)]
        score = get_score(src, dst, self.mu, self.node_biases,
                          h_dst, h_src)

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


def build_dgl_graph(ratings, total_nodes, content_vectors):
    g = dgl.DGLGraph(multigraph=True)
    g.add_nodes(total_nodes)
    g.ndata['content'] = torch.FloatTensor(content_vectors)
    rating_user_vertices = [u for u, i, r in ratings]
    rating_product_vertices = [i for u, i, r in ratings]
    ratings = [r for u, i, r in ratings]
    g.add_edges(
        rating_user_vertices,
        rating_product_vertices,
        data={'inv': torch.zeros(len(ratings), dtype=torch.uint8),
              'rating': torch.FloatTensor(ratings)})
    g.add_edges(
        rating_product_vertices,
        rating_user_vertices,
        data={'inv': torch.ones(len(ratings), dtype=torch.uint8),
              'rating': torch.FloatTensor(ratings)})
    return g
