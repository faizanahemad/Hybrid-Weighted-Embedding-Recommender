
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as FN
import time

# Load Pytorch as backend
dgl.load_backend('pytorch')

import movielens_torch as movielens
import stanfordnlp

# If you don't have stanfordnlp installed and the English models downloaded, please uncomment this statement
# stanfordnlp.download('en', force=True)

ml = movielens.MovieLens('ml-100k')


def mix_embeddings(ndata, emb, proj, dense):
    """Adds external (categorical and numeric) features into node representation G.ndata['h']"""
    extra_repr = []
    for key, value in ndata.items():
        if (value.dtype == torch.int64) and key in emb:
            result = emb[key](value)
            if result.dim() == 3:  # bag of words: the result would be a (n_nodes x seq_len x feature_size) tensor
                result = result.mean(1)
            extra_repr.append(result)
        elif (value.dtype == torch.float32) and key in proj:
            result = proj[key](value)
            extra_repr.append(result)
    ndata['h'] = ndata['h'] + dense(torch.stack(extra_repr, 0).sum(0))


def init_weight(param, initializer, nonlinearity, nonlinearity_param=None):
    initializer = getattr(nn.init, initializer)
    if nonlinearity is None:
        initializer(param)
    else:
        initializer(param, nn.init.calculate_gain(nonlinearity, nonlinearity_param))


def init_bias(param):
    nn.init.normal_(param, 0, 0.001)
    
from hwer.gcn import GaussianNoise
from hwer.gcn import GraphSageConvWithSamplingBase as GraphSageConvWithSampling


class GraphSageWithSampling(nn.Module):
    def __init__(self, feature_size, n_layers, G):
        super(GraphSageWithSampling, self).__init__()

        self.feature_size = feature_size
        self.n_layers = n_layers

        self.convs = nn.ModuleList([GraphSageConvWithSampling(feature_size, i == n_layers - 1, gaussian_noise, conv_depth) for i in range(n_layers)])
        noise = GaussianNoise(gaussian_noise)
        self.emb = nn.ModuleDict()
        self.proj = nn.ModuleDict()
        
        w1 = nn.Linear(feature_size, feature_size)

        dense = [w1, nn.LeakyReLU(negative_slope=0.1), noise]
        init_weight(w1.weight, 'xavier_uniform_', 'leaky_relu', 0.1)
        init_bias(w1.bias)
        
        w = nn.Linear(feature_size, feature_size)
        init_weight(w.weight, 'xavier_uniform_', 'leaky_relu', 0.1)
        init_bias(w.bias)
        dense.extend([w, nn.LeakyReLU(negative_slope=0.1)])
        self.dense = nn.Sequential(*dense)

        for key, scheme in G.node_attr_schemes().items():
            if scheme.dtype == torch.int64:
                n_items = G.ndata[key].max().item()
                em = nn.Embedding(
                    n_items + 1,
                    self.feature_size,
                    padding_idx=0)
                self.emb[key] = nn.Sequential(em, GaussianNoise(gaussian_noise))
                nn.init.normal_(em.weight, 1 / self.feature_size)
            elif scheme.dtype == torch.float32:
                w = nn.Linear(scheme.shape[0], self.feature_size)
                init_weight(w.weight, 'xavier_uniform_', 'leaky_relu')
                init_bias(w.bias)
                self.proj[key] = nn.Sequential(w, nn.LeakyReLU(), GaussianNoise(gaussian_noise))

        self.G = G
        import math
        embedding_dim = 2 ** int(math.log2(feature_size/4))
        expansion = nn.Linear(embedding_dim, feature_size)
        init_bias(expansion.bias)
        init_weight(expansion.weight, 'xavier_uniform_', 'leaky_relu', 0.1)
        self.expansion = nn.Sequential(expansion, nn.LeakyReLU(negative_slope=0.1))
        self.node_emb = nn.Embedding(G.number_of_nodes() + 1, embedding_dim)
        nn.init.normal_(self.node_emb.weight, std=1 / embedding_dim)

    msg = [FN.copy_src('h', 'h'),
           FN.copy_src('one', 'one')]
    red = [FN.sum('h', 'h_agg'), FN.sum('one', 'w')]

    def forward(self, nf):
        '''
        nf: NodeFlow.
        '''
        nf.copy_from_parent(edge_embed_names=None)
        for i in range(nf.num_layers):
            nf.layers[i].data['h'] = self.expansion(self.node_emb(nf.layer_parent_nid(i) + 1))
            nf.layers[i].data['one'] = torch.ones(nf.layer_size(i))
            mix_embeddings(nf.layers[i].data, model.gcn.emb, model.gcn.proj, model.gcn.dense)
        if self.n_layers == 0:
            return nf.layers[i].data['h']
        for i in range(self.n_layers):
            nf.block_compute(i, self.msg, self.red, self.convs[i])

        result = nf.layers[self.n_layers].data['h']
        assert (result != result).sum() == 0
        return result


class GraphSAGERecommender(nn.Module):
    def __init__(self, gcn):
        super(GraphSAGERecommender, self).__init__()

        self.gcn = gcn
        self.node_biases = nn.Parameter(torch.zeros(gcn.G.number_of_nodes() + 1))

    def forward(self, nf, src, dst):
        h_output = self.gcn(nf)
        h_src = h_output[nodeflow.map_from_parent_nid(-1, src, True)]
        h_dst = h_output[nodeflow.map_from_parent_nid(-1, dst, True)]
        score = (h_src * h_dst).sum(1) + self.node_biases[src + 1] + self.node_biases[dst + 1]
        return score


import tqdm
import spotlight
import pickle

g = ml.g
# Find the subgraph of all "training" edges
g_train = g.edge_subgraph(g.filter_edges(lambda edges: edges.data['train']), True)
g_train.copy_from_parent()
g_train.readonly()
eid_test = g.filter_edges(lambda edges: edges.data['test'])
src_test, dst_test = g.find_edges(eid_test)
src, dst = g_train.all_edges()
rating = g_train.edata['rating']
rating_test = g.edges[eid_test].data['rating']

gaussian_noise = 0.2
batch_size = 512
epochs = 100
n_dims = 128
weight_decay = 1e-8
lr = 0.001
layers = 3
conv_depth = 2

model = GraphSAGERecommender(GraphSageWithSampling(n_dims, layers, g_train))
opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


n_users = len(ml.user_ids)
n_products = len(ml.product_ids)

for epoch in range(epochs):
    start = time.time()
    model.eval()

    # Validation & Test, we precompute GraphSage output for all nodes first.
    sampler = dgl.contrib.sampling.NeighborSampler(
        g_train,
        batch_size,
        5,
        layers,
        seed_nodes=torch.arange(g.number_of_nodes()),
        prefetch=True,
        add_self_loop=True,
        shuffle=False,
        num_workers=4
    )

    with torch.no_grad():
        h = []
        for nf in sampler:
            # import pdb
            # pdb.set_trace()
            h.append(model.gcn.forward(nf))
        h = torch.cat(h)

        # Compute validation RMSE
        score = torch.zeros(len(src))
        for i in range(0, len(src), batch_size):
            s = src[i:i + batch_size]
            d = dst[i:i + batch_size]
            score[i:i + batch_size] = (h[s] * h[d]).sum(1) + model.node_biases[s + 1] + model.node_biases[d + 1]
        train_rmse = ((score - rating) ** 2).mean().sqrt()

        # Compute test RMSE
        score = torch.zeros(len(src_test))
        for i in range(0, len(src_test), batch_size):
            s = src_test[i:i + batch_size]
            d = dst_test[i:i + batch_size]
            score[i:i + batch_size] = (h[s] * h[d]).sum(1) + model.node_biases[s + 1] + model.node_biases[d + 1]
        test_rmse = ((score - rating_test) ** 2).mean().sqrt()

    model.train()

    shuffle_idx = torch.randperm(g_train.number_of_edges())
    src_shuffled = src[shuffle_idx]
    dst_shuffled = dst[shuffle_idx]
    rating_shuffled = rating[shuffle_idx]
    src_batches = src_shuffled.split(batch_size)
    dst_batches = dst_shuffled.split(batch_size)
    rating_batches = rating_shuffled.split(batch_size)

    seed_nodes = torch.cat(sum([[s, d] for s, d in zip(src_batches, dst_batches)], []))

    sampler = dgl.contrib.sampling.NeighborSampler(
        g_train,  # the graph
        batch_size * 2,  # number of nodes to compute at a time, HACK 2
        5,  # number of neighbors for each node
        layers,  # number of layers in GCN
        seed_nodes=seed_nodes,  # list of seed nodes, HACK 2
        prefetch=True,  # whether to prefetch the NodeFlows
        add_self_loop=True,  # whether to add a self-loop in the NodeFlows, HACK 1
        shuffle=False,  # whether to shuffle the seed nodes.  Should be False here.
        num_workers=4,
    )

    # Training
    for s, d, r, nodeflow in zip(src_batches, dst_batches, rating_batches, sampler):
        score = model.forward(nodeflow, s, d)
        loss = ((score - r) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()
    total_time = time.time() - start

    print('Epoch: %2d' % int(epoch+1), 'Training loss: %.4f ||' % loss.item(), 'Train RMSE: %.4f' % train_rmse.item(), 'Test RMSE: %.4f,' % test_rmse.item(), 'Time Taken: %.1f' % total_time)

import numpy as np
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Built Prediction Network, model params = %s", params)