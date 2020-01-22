
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as FN
import time

# Load Pytorch as backend
dgl.load_backend('pytorch')

import movielens_torch_vectorised as movielens
import stanfordnlp
import numpy as np

# If you don't have stanfordnlp installed and the English models downloaded, please uncomment this statement
# stanfordnlp.download('en', force=True)
feature_size = 100
n_content_dims = 200
ml = movielens.MovieLens('ml-100k', directory="100K/ml-100k", feature_size=n_content_dims)


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
    def __init__(self, feature_size, dropout):
        super(GraphSageConvWithSampling, self).__init__()

        self.feature_size = feature_size
        self.W = nn.Linear(feature_size * 2, feature_size)
        self.drop = nn.Dropout(dropout)
        init_weight(self.W.weight, 'xavier_uniform_', 'leaky_relu')
        init_bias(self.W.bias)

    def forward(self, nodes):
        h_agg = nodes.data['h_agg']
        h = nodes.data['h']
        w = nodes.data['w'][:, None]
        h_agg = (h_agg - h) / (w - 1).clamp(min=1)  # HACK 1
        h_concat = torch.cat([h, h_agg], 1)
        h_concat = self.drop(h_concat)
        h_new = F.leaky_relu(self.W(h_concat))
        return {'h': h_new / h_new.norm(dim=1, keepdim=True).clamp(min=1e-6)}


class GraphSageWithSampling(nn.Module):
    def __init__(self, n_content_dims, feature_size, n_layers, dropout, G):
        super(GraphSageWithSampling, self).__init__()

        self.feature_size = feature_size
        self.n_layers = n_layers

        self.convs = nn.ModuleList([GraphSageConvWithSampling(feature_size, dropout) for _ in range(n_layers)])
        proj = []
        for i in range(n_layers + 1):
            w = nn.Linear(n_content_dims, feature_size)
            init_weight(w.weight, 'xavier_uniform_', 'leaky_relu')
            init_bias(w.bias)
            drop = nn.Dropout(dropout)
            proj.append(nn.Sequential(drop, w, nn.LeakyReLU()))
        self.proj = nn.ModuleList(proj)

        self.G = G

        self.node_emb = nn.Embedding(G.number_of_nodes() + 1, feature_size)
        nn.init.normal_(self.node_emb.weight, std=1 / self.feature_size)

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
            mix_embeddings(nf.layers[i].data, self.proj[i])
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
        h_src = h_output[nf.map_from_parent_nid(-1, src, True)]
        h_dst = h_output[nf.map_from_parent_nid(-1, dst, True)]
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
# eid_train = g.filter_edges(lambda edges: edges.data['train'])
eid_valid = g.filter_edges(lambda edges: edges.data['valid'])
eid_test = g.filter_edges(lambda edges: edges.data['test'])
src_valid, dst_valid = g.find_edges(eid_valid)
src_test, dst_test = g.find_edges(eid_test)
src, dst = g_train.all_edges()
rating = g_train.edata['rating']
mu = torch.mean(rating)
rating_valid = g.edges[eid_valid].data['rating']
rating_test = g.edges[eid_test].data['rating']

n_layers = 2
dropout = 0.05
epochs = 50
batch_size = 1024
bias_reg = 1e-6
lr = 0.3

model = GraphSAGERecommender(GraphSageWithSampling(n_content_dims, feature_size, n_layers, dropout, g_train))
weight_reg = 1e-6
# opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_reg, nesterov=False)
opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_reg)
# scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=epochs,
#                                                 steps_per_epoch=int(np.ceil(len(src)/batch_size)),
#                                                 div_factor=20, final_div_factor=20)


for epoch in range(epochs):
    start = time.time()
    model.eval()

    # Validation & Test, we precompute GraphSage output for all nodes first.
    sampler = dgl.contrib.sampling.NeighborSampler(
        g_train,
        batch_size,
        5,
        n_layers,
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

        # Compute Train RMSE
        score = torch.zeros(len(src))
        for i in range(0, len(src), batch_size):
            s = src[i:i + batch_size]
            d = dst[i:i + batch_size]
            score[i:i + batch_size] = (h[s] * h[d]).sum(1) + model.node_biases[s + 1] + model.node_biases[d + 1]
        train_rmse = ((score - rating) ** 2).mean().sqrt()

        # Compute validation RMSE
        score = torch.zeros(len(src_valid))
        for i in range(0, len(src_valid), batch_size):
            s = src_valid[i:i + batch_size]
            d = dst_valid[i:i + batch_size]
            score[i:i + batch_size] = (h[s] * h[d]).sum(1) + model.node_biases[s + 1] + model.node_biases[d + 1]
        valid_rmse = ((score - rating_valid) ** 2).mean().sqrt()

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
        n_layers,  # number of layers in GCN
        seed_nodes=seed_nodes,  # list of seed nodes, HACK 2
        prefetch=True,  # whether to prefetch the NodeFlows
        add_self_loop=True,  # whether to add a self-loop in the NodeFlows, HACK 1
        shuffle=False,  # whether to shuffle the seed nodes.  Should be False here.
        num_workers=4,
    )

    # Training
    for s, d, r, nodeflow in zip(src_batches, dst_batches, rating_batches, sampler):
        score = model.forward(nodeflow, s, d)
        reg = bias_reg * (torch.norm(model.node_biases[src + 1], 2) + torch.norm(model.node_biases[dst + 1], 2))
        loss = ((score - r) ** 2).mean() + reg

        opt.zero_grad()
        loss.backward()
        opt.step()
        # scheduler.step()
    total_time = time.time() - start

    print('Epoch %2d: ' % int(epoch+1), 'Training loss: %.4f' % loss.item(), 'Train RMSE: %.4f ||' % train_rmse.item(), 'Validation RMSE: %.4f' % valid_rmse.item(), 'Test RMSE: %.4f,' % test_rmse.item(), 'Time Taken: %.1f' % total_time)