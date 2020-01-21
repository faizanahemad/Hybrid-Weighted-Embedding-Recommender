
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

ml = movielens.MovieLens('ml-100k', directory="ml-100k")


def mix_embeddings(ndata, emb, proj):
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
    ndata['h'] = ndata['h'] + torch.stack(extra_repr, 0).sum(0)


def init_weight(param, initializer, nonlinearity):
    initializer = getattr(nn.init, initializer)
    if nonlinearity is not None:
        initializer(param)
    else:
        initializer(param, nn.init.calculate_gain(nonlinearity))


def init_bias(param):
    nn.init.constant_(param, 0)


class GraphSageConvWithSampling(nn.Module):
    def __init__(self, feature_size):
        super(GraphSageConvWithSampling, self).__init__()

        self.feature_size = feature_size
        self.W = nn.Linear(feature_size * 2, feature_size)
        init_weight(self.W.weight, 'xavier_uniform_', 'leaky_relu')
        init_bias(self.W.bias)

    def forward(self, nodes):
        h_agg = nodes.data['h_agg']
        h = nodes.data['h']
        w = nodes.data['w'][:, None]
        h_agg = (h_agg - h) / (w - 1).clamp(min=1)  # HACK 1
        h_concat = torch.cat([h, h_agg], 1)
        h_new = F.leaky_relu(self.W(h_concat))
        return {'h': h_new / h_new.norm(dim=1, keepdim=True).clamp(min=1e-6)}


class GraphSageWithSampling(nn.Module):
    def __init__(self, feature_size, n_layers, G):
        super(GraphSageWithSampling, self).__init__()

        self.feature_size = feature_size
        self.n_layers = n_layers

        self.convs = nn.ModuleList([GraphSageConvWithSampling(feature_size) for _ in range(n_layers)])

        self.emb = nn.ModuleDict()
        self.proj = nn.ModuleDict()

        for key, scheme in G.node_attr_schemes().items():
            if scheme.dtype == torch.int64:
                n_items = G.ndata[key].max().item()
                self.emb[key] = nn.Embedding(
                    n_items + 1,
                    self.feature_size,
                    padding_idx=0)
                nn.init.normal_(self.emb[key].weight, 1 / self.feature_size)
            elif scheme.dtype == torch.float32:
                w = nn.Linear(scheme.shape[0], self.feature_size)
                init_weight(w.weight, 'xavier_uniform_', 'leaky_relu')
                init_bias(w.bias)
                self.proj[key] = nn.Sequential(w, nn.LeakyReLU())

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
            mix_embeddings(nf.layers[i].data, model.gcn.emb, model.gcn.proj)
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
eid_valid = g.filter_edges(lambda edges: edges.data['valid'])
eid_test = g.filter_edges(lambda edges: edges.data['test'])
src_valid, dst_valid = g.find_edges(eid_valid)
src_test, dst_test = g.find_edges(eid_test)
src, dst = g_train.all_edges()
rating = g_train.edata['rating']
rating_valid = g.edges[eid_valid].data['rating']
rating_test = g.edges[eid_test].data['rating']

model = GraphSAGERecommender(GraphSageWithSampling(100, 2, g_train))
opt = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-9)

batch_size = 1024
n_users = len(ml.user_ids)
n_products = len(ml.product_ids)

for epoch in range(50):
    start = time.time()
    model.eval()

    # Validation & Test, we precompute GraphSage output for all nodes first.
    sampler = dgl.contrib.sampling.NeighborSampler(
        g_train,
        batch_size,
        5,
        2,
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
        2,  # number of layers in GCN
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

    print('Epoch: %2d' % int(epoch+1), 'Training loss: %.4f ||' % loss.item(), 'Validation RMSE: %.4f' % valid_rmse.item(), 'Test RMSE: %.4f,' % test_rmse.item(), 'Time Taken: %.1f' % total_time)