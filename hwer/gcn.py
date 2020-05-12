import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as FN
import numpy as np
dgl.load_backend('pytorch')


class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0.0)

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x


def mix_embeddings(ndata, proj):
    """Adds external (categorical and numeric) features into node representation G.ndata['h']"""
    h = ndata['h']
    c = proj(ndata['content'])
    ndata['h'] = h + c[:, :h.shape[1]]


def init_weight(param, initializer, nonlinearity, nonlinearity_param=None):
    initializer = getattr(nn.init, initializer)
    if nonlinearity is None:
        initializer(param)
    else:
        initializer(param, nn.init.calculate_gain(nonlinearity, nonlinearity_param))


def init_bias(param):
    nn.init.normal_(param, 0, 0.001)


def build_content_layer(in_dims, out_dims):
    f = lambda x: 2 ** int((np.log2(x)))
    g = lambda x: 2 ** int((np.log2(x * 2)))

    h = lambda x: 2 ** int((np.log2(x / 2)))
    i = lambda x: 2 ** int((np.log2(x / 4)))

    if f(in_dims) + i(in_dims) > in_dims:
        inter_dims = f(in_dims) + i(in_dims)
    elif f(in_dims) + h(in_dims) > in_dims:
        inter_dims = f(in_dims) + h(in_dims)
    else:
        inter_dims = g(in_dims)

    w1 = nn.Linear(in_dims, inter_dims)
    init_fc(w1, 'xavier_uniform_', 'leaky_relu', 0.1)
    w = nn.Linear(inter_dims, out_dims)
    init_fc(w, 'xavier_uniform_', 'linear', 0.1)
    proj = [w1, nn.LeakyReLU(negative_slope=0.1), w]
    return nn.Sequential(*proj)


def init_fc(layer, initializer, nonlinearity, nonlinearity_param=None):
    init_weight(layer.weight, initializer, nonlinearity, nonlinearity_param)
    try:
        init_bias(layer.bias)
    except AttributeError:
        pass


class GraphConv(nn.Module):
    def __init__(self, in_dims, out_dims, gaussian_noise, prediction_layer):
        super(GraphConv, self).__init__()
        layers = [GaussianNoise(gaussian_noise)]
        if prediction_layer:
            expand = nn.Linear(in_dims, in_dims * 2)
            init_fc(expand, 'xavier_uniform_', 'leaky_relu', 0.1)
            layers.extend([expand, nn.LeakyReLU(negative_slope=0.1)])
            contract = nn.Linear(in_dims * 2, out_dims)
            init_fc(expand, 'xavier_uniform_', 'linear', 0.1)
            layers.append(contract)
            self.W = nn.Sequential(*layers)
        
        self.prediction_layer = prediction_layer

    def forward(self, nodes):
        h_agg = nodes.data['h_agg']
        h = nodes.data['h']
        w = nodes.data['w'][:, None]
        h_agg = h_agg / w
        h_new = torch.cat([h_agg, h], 1)
        if self.prediction_layer:
            h_new = self.W(h_new)
        h_new = h_new / h_new.norm(dim=1, keepdim=True).clamp(min=1e-5)
        return {'h': h_new}


class GraphConvModule(nn.Module):
    def __init__(self, n_content_dims, feature_size, n_layers, G,
                 gaussian_noise):
        super(GraphConvModule, self).__init__()

        self.feature_size = feature_size
        self.n_layers = n_layers
        self.proj = build_content_layer(n_content_dims, feature_size)
        self.G = G
        embedding_dim = int(feature_size/2)
        assert feature_size % 2 == 0
        assert feature_size % 16 == 0
        self.node_emb = nn.Embedding(G.number_of_nodes() + 1, embedding_dim)
        nn.init.normal_(self.node_emb.weight, std=1 / embedding_dim)
        self.embedding_dim = embedding_dim
        convs = []
        for i in range(n_layers):
            in_dims = max(4, int(self.feature_size/(4 ** (n_layers - i - 1))))
            out_dims = max(4, int(self.feature_size/(4 ** max(0, n_layers - i - 2))))
            in_dims = (in_dims + embedding_dim) if i == n_layers - 1 else in_dims
            conv = GraphConv(in_dims,
                             out_dims,
                             gaussian_noise if i == 0 else 0, i == n_layers - 1)
            convs.append(conv)

        self.convs = nn.ModuleList(convs)
        layer_dims = [0] + [max(4, min(embedding_dim, int(self.feature_size/(4 ** max(0, n_layers - i - 1))))) for i in range(n_layers + 1)]
        layer_dims = [[layer_dims[idx], layer_dims[min(idx+1, len(layer_dims) - 1)]] for idx, dim in enumerate(layer_dims)]
        self.layer_dims = [[s, e] if s != e else [0, e] for s, e in layer_dims]

    msg = [FN.copy_src('h', 'h'),
           FN.copy_src('one', 'one')]
    red = [FN.sum('h', 'h_agg'), FN.sum('one', 'w')]

    def forward(self, nf):
        '''
        nf: NodeFlow.
        '''
        nf.copy_from_parent(edge_embed_names=None)
        for i in range(nf.num_layers):
            nh = self.node_emb(nf.layer_parent_nid(i) + 1)
            nf.layers[i].data['h'] = nh[:, self.layer_dims[i][0]:self.layer_dims[i][1]]
            nf.layers[i].data['one'] = torch.ones(nf.layer_size(i))
            mix_embeddings(nf.layers[i].data, self.proj)
        if i == nf.num_layers - 1:
            h = nf.layers[i].data['h']
            nf.layers[i].data['h'] = torch.cat([self.node_emb.weight.mean(0).unsqueeze(0).expand(*h.shape), h], 1)
        #     ## nf.layers[i].data['h'] = torch.cat([h.mean(0).unsqueeze(0).expand(*h.shape), h], 1)
        if self.n_layers == 0:
            return nf.layers[i].data['h']
        for i in range(self.n_layers):
            nf.block_compute(i, self.msg, self.red, self.convs[i])

        result = nf.layers[self.n_layers].data['h']
        assert (result != result).sum() == 0
        return result

# h_src.unsqueeze(1).expand(h_src.size(0), 1, h_src.size(1)).bmm(torch.transpose(h_negs, 1, 2)))


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

