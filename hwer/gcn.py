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
    ndata['h'] = ndata['h'] + proj(ndata['content'])


def init_weight(param, initializer, nonlinearity, nonlinearity_param=None):
    initializer = getattr(nn.init, initializer)
    if nonlinearity is None:
        initializer(param)
    else:
        initializer(param, nn.init.calculate_gain(nonlinearity, nonlinearity_param))


def init_bias(param):
    nn.init.normal_(param, 0, 0.001)


def build_content_layer(in_dims, out_dims):
    inter_dims = 2 ** int((np.log2(in_dims*2)))
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
    def __init__(self, in_dims, out_dims, prediction_layer, gaussian_noise, depth):
        super(GraphConv, self).__init__()
        layers = [GaussianNoise(gaussian_noise)]
        expand = nn.Linear(in_dims * 2, in_dims * 4)
        init_fc(expand, 'xavier_uniform_', 'leaky_relu', 0.1)
        layers.extend([expand, nn.LeakyReLU(negative_slope=0.1)])
        contract = nn.Linear(in_dims * 4, out_dims)
        init_fc(expand, 'xavier_uniform_', 'linear', 0.1)
        layers.append(contract)
        self.W = nn.Sequential(*layers)

    def pre_process(self, nodes):
        h_agg = nodes.data['h_agg']
        h = nodes.data['h']
        w = nodes.data['w'][:, None]
        h_agg = (h_agg - h) / (w - 1).clamp(min=1)  # HACK 1
        return h, h_agg

    def post_process(self, h_concat, h, h_agg):
        h_new = self.W(h_concat)
        h_new = h_new / h_new.norm(dim=1, keepdim=True).clamp(min=1e-5)
        return {'h': h_new}

    def forward(self, nodes):
        h, h_agg = self.pre_process(nodes)
        h_concat = torch.cat([h, h_agg], 1)
        return self.post_process(h_concat, h, h_agg)


class GraphConvModule(nn.Module):
    def __init__(self, n_content_dims, feature_size, n_layers, G,
                 gaussian_noise, conv_depth,
                 init_node_vectors=None,):
        super(GraphConvModule, self).__init__()

        self.feature_size = feature_size
        self.n_layers = n_layers
        self.proj = build_content_layer(n_content_dims, feature_size)
        self.G = G
        embedding_dim = feature_size
        self.node_emb = nn.Embedding(G.number_of_nodes() + 1, embedding_dim)
        if init_node_vectors is None:
            nn.init.normal_(self.node_emb.weight, std=1 / embedding_dim)
        else:
            if embedding_dim != feature_size:
                from sklearn.decomposition import PCA
                init_node_vectors = torch.FloatTensor(PCA(n_components=embedding_dim, ).fit_transform(init_node_vectors))
            self.node_emb = nn.Embedding.from_pretrained(init_node_vectors, freeze=False)

        convs = []
        for i in range(n_layers):
            conv = GraphConv(feature_size, feature_size,
                             i == n_layers - 1, gaussian_noise, conv_depth)
            convs.append(conv)

        self.convs = nn.ModuleList(convs)

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


class ResnetConv(nn.Module):
    def __init__(self, in_dims, out_dims, first_layer, prediction_layer, gaussian_noise, depth):
        super(ResnetConv, self).__init__()
        layers = []
        depth = max(1, depth)
        if first_layer:
            inp = in_dims * 2
        else:
            inp = in_dims * 3
        for i in range(depth - 1):
            width = 1 if i == 0 else 2
            weights = nn.Linear(inp * width, inp * 2)
            init_fc(weights, 'xavier_uniform_', 'leaky_relu', 0.1)
            layers.append(GaussianNoise(gaussian_noise))
            layers.append(weights)
            layers.append(nn.LeakyReLU(negative_slope=0.1))

        layers.append(GaussianNoise(gaussian_noise))
        W = nn.Linear(inp * (2 if depth > 1 else 1), out_dims)
        layers.append(W)
        self.prediction_layer = prediction_layer
        init_fc(W, 'xavier_uniform_', 'leaky_relu', 0.1)
        layers.append(nn.LeakyReLU(negative_slope=0.1))

        if not prediction_layer:
            self.skip = None
            if in_dims != out_dims:
                skip = nn.Linear(in_dims, out_dims)
                init_fc(skip, 'xavier_uniform_', 'leaky_relu', 0.1)
                self.skip = nn.Sequential(skip, nn.LeakyReLU(negative_slope=0.1))
        else:
            pred = nn.Linear(in_dims, out_dims)
            init_fc(pred, 'xavier_uniform_', 'linear', 0.1)
            self.pred = pred
        self.W = nn.Sequential(*layers)
        self.first_layer = first_layer

    def forward(self, nodes):
        h_agg = nodes.data['h_agg']
        h = nodes.data['h']
        w = nodes.data['w'][:, None]
        h_agg = (h_agg - h) / (w - 1).clamp(min=1)  # HACK 1
        if self.first_layer:
            h_concat = torch.cat([h, h_agg], 1)
        else:
            h_residue = nodes.data['h_residue']
            h_concat = torch.cat([h, h_agg, h_residue], 1)
        h_new = self.W(h_concat)
        h_new = h + h_new
        h_new = h_new / h_new.norm(dim=1, keepdim=True).clamp(min=1e-6)

        if self.prediction_layer:
            h_new = self.pred(h_new)
            return {'h': h_new}
        # skip
        h_agg = self.skip(h_agg) if self.skip is not None else h_agg
        h_agg = h_agg / h_agg.norm(dim=1, keepdim=True).clamp(min=1e-6)
        return {'h': h_new, 'h_residue': h_agg}


class GraphResnetConvModule(nn.Module):
    def __init__(self, n_content_dims, feature_size, n_layers, G,
                 gaussian_noise, conv_depth,
                 init_node_vectors=None,):
        super(GraphResnetConvModule, self).__init__()

        self.feature_size = feature_size
        self.n_layers = n_layers
        noise = GaussianNoise(gaussian_noise)

        width = 2
        self.proj = build_content_layer(n_content_dims, feature_size * width, noise)
        self.G = G
        embedding_dim = feature_size
        self.node_emb = nn.Embedding(G.number_of_nodes() + 1, embedding_dim)
        if init_node_vectors is None:
            nn.init.normal_(self.node_emb.weight, std=1 / embedding_dim)
        else:
            if embedding_dim != feature_size:
                from sklearn.decomposition import PCA
                init_node_vectors = torch.FloatTensor(PCA(n_components=embedding_dim, ).fit_transform(init_node_vectors))
            self.node_emb = nn.Embedding.from_pretrained(init_node_vectors, freeze=False)
        self.convs = nn.ModuleList(
            [ResnetConv(feature_size * width, feature_size * (1 if i >= n_layers - 1 else width), i == 0, i == n_layers - 1, gaussian_noise, conv_depth) for i
             in
             range(n_layers)])

        m_init = [FN.copy_src('h', 'h'),
               FN.copy_src('one', 'one')]
        r_init = [FN.sum('h', 'h_agg'), FN.sum('one', 'w')]
        m = [FN.copy_src('h', 'h'),
             FN.copy_src('h_residue', 'h_residue'),
             FN.copy_src('one', 'one')]
        r = [FN.sum('h', 'h_agg'), FN.sum('h_residue', 'h_residue'), FN.sum('one', 'w')]
        self.msg = [m_init if i == 0 else m for i in range(n_layers)]
        self.red = [r_init if i == 0 else r for i in range(n_layers)]

        expansion = nn.Linear(feature_size, feature_size * width)
        init_fc(expansion, 'xavier_uniform_', 'leaky_relu', 0.1)
        self.expansion = nn.Sequential(noise, expansion, nn.LeakyReLU(0.1))

    def forward(self, nf):
        '''
        nf: NodeFlow.
        '''
        nf.copy_from_parent(edge_embed_names=None)
        for i in range(nf.num_layers):
            nf.layers[i].data['h'] = self.expansion(self.node_emb(nf.layer_parent_nid(i) + 1))
            nf.layers[i].data['one'] = torch.ones(nf.layer_size(i))
            mix_embeddings(nf.layers[i].data, self.proj)
        if self.n_layers == 0:
            return nf.layers[i].data['h']
        for i in range(self.n_layers):
            nf.block_compute(i, self.msg[i], self.red[i], self.convs[i])

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

