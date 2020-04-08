import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as FN
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


def init_fc(layer, initializer, nonlinearity, nonlinearity_param=None):
    init_weight(layer.weight, initializer, nonlinearity, nonlinearity_param)
    try:
        init_bias(layer.bias)
    except AttributeError:
        pass


class GraphSageConvWithSamplingBase(nn.Module):
    def __init__(self, feature_size, out_dims, prediction_layer, gaussian_noise, depth):
        super(GraphSageConvWithSamplingBase, self).__init__()
        layers = []
        depth = max(1, depth)
        for i in range(depth - 1):
            in_width = 2 if i == 0 else 4
            weights = nn.Linear(feature_size * in_width, feature_size * 4)
            init_fc(weights, 'xavier_uniform_', 'leaky_relu', 0.1)
            layers.append(GaussianNoise(gaussian_noise))
            layers.append(weights)
            layers.append(nn.LeakyReLU(negative_slope=0.1))

        layers.append(GaussianNoise(gaussian_noise))
        W = nn.Linear(feature_size * 4, out_dims)
        layers.append(W)
        self.prediction_layer = prediction_layer
        self.noise = GaussianNoise(gaussian_noise)

        if not prediction_layer:
            init_fc(W, 'xavier_uniform_', 'leaky_relu', 0.1)
            layers.append(nn.LeakyReLU(0.1)) # tanh
        else:
            init_fc(W, 'xavier_uniform_', 'linear')
        self.W = nn.Sequential(*layers)

    def pre_process(self, nodes):
        h_agg = nodes.data['h_agg']
        h = nodes.data['h']
        w = nodes.data['w'][:, None]
        h_agg = (h_agg - h) / (w - 1).clamp(min=1)  # HACK 1
        return h, h_agg

    def process_node_data(self, h):
        return h

    def process_neighbourhood_data(self, h_agg):
        return h_agg

    def post_process(self, h_concat, h, h_agg):
        h_new = self.W(h_concat)
        # h_new = h + h_new
        h_new = h_new / h_new.norm(dim=1, keepdim=True).clamp(min=1e-6)
        return {'h': h_new}

    def forward(self, nodes):
        h, h_agg = self.pre_process(nodes)
        h = self.process_node_data(h)
        h_agg = self.process_neighbourhood_data(h_agg)
        h_concat = torch.cat([h, h_agg], 1)
        return self.post_process(h_concat, h, h_agg)


class GraphSageWithSampling(nn.Module):
    def __init__(self, n_content_dims, feature_size, n_layers, G,
                 gaussian_noise, conv_depth,
                 init_node_vectors=None,):
        super(GraphSageWithSampling, self).__init__()

        self.feature_size = feature_size
        self.n_layers = n_layers
        GraphSageConvWithSampling = GraphSageConvWithSamplingBase

        noise = GaussianNoise(gaussian_noise)
        w1 = nn.Linear(n_content_dims, feature_size)
        init_fc(w1, 'xavier_uniform_', 'leaky_relu', 0.1)
        w = nn.Linear(feature_size, feature_size)
        w2 = nn.Linear(feature_size, feature_size)
        init_fc(w, 'xavier_uniform_', 'leaky_relu', 0.1)
        init_fc(w2, 'xavier_uniform_', 'leaky_relu', 0.1)
        proj = [w1, nn.LeakyReLU(negative_slope=0.1)]
        self.proj = nn.Sequential(*proj)


        self.G = G
        import math
        embedding_dim = feature_size
        self.node_emb = nn.Embedding(G.number_of_nodes() + 1, embedding_dim)
        if init_node_vectors is None:
            nn.init.normal_(self.node_emb.weight, std=1 / (100 * embedding_dim))
        else:
            if embedding_dim != feature_size:
                from sklearn.decomposition import PCA
                init_node_vectors = torch.FloatTensor(PCA(n_components=embedding_dim, ).fit_transform(init_node_vectors))
            self.node_emb = nn.Embedding.from_pretrained(init_node_vectors, freeze=False)

        convs = []
        # 1/8, 1/4, 1/2, 1
        # 1/4, 1/2, 1
        segments = min(n_layers, 4)
        segments = list(reversed([2**i for i in range(segments)]))
        for i in range(n_layers):
            s = segments[i] if i < len(segments) else segments[-1]
            dims = feature_size
            out_dims = feature_size

            w = nn.Linear(min(n_content_dims, feature_size * 2), dims)
            init_fc(w, 'xavier_uniform_', 'leaky_relu', 0.1)
            conv = GraphSageConvWithSampling(dims, out_dims, i == n_layers - 1, gaussian_noise, conv_depth)
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
        W = nn.Linear(inp * 2, out_dims)
        layers.append(W)
        self.prediction_layer = prediction_layer
        self.noise = GaussianNoise(gaussian_noise)

        if not prediction_layer:
            init_fc(W, 'xavier_uniform_', 'leaky_relu', 0.1)
            layers.append(nn.LeakyReLU(negative_slope=0.1))
            self.skip = None
            if in_dims != out_dims:
                skip = nn.Linear(in_dims, out_dims)
                init_fc(skip, 'xavier_uniform_', 'leaky_relu', 0.1)
                self.skip = nn.Sequential(skip, nn.LeakyReLU(negative_slope=0.1))
        else:
            init_fc(W, 'xavier_uniform_', 'linear')
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
        h_new = h_new / h_new.norm(dim=1, keepdim=True).clamp(min=1e-6)

        if self.prediction_layer:
            return {'h': h_new}
        # skip
        h_agg = self.skip(h_agg) if self.skip is not None else h_agg
        h_agg = h_agg / h_agg.norm(dim=1, keepdim=True).clamp(min=1e-6)
        return {'h': h_new, 'h_residue': h_agg}


class GraphResnetWithSampling(nn.Module):
    def __init__(self, n_content_dims, feature_size, n_layers, G,
                 gaussian_noise, conv_depth,
                 init_node_vectors=None,):
        super(GraphResnetWithSampling, self).__init__()

        self.feature_size = feature_size
        self.n_layers = n_layers
        noise = GaussianNoise(gaussian_noise)

        w1 = nn.Linear(n_content_dims, feature_size)
        proj = [w1, nn.LeakyReLU(negative_slope=0.1)]
        self.proj = nn.Sequential(*proj)

        self.G = G
        import math
        embedding_dim = feature_size
        self.node_emb = nn.Embedding(G.number_of_nodes() + 1, embedding_dim)
        if init_node_vectors is None:
            nn.init.normal_(self.node_emb.weight, std=1 / (100 * embedding_dim))
        else:
            if embedding_dim != feature_size:
                from sklearn.decomposition import PCA
                init_node_vectors = torch.FloatTensor(PCA(n_components=embedding_dim, ).fit_transform(init_node_vectors))
            self.node_emb = nn.Embedding.from_pretrained(init_node_vectors, freeze=False)
        self.convs = nn.ModuleList(
            [ResnetConv(feature_size, feature_size, i == 0, i == n_layers - 1, gaussian_noise, conv_depth) for i
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
            nf.block_compute(i, self.msg[i], self.red[i], self.convs[i])

        result = nf.layers[self.n_layers].data['h']
        assert (result != result).sum() == 0
        return result


def get_score(src, dst, mean, node_biases,
              h_dst, h_src):

    score = mean + (h_src * h_dst).sum(1) + node_biases[src + 1] + node_biases[dst + 1]
    return score


class GraphSAGERecommender(nn.Module):
    def __init__(self, gcn, mu, node_biases, zeroed_indices):
        super(GraphSAGERecommender, self).__init__()

        self.gcn = gcn
        if node_biases is not None:
            assert len(node_biases) == gcn.G.number_of_nodes() + 1
            node_biases[zeroed_indices] = 0.0
            self.node_biases = nn.Parameter(torch.FloatTensor(node_biases))
        else:
            self.node_biases = nn.Parameter(torch.zeros(gcn.G.number_of_nodes() + 1))
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
    def __init__(self, gcn, margin=1.0):
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
        score = d_a_b + self.margin - d_a_c

        f = 0.1
        a = 2*f
        b = -1 * f ** 2

        score = torch.where(score >= f, score ** 2, a * score + b)
        # score = torch.where(score >= 0.1, score ** 2, 0.2 * score)
        return score


class GraphSAGELogisticEmbedding(nn.Module):
    def __init__(self, gcn, **kwargs):
        super(GraphSAGELogisticEmbedding, self).__init__()

        self.gcn = gcn

    def forward(self, nf, src, dst, label):
        h_output = self.gcn(nf)
        h_src = h_output[nf.map_from_parent_nid(-1, src, True)]
        h_dst = h_output[nf.map_from_parent_nid(-1, dst, True)]
        dist = (h_src * h_dst).sum(1)
        dist = dist + 1  # 0 to 2 range
        dist = dist / 2  # 0 to 1 range
        score = (dist - label) ** 2
        return score


class GraphSAGELogisticEmbeddingv2(nn.Module):
    def __init__(self, gcn, **kwargs):
        super(GraphSAGELogisticEmbeddingv2, self).__init__()

        self.gcn = gcn

    def forward(self, nf, src, dst, label):
        h_output = self.gcn(nf)
        h_src = h_output[nf.map_from_parent_nid(-1, src, True)]
        h_dst = h_output[nf.map_from_parent_nid(-1, dst, True)]
        dist = (h_src * h_dst).sum(1)
        dist = dist + 1  # 0 to 2 range
        dist = dist / 2  # 0 to 1 range
        dist = dist.clamp(min=1e-7, max=1-1e-7)  # HACK 1
        score = -1 * (label * torch.log(dist) + (1-label) * torch.log(1 - dist))
        return score


class GraphSAGENegativeSamplingEmbedding(nn.Module):
    def __init__(self, gcn, negative_samples=10):
        super(GraphSAGENegativeSamplingEmbedding, self).__init__()

        self.gcn = gcn
        self.negative_samples = negative_samples

    def forward(self, nf, src, dst, neg):
        h_output = self.gcn(nf)
        h_src = h_output[nf.map_from_parent_nid(-1, src, True)]
        h_dst = h_output[nf.map_from_parent_nid(-1, dst, True)]
        h_neg = h_output[nf.map_from_parent_nid(-1, neg, True)]
        h_negs = h_neg[torch.randint(0, h_neg.shape[0], (h_neg.shape[0], 3))]
        neg_loss = -1 * self.negative_samples * F.logsigmoid(
            -1 * h_src.unsqueeze(1).expand(h_src.size(0), 1, h_src.size(1)).bmm(torch.transpose(h_negs, 1, 2))).sum(
            1).sum(1)

        pos_loss = -1 * F.logsigmoid((h_src * h_dst).sum(1))
        return pos_loss + neg_loss


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

