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


class GraphSageConvWithSamplingVanilla(nn.Module):
    def __init__(self, feature_size, prediction_layer, gaussian_noise, depth):
        super(GraphSageConvWithSamplingVanilla, self).__init__()
        layers = []

        W = nn.Linear(feature_size * 2, feature_size)
        layers.append(W)
        self.prediction_layer = prediction_layer

        if not prediction_layer:
            init_weight(W.weight, 'xavier_uniform_', 'leaky_relu', 0.1)
            layers.append(nn.LeakyReLU(negative_slope=0.1))
        else:
            init_weight(W.weight, 'xavier_uniform_', 'linear')
        init_bias(W.bias)
        self.W = nn.Sequential(*layers)

    def forward(self, nodes):
        h_agg = nodes.data['h_agg']
        h = nodes.data['h']
        w = nodes.data['w'][:, None]
        h_agg = (h_agg - h) / (w - 1).clamp(min=1)  # HACK 1
        h_concat = torch.cat([h, h_agg], 1)
        h_new = self.W(h_concat)
        h_new = h_new / h_new.norm(dim=1, keepdim=True).clamp(min=1e-6)
        return {'h': h_new}


class GraphSageConvWithSamplingBase(nn.Module):
    def __init__(self, feature_size, prediction_layer, gaussian_noise, depth):
        super(GraphSageConvWithSamplingBase, self).__init__()
        layers = []
        depth = min(1, depth)
        for i in range(depth - 1):
            weights = nn.Linear(feature_size * 2, feature_size * 2)
            init_weight(weights.weight, 'xavier_uniform_', 'leaky_relu', 0.1)
            init_bias(weights.bias)
            layers.append(weights)
            layers.append(nn.LeakyReLU(negative_slope=0.1))

        W = nn.Linear(feature_size * 2, feature_size)
        layers.append(W)
        self.prediction_layer = prediction_layer
        self.noise = GaussianNoise(gaussian_noise)

        if not prediction_layer:
            init_weight(W.weight, 'xavier_uniform_', 'leaky_relu', 0.1)
            layers.append(nn.LeakyReLU(negative_slope=0.1))
        else:
            init_weight(W.weight, 'xavier_uniform_', 'linear')
        init_bias(W.bias)
        self.W = nn.Sequential(*layers)

        if not prediction_layer:
            W_out = nn.Linear(feature_size, feature_size)
            init_weight(W_out.weight, 'xavier_uniform_', 'leaky_relu', 0.1)
            init_bias(W_out.bias)
            W_out2 = nn.Linear(feature_size, feature_size)
            init_weight(W_out2.weight, 'xavier_uniform_', 'linear')
            init_bias(W_out2.bias)
            self.W_out = nn.Sequential(GaussianNoise(gaussian_noise), W_out, nn.LeakyReLU(negative_slope=0.1), W_out2)

    def pre_process(self, nodes):
        h_agg = nodes.data['h_agg']
        h = nodes.data['h']
        w = nodes.data['w'][:, None]
        h_agg = (h_agg - h) / (w - 1).clamp(min=1)  # HACK 1
        return h, h_agg

    def process_node_data(self, h):
        h = self.noise(h)
        return h

    def process_neighbourhood_data(self, h_agg):
        return h_agg

    def post_process(self, h_concat, h, h_agg):
        h_new = self.W(h_concat)
        h_new = h_new / h_new.norm(dim=1, keepdim=True).clamp(min=1e-6)
        if self.prediction_layer:
            return {'h': h_new}
        h_new = self.W_out(h_new)
        return {'h': h_new}

    def forward(self, nodes):
        h, h_agg = self.pre_process(nodes)
        h = self.process_node_data(h)
        h_agg = self.process_neighbourhood_data(h_agg)
        h_concat = torch.cat([h, h_agg], 1)
        return self.post_process(h_concat, h, h_agg)


class GraphSageWithSampling(nn.Module):
    def __init__(self, n_content_dims, feature_size, n_layers, G,
                 conv_arch, gaussian_noise, conv_depth,
                 init_node_vectors=None,):
        super(GraphSageWithSampling, self).__init__()

        self.feature_size = feature_size
        self.n_layers = n_layers
        if conv_arch == 0:
            GraphSageConvWithSampling = GraphSageConvWithSamplingBase
        elif conv_arch == 3:
            GraphSageConvWithSampling = GraphSageConvWithSamplingV3
        else:
            GraphSageConvWithSampling = GraphSageConvWithSamplingVanilla
            conv_depth = 1

        self.convs = nn.ModuleList(
            [GraphSageConvWithSampling(feature_size, i == n_layers - 1, gaussian_noise, conv_depth) for i in
             range(n_layers)])
        noise = GaussianNoise(gaussian_noise)

        w1 = nn.Linear(n_content_dims, n_content_dims)

        proj = [w1, nn.LeakyReLU(negative_slope=0.1), noise]
        init_weight(w1.weight, 'xavier_uniform_', 'leaky_relu', 0.1)
        init_bias(w1.bias)
        if conv_arch == 3:
            proj.append(LinearResnet(n_content_dims, feature_size, gaussian_noise))
        else:
            w = nn.Linear(n_content_dims, feature_size)
            init_weight(w.weight, 'xavier_uniform_', 'leaky_relu', 0.1)
            init_bias(w.bias)
            proj.extend([w, nn.LeakyReLU(negative_slope=0.1)])
        self.proj = nn.Sequential(*proj)

        self.G = G
        import math
        embedding_dim = 2 ** int(math.log2(feature_size/4))
        self.node_emb = nn.Embedding(G.number_of_nodes() + 1, embedding_dim)
        if init_node_vectors is None:
            nn.init.normal_(self.node_emb.weight, std=1 / embedding_dim)
        else:
            if embedding_dim != feature_size:
                from sklearn.decomposition import PCA
                init_node_vectors = torch.FloatTensor(PCA(n_components=embedding_dim, ).fit_transform(init_node_vectors))
            self.node_emb = nn.Embedding.from_pretrained(init_node_vectors, freeze=False)
        expansion = nn.Linear(embedding_dim, feature_size)
        init_bias(expansion.bias)
        init_weight(expansion.weight, 'xavier_uniform_', 'leaky_relu', 0.1)
        self.expansion = nn.Sequential(expansion, nn.LeakyReLU(negative_slope=0.1))

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
    def __init__(self, gcn, mu, node_biases, zeroed_indices):
        super(GraphSAGERecommenderImplicit, self).__init__()

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
        score = F.leaky_relu(d_a_b + self.margin - d_a_c)
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


class LinearResnet(nn.Module):
    def __init__(self, input_size, output_size, gaussian_noise):
        super(LinearResnet, self).__init__()
        self.scaling = False
        if input_size != output_size:
            self.scaling = True
            self.iscale = nn.Linear(input_size, output_size)
            init_weight(self.iscale.weight, 'xavier_uniform_', 'linear')
            init_bias(self.iscale.bias)
        W1 = nn.Linear(input_size, input_size)
        noise = GaussianNoise(gaussian_noise)
        init_weight(W1.weight, 'xavier_uniform_', 'leaky_relu', 0.1)
        init_bias(W1.bias)

        W2 = nn.Linear(input_size, output_size)
        init_weight(W2.weight, 'xavier_uniform_', 'leaky_relu', 0.1)
        init_bias(W2.bias)

        self.W = nn.Sequential(noise, W1, nn.LeakyReLU(negative_slope=0.1), noise, W2, nn.LeakyReLU(negative_slope=0.1))

    def forward(self, h):
        identity = h
        if self.scaling:
            identity = self.iscale(identity)
        out = self.W(h)
        out = out + identity
        return out


class GraphSageConvWithSamplingV3(GraphSageConvWithSamplingBase):
    def __init__(self, feature_size, prediction_layer, gaussian_noise, depth):
        super(GraphSageConvWithSamplingV3, self).__init__(feature_size, prediction_layer, gaussian_noise, depth)
        #
        W1 = nn.Linear(feature_size * 2, feature_size * 2)
        init_weight(W1.weight, 'xavier_uniform_', 'leaky_relu', 0.1)
        init_bias(W1.bias)
        layers = [W1, nn.LeakyReLU(negative_slope=0.1)]
        depth = min(1, depth)
        for i in range(depth):
            weights = LinearResnet(feature_size * 2, feature_size * 2, gaussian_noise)
            layers.append(weights)

        W = nn.Linear(feature_size * 2, feature_size)
        layers.append(W)

        if not prediction_layer:
            init_weight(W.weight, 'xavier_uniform_', 'leaky_relu', 0.1)
            layers.append(nn.LeakyReLU(negative_slope=0.1))
        else:
            init_weight(W.weight, 'xavier_uniform_', 'linear')
        init_bias(W.bias)

        self.W = nn.Sequential(*layers)

        if not prediction_layer:
            W_out = nn.Linear(feature_size, feature_size)
            init_weight(W_out.weight, 'xavier_uniform_', 'leaky_relu', 0.1)
            init_bias(W_out.bias)
            self.W_out = nn.Sequential(GaussianNoise(gaussian_noise), W_out, nn.LeakyReLU(negative_slope=0.1),
                                       LinearResnet(feature_size, feature_size, gaussian_noise))
