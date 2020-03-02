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


def init_weight(param, initializer, nonlinearity):
    initializer = getattr(nn.init, initializer)
    if nonlinearity is not None:
        initializer(param)
    else:
        initializer(param, nn.init.calculate_gain(nonlinearity))


def init_bias(param):
    nn.init.normal_(param, 0, 0.01)


class GraphSageConvWithSamplingVanilla(nn.Module):
    def __init__(self, feature_size, dropout, activation, prediction_layer, gaussian_noise, depth):
        super(GraphSageConvWithSamplingVanilla, self).__init__()
        layers = []

        W = nn.Linear(feature_size * 2, feature_size)
        layers.append(W)
        self.activation = activation
        self.prediction_layer = prediction_layer

        if self.activation is not None:
            init_weight(W.weight, 'xavier_uniform_', 'leaky_relu')
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
        if self.prediction_layer:
            return {'h': h_new}
        return {'h': h_new / h_new.norm(dim=1, keepdim=True).clamp(min=1e-6)}


class GraphSageConvWithSamplingBase(nn.Module):
    def __init__(self, feature_size, dropout, activation, prediction_layer, gaussian_noise, depth):
        super(GraphSageConvWithSamplingBase, self).__init__()
        layers = []
        depth = min(1, depth)
        self.drop = nn.Dropout(dropout)
        for i in range(depth - 1):
            weights = nn.Linear(feature_size * 2, feature_size * 2)
            init_weight(weights.weight, 'xavier_uniform_', 'leaky_relu')
            init_bias(weights.bias)
            layers.append(weights)
            layers.append(nn.LeakyReLU(negative_slope=0.1))

        W = nn.Linear(feature_size * 2, feature_size)
        layers.append(W)
        self.activation = activation
        self.prediction_layer = prediction_layer
        self.noise = GaussianNoise(gaussian_noise)

        if self.activation is not None:
            init_weight(W.weight, 'xavier_uniform_', 'leaky_relu')
            layers.append(nn.LeakyReLU(negative_slope=0.1))
        else:
            init_weight(W.weight, 'xavier_uniform_', 'linear')
        init_bias(W.bias)
        self.W = nn.Sequential(*layers)
        
        W_out =  nn.Linear(feature_size, feature_size)
        init_weight(W_out.weight, 'xavier_uniform_', 'leaky_relu')
        init_bias(W_out.bias)
        self.W_out = nn.Sequential(GaussianNoise(gaussian_noise), W_out, nn.LeakyReLU(negative_slope=0.1))

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
        if self.prediction_layer:
            return {'h': h_new, 'h_out': h_new}
        h_new = h_new / h_new.norm(dim=1, keepdim=True).clamp(min=1e-6)
        return {'h': h_new, 'h_out': self.W_out(h_new)}

    def forward(self, nodes):
        h, h_agg = self.pre_process(nodes)
        h = self.process_node_data(h)
        h_agg = self.process_neighbourhood_data(h_agg)
        h_concat = torch.cat([h, h_agg], 1)
        h_concat = self.drop(h_concat)
        return self.post_process(h_concat, h, h_agg)


class GraphSageConvWithSamplingV1(GraphSageConvWithSamplingBase):
    def __init__(self, feature_size, dropout, activation, prediction_layer, gaussian_noise, depth):
        super(GraphSageConvWithSamplingV1, self).__init__(feature_size, dropout, activation, prediction_layer, gaussian_noise, depth)

        Wagg_1 = nn.Linear(feature_size, feature_size)
        Wagg = [Wagg_1, nn.LeakyReLU(negative_slope=0.1)]
        init_bias(Wagg_1.bias)
        init_weight(Wagg_1.weight, 'xavier_uniform_', 'leaky_relu')
        self.Wagg = nn.Sequential(*Wagg)

    def process_neighbourhood_data(self, h_agg):
        return self.Wagg(h_agg)


class GraphSageConvWithSamplingV2(GraphSageConvWithSamplingBase):
    def __init__(self, feature_size, dropout, activation, prediction_layer, gaussian_noise, depth):
        super(GraphSageConvWithSamplingV2, self).__init__(feature_size, dropout, activation, prediction_layer, gaussian_noise, depth)

        self.feature_size = feature_size

        #
        noise = GaussianNoise(gaussian_noise)
        self.noise = noise
        Wagg_1 = nn.Linear(feature_size, feature_size)
        Wagg = [Wagg_1, nn.LeakyReLU(negative_slope=0.1)]
        self.Wagg = nn.Sequential(*Wagg)

        Wh1 = nn.Linear(feature_size, feature_size)
        Wh = [Wh1, nn.LeakyReLU(negative_slope=0.1)]
        self.Wh = nn.Sequential(*Wh)

        init_bias(Wh1.bias)
        init_bias(Wagg_1.bias)
        init_weight(Wagg_1.weight, 'xavier_uniform_', 'leaky_relu')
        init_weight(Wh1.weight, 'xavier_uniform_', 'leaky_relu')

        #
    def process_neighbourhood_data(self, h_agg):
        return self.Wagg(h_agg)

    def process_node_data(self, h):
        h = self.noise(h)
        h = self.Wh(h)
        return h


class GraphSageWithSampling(nn.Module):
    def __init__(self, n_content_dims, feature_size, n_layers, dropout, prediction_layer, G,
                 conv_arch, gaussian_noise, conv_depth,
                 init_node_vectors=None,):
        super(GraphSageWithSampling, self).__init__()

        self.feature_size = feature_size
        self.n_layers = n_layers
        if conv_arch == 0:
            GraphSageConvWithSampling = GraphSageConvWithSamplingBase
        elif conv_arch == 1:
            GraphSageConvWithSampling = GraphSageConvWithSamplingV1
        elif conv_arch == 2:
            GraphSageConvWithSampling = GraphSageConvWithSamplingV2
        elif conv_arch == 3:
            GraphSageConvWithSampling = GraphSageConvWithSamplingV3
        elif conv_arch == 4:
            GraphSageConvWithSampling = GraphSageConvWithSamplingV4
        else:
            GraphSageConvWithSampling = GraphSageConvWithSamplingVanilla
            conv_depth = 1

        convs = []
        for i in range(n_layers):
            if i >= n_layers - 1:
                convs.append(GraphSageConvWithSampling(feature_size, dropout, None, prediction_layer, gaussian_noise, conv_depth))
            else:
                convs.append(GraphSageConvWithSampling(feature_size, dropout, F.leaky_relu, False, gaussian_noise, conv_depth))

        self.convs = nn.ModuleList(convs)
        noise = GaussianNoise(gaussian_noise)

        if conv_arch == 1 or conv_arch == 2:
            w1 = nn.Linear(n_content_dims, n_content_dims)
            w = nn.Linear(n_content_dims, feature_size)
            proj = [noise, w1, nn.LeakyReLU(negative_slope=0.1), w, nn.LeakyReLU(negative_slope=0.1)]
            init_weight(w1.weight, 'xavier_uniform_', 'leaky_relu')
            init_bias(w1.bias)
        elif conv_arch == 3 or conv_arch == 4:
            w = nn.Linear(n_content_dims, n_content_dims)
            proj = [noise, w, nn.LeakyReLU(negative_slope=0.1)]
            proj.append(LinearResnet(n_content_dims, n_content_dims, gaussian_noise))
            proj.append(LinearResnet(n_content_dims, feature_size, gaussian_noise))
        else:
            w = nn.Linear(n_content_dims, feature_size)
            proj = [noise, w, nn.LeakyReLU(negative_slope=0.1)]

        init_weight(w.weight, 'xavier_uniform_', 'leaky_relu')
        init_bias(w.bias)
        self.proj = nn.Sequential(*proj)

        self.G = G

        self.node_emb = nn.Embedding(G.number_of_nodes() + 1, feature_size)
        if init_node_vectors is None:
            nn.init.normal_(self.node_emb.weight, std=1 / self.feature_size)
        else:
            # self.node_emb.weight = nn.Parameter(init_node_vectors)
            self.node_emb = nn.Embedding.from_pretrained(init_node_vectors, freeze=False)

    msg = [FN.copy_src('h_out', 'h'),
           FN.copy_src('one', 'one')]
    red = [FN.sum('h', 'h_agg'), FN.sum('one', 'w')]

    def forward(self, nf):
        '''
        nf: NodeFlow.
        '''
        nf.copy_from_parent(edge_embed_names=None)
        for i in range(nf.num_layers):
            nf.layers[i].data['h_out'] = self.node_emb(nf.layer_parent_nid(i) + 1)
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


class NCFScorer(nn.Module):
    def __init__(self, feature_size, dropout, gaussian_noise, scorer_depth):
        super(NCFScorer, self).__init__()
        noise = GaussianNoise(gaussian_noise)
        self.noise = noise
        w1 = nn.Linear(feature_size * 2, feature_size * 2)
        pipeline_1 = nn.Sequential(w1, nn.LeakyReLU(negative_slope=0.1))
        self.pipeline_1 = pipeline_1
        w4 = nn.Linear(feature_size * 2, feature_size * 2)
        self.uvd_pipeline = nn.Sequential(w4, nn.LeakyReLU(negative_slope=0.1))
        w7 = nn.Linear(feature_size * 4, 1)

        layers = [LinearResnet(feature_size * 4, feature_size * 4, gaussian_noise)]
        for i in range(scorer_depth - 1):
            layers.append(LinearResnet(feature_size * 4, feature_size * 4, gaussian_noise))
        layers.append(w7)
        self.layers = nn.Sequential(*layers)

        # init weights
        init_weight(w7.weight, 'xavier_uniform_', 'linear')
        init_bias(w7.bias)
        weights = [w1, w4]
        for w in weights:
            init_weight(w.weight, 'xavier_uniform_', 'leaky_relu')
            init_bias(w.bias)

    def forward(self, src, dst, mu, node_biases, h_dst, h_src):
        user_item_vec_dot = h_src * h_dst
        user_item_vec_dist = (h_src - h_dst) ** 2
        user_bias = node_biases[src + 1]
        item_bias = node_biases[dst + 1]
        uvd = torch.cat([user_item_vec_dot, user_item_vec_dist], 1)
        uvd = self.uvd_pipeline(uvd)

        p1_out = self.pipeline_1(torch.cat([h_src, h_dst], 1))
        mains = torch.cat([uvd, p1_out], 1)
        score = self.layers(mains).flatten()
        return user_bias + item_bias + score


class GraphSAGERecommenderNCF(GraphSAGERecommenderImplicit):
    def __init__(self, gcn: GraphSageWithSampling, mu, node_biases, zeroed_indices, ncf: NCFScorer):
        super(GraphSAGERecommenderNCF, self).__init__(gcn, mu, node_biases, zeroed_indices)
        self.ncf = ncf

    def forward(self, nf, src, dst):
        h_output = self.gcn(nf)
        h_src = h_output[nf.map_from_parent_nid(-1, src, True)]
        h_dst = h_output[nf.map_from_parent_nid(-1, dst, True)]
        score = self.ncf(src, dst, self.mu, self.node_biases,
                          h_dst, h_src)

        return score


class VAENCFScorer(nn.Module):
    pass


class GraphSAGERecommenderVAENCF(nn.Module):
    pass


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


class LinearResnet(nn.Module):
    def __init__(self, input_size, output_size, gaussian_noise):
        super(LinearResnet, self).__init__()
        self.scaling = False
        if input_size != output_size:
            self.scaling = True
            self.iscale = nn.Linear(input_size, output_size)
            init_weight(self.iscale.weight, 'xavier_uniform_', 'linear')
            init_bias(self.iscale.bias)
        W1 = nn.Linear(input_size, output_size)
        bn1 = torch.nn.BatchNorm1d(output_size)
        noise = GaussianNoise(gaussian_noise)

        init_weight(W1.weight, 'xavier_uniform_', 'leaky_relu')
        init_bias(W1.bias)

        self.W = nn.Sequential(noise, W1, nn.LeakyReLU(negative_slope=0.1))

    def forward(self, h):
        identity = h
        if self.scaling:
            identity = self.iscale(identity)
        out = self.W(h)
        out = out + identity
        return out


class GraphSageConvWithSamplingV3(GraphSageConvWithSamplingV1):
    def __init__(self, feature_size, dropout, activation, prediction_layer, gaussian_noise, depth):
        super(GraphSageConvWithSamplingV3, self).__init__(feature_size, dropout, activation, prediction_layer, gaussian_noise, depth)
        #
        layers = []
        depth = min(1, depth)
        self.drop = nn.Dropout(dropout)
        for i in range(depth - 1):
            weights = LinearResnet(feature_size * 2, feature_size * 2, gaussian_noise)
            layers.append(weights)

        W = LinearResnet(feature_size * 2, feature_size, gaussian_noise)
        layers.append(W)
        self.W = nn.Sequential(*layers)

        Wagg_1 = nn.Linear(feature_size, feature_size)
        init_bias(Wagg_1.bias)
        init_weight(Wagg_1.weight, 'xavier_uniform_', 'leaky_relu')
        Wagg = [Wagg_1, nn.LeakyReLU(negative_slope=0.1), LinearResnet(feature_size, feature_size, gaussian_noise)]
        self.Wagg = nn.Sequential(*Wagg)

        W_out = nn.Linear(feature_size, feature_size)
        init_weight(W_out.weight, 'xavier_uniform_', 'leaky_relu')
        init_bias(W_out.bias)
        self.W_out = nn.Sequential(GaussianNoise(gaussian_noise), W_out, nn.LeakyReLU(negative_slope=0.1),
                                   LinearResnet(feature_size, feature_size, gaussian_noise))


class GraphSageConvWithSamplingV4(GraphSageConvWithSamplingV3):
    def __init__(self, feature_size, dropout, activation, prediction_layer, gaussian_noise, depth):
        super(GraphSageConvWithSamplingV4, self).__init__(feature_size, dropout, activation, prediction_layer, gaussian_noise, depth)

        self.feature_size = feature_size

        #
        noise = GaussianNoise(gaussian_noise)
        self.noise = noise

        Wh1 = nn.Linear(feature_size, feature_size)
        Wh = [Wh1, nn.LeakyReLU(negative_slope=0.1), LinearResnet(feature_size, feature_size, gaussian_noise)]
        self.Wh = nn.Sequential(*Wh)

        init_bias(Wh1.bias)
        init_weight(Wh1.weight, 'xavier_uniform_', 'leaky_relu')

    def process_node_data(self, h):
        h = self.noise(h)
        h = self.Wh(h)
        return h

