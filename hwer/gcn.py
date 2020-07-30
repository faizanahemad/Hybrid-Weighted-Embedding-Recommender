import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as FN
import numpy as np
dgl.load_backend('pytorch')
from torch.nn import TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, LayerNorm, TransformerEncoderLayer
import math


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
    w1 = nn.Linear(in_dims, out_dims)
    init_fc(w1, 'xavier_uniform_', 'leaky_relu', 0.1)
    proj = [w1, nn.LeakyReLU(negative_slope=0.1), nn.LayerNorm(out_dims)]
    return nn.Sequential(*proj)


def init_fc(layer, initializer, nonlinearity, nonlinearity_param=None):
    init_weight(layer.weight, initializer, nonlinearity, nonlinearity_param)
    try:
        init_bias(layer.bias)
    except AttributeError:
        pass

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class GraphConv(nn.Module):
    def __init__(self, in_dims, out_dims, layer, gaussian_noise, prediction_layer):
        super(GraphConv, self).__init__()
        # TODO: h transform for non-prediction layer
        if prediction_layer:
            self.gaussian_noise = GaussianNoise(gaussian_noise)
            self.n_layer = layer
            self.in_dims = in_dims
            self.pos_encoder = PositionalEncoding(in_dims)
            self.ln = nn.LayerNorm(in_dims)
            self.self_ln = nn.LayerNorm(in_dims)
            decoder_layer = TransformerDecoderLayer(in_dims, 2, in_dims * 4, 0.0, "gelu")
            self.decoder = TransformerDecoder(decoder_layer, 2)
        
        self.prediction_layer = prediction_layer

    def forward(self, nodes):
        h_agg = nodes.data['h_agg']
        h = nodes.data['h']
        w = nodes.data['w'][:, None]
        h_agg = h_agg / w

        if self.prediction_layer:
            h_agg = h_agg.reshape((h_agg.size(0), self.n_layer, self.in_dims))
            h_agg = h_agg.transpose(0, 1)
            h = h.unsqueeze(1).transpose(0, 1)
            h_new = self.decoder(h, h_agg).transpose(0, 1).squeeze()
        else:
            h_new = torch.cat([h_agg, h], 1)
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
        self.emb_expand = nn.Linear(embedding_dim, feature_size)
        init_fc(self.emb_expand, "xavier_uniform", "linear")
        convs = []
        for i in range(n_layers):
            conv = GraphConv(feature_size,
                             feature_size,
                             i+1,
                             gaussian_noise if i == 0 else 0, i == n_layers - 1)
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
            nh = self.emb_expand(self.node_emb(nf.layer_parent_nid(i) + 1))
            nf.layers[i].data['h'] = nh
            nf.layers[i].data['one'] = torch.ones(nf.layer_size(i))
            mix_embeddings(nf.layers[i].data, self.proj)
            nf.layers[i].data['h'] = nf.layers[i].data['h'] / nf.layers[i].data['h'].norm(dim=1, keepdim=True).clamp(min=1e-5)
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

