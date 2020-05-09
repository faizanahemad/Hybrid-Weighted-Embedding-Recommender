import torch
import torch.nn as nn
import torch.nn.functional as F
from .gcn import *


class NCF(nn.Module):
    def __init__(self, feature_size, depth, gaussian_noise):
        super(NCF, self).__init__()
        noise = GaussianNoise(gaussian_noise)
        layers = [noise]
        for layer_idx in range(1, depth + 1):
            iw = 4 if layer_idx == 2 else 2
            ow = 1 if layer_idx == depth else (4 if layer_idx == 1 else 2)
            wx = nn.Linear(feature_size * iw, feature_size * ow)
            init_fc(wx, 'xavier_uniform_', 'leaky_relu', 0.1)
            layers.extend([wx, nn.LeakyReLU(negative_slope=0.1)])

        w_out = nn.Linear(feature_size, 1)
        init_fc(w_out, 'xavier_uniform_', 'sigmoid', 0.1)
        layers.extend([w_out, nn.Sigmoid()])
        self.W = nn.Sequential(*layers)

    def forward(self, src, dst, g_src, g_dst):
        vec = torch.cat([g_src, g_dst], 1)
        ncf = self.W(vec).flatten()
        return ncf


class RecImplicit(nn.Module):
    def __init__(self, gcn: GraphConvModule, ncf: NCF):
        super(RecImplicit, self).__init__()
        self.gcn = gcn
        self.ncf = ncf

    def forward(self, nf, src, dst):
        h_output = self.gcn(nf)
        h_src = h_output[nf.map_from_parent_nid(-1, src, True)]
        h_dst = h_output[nf.map_from_parent_nid(-1, dst, True)]
        return self.ncf(src, dst, h_src, h_dst)


