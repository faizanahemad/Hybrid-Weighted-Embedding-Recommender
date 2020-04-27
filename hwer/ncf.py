import torch
import torch.nn as nn
import torch.nn.functional as F
from .gcn import *


class LinearResnet(nn.Module):
    def __init__(self, in_dims, out_dims, gaussian_noise=0.0):
        super(LinearResnet, self).__init__()
        noise = GaussianNoise(gaussian_noise)
        w1 = nn.Linear(in_dims, out_dims)
        init_fc(w1, 'xavier_uniform_', 'leaky_relu', 0.1)
        w2 = nn.Linear(out_dims, out_dims)
        init_fc(w2, 'xavier_uniform_', 'leaky_relu', 0.1)
        residuals = [w1, nn.LeakyReLU(negative_slope=0.1), noise, w2, nn.LeakyReLU(negative_slope=0.1)]
        self.residuals = nn.Sequential(*residuals)

        self.skip = None
        if in_dims != out_dims:
            skip = nn.Linear(in_dims, out_dims)
            init_fc(skip, 'xavier_uniform_', 'leaky_relu', 0.1)
            self.skip = nn.Sequential(skip, nn.LeakyReLU(negative_slope=0.1))

    def forward(self, x):
        r = self.residuals(x)
        x = x if self.skip is None else self.skip(x)
        return x + r


class NCF(nn.Module):
    def __init__(self, feature_size, depth, gaussian_noise, content, ncf_gcn_balance):
        super(NCF, self).__init__()
        noise = GaussianNoise(gaussian_noise)

        w1 = nn.Linear(feature_size * 2, feature_size * (2 ** (depth - 1)))
        init_fc(w1, 'xavier_uniform_', 'leaky_relu', 0.1)
        layers = [noise, w1, nn.LeakyReLU(negative_slope=0.1)]

        for i in reversed(range(depth - 1)):
            wx = nn.Linear(feature_size * (2 ** (i+1)), feature_size * (2 ** i))
            init_fc(wx, 'xavier_uniform_', 'leaky_relu', 0.1)
            layers.extend([noise, wx, nn.LeakyReLU(negative_slope=0.1)])

        w_out = nn.Linear(feature_size, 1)
        init_fc(w_out, 'xavier_uniform_', 'sigmoid', 0.1)
        layers.extend([w_out, nn.Sigmoid()])
        self.W = nn.Sequential(*layers)
        self.ncf_gcn_balance = ncf_gcn_balance

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


