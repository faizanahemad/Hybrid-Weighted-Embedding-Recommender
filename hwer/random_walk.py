from __future__ import print_function
import random
import numpy as np
import multiprocessing

# from time import time
import networkx as nx
import pickle as pkl
import numpy as np
import scipy.sparse as sp
import os
from joblib import Parallel, delayed


class Graph(object):
    def __init__(self):
        self.G = None
        self.look_up_dict = {}
        self.look_back_list = []
        self.node_size = 0

    def encode_node(self):
        look_up = self.look_up_dict
        look_back = self.look_back_list
        for node in self.G.nodes():
            look_up[node] = self.node_size
            look_back.append(node)
            self.node_size += 1
            self.G.nodes[node]['status'] = ''

    def read_g(self, g):
        self.G = g
        self.encode_node()
        return self

    def read_edgelist(self, edge_list, weighted=False):
        self.G = nx.DiGraph()

        def read_unweighted(l):
            src, dst = l[0], l[1]
            self.G.add_edge(src, dst)
            self.G.add_edge(dst, src)
            self.G[src][dst]['weight'] = 1.0
            self.G[dst][src]['weight'] = 1.0

        def read_weighted(l):
            src, dst, w = l[0], l[1], l[2]
            self.G.add_edge(src, dst)
            self.G.add_edge(dst, src)
            self.G[src][dst]['weight'] = float(w)
            self.G[dst][src]['weight'] = float(w)
        func = read_unweighted
        if weighted:
            func = read_weighted
        for x in edge_list:
            func(x)
        self.encode_node()
        return self


class Walker:
    def __init__(self, G, p, q, workers=None):
        self.G = G.G
        self.p = p
        self.q = q
        self.node_size = G.node_size
        self.look_up_dict = G.look_up_dict
        self.adjacency_list = {node: list(self.G.neighbors(node)) for node in self.G.nodes()}
        self.nodes = list(self.G.nodes())
        self.edges = list(self.G.edges())
        self.G = {node: dict(self.G[node]) for node in self.nodes}

    def node2vec_walk(self, walk_length, start_node):
        '''
        Simulate a random walk starting from start node.
        '''
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges
        adjacency_list = self.adjacency_list

        walk = [start_node]
        cur = walk[-1]
        cur_nbrs = adjacency_list[cur]
        if len(cur_nbrs) > 0:
            walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
        else:
            return walk

        for _ in range(walk_length - 1):
            cur = walk[-1]
            cur_nbrs = adjacency_list[cur]
            if len(cur_nbrs) == 0:
                return walk

            prev = walk[-2]
            pos = (prev, cur)
            next = cur_nbrs[alias_draw(alias_edges[pos][0],
                                       alias_edges[pos][1])]
            walk.append(next)

        return walk

    def simulate_walks(self, num_walks, walk_length):
        '''
        Repeatedly simulate random walks from each node.
        '''
        walks = []
        nodes = self.nodes
        print('Walk iteration:')
        for walk_iter in range(num_walks):
            print(str(walk_iter+1), '/', str(num_walks))
            for node in nodes:
                walks.append(self.node2vec_walk(
                    walk_length=walk_length, start_node=node))

        return walks

    def simulate_walks_generator(self, num_walks, walk_length):
        '''
        Repeatedly simulate random walks from each node.
        '''
        nodes = self.nodes
        for walk_iter in range(num_walks):
            for node in nodes:
                yield self.node2vec_walk(walk_length=walk_length, start_node=node)

    def get_alias_edge(self, src, dst):
        '''
        Get the alias edge setup lists for a given edge.
        '''
        G = self.G
        p = self.p
        q = self.q
        g_dst = G[dst]

        zipped = [(dst_nbr, dst_nbr == src, dst_nbr == dst, g_dst[dst_nbr]['weight']) for dst_nbr in g_dst]
        unnormalized_probs = []
        for dst_nbr, b1, b2, w in zipped:
            if b1:
                unnormalized_probs.append(w/p)
            elif b2:
                unnormalized_probs.append(w)
            else:
                unnormalized_probs.append(w / q)
        #
        norm_const = sum(unnormalized_probs)
        normalized_probs = [u_prob/norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        '''
        Preprocessing of transition probabilities for guiding the random walks.
        '''
        G = self.G
        cpu = int(os.cpu_count()/2)

        def node_processor(node):
            n = G[node]
            unnormalized_probs = [n[nbr]['weight'] for nbr in n]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [u_prob / norm_const for u_prob in unnormalized_probs]
            node_alias = alias_setup(normalized_probs)
            return node_alias

        # node_aliases = Parallel(n_jobs=cpu, prefer="threads")(delayed(node_processor)(node) for node in G.nodes())
        # alias_nodes = dict(zip([node for node in G.nodes()], node_aliases))

        alias_nodes = {}
        for node in self.nodes:
            alias_nodes[node] = node_processor(node)

        # edge_aliases = Parallel(n_jobs=cpu, prefer="threads")(delayed(lambda edge: self.get_alias_edge(edge[0], edge[1]))(edge) for edge in G.edges())
        # alias_edges = dict(zip([edge for edge in G.edges()], edge_aliases))

        alias_edges = {}
        for edge in self.edges:
            alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return


def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    J = np.zeros(K, dtype=np.int32)

    q = K * np.array(probs)
    c = q < 1.0
    smaller = list(np.where(c)[0])
    larger = list(np.where(~c)[0])

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    kk = int(np.random.rand()*len(J))
    if np.random.rand() < q[kk]:
        return kk
    return J[kk]