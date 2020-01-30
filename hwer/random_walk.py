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
import random


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
        self.adjacency_list = {node: list(v.keys()) for node, v in self.G.items()}

    def node2vec_walk(self, walk_length, start_node):
        '''
        Simulate a random walk starting from start node.
        '''
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges
        adjacency_list = self.adjacency_list

        walk = [start_node]
        # put next 5 lines as fn
        cur_nbrs = adjacency_list[start_node]
        n_neighbors = len(cur_nbrs)
        if n_neighbors == 0:
            return walk
        else:
            walk.append(cur_nbrs[random.choices(range(n_neighbors), alias_nodes[start_node])[0]])

        for _ in range(walk_length - 1):
            cur = walk[-1]
            cur_nbrs = adjacency_list[cur]
            n_neighbors = len(cur_nbrs)
            if n_neighbors == 0:
                return walk

            prev = walk[-2]
            pos = (prev, cur)
            next = cur_nbrs[random.choices(range(n_neighbors), alias_edges[pos])[0]]
            walk.append(next)

        return walk

    def simulate_walks(self, num_walks, walk_length):
        '''
        Repeatedly simulate random walks from each node.
        '''
        walks = []
        nodes = self.nodes
        node2vec_walk = self.node2vec_walk
        for walk_iter in range(num_walks):
            for node in nodes:
                walks.append(node2vec_walk(
                    walk_length=walk_length, start_node=node))

        return walks

    def simulate_walks_generator(self, num_walks, walk_length):
        '''
        Repeatedly simulate random walks from each node.
        '''
        nodes = self.nodes
        node2vec_walk = self.node2vec_walk
        for walk_iter in range(num_walks):
            for node in nodes:
                yield node2vec_walk(walk_length=walk_length, start_node=node)

    def preprocess_transition_probs(self):
        '''
        Preprocessing of transition probabilities for guiding the random walks.
        '''
        G = self.G
        cpu = int(os.cpu_count()/2)

        def node_processor(n):
            unnormalized_probs = [nbr['weight'] for nbr in n.values()]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [u_prob / norm_const for u_prob in unnormalized_probs]
            return normalized_probs

        # node_aliases = Parallel(n_jobs=cpu, prefer="threads")(delayed(node_processor)(node) for node in G.nodes())
        # alias_nodes = dict(zip([node for node in G.nodes()], node_aliases))

        alias_nodes = {}
        for node, nd in self.G.items():
            alias_nodes[node] = node_processor(nd)

        # edge_aliases = Parallel(n_jobs=cpu, prefer="threads")(delayed(lambda edge: self.get_alias_edge(edge[0], edge[1]))(edge) for edge in G.edges())
        # alias_edges = dict(zip([edge for edge in G.edges()], edge_aliases))

        alias_edges = {}

        def edge_processor(src, dst, g_dst, p, q):
            '''
            Get the alias edge setup lists for a given edge.
            '''

            zipped = [(dst_nbr == src, dst_nbr == dst, g_dst[dst_nbr]['weight']) for dst_nbr in g_dst]
            unnormalized_probs = []
            for b1, b2, w in zipped:
                if b1:
                    unnormalized_probs.append(w / p)
                elif b2:
                    unnormalized_probs.append(w)
                else:
                    unnormalized_probs.append(w / q) # HERE
            #
            norm_const = sum(unnormalized_probs)
            normalized_probs = [u_prob / norm_const for u_prob in unnormalized_probs]
            return normalized_probs

        p = self.p
        q = self.q
        for edge in self.edges:
            alias_edges[edge] = edge_processor(edge[0], edge[1], G[edge[1]], p, q)

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges