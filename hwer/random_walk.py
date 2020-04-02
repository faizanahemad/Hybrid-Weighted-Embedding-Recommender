from __future__ import print_function

import os
import random

# from time import time


class DiGraph:
    def __init__(self):
        self.e = dict()
        self.n = dict()

    def add_edge(self, src, dst, weight):
        self.e[(src, dst)] = weight
        self.e[(dst, src)] = weight
        if src in self.n:
            self.n[src][dst] = dict(weight=weight)
        else:
            self.n[src] = {dst: dict(weight=weight)}

        if dst in self.n:
            self.n[dst][src] = dict(weight=weight)
        else:
            self.n[dst] = {src: dict(weight=weight)}

    def __getitem__(self, item):
        return self.n[item]

    def nodes(self):
        return list(self.n.keys())

    def edges(self):
        return list(self.e.keys())

    def items(self):
        return list(self.n.items())


def read_edgelist(edge_list, weighted=False):
    G = DiGraph()

    def read_unweighted(l):
        src, dst = l[0], l[1]
        G.add_edge(src, dst, 1.0)

    def read_weighted(l):
        src, dst, w = l[0], l[1], l[2]
        G.add_edge(src, dst, float(w))

    func = read_unweighted
    if weighted:
        func = read_weighted
    for x in edge_list:
        func(x)

    for src, dst in zip(G.nodes(), G.nodes()):
        G.add_edge(src, dst, 1.0)

    return G


class Node2VecWalker:
    def __init__(self, G, p, q, workers=None):
        assert type(G) == DiGraph
        self.p = p
        self.q = q
        self.nodes = list(G.nodes())
        self.edges = list(G.edges())
        self.G = G
        self.adjacency_list = {node: list(v.keys()) for node, v in self.G.items()}

    def simulate_walks_generator_optimised(self, num_walks, walk_length):
        nodes = self.nodes
        import numpy as np
        np.random.shuffle(nodes)
        adjacency_list = self.adjacency_list
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        for _ in range(num_walks):
            for node in nodes:
                node_cur_nbrs = adjacency_list[node]
                cur, back = random.choices(node_cur_nbrs, alias_nodes[node])[0], node
                walk = [back, cur]
                for _ in range(walk_length - 1):
                    cur_nbrs = adjacency_list[cur]
                    cur, back = random.choices(cur_nbrs, alias_edges[(back, cur)])[0], cur
                    walk.append(cur)
                yield walk

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
                    unnormalized_probs.append(1.0)
                else:
                    unnormalized_probs.append(w / q) # HERE
            #
            norm_const = sum(unnormalized_probs)
            normalized_probs = [u_prob / norm_const for u_prob in unnormalized_probs]
            return normalized_probs

        p = self.p
        q = self.q
        for i, edge in enumerate(self.edges):
            alias_edges[edge] = edge_processor(edge[0], edge[1], G[edge[1]], p, q)

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges


class MemoryOptimisedNode2VecWalker:
    def __init__(self, G, p, q, workers=None):
        assert type(G) == DiGraph
        self.p = p
        self.q = q
        self.nodes = list(G.nodes())
        self.edges = list(G.edges())
        self.G = G
        self.adjacency_list = {node: list(v.keys()) for node, v in self.G.items()}

    def simulate_walks_generator_optimised(self, num_walks, walk_length):
        nodes = self.nodes
        import numpy as np
        np.random.shuffle(nodes)
        adjacency_list = self.adjacency_list

        for _ in range(num_walks):
            for node in nodes:
                node_cur_nbrs = adjacency_list[node]
                cur, back = random.choices(node_cur_nbrs, self.node_proba(node))[0], node
                walk = [back, cur]
                for _ in range(walk_length - 1):
                    cur_nbrs = adjacency_list[cur]
                    cur, back = random.choices(cur_nbrs, self.edge_proba((back, cur)))[0], cur
                    walk.append(cur)
                yield walk

    def node_proba(self, node):
        neigh_dict = self.G[node]
        unnormalized_probs = [nbr['weight'] for nbr in neigh_dict.values()]
        norm_const = sum(unnormalized_probs)
        normalized_probs = [u_prob / norm_const for u_prob in unnormalized_probs]
        return normalized_probs

    def edge_proba(self, edge):
        src, dst = edge[0], edge[1]
        g_dst = self.G[edge[1]]
        zipped = [(dst_nbr == src, dst_nbr == dst, g_dst[dst_nbr]['weight']) for dst_nbr in g_dst]
        unnormalized_probs = []
        for b1, b2, w in zipped:
            if b1:
                unnormalized_probs.append(w / self.p)
            elif b2:
                unnormalized_probs.append(1.0)
            else:
                unnormalized_probs.append(w / self.q)  # HERE
        #
        norm_const = sum(unnormalized_probs)
        normalized_probs = [u_prob / norm_const for u_prob in unnormalized_probs]
        return normalized_probs

    def preprocess_transition_probs(self):
        pass


class RandomWalker:
    def __init__(self, G, p, q, workers=None):
        assert type(G) == DiGraph
        self.p = p
        self.q = q
        self.nodes = list(G.nodes())
        self.G = G
        self.adjacency_list = {node: list(v.keys()) for node, v in self.G.items()}

    def simulate_walks_generator_optimised(self, num_walks, walk_length):
        nodes = self.nodes
        import numpy as np
        np.random.shuffle(nodes)
        adjacency_list = self.adjacency_list

        for _ in range(num_walks):
            for node in nodes:
                node_cur_nbrs = adjacency_list[node]
                cur, back = random.choice(node_cur_nbrs), node
                walk = [back, cur]
                for _ in range(walk_length - 1):
                    cur_nbrs = adjacency_list[cur]
                    cur, back = random.choice(cur_nbrs), cur
                    walk.append(cur)
                yield walk

    def preprocess_transition_probs(self):
        pass

