from __future__ import print_function

import os
import random

# from time import time


def read_edgelist(edge_list, weighted=False):
    import networkx as nx
    G = nx.DiGraph()

    def read_unweighted(l):
        src, dst = l[0], l[1]
        G.add_edge(src, dst)
        G.add_edge(dst, src)
        G[src][dst]['weight'] = 1.0
        G[dst][src]['weight'] = 1.0

    def read_weighted(l):
        src, dst, w = l[0], l[1], l[2]
        G.add_edge(src, dst)
        G.add_edge(dst, src)
        G[src][dst]['weight'] = float(w)
        G[dst][src]['weight'] = float(w)

    func = read_unweighted
    if weighted:
        func = read_weighted
    for x in edge_list:
        func(x)
    G.add_edges_from(zip(G.nodes(), G.nodes()), weight=1)

    return G


class Walker:
    def __init__(self, G, p, q, workers=None):
        import networkx as nx
        assert type(G) == nx.classes.digraph.DiGraph
        self.p = p
        self.q = q
        self.nodes = list(G.nodes())
        self.edges = list(G.edges())
        self.G = {node: dict(G[node]) for node in self.nodes}
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

        walk.append(cur_nbrs[random.choices(range(n_neighbors), alias_nodes[start_node])[0]])
        for _ in range(walk_length - 1):
            cur = walk[-1]
            cur_nbrs = adjacency_list[cur]
            n_neighbors = len(cur_nbrs)
            if n_neighbors == 0:
                return walk
            prev = walk[-2]
            pos = (prev, cur)
            walk.append(cur_nbrs[random.choices(range(n_neighbors), alias_edges[pos])[0]])

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

    def simulate_walks_generator_optimised(self, num_walks, walk_length):
        '''
        Repeatedly simulate random walks from each node.
        '''
        nodes = self.nodes
        import numpy as np
        np.random.shuffle(nodes)
        adjacency_list = self.adjacency_list
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        for node in nodes:
            node_cur_nbrs = adjacency_list[node]
            for _ in range(num_walks):
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
        for edge in self.edges:
            alias_edges[edge] = edge_processor(edge[0], edge[1], G[edge[1]], p, q)

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges