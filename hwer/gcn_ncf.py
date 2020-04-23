import time
from typing import List, Dict, Tuple, Optional, Set

import numpy as np
from more_itertools import flatten

from .logging import getLogger
from .random_walk import *
from .utils import unit_length_violations
import logging
import dill
import sys
logger = getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
from .gcn_recommender import GCNRecommender

from .recommendation_base import RecommendationBase, NodeType, Node, Edge, FeatureName
from .embed import BaseEmbed
from .gcn import *
from .ncf import *

# Take out the training loop with a loss builder fn

class GcnNCF(GCNRecommender):
    def __init__(self, embedding_mapper: Dict[NodeType, Dict[str, BaseEmbed]], node_types: Set[str],
                 n_dims: int = 32):
        super().__init__(embedding_mapper, node_types, n_dims)
        self.log = getLogger(type(self).__name__)
        assert n_dims % 2 == 0
        self.cpu = int(os.cpu_count() / 2)

    def __positive_pair_generator__(self, nodes: List[Node],
                                    edges: List[Edge],
                                    hyperparams):
        ps_proportion = hyperparams["ps_proportion"] if "ps_proportion" in hyperparams else 1
        ps_threshold = hyperparams["ps_threshold"] if "ps_threshold" in hyperparams else 0.1
        assert ps_threshold < 1
        positive_samples = len(edges) * ps_proportion
        node_to_index = self.nodes_to_idx
        p = 0.25
        q = hyperparams["q"] if "q" in hyperparams else 0.25
        total_nodes = len(nodes)
        edge_list = [(node_to_index[e.src], node_to_index[e.dst], e.weight) for e in edges]
        edge_list.extend([(i, i, 1) for i in range(total_nodes)])
        Walker = RandomWalker
        walker = Walker(read_edgelist(edge_list, weighted=False), p=p, q=q)
        walker.preprocess_transition_probs()
        samples_per_node = int(np.ceil(positive_samples / total_nodes))
        from collections import Counter
        random_walks = max(500, samples_per_node * 50)

        def results_filter(cnt):
            results = cnt.most_common()
            top_n = results[:5]
            results = list(filter(lambda res: res[1] / random_walks >= ps_threshold, results))
            results = results[:samples_per_node]
            if len(results) == 0:
                results = top_n
            return results



        def sampler():
            for i in range(total_nodes):
                cnt_2 = Counter()
                cnt_3 = Counter()
                for walk in walker.simulate_walks_single_node(i, random_walks, 4):
                    if len(walk) >= 3 and walk[2] != i:
                        cnt_2.update([walk[2]])
                    if len(walk) == 4 and walk[3] != i:
                        cnt_3.update([walk[3]])
                results = results_filter(cnt_2) + results_filter(cnt_3)
                for r, w in results:
                    yield i, r, w/random_walks

        return sampler

    def __negative_pair_generator__(self, nodes: List[Node],
                                    edges: List[Edge],
                                    hyperparams):
        nsh = hyperparams["nsh"] if "nsh" in hyperparams else 1
        positive_samples = len(edges)
        p = 0.25
        q = hyperparams["q"] if "q" in hyperparams else 0.25
        negative_samples = int(nsh * positive_samples)
        total_nodes = len(nodes)
        node_to_index = self.nodes_to_idx

        edge_list = [(node_to_index[e.src], node_to_index[e.dst], e.weight) for e in edges]
        edge_list.extend([(i, i, 1) for i in range(total_nodes)])
        Walker = RandomWalker
        walker = Walker(read_edgelist(edge_list, weighted=False), p=p, q=q)
        walker.preprocess_transition_probs()
        # samples_per_node = {i: int(len(walker.adjacency_list[i]) * nsh) for i in range(total_users + total_items)}
        spn = int(np.ceil(negative_samples / total_nodes))
        samples_per_node = {i: spn for i in range(total_nodes)}
        all_nodes = set([i for i in range(total_nodes)])

        def nsg():
            for i in range(total_nodes):
                neighbours = {i}
                for walk in walker.simulate_walks_single_node(i, 20, 5):
                    neighbours.update(walk)
                candidates = list(all_nodes - neighbours)
                results = random.choices(candidates, k=samples_per_node[i])
                for r in results:
                    yield i, r, 1.0

        return nsg

    def __word2vec_neg_sampler(self, nodes: List[Node],
                        edges: List[Edge], hyperparams):

        ns_w2v_proportion = hyperparams["ns_w2v_proportion"] if "ns_w2v_proportion" in hyperparams else 0
        ns_w2v_exponent = hyperparams["ns_w2v"] if "ns_w2v" in hyperparams else 3.0 / 4.0
        proportion = int(len(edges) * ns_w2v_proportion)
        from collections import Counter
        total_nodes = len(nodes)
        node_to_index = self.nodes_to_idx
        edge_list = [(node_to_index[e.src], node_to_index[e.dst], e.weight) for e in edges]
        edge_list.extend([(i, i, 1) for i in range(total_nodes)])
        proba_dict = Counter([u for u, i, r in edge_list])
        proba_dict.update(Counter([i for u, i, r in edge_list]))

        probas = np.array([proba_dict[i] for i in range(total_nodes)])
        probas = probas ** ns_w2v_exponent
        probas = probas / probas.sum()
        probas = torch.tensor(probas)

        def sampler():
            src_neg = torch.multinomial(probas, proportion, replacement=True)
            dst_neg = torch.multinomial(probas, proportion, replacement=True)
            weights_neg = torch.tensor([1.0] * len(src_neg))
            return src_neg, dst_neg, weights_neg
        return sampler

    def __simple_neg_sampler__(self, nodes: List[Node],
                               edges: List[Edge], hyperparams):
        ns_proportion = hyperparams["ns_proportion"] if "ns_proportion" in hyperparams else 1
        total_nodes = len(nodes)
        positive_samples = len(edges)
        ns = ns_proportion
        negative_samples = int(ns * positive_samples)

        def sampler():
            src_neg = torch.randint(0, total_nodes, (negative_samples,))
            dst_neg = torch.randint(0, total_nodes, (negative_samples,))
            weights_neg = torch.tensor([1.0] * negative_samples)
            return src_neg, dst_neg, weights_neg

        return sampler

    def __data_gen_fn__(self, nodes: List[Node],
                        edges: List[Edge], node_to_index: Dict[Node, int],
                        hyperparams):
        nsh = hyperparams["nsh"] if "nsh" in hyperparams else 0
        ns_proportion = hyperparams["ns_proportion"] if "ns_proportion" in hyperparams else 1
        ps_proportion = hyperparams["ps_proportion"] if "ps_proportion" in hyperparams else 0
        ps_threshold = hyperparams["ps_threshold"] if "ps_threshold" in hyperparams else 0.1
        ns_w2v_proportion = hyperparams["ns_w2v_proportion"] if "ns_w2v_proportion" in hyperparams else 0
        ns_w2v_exponent = hyperparams["ns_w2v_exponent"] if "ns_w2v_exponent" in hyperparams else 3.0/4.0

        affinities = [(node_to_index[e.src], node_to_index[e.dst], e.weight) for e in edges]
        total_nodes = len(nodes)
        hard_negative_gen = self.__negative_pair_generator__(nodes, edges, hyperparams)
        pos_gen = self.__positive_pair_generator__(nodes, edges, hyperparams)
        w2v_neg_gen = self.__word2vec_neg_sampler(nodes, edges, hyperparams)
        simple_neg_gen = self.__simple_neg_sampler__(nodes, edges, hyperparams)

        def get_samples():
            np.random.shuffle(affinities)
            src, dst, weights, ratings = [], [], [], []
            for u, v, r in affinities:
                src.append(u)
                dst.append(v)
                weights.append(r)
                ratings.append(1.0)

            src = torch.LongTensor(src)
            dst = torch.LongTensor(dst)
            weights = torch.FloatTensor(weights)
            ratings = torch.FloatTensor(ratings)

            if ns_proportion > 0:
                src_neg, dst_neg, weights_neg = simple_neg_gen()
                ratings_neg = torch.zeros_like(src_neg, dtype=torch.float)
                src = torch.cat((src, src_neg), 0)
                dst = torch.cat((dst, dst_neg), 0)
                weights = torch.cat((weights, weights_neg), 0)
                ratings = torch.cat((ratings, ratings_neg), 0)

            if ns_w2v_proportion > 0:
                src_neg, dst_neg, weights_neg = w2v_neg_gen()
                src = torch.cat((src, src_neg), 0)
                dst = torch.cat((dst, dst_neg), 0)
                weights = torch.cat((weights, weights_neg), 0)
                ratings_neg = torch.zeros_like(src_neg, dtype=torch.float)
                ratings = torch.cat((ratings, ratings_neg), 0)

            if nsh > 0:
                h_src_neg, h_dst_neg, weights_hneg = zip(*hard_negative_gen())
                weights_hneg = torch.FloatTensor(weights_hneg)
                h_src_neg = torch.LongTensor(h_src_neg)
                h_dst_neg = torch.LongTensor(h_dst_neg)
                src = torch.cat((src, h_src_neg), 0)
                dst = torch.cat((dst, h_dst_neg), 0)
                weights = torch.cat((weights, weights_hneg), 0)
                ratings_neg = torch.zeros_like(h_src_neg, dtype=torch.float)
                ratings = torch.cat((ratings, ratings_neg), 0)

            if ps_proportion > 0:
                h_src_pos, h_dst_pos, h_weight_pos = zip(*pos_gen())
                weights_pos = torch.FloatTensor(h_weight_pos)
                h_src_pos = torch.LongTensor(h_src_pos)
                h_dst_pos = torch.LongTensor(h_dst_pos)
                src = torch.cat((src, h_src_pos), 0)
                dst = torch.cat((dst, h_dst_pos), 0)
                weights = torch.cat((weights, weights_pos), 0)
                ratings_pos = torch.ones_like(h_src_pos, dtype=torch.float)
                ratings = torch.cat((ratings, ratings_pos), 0)

            shuffle_idx = torch.randperm(len(src))
            src = src[shuffle_idx]
            dst = dst[shuffle_idx]
            weights = weights[shuffle_idx]
            ratings = ratings[shuffle_idx]
            return src, dst, weights, ratings
        return get_samples

    def __build_collaborative_embeddings__(self,
                                           nodes: List[Node],
                                           edges: List[Edge],
                                           content_vectors: np.ndarray,
                                           hyperparams: Dict) -> np.ndarray:
        from .gcn import build_dgl_graph
        import torch
        import torch.nn.functional as F
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cpu = torch.device('cpu')
        import dgl
        n_dims = self.n_dims
        node_to_index = self.nodes_to_idx
        self.log.debug(
            "Started Building Collaborative Embeddings, n_nodes = %s, n_edges = %s, in_dims = %s, out_dims = %s",
            len(nodes), len(edges), content_vectors.shape[1], n_dims)
        assert len(nodes) == len(content_vectors)

        lr = hyperparams["lr"] if "lr" in hyperparams else 0.1
        epochs = hyperparams["epochs"] if "epochs" in hyperparams else 1
        gcn_layers = hyperparams["gcn_layers"] if "gcn_layers" in hyperparams else 2
        ncf_layers = hyperparams["ncf_layers"] if "ncf_layers" in hyperparams else 2
        batch_size = hyperparams["batch_size"] if "batch_size" in hyperparams else 512
        verbose = hyperparams["verbose"] if "verbose" in hyperparams else 1
        margin = hyperparams["margin"] if "margin" in hyperparams else 0.0
        kernel_l2 = hyperparams["kernel_l2"] if "kernel_l2" in hyperparams else 0.0
        conv_depth = hyperparams["conv_depth"] if "conv_depth" in hyperparams else 1
        gaussian_noise = hyperparams["gaussian_noise"] if "gaussian_noise" in hyperparams else 0.0
        ns_proportion = hyperparams["ns_proportion"] if "ns_proportion" in hyperparams else 1
        total_nodes = len(nodes)
        assert np.sum(np.isnan(content_vectors)) == 0
        import gc
        gc.collect()
        if epochs <= 0:
            return content_vectors

        edge_list = [(node_to_index[e.src], node_to_index[e.dst], e.weight) for e in edges]
        g_train = build_dgl_graph(edge_list, total_nodes, content_vectors)
        g_train.readonly()
        n_content_dims = content_vectors.shape[1]
        ncf = NCFEmbedding(self.n_dims, ncf_layers, gaussian_noise,
                           content_vectors)
        gcn = GraphSageWithSampling(n_content_dims, self.n_dims, gcn_layers, g_train,
                                    gaussian_noise, conv_depth)
        model = RecImplicitEmbedding(gcn=gcn, ncf=ncf)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=kernel_l2)
        generate_training_samples = self.__data_gen_fn__(nodes, edges, node_to_index,
                                                         hyperparams)

        src, dst, weights, ratings = generate_training_samples()
        model.train()
        positive_examples, negative_examples = torch.sum(ratings == 1).item(), torch.sum(ratings == 0).item()
        model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.log.info("Built KNN Network, model params = %s, examples = %s, positive = %s, negative = %s, model = \n%s",
                      params, len(src), positive_examples, negative_examples, model)
        gc.collect()
        for epoch in range(epochs):
            model.train()
            start = time.time()
            loss = 0.0
            def train(src, dst, weights, ratings):

                src_batches = src.split(batch_size)
                dst_batches = dst.split(batch_size)
                weights_batches = weights.split(batch_size)
                ratings_batches = ratings.split(batch_size)
                seed_nodes = torch.cat(sum([[s, d] for s, d in zip(src_batches, dst_batches)], []))
                sampler = dgl.contrib.sampling.NeighborSampler(
                    g_train,  # the graph
                    batch_size * 2,  # number of nodes to compute at a time, HACK 2
                    5,  # number of neighbors for each node
                    gcn_layers,  # number of layers in GCN
                    seed_nodes=seed_nodes,  # list of seed nodes, HACK 2
                    prefetch=True,  # whether to prefetch the NodeFlows
                    add_self_loop=True,  # whether to add a self-loop in the NodeFlows, HACK 1
                    shuffle=False,  # whether to shuffle the seed nodes.  Should be False here.
                    num_workers=self.cpu,
                )

                # Training
                total_loss = 0.0
                for s, d, nodeflow, w, r in zip(src_batches, dst_batches, sampler, weights_batches, ratings_batches):
                    h_src, h_dst = model.forward(nodeflow, s, d)
                    score = (h_src * h_dst).sum(1)
                    score = (score + 1)/2
                    loss = -1 * (r * torch.log(score + margin) + (1 - r) * torch.log(1 - score + margin))
                    loss = loss.mean()
                    total_loss = total_loss + loss.item()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                return total_loss / len(src_batches)

            loss += train(src, dst, weights, ratings)
            gen_time = time.time()
            if epoch < epochs - 1:
                src, dst, weights, ratings = generate_training_samples()
            gen_time = time.time() - gen_time

            total_time = time.time() - start
            self.log.info('Epoch %2d/%2d: ' % (int(epoch + 1),
                                               epochs) + ' Training loss: %.4f' % loss +
                          ' || Time Taken: %.1f' % total_time + " Generator time: %.1f" % gen_time)

            #
        model.eval()
        sampler = dgl.contrib.sampling.NeighborSampler(
            g_train,
            batch_size,
            5,
            gcn_layers,
            seed_nodes=torch.arange(g_train.number_of_nodes()),
            prefetch=True,
            add_self_loop=True,
            shuffle=False,
            num_workers=self.cpu
        )
        src_batches = torch.arange(g_train.number_of_nodes()).split(batch_size)
        with torch.no_grad():
            h = []
            for src, nf in zip(src_batches, sampler):
                h_src = model.gcn.forward(nf)
                h_src = model.ncf.forward(src, h_src)
                h.append(h_src)
        h = torch.cat(h).numpy()

        collaborative_node_vectors = h
        self.log.info(
            "End Training Collaborative Embeddings, Unit Length Violations:: nodes = %s",
            unit_length_violations(collaborative_node_vectors, axis=1))

        gc.collect()
        assert np.sum(np.isnan(collaborative_node_vectors)) == 0
        return collaborative_node_vectors

    def __build_prediction_network__(self, nodes: List[Node],
                                     edges: List[Edge],
                                     content_vectors: np.ndarray, collaborative_vectors: np.ndarray,
                                     nodes_to_idx: Dict[Node, int],
                                     hyperparams: Dict):
        from .gcn import build_dgl_graph, GraphSageWithSampling
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cpu = torch.device('cpu')
        import dgl
        import gc
        self.log.debug(
            "Start Building Prediction Network, collaborative vectors shape = %s, content vectors shape = %s",
            collaborative_vectors.shape, content_vectors.shape)

        lr = hyperparams["lr"] if "lr" in hyperparams else 0.001
        epochs = hyperparams["epochs"] if "epochs" in hyperparams else 15
        batch_size = hyperparams["batch_size"] if "batch_size" in hyperparams else 512
        verbose = hyperparams["verbose"] if "verbose" in hyperparams else 2
        use_content = hyperparams["use_content"] if "use_content" in hyperparams else False
        kernel_l2 = hyperparams["kernel_l2"] if "kernel_l2" in hyperparams else 0.0
        gcn_layers = hyperparams["gcn_layers"] if "gcn_layers" in hyperparams else 3
        ncf_layers = hyperparams["ncf_layers"] if "ncf_layers" in hyperparams else 2
        conv_depth = hyperparams["conv_depth"] if "conv_depth" in hyperparams else 1
        ns_proportion = hyperparams["ns_proportion"] if "ns_proportion" in hyperparams else 1
        gaussian_noise = hyperparams["gaussian_noise"] if "gaussian_noise" in hyperparams else 0.0
        margin = hyperparams["margin"] if "margin" in hyperparams else 0.0
        nsh = hyperparams["nsh"] if "nsh" in hyperparams else 1.0
        ps_proportion = hyperparams["ps_proportion"] if "ps_proportion" in hyperparams else 1
        ncf_gcn_balance = hyperparams["ncf_gcn_balance"] if "ncf_gcn_balance" in hyperparams else 1.0

        # For unseen users and items creating 2 mock nodes
        content_vectors = np.concatenate((np.zeros((1, content_vectors.shape[1])), content_vectors))
        gc.collect()
        assert np.sum(np.isnan(content_vectors)) == 0

        total_nodes = len(nodes) + 1
        if not use_content:
            content_vectors = np.zeros((content_vectors.shape[0], 1))

        import gc
        gc.collect()
        edge_list = [(nodes_to_idx[e.src] + 1, nodes_to_idx[e.dst] + 1, e.weight) for e in edges]
        edge_list.extend([(i, i, 1) for i in range(total_nodes)])
        g_train = build_dgl_graph(edge_list, total_nodes, content_vectors)
        n_content_dims = content_vectors.shape[1]
        g_train.readonly()
        ncf = NCF(self.n_dims, ncf_layers, gaussian_noise,
                  content_vectors, ncf_gcn_balance)
        gcn = GraphSageWithSampling(n_content_dims, self.n_dims, gcn_layers, g_train,
                                    gaussian_noise, conv_depth)
        model = RecImplicit(gcn=gcn, ncf=ncf)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=kernel_l2)

        generate_training_samples = self.__data_gen_fn__(nodes, edges, self.nodes_to_idx,
                                                         hyperparams)

        def get_samples():
            src, dst, weights, ratings = generate_training_samples()
            src = src + 1
            dst = dst + 1
            return src, dst, weights, ratings

        src, dst, weights, ratings = get_samples()
        # total_examples = len(src) - 10_000
        # src, dst, rating = src[:total_examples], dst[:total_examples], rating[:total_examples]
        # opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=kernel_l2, momentum=0.9, nesterov=True)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=epochs,
        #                                                 steps_per_epoch=int(
        #                                                     np.ceil(total_examples / batch_size)),
        #                                                 div_factor=50, final_div_factor=100)

        model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        params = sum([np.prod(p.size()) for p in model_parameters])
        positive_examples, negative_examples = torch.sum(ratings == 1).item(), torch.sum(ratings == 0).item()
        self.log.info("Built Prediction Network, model params = %s, examples = %s, positive = %s, negative = %s, model = \n%s",
                      params, len(src), positive_examples, negative_examples, model)

        for epoch in range(epochs):
            gc.collect()
            start = time.time()
            model.train()

            def train(src, dst, weights, rating):
                import gc
                gc.collect()

                src_batches = src.split(batch_size)
                dst_batches = dst.split(batch_size)
                rating_batches = rating.split(batch_size)
                weights_batches = weights.split(batch_size)

                seed_nodes = torch.cat(sum([[s, d] for s, d in zip(src_batches, dst_batches)], []))

                sampler = dgl.contrib.sampling.NeighborSampler(
                    g_train,  # the graph
                    batch_size * 2,  # number of nodes to compute at a time, HACK 2
                    5,  # number of neighbors for each node
                    gcn_layers,  # number of layers in GCN
                    seed_nodes=seed_nodes,  # list of seed nodes, HACK 2
                    prefetch=True,  # whether to prefetch the NodeFlows
                    add_self_loop=True,  # whether to add a self-loop in the NodeFlows, HACK 1
                    shuffle=False,  # whether to shuffle the seed nodes.  Should be False here.
                    num_workers=self.cpu,
                )

                # Training
                total_loss = 0.0
                for s, d, r, w, nf in zip(src_batches, dst_batches, rating_batches, weights_batches, sampler):
                    score = model(nf, s, d)
                    # loss = ((score - r) ** 2)
                    loss = -1 * (r * torch.log(score + margin) + (1-r)*torch.log(1 - score + margin))
                    loss = loss.mean()
                    total_loss = total_loss + loss.item()

                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    # scheduler.step()
                return total_loss / len(src_batches)

            loss = train(src, dst, weights, ratings)
            if epoch < epochs - 1:
                src, dst, weights, ratings = get_samples()
                # src, dst, rating = src[:total_examples], dst[:total_examples], rating[:total_examples]

            total_time = time.time() - start

            self.log.info('Epoch %2d/%2d: ' % (int(epoch + 1),
                                               epochs) + ' Training loss: %.4f' % loss + '|| Time Taken: %.1f' % total_time)

        gc.collect()
        model.eval()
        sampler = dgl.contrib.sampling.NeighborSampler(
            g_train,
            batch_size,
            5,
            gcn_layers,
            seed_nodes=torch.arange(g_train.number_of_nodes()),
            prefetch=True,
            add_self_loop=True,
            shuffle=False,
            num_workers=self.cpu
        )

        with torch.no_grad():
            h = []
            for nf in sampler:
                h.append(model.gcn.forward(nf))
            h = torch.cat(h)

        prediction_artifacts = {"model": model.ncf,
                                "h": h,
                                "total_nodes": total_nodes}
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.log.info("Built Prediction Network, model params = %s", params)
        gc.collect()
        return prediction_artifacts

    def predict(self, node_pairs: List[Tuple[Node, Node]]) -> List[float]:
        model = self.prediction_artifacts["model"]
        h = self.prediction_artifacts["h"]
        total_nodes = self.prediction_artifacts["total_nodes"]
        batch_size = 512

        uip = [(self.nodes_to_idx[u] + 1 if u in self.nodes_to_idx else 0,
                self.nodes_to_idx[i] + 1 if i in self.nodes_to_idx else 0) for u, i in node_pairs]

        assert np.sum(np.isnan(uip)) == 0

        src, dst = zip(*uip)

        src = torch.tensor(src)
        dst = torch.tensor(dst)

        predictions = []
        with torch.no_grad():
            src = src.split(batch_size)
            dst = dst.split(batch_size)

            for u, i in zip(src, dst):
                g_src = h[u]
                g_dst = h[i]
                scores = model.forward(u, i, g_src, g_dst)
                scores = list(scores.numpy())
                predictions.extend(scores)
        return predictions

