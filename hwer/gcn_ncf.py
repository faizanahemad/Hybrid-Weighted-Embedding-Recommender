import time
from typing import List, Dict, Tuple, Optional, Set

import numpy as np
from more_itertools import flatten

from .logging import getLogger
from .random_walk import *
from .utils import unit_length_violations
from .utils import NodeNotFoundException
import operator
import logging
import dill
import sys
logger = getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

from .recommendation_base import RecommendationBase, NodeType, Node, Edge, FeatureName
from .utils import unit_length, unit_length_violations
from .embed import BaseEmbed
from .gcn import *
from .ncf import *
from .content_recommender import ContentRecommendation


class GcnNCF(RecommendationBase):
    def __init__(self, embedding_mapper: Dict[NodeType, Dict[str, BaseEmbed]], node_types: Set[str],
                 n_dims: int = 32):
        super().__init__(node_types, n_dims)
        self.log = getLogger(type(self).__name__)
        assert n_dims % 2 == 0
        self.cpu = int(os.cpu_count() / 2)
        self.cb = ContentRecommendation(embedding_mapper, node_types, np.inf)
        self.content_data_used = None
        self.prediction_artifacts = dict()
        self.ncf_enabled = False

    def __word2vec_neg_sampler(self, nodes: List[Node],
                        edges: List[Edge], hyperparams):

        ns_w2v_proportion = hyperparams["ns_w2v_proportion"] if "ns_w2v_proportion" in hyperparams else 0
        ns_w2v_exponent = hyperparams["ns_w2v_exponent"] if "ns_w2v_exponent" in hyperparams else 3.0 / 4.0
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
        ns_proportion = hyperparams["ns_proportion"] if "ns_proportion" in hyperparams else 1
        ns_w2v_proportion = hyperparams["ns_w2v_proportion"] if "ns_w2v_proportion" in hyperparams else 0

        affinities = [(node_to_index[e.src], node_to_index[e.dst], e.weight) for e in edges]
        total_nodes = len(nodes)
        w2v_neg_gen = self.__word2vec_neg_sampler(nodes, edges, hyperparams)
        simple_neg_gen = self.__simple_neg_sampler__(nodes, edges, hyperparams)

        def get_samples():
            np.random.shuffle(affinities)
            src, dst, weights = zip(*affinities)
            ratings = [1.0] * len(src)

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

            shuffle_idx = torch.randperm(len(src))
            src = src[shuffle_idx]
            dst = dst[shuffle_idx]
            weights = weights[shuffle_idx]
            ratings = ratings[shuffle_idx]
            return src, dst, weights, ratings
        return get_samples

    def __train__(self, model, g_train, data_generator, hyperparams, loss_fn):
        lr = hyperparams["lr"] if "lr" in hyperparams else 0.001
        epochs = hyperparams["epochs"] if "epochs" in hyperparams else 15
        batch_size = hyperparams["batch_size"] if "batch_size" in hyperparams else 512
        kernel_l2 = hyperparams["kernel_l2"] if "kernel_l2" in hyperparams else 0.0
        gcn_layers = hyperparams["gcn_layers"] if "gcn_layers" in hyperparams else 2
        src, dst, weights, ratings = data_generator()
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=kernel_l2)
        import gc
        gc.collect()
        positive_examples, negative_examples = torch.sum(ratings == 1).item(), torch.sum(ratings == 0).item()
        model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.log.info("Start Model training...,\nmodel params = %s, examples = %s, positive = %s, negative = %s, \nModel = \n%s",
                      params, len(src), positive_examples, negative_examples, model)
        gc.collect()
        model.train()

        def train_one_epoch(src, dst, weights, ratings):
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

            total_loss = 0.0
            for s, d, nodeflow, w, r in zip(src_batches, dst_batches, sampler, weights_batches, ratings_batches):
                loss, _, _ = loss_fn(model, s, d, nodeflow, w, r)
                total_loss = total_loss + loss.item()
                opt.zero_grad()
                loss.backward()
                opt.step()
            return total_loss / len(src_batches)

        for epoch in range(epochs):
            start = time.time()
            loss = train_one_epoch(src, dst, weights, ratings)
            gen_time = time.time()
            if epoch < epochs - 1:
                src, dst, weights, ratings = data_generator()
            gen_time = time.time() - gen_time
            total_time = time.time() - start
            self.log.info('Epoch %2d/%2d: ' % (int(epoch + 1),
                                               epochs) + ' Training loss: %.4f' % loss +
                          ' || Time Taken: %.1f' % total_time + " Generator time: %.1f" % gen_time)

    def __build_prediction_network__(self, nodes: List[Node],
                                     edges: List[Edge],
                                     content_vectors: np.ndarray,
                                     nodes_to_idx: Dict[Node, int],
                                     hyperparams: Dict):
        from .gcn import build_dgl_graph, GraphConvModule
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cpu = torch.device(device)
        import dgl
        import gc
        self.log.debug(
            "Start Building Prediction Network, content vectors shape = %s", content_vectors.shape)

        lr = hyperparams["lr"] if "lr" in hyperparams else 0.001
        epochs = hyperparams["epochs"] if "epochs" in hyperparams else 15
        batch_size = hyperparams["batch_size"] if "batch_size" in hyperparams else 512
        verbose = hyperparams["verbose"] if "verbose" in hyperparams else 2
        kernel_l2 = hyperparams["kernel_l2"] if "kernel_l2" in hyperparams else 0.0
        gcn_layers = hyperparams["gcn_layers"] if "gcn_layers" in hyperparams else 3
        ncf_layers = hyperparams["ncf_layers"] if "ncf_layers" in hyperparams else 2
        ns_proportion = hyperparams["ns_proportion"] if "ns_proportion" in hyperparams else 1
        gaussian_noise = hyperparams["gaussian_noise"] if "gaussian_noise" in hyperparams else 0.0
        ncf_epochs = hyperparams["ncf_epochs"] if "ncf_epochs" in hyperparams else 0
        gcn_epochs = hyperparams["gcn_epochs"] if "gcn_epochs" in hyperparams else 1

        # For unseen users and items creating 2 mock nodes
        content_vectors = np.concatenate((np.zeros((1, content_vectors.shape[1])), content_vectors))
        gc.collect()
        assert np.sum(np.isnan(content_vectors)) == 0

        total_nodes = len(nodes) + 1

        import gc
        gc.collect()
        edge_list = [(nodes_to_idx[e.src] + 1, nodes_to_idx[e.dst] + 1, e.weight) for e in edges]
        edge_list.extend([(i, i, 1) for i in range(total_nodes)])
        n_content_dims = content_vectors.shape[1]
        g_train = build_dgl_graph(edge_list, total_nodes, content_vectors)
        g_train.readonly()
        if ncf_epochs == 0:
            ncf = None
        else:
            ncf = NCF(self.n_dims, ncf_layers, gaussian_noise)
        gcn = GraphConvModule(n_content_dims, self.n_dims, gcn_layers, g_train, gaussian_noise)
        model = RecImplicit(gcn=gcn, ncf=ncf)
        generate_training_samples = self.__data_gen_fn__(nodes, edges, self.nodes_to_idx,
                                                         hyperparams)

        def get_samples():
            src, dst, weights, ratings = generate_training_samples()
            src = src + 1
            dst = dst + 1
            return src, dst, weights, ratings

        eps = 5e-7

        def loss_fn_gcn(model, src, dst, nodeflow, weights, ratings):
            h_output = model.gcn(nodeflow)
            h_src = h_output[nodeflow.map_from_parent_nid(-1, src, True)]
            h_dst = h_output[nodeflow.map_from_parent_nid(-1, dst, True)]
            gcn_score = (h_src * h_dst).sum(1)
            gcn_score = ((gcn_score + 1.0)/2.0).clamp(min=eps, max=1-eps)
            gcn_loss = -1 * (ratings * torch.log(gcn_score + eps) + (1 - ratings) * torch.log(1 - (gcn_score - eps)))
            gcn_loss = (gcn_loss * weights).mean()
            gcn_loss = gcn_loss
            return gcn_loss, h_src, h_dst

        def loss_fn_ncf(model, src, dst, nodeflow, weights, ratings):
            gcn_loss, h_src, h_dst = loss_fn_gcn(model, src, dst, nodeflow, weights, ratings)
            score = model.ncf(src, dst, h_src, h_dst)
            loss = -1 * (ratings * torch.log(score) + (1 - ratings) * torch.log(1 - score))
            loss = loss * weights
            loss = loss.mean()
            return loss, h_src, h_dst

        def get_gcn_vectors():
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
            return h

        import copy
        hp_gcn = copy.deepcopy(hyperparams)
        hp_ncf = copy.deepcopy(hyperparams)
        hp_gcn["epochs"] = gcn_epochs
        hp_ncf["epochs"] = ncf_epochs
        self.__train__(model, g_train, get_samples, hp_gcn, loss_fn_gcn)
        knn_vectors = get_gcn_vectors()
        if ncf_epochs > 0:
            self.__train__(model, g_train, get_samples, hp_ncf, loss_fn_ncf)
        gc.collect()

        prediction_artifacts = {"model": model.ncf,
                                "h": get_gcn_vectors(),
                                "knn_vectors": knn_vectors}
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.log.info("Built Prediction Network, model params = %s", params)
        gc.collect()
        return prediction_artifacts

    def predict(self, node_pairs: List[Tuple[Node, Node]]) -> List[float]:
        if not self.ncf_enabled:
            src, dst = zip(*node_pairs)
            results = (self.get_embeddings(src) * self.get_embeddings(dst)).sum(1)
            results = (results + 1) / 2
            return results

        model = self.prediction_artifacts["model"]
        h = self.prediction_artifacts["h"]
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

    def find_closest_neighbours(self, node_type: str, anchor: Node, positive: List[Node] = None,
                                negative: List[Node] = None, k=200) -> List[Tuple[Node, float]]:
        assert self.fit_done
        assert node_type in self.node_types and node_type in self.knn.knn
        if anchor not in self.nodes_to_idx:
            raise NodeNotFoundException("Node = %s, was not provided in training" % anchor)

        embedding_list = [self.get_average_embeddings([anchor])]
        if positive is not None and len(positive) > 0:
            embedding_list.append(self.get_average_embeddings(positive))
        if negative is not None and len(negative) > 0:
            embedding_list.append(-1 * self.get_average_embeddings(negative))

        embedding = np.average(embedding_list, axis=0)
        node_dist_list = self.knn.query(embedding, node_type, k=k)
        if not self.ncf_enabled:
            nodes, dist = zip(*node_dist_list)
            dist = np.array(dist)
            dist = (-1 * dist + 2)/2
            node_dist_list = zip(nodes, dist)
            results = list(sorted(node_dist_list, key=operator.itemgetter(1), reverse=True))
        else:
            scores = self.predict([(anchor, node) for node, dist in node_dist_list])
            results = list(sorted(zip([n for n, d in node_dist_list], scores), key=operator.itemgetter(1), reverse=True))
        return results

    def fit(self,
            nodes: List[Node],
            edges: List[Edge],
            node_data: Dict[Node, Dict[FeatureName, object]],
            **kwargs):
        start_time = time.time()
        _ = super().fit(nodes, edges, node_data, **kwargs)
        self.log.debug("Hybrid Base: Fit Method Started")
        hyperparameters = {} if "hyperparameters" not in kwargs else kwargs["hyperparameters"]
        gcn_ncf_params = {} if "gcn_ncf_params" not in hyperparameters else \
            hyperparameters["gcn_ncf_params"]
        ncf_epochs = gcn_ncf_params["ncf_epochs"] if "ncf_epochs" in gcn_ncf_params else 0
        gcn_epochs = gcn_ncf_params["gcn_epochs"] if "gcn_epochs" in gcn_ncf_params else 1
        assert gcn_epochs >= 1

        use_content = hyperparameters["use_content"] if "use_content" in hyperparameters else False
        content_data_used = len(node_data) != 0 and use_content
        self.content_data_used = content_data_used

        self.log.debug("Hybrid Base: Fit Method: content_data_used = %s", content_data_used)
        start = time.time()
        if content_data_used:
            super(type(self.cb), self.cb).fit(nodes, edges, node_data, **kwargs)
            content_vectors = self.cb.__build_content_embeddings__(nodes, edges, node_data, np.inf)
            self.cb = None
            del self.cb

        else:
            content_vectors = np.random.rand(len(nodes), 1)
        self.log.info("Hybrid Base: Built Content Embedding., shape = %s, Time = %.1f" %
                       (content_vectors.shape, time.time() - start))
        import gc
        gc.collect()
        prediction_artifacts = self.__build_prediction_network__(nodes, edges,
                                                                 content_vectors,
                                                                 self.nodes_to_idx, gcn_ncf_params)

        if prediction_artifacts is not None:
            self.prediction_artifacts.update(dict(prediction_artifacts))
        gc.collect()
        self.log.debug("Hybrid Base: Built Prediction Network.")

        collaborative_vectors = self.prediction_artifacts["knn_vectors"][1:].numpy()
        self.prediction_artifacts["knn_vectors"] = None
        del self.prediction_artifacts["knn_vectors"]
        if ncf_epochs == 0:
            self.prediction_artifacts = None
            del self.prediction_artifacts
        self.ncf_enabled = ncf_epochs > 0

        knn_vectors = self.prepare_for_knn(content_vectors, collaborative_vectors)
        self.__build_knn__(knn_vectors)
        self.fit_done = True
        self.log.info("End Fitting Recommender, vectors shape = %s, Time to fit = %.1f",
                      self.vectors.shape, time.time() - start_time)
        gc.collect()
        return self.vectors

    def prepare_for_knn(self, content_vectors: np.ndarray, collaborative_vectors: np.ndarray) -> np.ndarray:
        from .utils import unit_length
        from sklearn.decomposition import PCA
        if collaborative_vectors.shape[1] > self.n_dims:
            pca = PCA(n_components=self.n_dims)
            collaborative_vectors = pca.fit_transform(collaborative_vectors)
        elif collaborative_vectors.shape[1] < self.n_dims:
            raise ValueError()
        collaborative_vectors = unit_length(collaborative_vectors, axis=1)
        return collaborative_vectors

