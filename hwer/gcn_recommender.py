import time
from typing import List, Dict, Tuple, Optional, Set

import numpy as np

from .logging import getLogger
from .random_walk import *
from .hybrid_recommender import HybridRecommender
from .utils import unit_length_violations
import logging
import dill
import sys
from .recommendation_base import RecommendationBase, NodeType, Node, Edge, FeatureName
from .embed import BaseEmbed
logger = getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))


class GCNRecommender(HybridRecommender):
    def __init__(self, embedding_mapper: Dict[NodeType, Dict[str, BaseEmbed]], node_types: Set[str],
                 n_dims: int = 32):
        super().__init__(embedding_mapper, node_types, n_dims)
        self.log = getLogger(type(self).__name__)
        assert n_dims % 2 == 0
        self.cpu = int(os.cpu_count() / 2)

    def __get_triplet_gcn_model__(self, n_content_dims, n_collaborative_dims, gcn_layers,
                                  conv_depth, g_train, triplet_vectors, margin, gaussian_noise):
        from .gcn import GraphSAGETripletEmbedding, GraphSageWithSampling, GraphSAGENegativeSamplingEmbedding, \
            GraphResnetWithSampling, GraphSAGELogisticEmbedding, GraphSAGELogisticEmbeddingv2
        self.log.info("Getting Triplet Model for GCN")
        model = GraphSAGELogisticEmbedding(GraphSageWithSampling(n_content_dims, n_collaborative_dims,
                                                                gcn_layers, g_train,
                                                                gaussian_noise, conv_depth, triplet_vectors))
        return model

    def __data_gen_fn__(self, nodes: List[Node],
                        edges: List[Edge], node_to_index: Dict[Node, int],
                        hyperparams):
        affinities = [(node_to_index[e.src], node_to_index[e.dst], e.weight) for e in edges]

        def affinities_generator():
            np.random.shuffle(affinities)
            for i, j, r in affinities:
                yield (i, j), r

        return affinities_generator

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
        layers = hyperparams["layers"] if "layers" in hyperparams else 3
        batch_size = hyperparams["batch_size"] if "batch_size" in hyperparams else 512
        verbose = hyperparams["verbose"] if "verbose" in hyperparams else 1
        margin = hyperparams["margin"] if "margin" in hyperparams else 1.0
        kernel_l2 = hyperparams["kernel_l2"] if "kernel_l2" in hyperparams else 0.0
        enable_gcn = hyperparams["enable_gcn"] if "enable_gcn" in hyperparams else False
        ns_proportion = hyperparams["ns_proportion"] if "ns_proportion" in hyperparams else 2
        conv_depth = hyperparams["conv_depth"] if "conv_depth" in hyperparams else 1
        gaussian_noise = hyperparams["gaussian_noise"] if "gaussian_noise" in hyperparams else 0.0
        total_nodes = len(nodes)
        assert np.sum(np.isnan(content_vectors)) == 0
        import gc
        gc.collect()
        if not enable_gcn or epochs <= 0:
            return content_vectors

        edge_list = [(node_to_index[e.src], node_to_index[e.dst], e.weight) for e in edges]

        g_train = build_dgl_graph(edge_list, total_nodes, content_vectors)
        g_train.readonly()
        n_content_dims = content_vectors.shape[1]
        model = self.__get_triplet_gcn_model__(n_content_dims, self.n_dims, layers,
                                               conv_depth, g_train, None, margin,
                                               gaussian_noise)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=kernel_l2)
        generate_training_samples = self.__data_gen_fn__(nodes, edges, node_to_index,
                                                         hyperparams)

        def get_samples():
            src, dst, weights, error_weights = [], [], [], []
            for (u, v), r in generate_training_samples():
                src.append(u)
                dst.append(v)
                weights.append(1.0)
                error_weights.append(r)

            src = torch.LongTensor(src)
            dst = torch.LongTensor(dst)
            weights = torch.FloatTensor(weights)

            ns = int(ns_proportion * len(src))
            src_neg = torch.randint(0, total_nodes, (ns,))
            dst_neg = torch.randint(0, total_nodes, (ns,))
            weights_neg = torch.tensor([0.0] * ns)
            src = torch.cat((src, src_neg), 0)
            dst = torch.cat((dst, dst_neg), 0)
            weights = torch.cat((weights, weights_neg), 0)

            shuffle_idx = torch.randperm(len(src))
            src = src[shuffle_idx]
            dst = dst[shuffle_idx]
            weights = weights[shuffle_idx]
            weights = weights.clamp(min=1e-7, max=1-1e-7)
            return src, dst, weights

        src, dst, weights = get_samples()
        model.train()
        model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.log.info("Built KNN Network, model params = %s, examples = %s, model = \n%s", params, len(src), model)
        gc.collect()
        for epoch in range(epochs):
            start = time.time()
            loss = 0.0
            def train(src, dst):

                src_batches = src.split(batch_size)
                dst_batches = dst.split(batch_size)
                weights_batches = weights.split(batch_size)
                model.train()
                seed_nodes = torch.cat(sum([[s, d] for s, d in zip(src_batches, dst_batches)], []))
                sampler = dgl.contrib.sampling.NeighborSampler(
                    g_train,  # the graph
                    batch_size * 2,  # number of nodes to compute at a time, HACK 2
                    5,  # number of neighbors for each node
                    layers,  # number of layers in GCN
                    seed_nodes=seed_nodes,  # list of seed nodes, HACK 2
                    prefetch=True,  # whether to prefetch the NodeFlows
                    add_self_loop=True,  # whether to add a self-loop in the NodeFlows, HACK 1
                    shuffle=False,  # whether to shuffle the seed nodes.  Should be False here.
                    num_workers=self.cpu,
                )

                # Training
                total_loss = 0.0
                for s, d, nodeflow, ws in zip(src_batches, dst_batches, sampler, weights_batches):
                    score = model.forward(nodeflow, s, d, ws)
                    loss = score.sum()
                    total_loss = total_loss + loss

                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                return total_loss / len(src_batches)

            loss += train(src, dst)
            gen_time = time.time()
            if epoch < epochs - 1:
                src, dst, weights = get_samples()
            gen_time = time.time() - gen_time

            total_time = time.time() - start
            self.log.info('Epoch %2d/%2d: ' % (int(epoch + 1),
                                               epochs) + ' Training loss: %.4f' % loss.item() +
                          ' || Time Taken: %.1f' % total_time + " Generator time: %.1f" % gen_time)

            #
        model.eval()
        sampler = dgl.contrib.sampling.NeighborSampler(
            g_train,
            batch_size,
            5,
            layers,
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

        pass
