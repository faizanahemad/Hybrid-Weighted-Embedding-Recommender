import time
from collections import Counter
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from bidict import bidict
from more_itertools import flatten
from sklearn.model_selection import StratifiedKFold
from surprise import Dataset
from surprise import Reader
from surprise import SVDpp
from tensorflow import keras
from tensorflow.keras import layers
from .random_walk import *
import os

import networkx as nx
import tensorflow as tf
from dgl import DGLGraph
import dgl.function as fn
import dgl
from dgl.data import register_data_args, load_data


from .svdpp_hybrid import SVDppHybrid
from .logging import getLogger
from .recommendation_base import EntityType
from .utils import RatingPredRegularization, get_rng, \
    LRSchedule, resnet_layer_with_content, ScaledGlorotNormal, root_mean_squared_error, mean_absolute_error, \
    normalize_affinity_scores_by_user_item_bs, get_clipped_rmse, unit_length_violations, UnitLengthRegularization, \
    unit_length

from .gcn import *


class HybridGCNRec(SVDppHybrid):
    def __init__(self, embedding_mapper: dict, knn_params: Optional[dict], rating_scale: Tuple[float, float],
                 n_content_dims: int = 32, n_collaborative_dims: int = 32, fast_inference: bool = False,
                 super_fast_inference: bool = False):
        super().__init__(embedding_mapper, knn_params, rating_scale, n_content_dims, n_collaborative_dims, fast_inference)
        self.log = getLogger(type(self).__name__)
        self.super_fast_inference = super_fast_inference
        self.cpu = int(os.cpu_count()/2)

    def user_item_affinities_triplet_trainer_data_gen_fn__(self,
                                                 user_ids: List[str], item_ids: List[str],
                                                 user_id_to_index: Dict[str, int], item_id_to_index: Dict[str, int],
                                                 hyperparams: Dict):
        batch_size = hyperparams["batch_size"] if "batch_size" in hyperparams else 512
        total_users = len(user_ids)
        total_items = len(item_ids)

        def generate_training_samples(affinities: List[Tuple[str, str, float]]):
            affinities = [(user_id_to_index[i], total_users + item_id_to_index[j], r) for i, j, r in affinities]
            adjacency_set = defaultdict(set)

            graph = Graph().read_edgelist(affinities)
            walker = Walker(graph, p=4, q=2)
            walker.preprocess_transition_probs()

            for u, i, r in affinities:
                adjacency_set[u].add(i)
                adjacency_set[i].add(u)

            def get_negative_example(i, j):
                random_items = np.random.randint(0, total_users + total_items, 10)
                random_items = set(random_items) - adjacency_set[i]
                random_items = random_items - adjacency_set[j]
                random_items = np.array(list(random_items))
                distant_item = np.random.randint(0, total_users + total_items)
                distant_item = random_items[0] if len(random_items) > 0 else distant_item
                return distant_item

            def get_one_example(i, j):
                user = i
                second_item = j
                distant_item = get_negative_example(i, j)
                return (user, second_item, distant_item), 0

            def generator():
                for walk in walker.simulate_walks_generator(5, walk_length=3):
                    nodes = list(set(walk))
                    combinations = [(x, y) for i, x in enumerate(nodes) for y in nodes[i + 1:]]
                    generated = [get_one_example(u, v) for u, v in combinations]
                    for g in generated:
                        yield g

            def generator():
                for i in range(0, len(affinities), batch_size):
                    start = i
                    end = min(i + batch_size, len(affinities))
                    combinations = []
                    for u, v, w in affinities[start:end]:
                        w1 = walker.node2vec_walk(3, u)
                        nxt = w1[1]
                        c = [(x, y) for i, x in enumerate(w1) for y in w1[i + 1:]]
                        c = c + [(u, v)] + [(v, nxt)] if v != nxt else []
                        c = list(set(c))
                        combinations.extend(c)

                    generated = [get_one_example(u, v) for u, v in combinations]
                    for g in generated:
                        yield g

            return generator
        return generate_training_samples

    def __user_item_affinities_triplet_trainer__(self,
                                         user_ids: List[str], item_ids: List[str],
                                         user_item_affinities: List[Tuple[str, str, float]],
                                         user_vectors: np.ndarray, item_vectors: np.ndarray,
                                         user_id_to_index: Dict[str, int], item_id_to_index: Dict[str, int],
                                         n_output_dims: int,
                                         hyperparams: Dict) -> Tuple[np.ndarray, np.ndarray]:
        self.log.debug("Start Training User-Item Affinities, n_users = %s, n_items = %s, n_samples = %s, in_dims = %s, out_dims = %s",
                       len(user_ids), len(item_ids), len(user_item_affinities), user_vectors.shape[1], n_output_dims)

        lr = hyperparams["lr"] if "lr" in hyperparams else 0.001
        gcn_lr = hyperparams["gcn_lr"] if "gcn_lr" in hyperparams else 0.1
        epochs = hyperparams["epochs"] if "epochs" in hyperparams else 15
        gcn_epochs = hyperparams["gcn_epochs"] if "gcn_epochs" in hyperparams else 5
        gcn_layers = hyperparams["gcn_layers"] if "gcn_layers" in hyperparams else 5
        gcn_dropout = hyperparams["gcn_dropout"] if "gcn_dropout" in hyperparams else 0.0
        batch_size = hyperparams["batch_size"] if "batch_size" in hyperparams else 512
        gcn_batch_size = hyperparams["gcn_batch_size"] if "gcn_batch_size" in hyperparams else 512
        verbose = hyperparams["verbose"] if "verbose" in hyperparams else 1
        margin = hyperparams["margin"] if "margin" in hyperparams else 0.5
        gcn_kernel_l2 = hyperparams["gcn_kernel_l2"] if "gcn_kernel_l2" in hyperparams else 0.0

        assert np.sum(np.isnan(user_vectors)) == 0
        assert np.sum(np.isnan(item_vectors)) == 0

        user_triplet_vectors, item_triplet_vectors = super().__user_item_affinities_triplet_trainer__(user_ids, item_ids, user_item_affinities,
                              user_vectors, item_vectors,
                              user_id_to_index,
                              item_id_to_index,
                              n_output_dims,
                              hyperparams)
        total_users = len(user_ids)
        total_items = len(item_ids)

        triplet_vectors = np.concatenate((np.zeros((1, user_triplet_vectors.shape[1])), user_triplet_vectors, item_triplet_vectors))
        triplet_vectors = torch.FloatTensor(triplet_vectors)

        edge_list = [(user_id_to_index[u], total_users + item_id_to_index[i], r) for u, i, r in user_item_affinities]

        g_train = build_dgl_graph(edge_list, len(user_ids) + len(item_ids), np.concatenate((user_vectors, item_vectors)))
        g_train.readonly()
        n_content_dims = user_vectors.shape[1]
        model = GraphSAGETripletEmbedding(GraphSageWithSampling(n_content_dims, self.n_collaborative_dims,
                                                                gcn_layers, gcn_dropout, g_train, triplet_vectors), margin)
        opt = torch.optim.Adam(model.parameters(), lr=gcn_lr, weight_decay=gcn_kernel_l2)
        generate_training_samples = self.__user_item_affinities_triplet_trainer_data_gen_fn__(user_ids, item_ids,
                                                                                              user_id_to_index,
                                                                                              item_id_to_index,
                                                                                              hyperparams)
        generator = generate_training_samples(user_item_affinities)
        model.train()
        for epoch in range(gcn_epochs):
            start = time.time()
            start_gen = time.time()
            src, dst, neg = [], [], []
            for (u, v, w), r in generator():
                src.append(u)
                dst.append(v)
                neg.append(w)
            #
            total_gen = time.time() - start_gen
            src = torch.LongTensor(src)
            dst = torch.LongTensor(dst)
            neg = torch.LongTensor(neg)

            def train(src, dst, neg):

                shuffle_idx = torch.randperm(len(src))
                src_shuffled = src[shuffle_idx]
                dst_shuffled = dst[shuffle_idx]
                neg_shuffled = neg[shuffle_idx]

                src_batches = src_shuffled.split(gcn_batch_size)
                dst_batches = dst_shuffled.split(gcn_batch_size)
                neg_batches = neg_shuffled.split(gcn_batch_size)

                seed_nodes = torch.cat(sum([[s, d, n] for s, d, n in zip(src_batches, dst_batches, neg_batches)], []))

                sampler = dgl.contrib.sampling.NeighborSampler(
                    g_train,  # the graph
                    gcn_batch_size * 2,  # number of nodes to compute at a time, HACK 2
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
                for s, d, n, nodeflow in zip(src_batches, dst_batches, neg_batches, sampler):
                    score = model.forward(nodeflow, s, d, n)
                    loss = (score ** 2).mean()
                    total_loss = total_loss + loss

                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                return total_loss/len(src_batches)

            if epoch % 2 == 1:
                loss = train(src, dst, neg)
            else:
                # Reverse Training
                loss = train(dst, src, neg)

            total_time = time.time() - start
            self.log.info('Epoch %2d/%2d: ' % (int(epoch + 1), gcn_epochs) + ' Training loss: %.4f' % loss.item() + ' Generator Time: %.1f' % total_gen + ' Time Taken: %.1f' % total_time)

            #
        model.eval()
        sampler = dgl.contrib.sampling.NeighborSampler(
            g_train,
            gcn_batch_size,
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
            h = torch.cat(h).numpy()

        user_vectors, item_vectors = h[:total_users], h[total_users:]
        self.log.info(
            "End Training User-Item Affinities, Unit Length Violations:: user = %s, item = %s, margin = %.4f",
            unit_length_violations(user_vectors, axis=1), unit_length_violations(item_vectors, axis=1), margin)
        return user_vectors, item_vectors

    def __build_prediction_network__(self, user_ids: List[str], item_ids: List[str],
                                     user_item_affinities: List[Tuple[str, str, float]],
                                     user_content_vectors: np.ndarray, item_content_vectors: np.ndarray,
                                     user_vectors: np.ndarray, item_vectors: np.ndarray,
                                     user_id_to_index: Dict[str, int], item_id_to_index: Dict[str, int],
                                     rating_scale: Tuple[float, float], hyperparams: Dict):
        self.log.debug(
            "Start Building Prediction Network, collaborative vectors shape = %s, content vectors shape = %s",
            (user_vectors.shape, item_vectors.shape), (user_content_vectors.shape, item_content_vectors.shape))

        lr = hyperparams["lr"] if "lr" in hyperparams else 0.001
        epochs = hyperparams["epochs"] if "epochs" in hyperparams else 15
        batch_size = hyperparams["batch_size"] if "batch_size" in hyperparams else 512
        verbose = hyperparams["verbose"] if "verbose" in hyperparams else 1
        bias_regularizer = hyperparams["bias_regularizer"] if "bias_regularizer" in hyperparams else 0.0
        padding_length = hyperparams["padding_length"] if "padding_length" in hyperparams else 100
        use_content = hyperparams["use_content"] if "use_content" in hyperparams else False
        kernel_l2 = hyperparams["kernel_l2"] if "kernel_l2" in hyperparams else 0.0
        network_depth = hyperparams["network_depth"] if "network_depth" in hyperparams else 3
        dropout = hyperparams["dropout"] if "dropout" in hyperparams else 0.0
        use_resnet = hyperparams["use_resnet"] if "use_resnet" in hyperparams else False
        enable_implicit = hyperparams["enable_implicit"] if "enable_implicit" in hyperparams else False

        assert user_content_vectors.shape[1] == item_content_vectors.shape[1]
        assert user_vectors.shape[1] == item_vectors.shape[1]
        # For unseen users and items creating 2 mock nodes
        user_content_vectors = np.concatenate((np.zeros((1, user_content_vectors.shape[1])), user_content_vectors))
        item_content_vectors = np.concatenate((np.zeros((1, item_content_vectors.shape[1])), item_content_vectors))
        user_vectors = np.concatenate((np.zeros((1, user_vectors.shape[1])), user_vectors))
        item_vectors = np.concatenate((np.zeros((1, item_vectors.shape[1])), item_vectors))

        mu, user_bias, item_bias, train, \
        ratings_count_by_user, ratings_count_by_item, \
        min_affinity, \
        max_affinity, user_item_list, item_user_list, \
        gen_fn, prediction_output_shape, prediction_output_types = self.__build_dataset__(user_ids, item_ids,
                                                                              user_item_affinities,
                                                                              user_content_vectors,
                                                                              item_content_vectors,
                                                                              user_vectors, item_vectors,
                                                                              user_id_to_index,
                                                                              item_id_to_index,
                                                                              rating_scale, hyperparams)
        assert np.sum(np.isnan(user_bias)) == 0
        assert np.sum(np.isnan(item_bias)) == 0
        assert np.sum(np.isnan(user_content_vectors)) == 0
        assert np.sum(np.isnan(item_content_vectors)) == 0
        assert np.sum(np.isnan(user_vectors)) == 0
        assert np.sum(np.isnan(item_vectors)) == 0

        total_users = len(user_ids) + 1
        total_items = len(item_ids) + 1
        if use_content:
            user_vectors = np.concatenate((user_vectors, user_content_vectors), axis=1)
            item_vectors = np.concatenate((item_vectors, item_content_vectors), axis=1)
        else:
            user_vectors = np.zeros_like(user_vectors)
            item_vectors = np.zeros_like(item_vectors)
        edge_list = [(user_id_to_index[u] + 1, total_users + item_id_to_index[i] + 1, r) for u, i, r in
                     user_item_affinities]
        biases = np.concatenate(([0.0], user_bias, item_bias))
        g_train = build_dgl_graph(edge_list, total_users + total_items, np.concatenate((user_vectors, item_vectors)))
        n_content_dims = user_vectors.shape[1]
        g_train.readonly()
        zeroed_indices = [0, 1, total_users + 1]
        model = GraphSAGERecommenderImplicit(GraphSageWithSampling(n_content_dims, self.n_collaborative_dims, network_depth, dropout, g_train),
                                             mu, biases, padding_length=padding_length, zeroed_indices=zeroed_indices, enable_implicit=enable_implicit)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=kernel_l2)

        generate_training_samples, gen_fn, ratings_count_by_user, ratings_count_by_item, user_item_list, item_user_list = self.__prediction_network_datagen__(
            user_ids, item_ids,
            user_item_affinities,
            user_content_vectors,
            item_content_vectors,
            user_vectors, item_vectors,
            user_id_to_index,
            item_id_to_index,
            rating_scale, batch_size, padding_length,
            0, False)
        user_item_affinities = [(user_id_to_index[u] + 1, item_id_to_index[i] + 1,
                                 ratings_count_by_user[user_id_to_index[u] + 1],
                                 ratings_count_by_item[item_id_to_index[i] + 1], r) for u, i, r in user_item_affinities]
        generator = generate_training_samples(user_item_affinities)

        for epoch in range(epochs):
            start = time.time()

            src = []
            dst = []
            dst_to_srcs = []
            src_to_dsts = []
            src_to_dsts_count = []
            dst_to_srcs_count = []
            rating = []
            start_gen = time.time()
            for (u, i, us, iis, nu, ni), r in generator():
                src.append(u)
                dst.append(i)
                dst_to_srcs.append(us)
                src_to_dsts.append(iis)
                src_to_dsts_count.append(nu)
                dst_to_srcs_count.append(ni)
                rating.append(r)
            total_gen = time.time() - start_gen

            src = torch.LongTensor(src)
            dst = torch.LongTensor(dst) + total_users
            dst_to_srcs = torch.LongTensor(dst_to_srcs)
            src_to_dsts = torch.LongTensor(src_to_dsts) + total_users
            src_to_dsts_count = torch.DoubleTensor(src_to_dsts_count)
            dst_to_srcs_count = torch.DoubleTensor(dst_to_srcs_count)
            rating = torch.DoubleTensor(rating)

            model.eval()

            # Validation & Test, we precompute GraphSage output for all nodes first.
            sampler = dgl.contrib.sampling.NeighborSampler(
                g_train,
                batch_size,
                5,
                network_depth,
                seed_nodes=torch.arange(g_train.number_of_nodes()),
                prefetch=True,
                add_self_loop=True,
                shuffle=False,
                num_workers=self.cpu
            )

            with torch.no_grad():
                h = []
                for nf in sampler:
                    # import pdb
                    # pdb.set_trace()
                    h.append(model.gcn.forward(nf))
                h = torch.cat(h)

                # Compute Train RMSE
                score = torch.zeros(len(src))
                for i in range(0, len(src), batch_size):
                    s = src[i:i + batch_size]
                    d = dst[i:i + batch_size]
                    d2s = dst_to_srcs[i:i + batch_size]
                    s2d = src_to_dsts[i:i + batch_size]
                    s2dc = src_to_dsts_count[i:i + batch_size]
                    d2sc = dst_to_srcs_count[i:i + batch_size]
                    s2d_imp = h[s2d]
                    d2s_imp = h[d2s]
                    #

                    res = get_score(s, d, model.mu, model.node_biases,
                                    h[d], s2d, s2dc, s2d_imp,
                                    h[s], d2s, d2sc, d2s_imp,
                                    zeroed_indices, enable_implicit=enable_implicit)
                    score[i:i + batch_size] = res
                train_rmse = ((score - rating) ** 2).mean().sqrt()

            model.train()

            def train(src, dst, src_to_dsts, dst_to_srcs, src_to_dsts_count, dst_to_srcs_count):
                shuffle_idx = torch.randperm(len(src))
                src_shuffled = src[shuffle_idx]
                dst_shuffled = dst[shuffle_idx]
                dst_to_srcs_shuffled = dst_to_srcs[shuffle_idx]
                src_to_dsts_shuffled = src_to_dsts[shuffle_idx]
                src_to_dsts_count_shuffled = src_to_dsts_count[shuffle_idx]
                dst_to_srcs_count_shuffled = dst_to_srcs_count[shuffle_idx]
                rating_shuffled = rating[shuffle_idx]

                src_batches = src_shuffled.split(batch_size)
                dst_batches = dst_shuffled.split(batch_size)
                dst_to_srcs_batches = dst_to_srcs_shuffled.split(batch_size)
                src_to_dsts_batches = src_to_dsts_shuffled.split(batch_size)
                src_to_dsts_count_batches = src_to_dsts_count_shuffled.split(batch_size)
                dst_to_srcs_count_batches = dst_to_srcs_count_shuffled.split(batch_size)
                rating_batches = rating_shuffled.split(batch_size)

                seed_nodes = torch.cat(sum([[s, d,] for s, d in zip(src_batches, dst_batches)], []))

                sampler = dgl.contrib.sampling.NeighborSampler(
                    g_train,  # the graph
                    batch_size * 2,  # number of nodes to compute at a time, HACK 2
                    5,  # number of neighbors for each node
                    network_depth,  # number of layers in GCN
                    seed_nodes=seed_nodes,  # list of seed nodes, HACK 2
                    prefetch=True,  # whether to prefetch the NodeFlows
                    add_self_loop=True,  # whether to add a self-loop in the NodeFlows, HACK 1
                    shuffle=False,  # whether to shuffle the seed nodes.  Should be False here.
                    num_workers=self.cpu,
                )

                # Training
                total_loss = 0.0
                for s, d, d2s, s2d, s2dc, d2sc,  r, nodeflow in zip(src_batches, dst_batches, dst_to_srcs_batches, src_to_dsts_batches, src_to_dsts_count_batches, dst_to_srcs_count_batches, rating_batches, sampler):
                    score = model.forward(nodeflow, s, d, s2d, s2dc, d2s, d2sc)
                    loss = ((score - r) ** 2).mean()
                    total_loss = total_loss + loss
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                return total_loss/len(src_batches)
            if epoch % 2 == 1:
                loss = train(src, dst, src_to_dsts, dst_to_srcs, src_to_dsts_count, dst_to_srcs_count)
            else:
                loss = train(dst, src, dst_to_srcs, src_to_dsts, dst_to_srcs_count, src_to_dsts_count)

            total_time = time.time() - start

            self.log.info('Epoch %2d/%2d: ' % (int(epoch + 1), epochs) + ' Training loss: %.4f' % loss.item() + ' Train RMSE: %.4f ||' % train_rmse.item() + ' Generator Time: %.1f' % total_gen + '|| Time Taken: %.1f' % total_time)

        model.eval()
        sampler = dgl.contrib.sampling.NeighborSampler(
            g_train,
            batch_size,
            5,
            network_depth,
            seed_nodes=torch.arange(g_train.number_of_nodes()),
            prefetch=True,
            add_self_loop=True,
            shuffle=False,
            num_workers=self.cpu
        )

        with torch.no_grad():
            h = []
            for nf in sampler:
                # import pdb
                # pdb.set_trace()
                h.append(model.gcn.forward(nf))
            h = torch.cat(h).numpy()

        bias = model.node_biases.detach().numpy()
        assert len(bias) == total_users + total_items + 1
        mu = model.mu.detach().numpy()

        prediction_artifacts = {"vectors": h, "user_item_list": user_item_list,
                                "item_user_list": item_user_list, "mu": mu,
                                "bias": bias,
                                "total_users": total_users,
                                "ratings_count_by_user": ratings_count_by_user, "padding_length": padding_length,
                                "ratings_count_by_item": ratings_count_by_item, "enable_implicit": enable_implicit,
                                "batch_size": batch_size, "gen_fn": gen_fn, "zeroed_indices": zeroed_indices}
        # self.log.info("Built Prediction Network, model params = %s", model.count_params())
        return prediction_artifacts

    def predict(self, user_item_pairs: List[Tuple[str, str]], clip=True) -> List[float]:
        h = self.prediction_artifacts["vectors"]
        mu = self.prediction_artifacts["mu"]
        bias = self.prediction_artifacts["bias"]
        total_users = self.prediction_artifacts["total_users"]
        zeroed_indices = self.prediction_artifacts["zeroed_indices"]
        enable_implicit = self.prediction_artifacts["enable_implicit"]

        ratings_count_by_user = self.prediction_artifacts["ratings_count_by_user"]
        ratings_count_by_item = self.prediction_artifacts["ratings_count_by_item"]
        batch_size = self.prediction_artifacts["batch_size"]
        gen_fn = self.prediction_artifacts["gen_fn"]
        batch_size = max(512, batch_size)

        def generate_prediction_samples(affinities):
            def generator():
                for i in range(0, len(affinities), batch_size):
                    start = i
                    end = min(i + batch_size, len(affinities))
                    generated = np.array([gen_fn(u, v, nu, ni) for u, v, nu, ni in affinities[start:end]])
                    for g in generated:
                        yield g
            return generator

        if self.fast_inference:
            return self.fast_predict(user_item_pairs)

        if self.super_fast_inference:
            assert self.mu is not None
            predictions = [self.mu + self.bu[u] + self.bi[i] for u, i, in user_item_pairs]
            return np.clip(predictions, self.rating_scale[0], self.rating_scale[1])

        uip = [(self.user_id_to_index[u] + 1 if u in self.user_id_to_index else 0,
                self.item_id_to_index[i] + 1 if i in self.item_id_to_index else 0,
                ratings_count_by_user[self.user_id_to_index[u] + 1 if u in self.user_id_to_index else 1],
                ratings_count_by_item[self.item_id_to_index[i] + 1 if i in self.item_id_to_index else 1]) for u, i in user_item_pairs]

        assert np.sum(np.isnan(uip)) == 0
        generator = generate_prediction_samples(uip)

        user = []
        item = []
        users = []
        items = []
        nus = []
        nis = []
        for u, i, us, iis, nu, ni in generator():
            user.append(u)
            item.append(i)
            users.append(us)
            items.append(iis)
            nus.append(nu)
            nis.append(ni)

        user = np.array(user).astype(int)
        item = np.array(item).astype(int) + total_users
        users = np.array(users).astype(int)
        items = np.array(items).astype(int) + total_users
        nus = np.array(nus)
        nis = np.array(nis)

        score = np.zeros(len(user))
        for i in range(0, len(user), batch_size):
            s = user[i:i + batch_size]
            d = item[i:i + batch_size]
            d2s = users[i:i + batch_size]
            s2d = items[i:i + batch_size]
            s2dc = nus[i:i + batch_size]
            d2sc = nis[i:i + batch_size]
            s2d_imp = h[s2d]
            d2s_imp = h[d2s]

            res = get_score(s, d, mu, bias,
                            h[d], s2d, s2dc, s2d_imp,
                            h[s], d2s, d2sc, d2s_imp,
                            zeroed_indices, enable_implicit=enable_implicit)
            score[i:i + batch_size] = res

        predictions = score
        predictions = np.array(predictions)
        assert len(predictions) == len(user_item_pairs)
        if clip:
            predictions = np.clip(predictions, self.rating_scale[0], self.rating_scale[1])
        return predictions
