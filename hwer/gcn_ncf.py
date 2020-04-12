import time
from typing import List, Dict, Tuple, Optional

import numpy as np
from more_itertools import flatten

from .logging import getLogger
from .random_walk import *
from .svdpp_hybrid import SVDppHybrid
from .utils import unit_length_violations
import logging
import dill
import sys
logger = getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
from .hybrid_graph_recommender import HybridGCNRec
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
        cos = (g_src * g_dst).sum(1)
        cos = ((cos + 1)/2).flatten()
        ncf = self.W(vec).flatten()
        out = ncf * self.ncf_gcn_balance + (1 - self.ncf_gcn_balance) * cos
        return out


class NCFEmbedding(nn.Module):
    def __init__(self, feature_size, depth, gaussian_noise, content):
        super(NCFEmbedding, self).__init__()
        noise = GaussianNoise(gaussian_noise)

        wc1 = nn.Linear(content.shape[1], feature_size)
        init_fc(wc1, 'xavier_uniform_', 'leaky_relu', 0.1)
        wc2 = nn.Linear(feature_size, feature_size)
        init_fc(wc2, 'xavier_uniform_', 'leaky_relu', 0.1)
        content_emb = nn.Embedding.from_pretrained(torch.tensor(content, dtype=torch.float), freeze=True)
        self.cem = nn.Sequential(content_emb, wc1, nn.LeakyReLU(0.1), noise, wc2, nn.LeakyReLU(0.1))

        w1 = nn.Linear(feature_size * 2, feature_size * 2)
        init_fc(w1, 'xavier_uniform_', 'leaky_relu', 0.1)
        layers = [noise, w1, nn.LeakyReLU(negative_slope=0.1)]

        for _ in range(depth - 1):
            wx = nn.Linear(feature_size * 2, feature_size * 2)
            init_fc(wx, 'xavier_uniform_', 'leaky_relu', 0.1)
            layers.extend([noise, wx, nn.LeakyReLU(negative_slope=0.1)])

        w_out = nn.Linear(feature_size * 2, feature_size)
        init_fc(w_out, 'xavier_uniform_', 'tanh', 0.1)
        layers.extend([w_out, nn.Tanh()])
        self.W = nn.Sequential(*layers)

    def forward(self, node, g_node):
        hc_src = self.cem(node)
        vec = torch.cat([hc_src, g_node], 1)
        vec = self.W(vec)
        vec = vec / vec.norm(dim=1, keepdim=True).clamp(min=1e-5)
        return vec


class RecImplicitEmbedding(nn.Module):
    def __init__(self, gcn: GraphSageWithSampling, ncf: NCFEmbedding):
        super(RecImplicitEmbedding, self).__init__()
        self.gcn = gcn
        self.ncf = ncf

    def forward(self, nf, src, dst):
        h_output = self.gcn(nf)
        h_src = h_output[nf.map_from_parent_nid(-1, src, True)]
        h_dst = h_output[nf.map_from_parent_nid(-1, dst, True)]
        return self.ncf(src, h_src), self.ncf(dst, h_dst)


class RecImplicit(nn.Module):
    def __init__(self, gcn: GraphSageWithSampling, ncf: NCF):
        super(RecImplicit, self).__init__()
        self.gcn = gcn
        self.ncf = ncf

    def forward(self, nf, src, dst):
        h_output = self.gcn(nf)
        h_src = h_output[nf.map_from_parent_nid(-1, src, True)]
        h_dst = h_output[nf.map_from_parent_nid(-1, dst, True)]
        return self.ncf(src, dst, h_src, h_dst)


class GcnNCF(HybridGCNRec):
    def __init__(self, embedding_mapper: dict, knn_params: Optional[dict], rating_scale: Tuple[float, float],
                 n_content_dims: int = 32, n_collaborative_dims: int = 32, fast_inference: bool = False,
                 super_fast_inference: bool = False):
        super().__init__(embedding_mapper, knn_params, rating_scale, n_content_dims, n_collaborative_dims,
                         fast_inference, super_fast_inference)
        self.log = getLogger(type(self).__name__)
        assert n_collaborative_dims % 2 == 0
        self.cpu = int(os.cpu_count() / 2)

    def __positive_pair_generator__(self, total_users, total_items,
                                    affinities: List[Tuple[int, int, float]],
                                    hyperparams):
        ps_proportion = hyperparams["ps_proportion"] if "ps_proportion" in hyperparams else 1
        positive_samples = len(affinities) * ps_proportion
        p = 0.25
        q = hyperparams["q"] if "q" in hyperparams else 0.25
        affinities = [(i, j, r) for i, j, r in affinities]
        affinities.extend([(i, i, 1) for i in range(total_users + total_items)])
        Walker = RandomWalker
        walker = Walker(read_edgelist(affinities, weighted=False), p=p, q=q)
        walker.preprocess_transition_probs()
        samples_per_node = int(np.ceil(positive_samples / (total_users + total_items)))
        from collections import Counter
        random_walks = max(100, samples_per_node * 10)

        def sampler():
            for i in range(total_users):
                cnt = Counter()
                for walk in walker.simulate_walks_single_node(i, random_walks, 4):
                    if len(walk) == 5 and walk[4] != i:
                        cnt.update([walk[4]])
                results = cnt.most_common(samples_per_node)
                if len(results) > 0:
                    results, _ = zip(*results)
                for r in results:
                    yield i, r

            for j in range(total_items):
                j = j + total_users
                cnt = Counter()
                for walk in walker.simulate_walks_single_node(j, random_walks, 4):
                    if len(walk) == 5 and walk[4] != j:
                        cnt.update([walk[4]])
                results = cnt.most_common(samples_per_node)
                if len(results) > 0:
                    results, _ = zip(*results)
                for r in results:
                    yield j, r

        return sampler

    def __negative_pair_generator__(self, total_users, total_items,
                                    affinities: List[Tuple[int, int, float]],
                                    hyperparams):
        nsh = hyperparams["nsh"] if "nsh" in hyperparams else 1
        positive_samples = len(affinities)
        p = 0.25
        q = hyperparams["q"] if "q" in hyperparams else 0.25
        negative_samples = int(nsh * positive_samples)
        affinities = [(i, j, r) for i, j, r in affinities]
        affinities.extend([(i, i, 1) for i in range(total_users + total_items)])
        Walker = RandomWalker
        walker = Walker(read_edgelist(affinities, weighted=False), p=p, q=q)
        walker.preprocess_transition_probs()
        # samples_per_node = {i: int(len(walker.adjacency_list[i]) * nsh) for i in range(total_users + total_items)}
        spn = int(np.ceil(negative_samples / (total_users + total_items)))
        samples_per_node = {i: spn for i in range(total_users + total_items)}
        all_nodes = set([i for i in range(total_users + total_items)])
        all_users = set([i for i in range(total_users)])
        all_items = set([i+total_users for i in range(total_items)])

        def nsg():
            for i in range(total_users):
                neighbours = {i}
                for walk in walker.simulate_walks_single_node(i, 20, 5):
                    neighbours.update(walk)
                candidates = list(all_items - neighbours)
                results = random.choices(candidates, k=samples_per_node[i])
                for r in results:
                    yield i, r

            for j in range(total_items):
                j = j + total_users
                neighbours = {j}
                for walk in walker.simulate_walks_single_node(j, 20, 5):
                    neighbours.update(walk)
                candidates = list(all_users - neighbours)
                results = random.choices(candidates, k=samples_per_node[j])
                for r in results:
                    yield j, r

        def nsg2():
            for i in range(total_users+total_items):
                neighbours = {i}
                for walk in walker.simulate_walks_single_node(i, 20, 5):
                    neighbours.update(walk)
                candidates = list(all_nodes - neighbours)
                results = random.choices(candidates, k=samples_per_node[i])
                for r in results:
                    yield i, r

        return nsg

    def __user_item_affinities_triplet_trainer_data_gen_fn__(self, user_ids, item_ids,
                                                             user_id_to_index,
                                                             item_id_to_index,
                                                             affinities: List[Tuple[str, str, float]],
                                                             hyperparams):

        walk_length = 3  # Fixed, change node2vec_generator if changed, see grouper from more itertools or commit 0a7d3b0755ae3ccafec5077edb4c8bf1ed1e3b34
        # for a generic implementation
        num_walks = hyperparams["num_walks"] if "num_walks" in hyperparams else 10
        p = 0.25
        q = hyperparams["q"] if "q" in hyperparams else 0.25
        total_users = len(user_ids)
        total_items = len(item_ids)
        ratings = np.array([r for i, j, r in affinities])
        min_rating, max_rating = np.min(ratings), np.max(ratings)
        affinities = [(user_id_to_index[i], total_users + item_id_to_index[j], 1 + r - min_rating) for i, j, r in affinities]
        affinities_gen_data = [(i, j, r) for i, j, r in affinities]

        def affinities_generator():
            np.random.shuffle(affinities_gen_data)
            for i, j, r in affinities_gen_data:
                yield (i, j), r

        if num_walks == 0:
            return affinities_generator

        affinities.extend([(i, i, 1) for i in range(total_users + total_items)])
        Walker = RandomWalker
        walker = Walker(read_edgelist(affinities, weighted=True), p=p, q=q)
        walker.preprocess_transition_probs()

        def node2vec_generator():
            g = walker.simulate_walks_generator_optimised(num_walks, walk_length=walk_length)
            for walk in g:
                yield (walk[0], walk[2]), 1

        from more_itertools import distinct_combinations, chunked, grouper, interleave_longest

        def sentences_generator():
            return interleave_longest(node2vec_generator(), affinities_generator())

        return sentences_generator

    def __user_item_affinities_triplet_trainer__(self,
                                                 user_ids: List[str], item_ids: List[str],
                                                 user_item_affinities: List[Tuple[str, str, float]],
                                                 user_vectors: np.ndarray, item_vectors: np.ndarray,
                                                 user_id_to_index: Dict[str, int], item_id_to_index: Dict[str, int],
                                                 n_output_dims: int,
                                                 hyperparams: Dict) -> Tuple[np.ndarray, np.ndarray]:
        from .gcn import build_dgl_graph
        import torch
        import torch.nn.functional as F
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cpu = torch.device('cpu')
        import dgl
        self.log.debug(
            "Start Training User-Item Affinities, n_users = %s, n_items = %s, n_samples = %s, in_dims = %s, out_dims = %s",
            len(user_ids), len(item_ids), len(user_item_affinities), user_vectors.shape[1], n_output_dims)

        lr = hyperparams["lr"] if "lr" in hyperparams else 0.1
        epochs = hyperparams["epochs"] if "epochs" in hyperparams else 1
        gcn_layers = hyperparams["gcn_layers"] if "gcn_layers" in hyperparams else 2
        ncf_layers = hyperparams["ncf_layers"] if "ncf_layers" in hyperparams else 2
        gcn_batch_size = hyperparams["gcn_batch_size"] if "gcn_batch_size" in hyperparams else 512
        verbose = hyperparams["verbose"] if "verbose" in hyperparams else 1
        margin = hyperparams["margin"] if "margin" in hyperparams else 0.0
        gcn_kernel_l2 = hyperparams["gcn_kernel_l2"] if "gcn_kernel_l2" in hyperparams else 0.0
        conv_depth = hyperparams["conv_depth"] if "conv_depth" in hyperparams else 1
        gaussian_noise = hyperparams["gaussian_noise"] if "gaussian_noise" in hyperparams else 0.0
        ns_proportion = hyperparams["ns_proportion"] if "ns_proportion" in hyperparams else 1
        total_users = len(user_ids)
        total_items = len(item_ids)

        assert np.sum(np.isnan(user_vectors)) == 0
        assert np.sum(np.isnan(item_vectors)) == 0

        if epochs <= 0:
            from .utils import unit_length
            from sklearn.decomposition import PCA
            user_vectors_length = len(user_vectors)
            all_vectors = np.concatenate((user_vectors, item_vectors), axis=0)
            if user_vectors.shape[1] > self.n_collaborative_dims:
                pca = PCA(n_components=self.n_collaborative_dims, )
                all_vectors = pca.fit_transform(all_vectors)
            elif user_vectors.shape[1] < self.n_collaborative_dims:
                raise ValueError()
            all_vectors = unit_length(all_vectors, axis=1)
            user_vectors = all_vectors[:user_vectors_length]
            item_vectors = all_vectors[user_vectors_length:]
            return user_vectors, item_vectors
        import gc
        gc.collect()

        total_users = len(user_ids)
        edge_list = [(user_id_to_index[u], total_users + item_id_to_index[i], r) for u, i, r in user_item_affinities]
        content_vectors = np.concatenate((user_vectors, item_vectors))
        g_train = build_dgl_graph(edge_list, len(user_ids) + len(item_ids), content_vectors)
        g_train.readonly()
        n_content_dims = content_vectors.shape[1]
        ncf = NCFEmbedding(self.n_collaborative_dims, ncf_layers, gaussian_noise,
                           np.concatenate((user_vectors, item_vectors)))
        gcn = GraphSageWithSampling(n_content_dims, self.n_collaborative_dims, gcn_layers, g_train,
                                    gaussian_noise, conv_depth)
        model = RecImplicitEmbedding(gcn=gcn, ncf=ncf)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=gcn_kernel_l2)
        generate_training_samples = self.__user_item_affinities_triplet_trainer_data_gen_fn__(user_ids, item_ids,
                                                                                              user_id_to_index,
                                                                                              item_id_to_index,
                                                                                              user_item_affinities,
                                                                                              hyperparams)

        def get_samples():
            src, dst, weights, error_weights = [], [], [], []
            for (u, v), r in generate_training_samples():
                src.append(u)
                dst.append(v)
                weights.append(1.0)
                error_weights.append(r)

            positive_samples = len(src)
            src = torch.LongTensor(src)
            dst = torch.LongTensor(dst)
            weights = torch.FloatTensor(weights)

            ns = ns_proportion
            negative_samples = int(ns * positive_samples)
            src_neg = torch.randint(0, total_users+total_items, (negative_samples,))
            dst_neg = torch.randint(0, total_users+total_items, (negative_samples,))
            weights_neg = torch.tensor([0.0] * negative_samples)
            src = torch.cat((src, src_neg), 0)
            dst = torch.cat((dst, dst_neg), 0)
            weights = torch.cat((weights, weights_neg), 0)

            shuffle_idx = torch.randperm(positive_samples + negative_samples)
            src = src[shuffle_idx]
            dst = dst[shuffle_idx]
            weights = weights[shuffle_idx]
            self.log.info("Generate Samples: Positive = %s, Negative = %s", positive_samples, negative_samples)
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
            def train(src, dst, weights):

                src_batches = src.split(gcn_batch_size)
                dst_batches = dst.split(gcn_batch_size)
                weights_batches = weights.split(gcn_batch_size)
                model.train()
                seed_nodes = torch.cat(sum([[s, d] for s, d in zip(src_batches, dst_batches)], []))
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
                for s, d, nodeflow, r in zip(src_batches, dst_batches, sampler, weights_batches):
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

            loss += train(src, dst, weights)
            gen_time = time.time()
            if epoch < epochs - 1:
                src, dst, weights = get_samples()
            gen_time = time.time() - gen_time

            total_time = time.time() - start
            self.log.info('Epoch %2d/%2d: ' % (int(epoch + 1),
                                               epochs) + ' Training loss: %.4f' % loss +
                          ' || Time Taken: %.1f' % total_time + " Generator time: %.1f" % gen_time)

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
        src_batches = torch.arange(g_train.number_of_nodes()).split(gcn_batch_size)
        with torch.no_grad():
            h = []
            for src, nf in zip(src_batches, sampler):
                h_src = model.gcn.forward(nf)
                h_src = model.ncf.forward(src, h_src)
                h.append(h_src)
        h = torch.cat(h).numpy()

        user_vectors, item_vectors = h[:total_users], h[total_users:]
        self.log.info(
            "End Training User-Item Affinities, Unit Length Violations:: user = %s, item = %s, margin = %.4f",
            unit_length_violations(user_vectors, axis=1), unit_length_violations(item_vectors, axis=1), margin)

        gc.collect()
        return user_vectors, item_vectors

    def __build_prediction_network__(self, user_ids: List[str], item_ids: List[str],
                                     user_item_affinities: List[Tuple[str, str, float]],
                                     user_content_vectors: np.ndarray, item_content_vectors: np.ndarray,
                                     user_vectors: np.ndarray, item_vectors: np.ndarray,
                                     user_id_to_index: Dict[str, int], item_id_to_index: Dict[str, int],
                                     rating_scale: Tuple[float, float], hyperparams: Dict):
        from .gcn import build_dgl_graph, GraphSageWithSampling, GraphSAGERecommender, get_score
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cpu = torch.device('cpu')
        import dgl
        import gc
        self.log.debug(
            "Start Building Prediction Network, collaborative vectors shape = %s, content vectors shape = %s",
            (user_vectors.shape, item_vectors.shape), (user_content_vectors.shape, item_content_vectors.shape))

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

        assert user_content_vectors.shape[1] == item_content_vectors.shape[1]
        assert user_vectors.shape[1] == item_vectors.shape[1]
        # For unseen users and items creating 2 mock nodes
        user_content_vectors = np.concatenate((np.zeros((1, user_content_vectors.shape[1])), user_content_vectors))
        item_content_vectors = np.concatenate((np.zeros((1, item_content_vectors.shape[1])), item_content_vectors))
        gc.collect()
        assert np.sum(np.isnan(user_content_vectors)) == 0
        assert np.sum(np.isnan(item_content_vectors)) == 0

        total_users = len(user_ids) + 1
        total_items = len(item_ids) + 1
        if not use_content:
            user_content_vectors = np.zeros((user_content_vectors.shape[0], 1))
            item_content_vectors = np.zeros((item_content_vectors.shape[0], 1))

        import gc
        gc.collect()
        edge_list = [(user_id_to_index[u] + 1, total_users + item_id_to_index[i] + 1, r) for u, i, r in
                     user_item_affinities]
        g_train = build_dgl_graph(edge_list, total_users + total_items, np.concatenate((user_content_vectors, item_content_vectors)))
        n_content_dims = user_content_vectors.shape[1]
        g_train.readonly()
        ncf = NCF(self.n_collaborative_dims, ncf_layers, gaussian_noise,
                  np.concatenate((user_content_vectors, item_content_vectors)), ncf_gcn_balance)
        gcn = GraphSageWithSampling(n_content_dims, self.n_collaborative_dims, gcn_layers, g_train,
                                    gaussian_noise, conv_depth)
        model = RecImplicit(gcn=gcn, ncf=ncf)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=kernel_l2)
        hard_negative_gen = self.__negative_pair_generator__(total_users, total_items, user_item_affinities, hyperparams)
        pos_gen = self.__positive_pair_generator__(total_users, total_items, user_item_affinities, hyperparams)

        user_item_affinities = [(user_id_to_index[u] + 1, total_users + item_id_to_index[i] + 1, r) for u, i, r in
                                user_item_affinities]

        def get_samples():
            src, dst, weights, error_weights = [], [], [], []
            for u, v, r in user_item_affinities:
                src.append(u)
                dst.append(v)
                weights.append(1.0)
                error_weights.append(r)

            src = torch.LongTensor(src)
            dst = torch.LongTensor(dst)
            weights = torch.FloatTensor(weights)

            ns = ns_proportion
            positive_samples = len(src)
            negative_samples = int(ns * positive_samples)
            src_neg = torch.randint(0, total_users, (negative_samples,))
            dst_neg = torch.randint(total_users+1, total_users+total_items, (negative_samples,))
            weights_neg = torch.tensor([0.0] * negative_samples)
            src = torch.cat((src, src_neg), 0)
            dst = torch.cat((dst, dst_neg), 0)
            weights = torch.cat((weights, weights_neg), 0)

            if nsh > 0:
                h_src_neg, h_dst_neg = zip(*hard_negative_gen())
                weights_hneg = torch.tensor([0.0] * len(h_src_neg))
                h_src_neg = torch.LongTensor(h_src_neg)
                h_dst_neg = torch.LongTensor(h_dst_neg)
                src = torch.cat((src, h_src_neg), 0)
                dst = torch.cat((dst, h_dst_neg), 0)
                weights = torch.cat((weights, weights_hneg), 0)

            if ps_proportion > 0:
                h_src_pos, h_dst_pos = zip(*pos_gen())
                weights_pos = torch.tensor([1.0] * len(h_src_pos))
                h_src_pos = torch.LongTensor(h_src_pos)
                h_dst_pos = torch.LongTensor(h_dst_pos)
                src = torch.cat((src, h_src_pos), 0)
                dst = torch.cat((dst, h_dst_pos), 0)
                weights = torch.cat((weights, weights_pos), 0)

            shuffle_idx = torch.randperm(len(src))
            src = src[shuffle_idx]
            dst = dst[shuffle_idx]
            weights = weights[shuffle_idx]
            return src, dst, weights

        src, dst, rating = get_samples()
        # total_examples = len(src) - 10_000
        # src, dst, rating = src[:total_examples], dst[:total_examples], rating[:total_examples]
        # opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=kernel_l2, momentum=0.9, nesterov=True)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=epochs,
        #                                                 steps_per_epoch=int(
        #                                                     np.ceil(total_examples / batch_size)),
        #                                                 div_factor=50, final_div_factor=100)

        model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.log.info("Built Prediction Network, model params = %s, examples = %s, model = \n%s", params, len(src), model)

        for epoch in range(epochs):
            gc.collect()
            start = time.time()
            model.train()

            # shuffle_idx = torch.randperm(len(src))
            # src = src[shuffle_idx]
            # dst = dst[shuffle_idx]
            # rating = rating[shuffle_idx]

            def train(src, dst, rating):
                import gc
                gc.collect()

                src_batches = src.split(batch_size)
                dst_batches = dst.split(batch_size)
                rating_batches = rating.split(batch_size)

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
                for s, d, r, nf in zip(src_batches, dst_batches, rating_batches, sampler):
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

            loss = train(src, dst, rating)
            if epoch < epochs - 1:
                src, dst, rating = get_samples()
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
                                "total_users": total_users}
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.log.info("Built Prediction Network, model params = %s", params)
        gc.collect()
        return prediction_artifacts

    def predict(self, user_item_pairs: List[Tuple[str, str]], clip=True) -> List[float]:
        from .gcn import get_score
        model = self.prediction_artifacts["model"]
        h = self.prediction_artifacts["h"]
        total_users = self.prediction_artifacts["total_users"]
        batch_size = 512

        uip = [(self.user_id_to_index[u] + 1 if u in self.user_id_to_index else 0,
                self.item_id_to_index[i] + 1 if i in self.item_id_to_index else 0) for u, i in user_item_pairs]

        assert np.sum(np.isnan(uip)) == 0

        user, item = zip(*uip)

        user = torch.tensor(user)
        item = torch.tensor(item) + total_users

        predictions = []
        with torch.no_grad():
            user = user.split(batch_size)
            item = item.split(batch_size)

            for u, i in zip(user, item):
                g_src = h[u]
                g_dst = h[i]
                scores = model.forward(u, i, g_src, g_dst)
                scores = list(scores.numpy())
                predictions.extend(scores)
        return predictions

    def __build_svd_model__(self, user_ids: List[str], item_ids: List[str],
                            user_item_affinities: List[Tuple[str, str, float]],
                            user_id_to_index: Dict[str, int], item_id_to_index: Dict[str, int],
                            rating_scale: Tuple[float, float], **svd_params):
        pass
