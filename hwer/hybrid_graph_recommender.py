import time
from typing import List, Dict, Tuple, Optional

import numpy as np

from .logging import getLogger
from .random_walk import *
from .hybrid_recommender import HybridRecommender
from .utils import unit_length_violations
import logging
import dill
import sys
logger = getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))


class HybridGCNRec(HybridRecommender):
    def __init__(self, embedding_mapper: dict, knn_params: Optional[dict], rating_scale: Tuple[float, float],
                 n_collaborative_dims: int = 32):
        super().__init__(embedding_mapper, knn_params, rating_scale, n_collaborative_dims)
        self.log = getLogger(type(self).__name__)
        assert n_collaborative_dims % 2 == 0
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

    def __user_item_affinities_triplet_trainer_data_gen_fn__(self, user_ids, item_ids,
                                                             user_id_to_index,
                                                             item_id_to_index,
                                                             affinities: List[Tuple[str, str, float]],
                                                             hyperparams):

        total_users = len(user_ids)
        ratings = np.array([r for i, j, r in affinities])
        min_rating, max_rating = np.min(ratings), np.max(ratings)
        affinities = [(user_id_to_index[i], total_users + item_id_to_index[j], 1 + r - min_rating) for i, j, r in affinities]
        affinities_gen_data = [(i, j, r) for i, j, r in affinities]

        def affinities_generator():
            np.random.shuffle(affinities_gen_data)
            for i, j, r in affinities_gen_data:
                yield (i, j), r

        return affinities_generator

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

        gcn_lr = hyperparams["gcn_lr"] if "gcn_lr" in hyperparams else 0.1
        gcn_epochs = hyperparams["gcn_epochs"] if "gcn_epochs" in hyperparams else 1
        gcn_layers = hyperparams["gcn_layers"] if "gcn_layers" in hyperparams else 5
        gcn_batch_size = hyperparams["gcn_batch_size"] if "gcn_batch_size" in hyperparams else 512
        verbose = hyperparams["verbose"] if "verbose" in hyperparams else 1
        margin = hyperparams["margin"] if "margin" in hyperparams else 1.0
        gcn_kernel_l2 = hyperparams["gcn_kernel_l2"] if "gcn_kernel_l2" in hyperparams else 0.0
        enable_gcn = hyperparams["enable_gcn"] if "enable_gcn" in hyperparams else False
        conv_depth = hyperparams["conv_depth"] if "conv_depth" in hyperparams else 1
        gaussian_noise = hyperparams["gaussian_noise"] if "gaussian_noise" in hyperparams else 0.0
        enable_svd = hyperparams["enable_svd"] if "enable_svd" in hyperparams else False
        total_users = len(user_ids)
        total_items = len(item_ids)

        assert np.sum(np.isnan(user_vectors)) == 0
        assert np.sum(np.isnan(item_vectors)) == 0

        import gc
        gc.collect()

        user_triplet_vectors, item_triplet_vectors = user_vectors, item_vectors
        if enable_svd:
            from surprise import Dataset
            from surprise import Reader
            from surprise import SVD

            reader = Reader(rating_scale=(-1, 1))
            import pandas as pd
            ratings = [(user_id_to_index[u], total_users + item_id_to_index[i], 1) for u, i, r in user_item_affinities]
            negs = list(zip(np.random.randint(0, total_users, len(ratings) * 5),
                            np.random.randint(total_users, total_users + total_items, len(ratings) * 5),
                            [-1] * (len(ratings) * 5)))
            ratings = ratings + negs
            np.random.shuffle(ratings)
            train = pd.DataFrame(ratings)
            train = Dataset.load_from_df(train, reader).build_full_trainset()
            svd_model = SVD(n_factors=self.n_collaborative_dims, biased=False)
            svd_model.fit(train)
            user_svd_vectors = [svd_model.pu[svd_model.trainset.to_inner_uid(i)] for i in range(total_users)]
            item_svd_vectors = [svd_model.qi[svd_model.trainset.to_inner_iid(i + total_users)] for i in range(total_items)]
            user_svd_vectors = np.vstack(user_svd_vectors)
            item_svd_vectors = np.vstack(item_svd_vectors)

            from .utils import unit_length
            user_svd_vectors = unit_length(user_svd_vectors, axis=1)
            item_svd_vectors = unit_length(item_svd_vectors, axis=1)

            # return user_svd_vectors, item_svd_vectors

        if not enable_gcn or gcn_epochs <= 0:
            if enable_svd:
                user_triplet_vectors, item_triplet_vectors = user_svd_vectors, item_svd_vectors
            return user_triplet_vectors, item_triplet_vectors

        triplet_vectors = None

        if enable_svd:
            triplet_vectors = np.concatenate(
                (np.zeros((1, user_svd_vectors.shape[1])), user_svd_vectors, item_svd_vectors))
            triplet_vectors = torch.FloatTensor(triplet_vectors)

        total_users = len(user_ids)
        edge_list = [(user_id_to_index[u], total_users + item_id_to_index[i], r) for u, i, r in user_item_affinities]
        content_vectors = np.concatenate((user_vectors, item_vectors))

        g_train = build_dgl_graph(edge_list, len(user_ids) + len(item_ids), content_vectors)
        g_train.readonly()
        n_content_dims = content_vectors.shape[1]
        model = self.__get_triplet_gcn_model__(n_content_dims, self.n_collaborative_dims, gcn_layers,
                                               conv_depth, g_train, triplet_vectors, margin,
                                               gaussian_noise)
        opt = torch.optim.Adam(model.parameters(), lr=gcn_lr, weight_decay=gcn_kernel_l2)
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

            src = torch.LongTensor(src)
            dst = torch.LongTensor(dst)
            weights = torch.FloatTensor(weights)

            ns = 2
            src_neg = torch.randint(0, total_users+total_items, (len(src) * ns,))
            dst_neg = torch.randint(0, total_users + total_items, (len(src) * ns,))
            weights_neg = torch.tensor([0.0] * len(src) * ns)
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

        def pretrain():
            opt = torch.optim.Adam(model.parameters(), lr=gcn_lr, weight_decay=gcn_kernel_l2)
            target = torch.tensor(np.concatenate((user_svd_vectors, item_svd_vectors)))
            for epoch in range(gcn_epochs * 2):
                seed_nodes = torch.randperm(g_train.number_of_nodes())
                sampler = dgl.contrib.sampling.NeighborSampler(
                    g_train,
                    gcn_batch_size,
                    5,
                    gcn_layers,
                    seed_nodes=seed_nodes,
                    prefetch=True,
                    add_self_loop=True,
                    shuffle=False,
                    num_workers=self.cpu
                )
                nodes = seed_nodes.split(gcn_batch_size)
                total_loss = 0
                for nf, node in zip(sampler, nodes):
                    vec = model.gcn.forward(nf)
                    loss = ((target[node] - vec) ** 2).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    total_loss += loss.item()
                self.log.info('Pretraining Epoch %2d/%2d: ' % (int(epoch + 1),
                                               gcn_epochs) + " loss = %.4f" % (total_loss))

        if enable_svd:
            pretrain()

        model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.log.info("Built KNN Network, model params = %s, examples = %s, model = \n%s", params, len(src), model)
        gc.collect()
        for epoch in range(gcn_epochs):
            start = time.time()
            loss = 0.0
            def train(src, dst):

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
            if epoch < gcn_epochs - 1:
                src, dst, weights = get_samples()
            gen_time = time.time() - gen_time

            total_time = time.time() - start
            self.log.info('Epoch %2d/%2d: ' % (int(epoch + 1),
                                               gcn_epochs) + ' Training loss: %.4f' % loss.item() +
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

        with torch.no_grad():
            h = []
            for nf in sampler:
                h.append(model.gcn.forward(nf))
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
        kernel_l2 = hyperparams["kernel_l2"] if "kernel_l2" in hyperparams else 0.0
        network_depth = hyperparams["network_depth"] if "network_depth" in hyperparams else 3
        conv_depth = hyperparams["conv_depth"] if "conv_depth" in hyperparams else 1
        gaussian_noise = hyperparams["gaussian_noise"] if "gaussian_noise" in hyperparams else 0.0

        assert user_content_vectors.shape[1] == item_content_vectors.shape[1]
        assert user_vectors.shape[1] == item_vectors.shape[1]
        # For unseen users and items creating 2 mock nodes
        user_content_vectors = np.concatenate((np.zeros((1, user_content_vectors.shape[1])), user_content_vectors))
        item_content_vectors = np.concatenate((np.zeros((1, item_content_vectors.shape[1])), item_content_vectors))
        user_vectors = np.concatenate((np.zeros((1, user_vectors.shape[1])), user_vectors))
        item_vectors = np.concatenate((np.zeros((1, item_vectors.shape[1])), item_vectors))
        gc.collect()
        mu, user_bias, item_bias = self.__calculate_bias__(user_ids, item_ids, user_item_affinities, rating_scale)
        assert np.sum(np.isnan(user_bias)) == 0
        assert np.sum(np.isnan(item_bias)) == 0
        assert np.sum(np.isnan(user_content_vectors)) == 0
        assert np.sum(np.isnan(item_content_vectors)) == 0
        assert np.sum(np.isnan(user_vectors)) == 0
        assert np.sum(np.isnan(item_vectors)) == 0

        total_users = len(user_ids) + 1
        total_items = len(item_ids) + 1
        user_vectors = np.concatenate((user_vectors, user_content_vectors), axis=1)
        item_vectors = np.concatenate((item_vectors, item_content_vectors), axis=1)

        edge_list = [(user_id_to_index[u] + 1, total_users + item_id_to_index[i] + 1, r) for u, i, r in
                     user_item_affinities]
        biases = np.concatenate(([0.0], user_bias, item_bias))
        import gc
        gc.collect()
        g_train = build_dgl_graph(edge_list, total_users + total_items, np.concatenate((user_vectors, item_vectors)))
        n_content_dims = user_vectors.shape[1]
        g_train.readonly()
        zeroed_indices = [0, 1, total_users + 1]
        model = GraphSAGERecommender(
            GraphSageWithSampling(n_content_dims, self.n_collaborative_dims, network_depth, g_train,
                                  gaussian_noise, conv_depth),
            mu, biases, zeroed_indices=zeroed_indices)
        opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=kernel_l2, momentum=0.9, nesterov=True)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=epochs,
                                                        steps_per_epoch=int(
                                                            np.ceil(len(user_item_affinities) / batch_size)),
                                                        div_factor=25, final_div_factor=100)
        # opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=kernel_l2)
        user_item_affinities = [(user_id_to_index[u] + 1, item_id_to_index[i] + 1, r) for u, i, r in
                                user_item_affinities]
        src, dst, rating = zip(*user_item_affinities)

        src = torch.LongTensor(src)
        dst = torch.LongTensor(dst) + total_users
        rating = torch.DoubleTensor(rating)
        model.to(device)
        model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.log.info("Built Prediction Network, model params = %s, examples = %s, model = \n%s", params, len(src), model)

        for epoch in range(epochs):
            gc.collect()
            start = time.time()

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
            eval_start_time = time.time()
            with torch.no_grad():
                h = []
                for nf in sampler:
                    h.append(model.gcn.forward(nf))
                h = torch.cat(h)

                # Compute Train RMSE
                score = torch.zeros(len(src))
                for i in range(0, len(src), batch_size):
                    s = src[i:i + batch_size]
                    d = dst[i:i + batch_size]
                    s.to(device)
                    d.to(device)
                    #
                    h_d = h[d]
                    h_s = h[s]
                    h_d.to(device)
                    h_s.to(device)

                    res = get_score(s, d, model.mu, model.node_biases, h_d, h_s)
                    res.to(cpu)
                    score[i:i + batch_size] = res
                train_rmse = ((score - rating) ** 2).mean().sqrt()
            eval_total = time.time() - eval_start_time

            model.train()

            def train(src, dst, rating):
                import gc
                gc.collect()
                shuffle_idx = torch.randperm(len(src))
                src_shuffled = src[shuffle_idx]
                dst_shuffled = dst[shuffle_idx]
                rating_shuffled = rating[shuffle_idx]

                src_batches = src_shuffled.split(batch_size)
                dst_batches = dst_shuffled.split(batch_size)
                rating_batches = rating_shuffled.split(batch_size)

                seed_nodes = torch.cat(sum([[s, d] for s, d in zip(src_batches, dst_batches)], []))

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
                odd_even = True
                for s, d, r, nodeflow in zip(src_batches, dst_batches, rating_batches, sampler):
                    s.to(device)
                    d.to(device)

                    score = model.forward(nodeflow, s, d) if odd_even else model.forward(nodeflow, d, s)
                    odd_even = not odd_even
                    loss = ((score - r) ** 2).mean()
                    total_loss = total_loss + loss.item()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    scheduler.step()
                return total_loss / len(src_batches)

            loss = train(src, dst, rating)

            total_time = time.time() - start

            self.log.info('Epoch %2d/%2d: ' % (int(epoch + 1),
                                               epochs) + ' Training loss: %.4f' % loss + ' Train RMSE: %.4f ||' % train_rmse.item() + ' Eval Time: %.1f ||' % eval_total + '|| Time Taken: %.1f' % total_time)

        gc.collect()
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
                h.append(model.gcn.forward(nf))
            h = torch.cat(h).numpy()

        bias = model.node_biases.detach().numpy()
        assert len(bias) == total_users + total_items + 1
        mu = model.mu.detach().numpy()

        prediction_artifacts = {"vectors": h, "mu": mu,
                                "bias": bias,
                                "total_users": total_users}
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.log.info("Built Prediction Network, model params = %s", params)
        gc.collect()
        return prediction_artifacts

    def predict(self, user_item_pairs: List[Tuple[str, str]], clip=True) -> List[float]:
        from .gcn import get_score
        h = self.prediction_artifacts["vectors"]
        mu = self.prediction_artifacts["mu"]
        bias = self.prediction_artifacts["bias"]
        total_users = self.prediction_artifacts["total_users"]
        batch_size = 512

        uip = [(self.user_id_to_index[u] + 1 if u in self.user_id_to_index else 0,
                self.item_id_to_index[i] + 1 if i in self.item_id_to_index else 0) for u, i in user_item_pairs]

        assert np.sum(np.isnan(uip)) == 0

        user, item = zip(*uip)

        user = np.array(user)
        item = np.array(item) + total_users

        predictions = np.full(len(user), self.mu)
        for i in range(0, len(user), batch_size):
            s = user[i:i + batch_size]
            d = item[i:i + batch_size]

            res = get_score(s, d, mu, bias,
                            h[d], h[s], )
            predictions[i:i + batch_size] = res

        if clip:
            predictions = np.clip(predictions, self.rating_scale[0], self.rating_scale[1])
        return predictions

    def prepare_for_knn(self, user_vectors, item_vectors):
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
