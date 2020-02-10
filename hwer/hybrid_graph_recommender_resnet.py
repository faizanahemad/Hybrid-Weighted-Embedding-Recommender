import os
import time
from typing import List, Dict, Tuple, Optional
import torch
import numpy as np

from .hybrid_graph_recommender import HybridGCNRec
from .logging import getLogger


class HybridGCNRecResnet(HybridGCNRec):
    def __init__(self, embedding_mapper: dict, knn_params: Optional[dict], rating_scale: Tuple[float, float],
                 n_content_dims: int = 32, n_collaborative_dims: int = 32, fast_inference: bool = False,
                 super_fast_inference: bool = False):
        super().__init__(embedding_mapper, knn_params, rating_scale, n_content_dims, n_collaborative_dims,
                         fast_inference, super_fast_inference)
        self.log = getLogger(type(self).__name__)
        self.cpu = int(os.cpu_count() / 2)

    def __get_triplet_gcn_model__(self, n_content_dims, n_collaborative_dims,
                                  gcn_layers, conv_depth, network_width,
                                  gcn_dropout, g_train, triplet_vectors, margin,
                                  conv_arch, gaussian_noise):
        from .gcn_res import GraphSAGETripletEmbedding, GraphSageWithSampling
        self.log.info("Getting Triplet Model for GCN RESNET")
        model = GraphSAGETripletEmbedding(GraphSageWithSampling(n_content_dims, n_collaborative_dims, network_width,
                                                                gcn_layers, conv_depth, gcn_dropout, g_train, gaussian_noise,
                                                                triplet_vectors),
                                          margin)
        return model

    def __scorer__(self, h, src, dst, mu, node_biases,
                   src_to_dsts, dst_to_srcs, src_to_dsts_count, dst_to_srcs_count,
                   src_collaborative_vectors, dst_collaborative_vectors,
                   zeroed_indices, scorer, batch_size):
        with torch.no_grad():
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

                sv = src_collaborative_vectors[i:i + batch_size]
                dv = dst_collaborative_vectors[i:i + batch_size]
                #

                res, _, _, _ = scorer(s, d, mu, node_biases,
                                      h[d], s2d, s2dc, s2d_imp,
                                      h[s], d2s, d2sc, d2s_imp,
                                      zeroed_indices, sv, dv)
                score[i:i + batch_size] = res
        return score

    def __build_prediction_network__(self, user_ids: List[str], item_ids: List[str],
                                     user_item_affinities: List[Tuple[str, str, float]],
                                     user_content_vectors: np.ndarray, item_content_vectors: np.ndarray,
                                     user_vectors: np.ndarray, item_vectors: np.ndarray,
                                     user_id_to_index: Dict[str, int], item_id_to_index: Dict[str, int],
                                     rating_scale: Tuple[float, float], hyperparams: Dict):
        from .gcn_res import GraphSAGERecommenderImplicitResnet, GraphSageWithSampling
        from .gcn import build_dgl_graph
        import dgl
        self.log.debug(
            "Start Building Prediction Network, collaborative vectors shape = %s, content vectors shape = %s",
            (user_vectors.shape, item_vectors.shape), (user_content_vectors.shape, item_content_vectors.shape))

        lr = hyperparams["lr"] if "lr" in hyperparams else 0.001
        epochs = hyperparams["epochs"] if "epochs" in hyperparams else 15
        batch_size = hyperparams["batch_size"] if "batch_size" in hyperparams else 512
        padding_length = hyperparams["padding_length"] if "padding_length" in hyperparams else 100
        use_content = hyperparams["use_content"] if "use_content" in hyperparams else False
        kernel_l2 = hyperparams["kernel_l2"] if "kernel_l2" in hyperparams else 0.0
        network_depth = hyperparams["network_depth"] if "network_depth" in hyperparams else 3
        conv_depth = hyperparams["conv_depth"] if "conv_depth" in hyperparams else 1
        scorer_depth = hyperparams["scorer_depth"] if "scorer_depth" in hyperparams else 1
        network_width = hyperparams["network_width"] if "network_width" in hyperparams else 128
        dropout = hyperparams["dropout"] if "dropout" in hyperparams else 0.0
        implicit_reg = hyperparams["implicit_reg"] if "implicit_reg" in hyperparams else 0.0
        residual_reg = hyperparams["residual_reg"] if "residual_reg" in hyperparams else 0.0
        bias_reg = hyperparams["bias_reg"] if "bias_reg" in hyperparams else 0.0
        conv_arch = hyperparams["conv_arch"] if "conv_arch" in hyperparams else 1
        gaussian_noise = hyperparams["gaussian_noise"] if "gaussian_noise" in hyperparams else 0.0

        assert user_content_vectors.shape[1] == item_content_vectors.shape[1]
        assert user_vectors.shape[1] == item_vectors.shape[1]
        # For unseen users and items creating 2 mock nodes
        user_content_vectors = np.concatenate((np.zeros((1, user_content_vectors.shape[1])), user_content_vectors))
        item_content_vectors = np.concatenate((np.zeros((1, item_content_vectors.shape[1])), item_content_vectors))
        user_vectors = np.concatenate((np.zeros((1, user_vectors.shape[1])), user_vectors))
        item_vectors = np.concatenate((np.zeros((1, item_vectors.shape[1])), item_vectors))

        mu, user_bias, item_bias = self.__calculate_bias__(user_ids, item_ids, user_item_affinities, rating_scale)
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
        model = GraphSAGERecommenderImplicitResnet(
            GraphSageWithSampling(n_content_dims, self.n_collaborative_dims, network_width, network_depth, conv_depth,
                                  dropout, g_train, gaussian_noise),
            mu, biases, padding_length, zeroed_indices,
            self.n_collaborative_dims, network_width, scorer_depth, dropout, n_content_dims, self.n_collaborative_dims,
            batch_size)
        # opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=kernel_l2)

        opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=kernel_l2, momentum=0.9, nesterov=True)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=epochs,
                                                        steps_per_epoch=int(
                                                            np.ceil(len(user_item_affinities) / batch_size)),
                                                        div_factor=50, final_div_factor=50)

        generate_training_samples, gen_fn, ratings_count_by_user, ratings_count_by_item, user_item_list, item_user_list = self.__prediction_network_datagen__(
            user_ids, item_ids,
            user_item_affinities,
            user_content_vectors,
            item_content_vectors,
            user_vectors, item_vectors,
            user_id_to_index,
            item_id_to_index,
            rating_scale, batch_size, padding_length,
            0, True)
        user_item_affinities = [(user_id_to_index[u] + 1, item_id_to_index[i] + 1,
                                 ratings_count_by_user[user_id_to_index[u] + 1],
                                 ratings_count_by_item[item_id_to_index[i] + 1], r) for u, i, r in user_item_affinities]
        generator = generate_training_samples(user_item_affinities)

        src = []
        dst = []
        dst_to_srcs = []
        src_to_dsts = []
        src_to_dsts_count = []
        dst_to_srcs_count = []

        src_vectors = []
        dst_vectors = []

        rating = []
        start_gen = time.time()
        for (u, i, us, iis, nu, ni, ucv, uv, icv, iv), r in generator():
            src.append(u)
            dst.append(i)
            dst_to_srcs.append(us)
            src_to_dsts.append(iis)
            src_to_dsts_count.append(nu)
            dst_to_srcs_count.append(ni)
            src_vectors.append(uv)
            dst_vectors.append(iv)

            rating.append(r)
        total_gen = time.time() - start_gen

        src = torch.LongTensor(src)
        dst = torch.LongTensor(dst) + total_users
        dst_to_srcs = torch.LongTensor(dst_to_srcs)
        src_to_dsts = torch.LongTensor(src_to_dsts) + total_users
        src_to_dsts_count = torch.FloatTensor(src_to_dsts_count)
        dst_to_srcs_count = torch.FloatTensor(dst_to_srcs_count)

        src_vectors = torch.FloatTensor(src_vectors)
        dst_vectors = torch.FloatTensor(dst_vectors)

        rating = torch.DoubleTensor(rating)

        for epoch in range(epochs):
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

                score = self.__scorer__(h, src, dst, model.mu, model.node_biases,
                                        src_to_dsts, dst_to_srcs, src_to_dsts_count, dst_to_srcs_count,
                                        src_vectors, dst_vectors,
                                        zeroed_indices, model.scorer, batch_size)

                train_rmse = ((score - rating) ** 2).mean().sqrt()
            eval_total = time.time() - eval_start_time

            model.train()

            def train(src, dst, src_to_dsts, dst_to_srcs, src_to_dsts_count, dst_to_srcs_count, src_vectors,
                      dst_vectors):
                shuffle_idx = torch.randperm(len(src))
                src_shuffled = src[shuffle_idx]
                dst_shuffled = dst[shuffle_idx]
                dst_to_srcs_shuffled = dst_to_srcs[shuffle_idx]
                src_to_dsts_shuffled = src_to_dsts[shuffle_idx]
                src_to_dsts_count_shuffled = src_to_dsts_count[shuffle_idx]
                dst_to_srcs_count_shuffled = dst_to_srcs_count[shuffle_idx]

                src_vectors_shuffled = src_vectors[shuffle_idx]
                dst_vectors_shuffled = dst_vectors[shuffle_idx]

                rating_shuffled = rating[shuffle_idx]

                src_batches = src_shuffled.split(batch_size)
                dst_batches = dst_shuffled.split(batch_size)
                dst_to_srcs_batches = dst_to_srcs_shuffled.split(batch_size)
                src_to_dsts_batches = src_to_dsts_shuffled.split(batch_size)
                src_to_dsts_count_batches = src_to_dsts_count_shuffled.split(batch_size)
                dst_to_srcs_count_batches = dst_to_srcs_count_shuffled.split(batch_size)

                src_vectors_batches = src_vectors_shuffled.split(batch_size)
                dst_vectors_batches = dst_vectors_shuffled.split(batch_size)

                rating_batches = rating_shuffled.split(batch_size)

                seed_nodes = torch.cat(sum([[s, d, ] for s, d in zip(src_batches, dst_batches)], []))

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
                for s, d, d2s, s2d, s2dc, d2sc, sv, dv, r, nodeflow in zip(src_batches, dst_batches,
                                                                           dst_to_srcs_batches,
                                                                           src_to_dsts_batches,
                                                                           src_to_dsts_count_batches,
                                                                           dst_to_srcs_count_batches,
                                                                           src_vectors_batches,
                                                                           dst_vectors_batches,
                                                                           rating_batches, sampler):
                    score, bias_loss, residual_loss, implicit_loss = model.forward(nodeflow, s, d, s2d, s2dc, d2s, d2sc,
                                                                                   sv, dv)
                    loss = ((score - r) ** 2).mean()
                    loss += residual_reg * residual_loss + bias_reg * bias_loss + implicit_reg * implicit_loss
                    opt.zero_grad()
                    loss.backward()
                    total_loss = total_loss + loss.item()
                    opt.step()
                    scheduler.step()
                return total_loss / len(src_batches)

            if epoch % 2 == 1:
                loss = train(src, dst, src_to_dsts, dst_to_srcs, src_to_dsts_count, dst_to_srcs_count, src_vectors,
                             dst_vectors)
            else:
                loss = train(dst, src, dst_to_srcs, src_to_dsts, dst_to_srcs_count, src_to_dsts_count, dst_vectors,
                             src_vectors)

            total_time = time.time() - start

            self.log.info('Epoch %2d/%2d: ' % (int(epoch + 1),
                                               epochs) + ' Training loss: %.4f' % loss + ' Train RMSE: %.4f ||' % train_rmse.item() + '|| Eval Time: %.1f' % eval_total + '|| Time Taken: %.1f' % total_time)

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
            h = torch.cat(h)

        bias = model.node_biases
        assert len(bias) == total_users + total_items + 1
        mu = model.mu

        prediction_artifacts = {"vectors": h, "user_item_list": user_item_list,
                                "item_user_list": item_user_list, "mu": mu,
                                "bias": bias, "scorer": model.scorer,
                                "total_users": total_users,
                                "ratings_count_by_user": ratings_count_by_user, "padding_length": padding_length,
                                "ratings_count_by_item": ratings_count_by_item,
                                "batch_size": batch_size, "gen_fn": gen_fn, "zeroed_indices": zeroed_indices}
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.log.info("Built Prediction Network, model params = %s", params)
        return prediction_artifacts

    def predict(self, user_item_pairs: List[Tuple[str, str]], clip=True) -> List[float]:
        h = self.prediction_artifacts["vectors"]
        scorer = self.prediction_artifacts["scorer"]
        mu = self.prediction_artifacts["mu"]
        bias = self.prediction_artifacts["bias"]
        total_users = self.prediction_artifacts["total_users"]
        zeroed_indices = self.prediction_artifacts["zeroed_indices"]

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
            return self.super_fast_predict(user_item_pairs)

        uip = [(self.user_id_to_index[u] + 1 if u in self.user_id_to_index else 0,
                self.item_id_to_index[i] + 1 if i in self.item_id_to_index else 0,
                ratings_count_by_user[self.user_id_to_index[u] + 1 if u in self.user_id_to_index else 1],
                ratings_count_by_item[self.item_id_to_index[i] + 1 if i in self.item_id_to_index else 1]) for u, i in
               user_item_pairs]

        assert np.sum(np.isnan(uip)) == 0
        generator = generate_prediction_samples(uip)

        src = []
        dst = []
        dst_to_srcs = []
        src_to_dsts = []
        src_to_dsts_count = []
        dst_to_srcs_count = []

        src_vectors = []
        dst_vectors = []

        for u, i, us, iis, nu, ni, ucv, uv, icv, iv in generator():
            src.append(u)
            dst.append(i)
            dst_to_srcs.append(us)
            src_to_dsts.append(iis)
            src_to_dsts_count.append(nu)
            dst_to_srcs_count.append(ni)
            src_vectors.append(uv)
            dst_vectors.append(iv)

        src = torch.LongTensor(src)
        dst = torch.LongTensor(dst) + total_users
        dst_to_srcs = torch.LongTensor(dst_to_srcs)
        src_to_dsts = torch.LongTensor(src_to_dsts) + total_users
        src_to_dsts_count = torch.FloatTensor(src_to_dsts_count)
        dst_to_srcs_count = torch.FloatTensor(dst_to_srcs_count)

        src_vectors = torch.FloatTensor(src_vectors)
        dst_vectors = torch.FloatTensor(dst_vectors)

        score = self.__scorer__(h, src, dst, mu, bias,
                                src_to_dsts, dst_to_srcs, src_to_dsts_count, dst_to_srcs_count,
                                src_vectors, dst_vectors,
                                zeroed_indices, scorer, batch_size)

        predictions = score.detach().numpy()
        assert len(predictions) == len(user_item_pairs)
        if clip:
            predictions = np.clip(predictions, self.rating_scale[0], self.rating_scale[1])
        return predictions
