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


class HybridGCNRec(SVDppHybrid):
    def __init__(self, embedding_mapper: dict, knn_params: Optional[dict], rating_scale: Tuple[float, float],
                 n_content_dims: int = 32, n_collaborative_dims: int = 32, fast_inference: bool = False,
                 super_fast_inference: bool = False):
        super().__init__(embedding_mapper, knn_params, rating_scale, n_content_dims, n_collaborative_dims,
                         fast_inference, super_fast_inference)
        self.log = getLogger(type(self).__name__)
        assert n_collaborative_dims % 2 == 0
        assert n_content_dims == n_collaborative_dims
        self.cpu = int(os.cpu_count() / 2)

    def __word2vec_trainer__(self,
                             user_ids: List[str], item_ids: List[str],
                             user_item_affinities: List[Tuple[str, str, float]],
                             user_id_to_index: Dict[str, int], item_id_to_index: Dict[str, int],
                             n_output_dims: int,
                             hyperparams: Dict):
        self.log.info("__word2vec_trainer__: Training Node2Vec base Random Walks with Word2Vec...")
        import gc
        from gensim.models import Word2Vec
        walk_length = 10
        num_walks = hyperparams["num_walks"] if "num_walks" in hyperparams else 20
        iter = hyperparams["iter"] if "iter" in hyperparams else 3
        p = 1.0
        q = hyperparams["q"] if "q" in hyperparams else 0.5

        total_users = len(user_ids)
        total_items = len(item_ids)

        affinities = [(user_id_to_index[i], total_users + item_id_to_index[j], r) for i, j, r in user_item_affinities]
        affinities.extend([(i, i, 1) for i in range(total_users + total_items)])
        walker = Walker(read_edgelist(affinities), p=p, q=q)
        walker.preprocess_transition_probs()

        sfile = "sentences-%s.txt" % str(np.random.randint(int(1e8)))
        from more_itertools import chunked
        from .utils import save_list_per_line

        def sentences_generator():
            g = walker.simulate_walks_generator_optimised(num_walks, walk_length=walk_length)
            save_list_per_line([[""]], sfile, 'w')
            total_sentences = 0
            for w in chunked(g, 128):
                total_sentences += len(w)
                save_list_per_line(w, sfile, 'a')
            return total_sentences

        gts = time.time()
        start = gts
        gt = 0.0
        sentences_generator()
        gt += time.time() - gts
        w2v = Word2Vec(corpus_file=sfile, min_count=1, sample=0.0002,
                       size=int(self.n_collaborative_dims/2), window=3, workers=self.cpu, sg=1,
                       negative=10, max_vocab_size=None, iter=2)
        # w2v.init_sims(True)

        w2v2 = Word2Vec(corpus_file=sfile, min_count=1, sample=0.0002,
                       size=int(self.n_collaborative_dims/2), window=5, workers=self.cpu, sg=1,
                       negative=10, max_vocab_size=None, iter=2)
        # w2v2.init_sims(True)

        for _ in range(iter):
            gts = time.time()
            total_examples = sentences_generator()

            gc.collect()
            gt += time.time() - gts
            # w2v.init_sims(True)
            w2v.train(corpus_file=sfile, epochs=2, total_examples=total_examples, total_words=len(walker.nodes))

            # w2v2.init_sims(True)
            w2v2.train(corpus_file=sfile, epochs=2, total_examples=total_examples, total_words=len(walker.nodes))

            gc.collect()
        uv1 = np.array([w2v.wv[str(self.user_id_to_index[u])] for u in user_ids])
        iv1 = np.array([w2v.wv[str(total_users + self.item_id_to_index[i])] for i in item_ids])

        uv2 = np.array([w2v2.wv[str(self.user_id_to_index[u])] for u in user_ids])
        iv2 = np.array([w2v2.wv[str(total_users + self.item_id_to_index[i])] for i in item_ids])

        from .utils import unit_length, unit_length_violations
        user_vectors = np.concatenate((uv1, uv2), axis=1)
        item_vectors = np.concatenate((iv1, iv2), axis=1)

        self.log.info(
            "Trained Word2Vec with Node2Vec Walks, Walks Generation time = %.1f, Total Word2Vec Time = %.1f" % (
            gt, time.time() - start))
        self.log.info("Node2Vec Unit Length violation: users = %s, items = %s"
                       % (unit_length_violations(user_vectors, axis=1), unit_length_violations(item_vectors, axis=1)))
        return user_vectors, item_vectors

    def __get_triplet_gcn_model__(self, n_content_dims, n_collaborative_dims, gcn_layers,
                                  conv_depth, g_train, triplet_vectors, margin,
                                  conv_arch, gaussian_noise):
        from .gcn import GraphSAGETripletEmbedding, GraphSageWithSampling, GraphSAGENegativeSamplingEmbedding
        self.log.info("Getting Triplet Model for GCN")
        model = GraphSAGETripletEmbedding(GraphSageWithSampling(n_content_dims, n_collaborative_dims,
                                                                gcn_layers, g_train,
                                                                conv_arch, gaussian_noise, conv_depth, triplet_vectors))
        return model

    def __user_item_affinities_triplet_trainer_data_gen_fn__(self, user_ids, item_ids,
                                                             user_id_to_index,
                                                             item_id_to_index,
                                                             affinities: List[Tuple[str, str, float]],
                                                             hyperparams):

        walk_length = 3
        num_walks = hyperparams["num_walks"] if "num_walks" in hyperparams else 150
        p = 0.25
        q = hyperparams["q"] if "q" in hyperparams else 0.25

        total_users = len(user_ids)
        total_items = len(item_ids)
        affinities = [(user_id_to_index[i], total_users + item_id_to_index[j], r) for i, j, r in affinities]
        affinities.extend([(i, i, 1) for i in range(total_users + total_items)])
        walker = Walker(read_edgelist(affinities), p=p, q=q)
        walker.preprocess_transition_probs()

        from more_itertools import distinct_combinations, chunked, grouper, interleave_longest
        from joblib import Parallel, delayed

        def sentences_generator():
            g = walker.simulate_walks_generator_optimised(num_walks, walk_length=walk_length)

            def iter_walk(walk):
                data_pairs = []
                walk = list(set(walk))
                np.random.shuffle(walk)
                for c in grouper(walk, 2, walk[0]):
                    data_pairs.append(((c[0], c[1], np.random.randint(0, total_users + total_items)), 0))
                return data_pairs

            def node2vec_generator():
                for walks in chunked(g, 2048):
                    data_points = Parallel(n_jobs=4)(delayed(iter_walk)(walk) for walk in walks)
                    for d in data_points:
                        for i in d:
                            yield i

            def affinities_generator():
                np.random.shuffle(affinities)
                for i, j, r in affinities:
                    yield (i, j, np.random.randint(0, total_users + total_items)), 0
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

        lr = hyperparams["lr"] if "lr" in hyperparams else 0.001
        gcn_lr = hyperparams["gcn_lr"] if "gcn_lr" in hyperparams else 0.1
        epochs = hyperparams["epochs"] if "epochs" in hyperparams else 1
        gcn_epochs = hyperparams["gcn_epochs"] if "gcn_epochs" in hyperparams else 1
        gcn_layers = hyperparams["gcn_layers"] if "gcn_layers" in hyperparams else 5
        gcn_dropout = hyperparams["gcn_dropout"] if "gcn_dropout" in hyperparams else 0.0
        batch_size = hyperparams["batch_size"] if "batch_size" in hyperparams else 512
        gcn_batch_size = hyperparams["gcn_batch_size"] if "gcn_batch_size" in hyperparams else 512
        verbose = hyperparams["verbose"] if "verbose" in hyperparams else 1
        margin = hyperparams["margin"] if "margin" in hyperparams else 1.0
        gcn_kernel_l2 = hyperparams["gcn_kernel_l2"] if "gcn_kernel_l2" in hyperparams else 0.0
        enable_node2vec = hyperparams["enable_node2vec"] if "enable_node2vec" in hyperparams else False
        enable_gcn = hyperparams["enable_gcn"] if "enable_gcn" in hyperparams else False
        conv_depth = hyperparams["conv_depth"] if "conv_depth" in hyperparams else 1
        network_width = hyperparams["network_width"] if "network_width" in hyperparams else 128
        node2vec_params = hyperparams["node2vec_params"] if "node2vec_params" in hyperparams else {}
        conv_arch = hyperparams["conv_arch"] if "conv_arch" in hyperparams else 1
        gaussian_noise = hyperparams["gaussian_noise"] if "gaussian_noise" in hyperparams else 0.0

        assert np.sum(np.isnan(user_vectors)) == 0
        assert np.sum(np.isnan(item_vectors)) == 0

        import gc
        gc.collect()

        user_triplet_vectors, item_triplet_vectors = user_vectors, item_vectors
        if enable_node2vec:
            w2v_user_vectors, w2v_item_vectors = self.__word2vec_trainer__(user_ids, item_ids, user_item_affinities,
                                                                           user_id_to_index,
                                                                           item_id_to_index,
                                                                           n_output_dims,
                                                                           node2vec_params)
            self.w2v_user_vectors = w2v_user_vectors
            self.w2v_item_vectors = w2v_item_vectors
            user_triplet_vectors, item_triplet_vectors = w2v_user_vectors, w2v_item_vectors

        if not enable_gcn or gcn_epochs <= 0:
            return user_triplet_vectors, item_triplet_vectors

        triplet_vectors = None
        if enable_node2vec:
            triplet_vectors = np.concatenate(
                (np.zeros((1, user_triplet_vectors.shape[1])), user_triplet_vectors, item_triplet_vectors))
            triplet_vectors = torch.FloatTensor(triplet_vectors)

        total_users = len(user_ids)
        edge_list = [(user_id_to_index[u], total_users + item_id_to_index[i], r) for u, i, r in user_item_affinities]
        g_train = build_dgl_graph(edge_list, len(user_ids) + len(item_ids),
                                  np.concatenate((user_vectors, item_vectors)))
        g_train.readonly()
        n_content_dims = user_vectors.shape[1]
        model = self.__get_triplet_gcn_model__(n_content_dims, self.n_collaborative_dims, gcn_layers,
                                               conv_depth, g_train, triplet_vectors, margin,
                                               conv_arch, gaussian_noise)
        opt = torch.optim.Adam(model.parameters(), lr=gcn_lr, weight_decay=gcn_kernel_l2)
        generate_training_samples = self.__user_item_affinities_triplet_trainer_data_gen_fn__(user_ids, item_ids,
                                                                                              user_id_to_index,
                                                                                              item_id_to_index,
                                                                                              user_item_affinities,
                                                                                              hyperparams)

        model.train()
        gc.collect()
        from more_itertools import chunked
        for epoch in range(gcn_epochs):
            start = time.time()
            loss = 0.0
            for big_batch in chunked(generate_training_samples(), gcn_batch_size * 10):
                src, dst, neg = [], [], []
                for (u, v, w), r in big_batch:
                    src.append(u)
                    dst.append(v)
                    neg.append(w)
            #

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
                    model.train()
                    seed_nodes = torch.cat(sum([[s, d, n] for s, d, n in zip(src_batches, dst_batches, neg_batches)], []))
                    sampler = dgl.contrib.sampling.NeighborSampler(
                        g_train,  # the graph
                        gcn_batch_size * 2,  # number of nodes to compute at a time, HACK 2
                        10,  # number of neighbors for each node
                        gcn_layers,  # number of layers in GCN
                        seed_nodes=seed_nodes,  # list of seed nodes, HACK 2
                        prefetch=True,  # whether to prefetch the NodeFlows
                        add_self_loop=True,  # whether to add a self-loop in the NodeFlows, HACK 1
                        shuffle=False,  # whether to shuffle the seed nodes.  Should be False here.
                        num_workers=self.cpu,
                    )

                    # Training
                    total_loss = 0.0
                    odd_even = True
                    for s, d, n, nodeflow in zip(src_batches, dst_batches, neg_batches, sampler):
                        score = model.forward(nodeflow, s, d, n) if odd_even else model.forward(nodeflow, d, s, n)
                        odd_even = not odd_even
                        loss = score.mean()
                        total_loss = total_loss + loss

                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                    return total_loss / len(src_batches)

                loss += train(src, dst, neg)

            total_time = time.time() - start
            self.log.info('Epoch %2d/%2d: ' % (int(epoch + 1),
                                               gcn_epochs) + ' Training loss: %.4f' % loss.item() + ' || Time Taken: %.1f' % total_time)

            #
        model.eval()
        sampler = dgl.contrib.sampling.NeighborSampler(
            g_train,
            gcn_batch_size,
            10,
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
        from .gcn import build_dgl_graph, GraphSageWithSampling, GraphSAGERecommenderImplicit, get_score
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
        bias_regularizer = hyperparams["bias_regularizer"] if "bias_regularizer" in hyperparams else 0.0
        use_content = hyperparams["use_content"] if "use_content" in hyperparams else False
        kernel_l2 = hyperparams["kernel_l2"] if "kernel_l2" in hyperparams else 0.0
        network_depth = hyperparams["network_depth"] if "network_depth" in hyperparams else 3
        dropout = hyperparams["dropout"] if "dropout" in hyperparams else 0.0
        conv_arch = hyperparams["conv_arch"] if "conv_arch" in hyperparams else 1
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
        if use_content:
            try:
                self.w2v_user_vectors = np.concatenate(
                    (np.zeros((1, self.w2v_user_vectors.shape[1])), self.w2v_user_vectors))
                self.w2v_item_vectors = np.concatenate(
                    (np.zeros((1, self.w2v_item_vectors.shape[1])), self.w2v_item_vectors))

                user_vectors = np.concatenate((self.w2v_user_vectors, user_vectors, user_content_vectors), axis=1)
                item_vectors = np.concatenate((self.w2v_item_vectors, item_vectors, item_content_vectors), axis=1)
                self.w2v_user_vectors = None
                self.w2v_item_vectors = None
                del self.w2v_user_vectors
                del self.w2v_item_vectors
            except AttributeError as e:
                user_vectors = np.concatenate((user_vectors, user_content_vectors), axis=1)
                item_vectors = np.concatenate((item_vectors, item_content_vectors), axis=1)
        else:
            user_vectors = np.zeros((user_vectors.shape[0], 1))
            item_vectors = np.zeros((item_vectors.shape[0], 1))

        edge_list = [(user_id_to_index[u] + 1, total_users + item_id_to_index[i] + 1, r) for u, i, r in
                     user_item_affinities]
        biases = np.concatenate(([0.0], user_bias, item_bias))
        import gc
        gc.collect()
        g_train = build_dgl_graph(edge_list, total_users + total_items, np.concatenate((user_vectors, item_vectors)))
        n_content_dims = user_vectors.shape[1]
        g_train.readonly()
        zeroed_indices = [0, 1, total_users + 1]
        model = GraphSAGERecommenderImplicit(
            GraphSageWithSampling(n_content_dims, self.n_collaborative_dims, network_depth, g_train,
                                  conv_arch, gaussian_noise, conv_depth),
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
                    # r = r + torch.randn(r.shape)
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

        if self.fast_inference:
            return self.fast_predict(user_item_pairs)

        if self.super_fast_inference:
            return self.super_fast_predict(user_item_pairs)

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

    @staticmethod
    def persist(model, path: str = "."):
        import hnswlib
        logger.info("save_model:: Saving Model...")
        user_knn_path = os.path.join(path, 'user_knn.bin')
        item_knn_path = os.path.join(path, 'item_knn.bin')

        model.user_knn.save_index(user_knn_path)
        model.item_knn.save_index(item_knn_path)
        logger.debug("save_model:: Saved Serialized HNSW KNN indices.")

        model.user_knn = None
        model.item_knn = None
        model_path = os.path.join(path, 'model.pth')
        with open(model_path, 'wb') as f:
            dill.dump(model, f)
        logger.info("save_model:: Saved Model to %s" % model_path)

    @staticmethod
    def load(path: str = "."):
        import hnswlib
        logger.info("load_model:: Loading Model...")
        logger.debug("Load Dir Contents = %s" % os.listdir(path))
        user_knn_path = os.path.join(path, 'user_knn.bin')
        item_knn_path = os.path.join(path, 'item_knn.bin')
        with open(os.path.join(path, 'model.pth'), 'rb') as f:
            model = dill.load(f)
        logger.debug("load_model:: Loaded Main Model without HNSW KNN indices.")

        user_knn = hnswlib.Index(space='cosine', dim=model.n_output_dims)
        user_knn.load_index(user_knn_path)
        item_knn = hnswlib.Index(space='cosine', dim=model.n_output_dims)
        item_knn.load_index(item_knn_path)
        logger.debug("load_model:: Loaded Serialized HNSW KNN indices.")
        model.user_knn = user_knn
        model.item_knn = item_knn
        logger.info("load_model:: Loaded Full Model.")
        return model
