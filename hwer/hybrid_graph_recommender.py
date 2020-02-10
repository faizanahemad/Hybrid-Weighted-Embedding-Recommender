import time
from typing import List, Dict, Tuple, Optional

import numpy as np
from more_itertools import flatten

from .logging import getLogger
from .random_walk import *
from .svdpp_hybrid import SVDppHybrid
from .utils import unit_length_violations


class HybridGCNRec(SVDppHybrid):
    def __init__(self, embedding_mapper: dict, knn_params: Optional[dict], rating_scale: Tuple[float, float],
                 n_content_dims: int = 32, n_collaborative_dims: int = 32, fast_inference: bool = False,
                 super_fast_inference: bool = False):
        super().__init__(embedding_mapper, knn_params, rating_scale, n_content_dims, n_collaborative_dims,
                         fast_inference, super_fast_inference)
        self.log = getLogger(type(self).__name__)
        self.cpu = int(os.cpu_count() / 2)

    def __word2vec_trainer__(self,
                             user_ids: List[str], item_ids: List[str],
                             user_item_affinities: List[Tuple[str, str, float]],
                             user_vectors: np.ndarray, item_vectors: np.ndarray,
                             user_id_to_index: Dict[str, int], item_id_to_index: Dict[str, int],
                             n_output_dims: int,
                             hyperparams: Dict):
        from gensim.models import Word2Vec
        walk_length = hyperparams["walk_length"] if "walk_length" in hyperparams else 40
        num_walks = hyperparams["num_walks"] if "num_walks" in hyperparams else 20
        window = hyperparams["window"] if "window" in hyperparams else 4
        iter = hyperparams["iter"] if "iter" in hyperparams else 2
        p = hyperparams["p"] if "p" in hyperparams else 0.5
        q = hyperparams["q"] if "q" in hyperparams else 0.5

        total_users = len(user_ids)
        total_items = len(item_ids)

        affinities = [(user_id_to_index[i], total_users + item_id_to_index[j], r) for i, j, r in user_item_affinities]
        _, present_items, _ = zip(*affinities)
        present_items = set(list(map(str, present_items)))
        affinities.extend([(i, i, 1) for i in range(total_users + total_items)])
        walker = Walker(read_edgelist(affinities), p=p, q=q)
        walker.preprocess_transition_probs()

        def sentences_generator():
            return walker.simulate_walks_generator_optimised(num_walks, walk_length=walk_length)

        gts = time.time()
        start = gts
        gt = 0.0

        # sentences = Parallel(n_jobs=self.cpu, prefer="threads")(delayed(lambda w: list(map(str, w)))(w) for w in sentences_generator())
        sentences = []
        for w in sentences_generator():
            sentences.append(list(map(str, w)))
        gt += time.time() - gts
        np.random.shuffle(sentences)
        w2v = Word2Vec(sentences, min_count=1,
                       size=self.n_collaborative_dims, window=window, workers=self.cpu, sg=1,
                       negative=10, max_vocab_size=None, iter=1)

        for _ in range(0):
            gts = time.time()
            sentences = []
            for w in sentences_generator():
                sentences.append(list(map(str, w)))
            # sentences = Parallel(n_jobs=self.cpu, prefer="threads")(delayed(lambda w: list(map(str, w)))(w) for w in sentences_generator())

            gt += time.time() - gts
            np.random.shuffle(sentences)
            w2v.train(sentences, total_examples=len(sentences), epochs=1)

        all_words = set(list(flatten(sentences)))
        nodes = set(walker.nodes)
        for i in item_ids:
            itm = str(total_users + self.item_id_to_index[i])
            if itm not in w2v.wv:
                print(i, self.item_id_to_index[i], itm, itm in all_words, itm in nodes, itm in present_items)
        user_vectors = np.array([w2v.wv[str(self.user_id_to_index[u])] for u in user_ids])
        item_vectors = np.array([w2v.wv[str(total_users + self.item_id_to_index[i])] for i in item_ids])
        self.log.info(
            "Trained Word2Vec with Node2Vec Walks, Walks Generation time = %.1f, Total Word2Vec Time = %.1f" % (
            gt, time.time() - start))
        return user_vectors, item_vectors

    def __node2vec_trainer__(self,
                             user_ids: List[str], item_ids: List[str],
                             user_item_affinities: List[Tuple[str, str, float]],
                             user_vectors: np.ndarray, item_vectors: np.ndarray,
                             user_id_to_index: Dict[str, int], item_id_to_index: Dict[str, int],
                             n_output_dims: int,
                             hyperparams: Dict):
        import networkx as nx
        from node2vec import Node2Vec
        walk_length = hyperparams["walk_length"] if "walk_length" in hyperparams else 60
        num_walks = hyperparams["num_walks"] if "num_walks" in hyperparams else 140
        window = hyperparams["window"] if "window" in hyperparams else 4
        iter = hyperparams["iter"] if "iter" in hyperparams else 5
        p = hyperparams["p"] if "p" in hyperparams else 0.5
        q = hyperparams["q"] if "q" in hyperparams else 0.5

        total_users = len(user_ids)
        total_items = len(item_ids)
        affinities = [(user_id_to_index[i], total_users + item_id_to_index[j], r) for i, j, r in user_item_affinities]
        affinities.extend([(i, i, 1) for i in range(total_users + total_items)])
        edges = [(str(x), str(y)) for x, y, w in affinities]
        graph = nx.DiGraph(edges)
        node2vec = Node2Vec(graph, dimensions=n_output_dims, p=p, q=q,
                            walk_length=walk_length, num_walks=num_walks, workers=self.cpu)
        n2v = node2vec.fit(window=window, min_count=1, batch_words=1000, iter=iter,
                           workers=self.cpu, sg=1, negative=10, max_vocab_size=None, )
        user_vectors = np.array([n2v.wv[str(self.user_id_to_index[u])] for u in user_ids])
        item_vectors = np.array([n2v.wv[str(total_users + self.item_id_to_index[i])] for i in item_ids])
        return user_vectors, item_vectors

    def __node2vec_triplet_trainer__(self,
                                     user_ids: List[str], item_ids: List[str],
                                     user_item_affinities: List[Tuple[str, str, float]],
                                     user_vectors: np.ndarray, item_vectors: np.ndarray,
                                     user_id_to_index: Dict[str, int], item_id_to_index: Dict[str, int],
                                     n_output_dims: int,
                                     hyperparams: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.log.debug(
            "Start Training User-Item Affinities, n_users = %s, n_items = %s, n_samples = %s, in_dims = %s, out_dims = %s",
            len(user_ids), len(item_ids), len(user_item_affinities), user_vectors.shape[1], n_output_dims)

        enable_node2vec = hyperparams["enable_node2vec"] if "enable_node2vec" in hyperparams else False
        enable_triplet_loss = hyperparams["enable_triplet_loss"] if "enable_triplet_loss" in hyperparams else False
        node2vec_params = hyperparams["node2vec_params"] if "node2vec_params" in hyperparams else {}

        assert np.sum(np.isnan(user_vectors)) == 0
        assert np.sum(np.isnan(item_vectors)) == 0
        w2v_user_vectors, w2v_item_vectors = user_vectors, item_vectors
        if enable_node2vec:
            w2v_user_vectors, w2v_item_vectors = self.__word2vec_trainer__(user_ids, item_ids, user_item_affinities,
                                                                           user_vectors, item_vectors,
                                                                           user_id_to_index,
                                                                           item_id_to_index,
                                                                           n_output_dims,
                                                                           node2vec_params)
        user_triplet_vectors, item_triplet_vectors = w2v_user_vectors, w2v_item_vectors

        if enable_triplet_loss:
            user_triplet_vectors, item_triplet_vectors = super().__user_item_affinities_triplet_trainer__(user_ids,
                                                                                                          item_ids,
                                                                                                          user_item_affinities,
                                                                                                          w2v_user_vectors,
                                                                                                          w2v_item_vectors,
                                                                                                          user_id_to_index,
                                                                                                          item_id_to_index,
                                                                                                          n_output_dims,
                                                                                                          hyperparams)
        return w2v_user_vectors, w2v_item_vectors, user_triplet_vectors, item_triplet_vectors

    def __get_triplet_gcn_model__(self, n_content_dims, n_collaborative_dims, gcn_layers,
                                  conv_depth, network_width,
                                  gcn_dropout, g_train, triplet_vectors, margin):
        from .gcn import GraphSAGETripletEmbedding, GraphSageWithSampling
        self.log.info("Getting Triplet Model for GCN")
        model = GraphSAGETripletEmbedding(GraphSageWithSampling(n_content_dims, n_collaborative_dims,
                                                                gcn_layers, gcn_dropout, False, g_train, triplet_vectors),
                                          margin)
        return model

    def __user_item_affinities_triplet_trainer__(self,
                                                 user_ids: List[str], item_ids: List[str],
                                                 user_item_affinities: List[Tuple[str, str, float]],
                                                 user_vectors: np.ndarray, item_vectors: np.ndarray,
                                                 user_id_to_index: Dict[str, int], item_id_to_index: Dict[str, int],
                                                 n_output_dims: int,
                                                 hyperparams: Dict) -> Tuple[np.ndarray, np.ndarray]:
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
        margin = hyperparams["margin"] if "margin" in hyperparams else 0.5
        gcn_kernel_l2 = hyperparams["gcn_kernel_l2"] if "gcn_kernel_l2" in hyperparams else 0.0
        enable_node2vec = hyperparams["enable_node2vec"] if "enable_node2vec" in hyperparams else False
        enable_gcn = hyperparams["enable_gcn"] if "enable_gcn" in hyperparams else False
        conv_depth = hyperparams["conv_depth"] if "conv_depth" in hyperparams else 1
        network_width = hyperparams["network_width"] if "network_width" in hyperparams else 128
        node2vec_params = hyperparams["node2vec_params"] if "node2vec_params" in hyperparams else {}

        assert np.sum(np.isnan(user_vectors)) == 0
        assert np.sum(np.isnan(item_vectors)) == 0

        w2v_user_vectors, w2v_item_vectors, user_triplet_vectors, item_triplet_vectors = self.__node2vec_triplet_trainer__(
            user_ids, item_ids, user_item_affinities,
            user_vectors, item_vectors,
            user_id_to_index,
            item_id_to_index,
            n_output_dims,
            hyperparams)

        if not enable_gcn:
            return user_triplet_vectors, item_triplet_vectors

        triplet_vectors = np.concatenate(
            (np.zeros((1, user_triplet_vectors.shape[1])), user_triplet_vectors, item_triplet_vectors))
        from .gcn import build_dgl_graph
        import torch
        import torch.nn.functional as F
        import dgl
        triplet_vectors = torch.FloatTensor(triplet_vectors)

        total_users = len(user_ids)
        edge_list = [(user_id_to_index[u], total_users + item_id_to_index[i], r) for u, i, r in user_item_affinities]
        graph_user_vectors, graph_item_vectors = user_vectors, item_vectors
        if enable_node2vec:
            graph_user_vectors = np.concatenate((user_vectors, w2v_user_vectors), axis=1)
            graph_item_vectors = np.concatenate((item_vectors, w2v_item_vectors), axis=1)
        g_train = build_dgl_graph(edge_list, len(user_ids) + len(item_ids),
                                  np.concatenate((graph_user_vectors, graph_item_vectors)))
        g_train.readonly()
        n_content_dims = graph_user_vectors.shape[1]
        model = self.__get_triplet_gcn_model__(n_content_dims, self.n_collaborative_dims, gcn_layers,
                                               conv_depth, network_width,
                                               gcn_dropout, g_train, triplet_vectors, margin)
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

                eval_start_time = time.time()
                with torch.no_grad():
                    h = []
                    for nf in sampler:
                        h.append(model.gcn.forward(nf))
                    h = torch.cat(h)

                    score = torch.zeros(len(src))
                    for i in range(0, len(src), batch_size):
                        s = src[i:i + batch_size]
                        d = dst[i:i + batch_size]
                        n = neg[i:i + batch_size]

                        h_src = h[s]
                        h_dst = h[d]
                        h_neg = h[n]
                        d_a_b = 1.0 - (h_src * h_dst).sum(1)
                        d_a_c = 1.0 - (h_src * h_neg).sum(1)
                        res = F.relu(d_a_b + margin - d_a_c)
                        score[i:i + batch_size] = res
                    train_rmse = (score ** 2).mean().sqrt()
                eval_total = time.time() - eval_start_time

                model.train()
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
                return total_loss / len(src_batches), train_rmse, eval_total

            if epoch % 2 == 1:
                loss, train_rmse, eval_total = train(src, dst, neg)
            else:
                # Reverse Training
                loss, train_rmse, eval_total = train(dst, src, neg)

            total_time = time.time() - start
            self.log.info('Epoch %2d/%2d: ' % (int(epoch + 1),
                                               gcn_epochs) + ' Training loss: %.4f' % loss.item() + ' Training RMSE: %.4f' % train_rmse.item() + '|| Generator Time: %.1f' % total_gen + ' Eval Time: %.1f' % eval_total + ' Time Taken: %.1f' % total_time)

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
        from .gcn import build_dgl_graph, GraphSageWithSampling, GraphSAGERecommenderImplicit, get_score
        import torch
        import dgl
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
        enable_implicit = hyperparams["enable_implicit"] if "enable_implicit" in hyperparams else False
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
        model = GraphSAGERecommenderImplicit(
            GraphSageWithSampling(n_content_dims, self.n_collaborative_dims, network_depth, dropout, False, g_train),
            mu, biases, zeroed_indices=zeroed_indices)
        opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=kernel_l2, momentum=0.9, nesterov=True)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=epochs,
                                                        steps_per_epoch=int(
                                                            np.ceil(len(user_item_affinities) / batch_size)),
                                                        div_factor=50, final_div_factor=50)
        # opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=kernel_l2)
        user_item_affinities = [(user_id_to_index[u] + 1, item_id_to_index[i] + 1, r) for u, i, r in
                                user_item_affinities]
        src, dst, rating = zip(*user_item_affinities)

        src = torch.LongTensor(src)
        dst = torch.LongTensor(dst) + total_users
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

                # Compute Train RMSE
                score = torch.zeros(len(src))
                for i in range(0, len(src), batch_size):
                    s = src[i:i + batch_size]
                    d = dst[i:i + batch_size]
                    #

                    res = get_score(s, d, model.mu, model.node_biases, h[d], h[s])
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
                for s, d, r, nodeflow in zip(src_batches, dst_batches, rating_batches, sampler):
                    score = model.forward(nodeflow, s, d)
                    # r = r + torch.randn(r.shape)
                    loss = ((score - r) ** 2).mean()
                    total_loss = total_loss + loss.item()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    scheduler.step()
                return total_loss / len(src_batches)

            if epoch % 2 == 1:
                loss = train(src, dst, rating)
            else:
                loss = train(dst, src, rating)

            total_time = time.time() - start

            self.log.info('Epoch %2d/%2d: ' % (int(epoch + 1),
                                               epochs) + ' Training loss: %.4f' % loss + ' Train RMSE: %.4f ||' % train_rmse.item() + ' Eval Time: %.1f ||' % eval_total + '|| Time Taken: %.1f' % total_time)

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
                                "total_users": total_users,
                                "batch_size": batch_size}
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.log.info("Built Prediction Network, model params = %s", params)
        return prediction_artifacts

    def predict(self, user_item_pairs: List[Tuple[str, str]], clip=True) -> List[float]:
        from .gcn import get_score
        h = self.prediction_artifacts["vectors"]
        mu = self.prediction_artifacts["mu"]
        bias = self.prediction_artifacts["bias"]
        total_users = self.prediction_artifacts["total_users"]
        batch_size = self.prediction_artifacts["batch_size"]
        batch_size = max(512, batch_size)

        if self.fast_inference:
            return self.fast_predict(user_item_pairs)

        if self.super_fast_inference:
            return self.super_fast_predict(user_item_pairs)

        uip = [(self.user_id_to_index[u] + 1 if u in self.user_id_to_index else 0,
                self.item_id_to_index[i] + 1 if i in self.item_id_to_index else 0) for u, i in user_item_pairs]

        assert np.sum(np.isnan(uip)) == 0

        user, item = zip(*uip)

        user = np.array(user).astype(int)
        item = np.array(item).astype(int) + total_users

        score = np.zeros(len(user))
        for i in range(0, len(user), batch_size):
            s = user[i:i + batch_size]
            d = item[i:i + batch_size]

            res = get_score(s, d, mu, bias,
                            h[d], h[s], )
            score[i:i + batch_size] = res

        predictions = score
        predictions = np.array(predictions)
        assert len(predictions) == len(user_item_pairs)
        if clip:
            predictions = np.clip(predictions, self.rating_scale[0], self.rating_scale[1])
        return predictions
