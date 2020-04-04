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


class NCF(nn.Module):
    def __init__(self, feature_size, depth, gaussian_noise,
                 content, total_users, total_items):
        super(NCF, self).__init__()
        noise = GaussianNoise(gaussian_noise)
        self.node_emb = nn.Embedding(total_users + total_items, feature_size)
        self.content_emb = nn.Embedding.from_pretrained(torch.tensor(content, dtype=torch.float), freeze=True)
        nn.init.normal_(self.node_emb.weight, std=1 / (10 * feature_size))

        wc1 = nn.Linear(content.shape[1] * 2, feature_size * 2)
        init_fc(wc1, 'xavier_uniform_', 'leaky_relu', 0.1)
        wc2 = nn.Linear(feature_size * 2, feature_size)
        init_fc(wc2, 'xavier_uniform_', 'leaky_relu', 0.1)

        self.cem = nn.Sequential(wc1, nn.LeakyReLU(0.1), noise, wc2, nn.LeakyReLU(0.1))

        w1 = nn.Linear(feature_size * 3, feature_size * 2)
        init_fc(w1, 'xavier_uniform_', 'leaky_relu', 0.1)
        layers = [noise, w1, nn.LeakyReLU(negative_slope=0.1)]

        for _ in range(depth):
            wx = nn.Linear(feature_size * 2, feature_size * 2)
            init_fc(wx, 'xavier_uniform_', 'leaky_relu', 0.1)
            layers.extend([noise, wx, nn.LeakyReLU(negative_slope=0.1)])

        w_out = nn.Linear(feature_size * 2, 1)
        init_fc(w_out, 'xavier_uniform_', 'sigmoid', 0.1)
        self.w_out = nn.Sequential(w_out)
        self.W = nn.Sequential(*layers)

    def forward(self, src, dst):
        h_src = self.node_emb(src)
        h_dst = self.node_emb(dst)

        hc_src = self.content_emb(src)
        hc_dst = self.content_emb(dst)
        hc = torch.cat([hc_src, hc_dst], 1)
        hc = self.cem(hc)
        vec = torch.cat([h_src, h_dst, hc], 1)
        out = self.W(vec)
        out = self.w_out(out).flatten()
        out = F.sigmoid(out)
        return out


class GcnNCF(HybridGCNRec):
    def __init__(self, embedding_mapper: dict, knn_params: Optional[dict], rating_scale: Tuple[float, float],
                 n_content_dims: int = 32, n_collaborative_dims: int = 32, fast_inference: bool = False,
                 super_fast_inference: bool = False):
        super().__init__(embedding_mapper, knn_params, rating_scale, n_content_dims, n_collaborative_dims,
                         fast_inference, super_fast_inference)
        self.log = getLogger(type(self).__name__)
        assert n_collaborative_dims % 2 == 0
        self.cpu = int(os.cpu_count() / 2)

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
            user_vectors = np.zeros((user_vectors.shape[0], 1))
            item_vectors = np.zeros((item_vectors.shape[0], 1))

        import gc
        gc.collect()
        model = NCF(self.n_collaborative_dims, 2, gaussian_noise,
                    np.concatenate((user_vectors, item_vectors)),
                    total_users, total_items)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=kernel_l2)
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

            ns = 1
            src_neg = torch.randint(0, total_users, (len(src) * ns,))
            dst_neg = torch.randint(total_users+1, total_users+total_items, (len(src) * ns,))
            weights_neg = torch.tensor([0.0] * len(src) * ns)
            src = torch.cat((src, src_neg), 0)
            dst = torch.cat((dst, dst_neg), 0)
            weights = torch.cat((weights, weights_neg), 0)

            shuffle_idx = torch.randperm(len(src))
            src = src[shuffle_idx]
            dst = dst[shuffle_idx]
            weights = weights[shuffle_idx]
            weights = weights.clamp(min=1e-4, max=1-1e-4)
            return src, dst, weights

        src, dst, rating = get_samples()

        model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.log.info("Built Prediction Network, model params = %s, examples = %s, model = \n%s", params, len(src), model)

        for epoch in range(epochs):
            gc.collect()
            start = time.time()
            model.train()

            def train(src, dst, rating):
                import gc
                gc.collect()

                src_batches = src.split(batch_size)
                dst_batches = dst.split(batch_size)
                rating_batches = rating.split(batch_size)

                # Training
                total_loss = 0.0
                for s, d, r in zip(src_batches, dst_batches, rating_batches):
                    score = model(s, d)
                    loss = ((score - r) ** 2)
                    # loss = -1 * (r * torch.log(score) + (1-r)*torch.log(1-score))
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

            total_time = time.time() - start

            self.log.info('Epoch %2d/%2d: ' % (int(epoch + 1),
                                               epochs) + ' Training loss: %.4f' % loss + '|| Time Taken: %.1f' % total_time)

        gc.collect()
        model.eval()

        prediction_artifacts = {"model": model,
                                "total_users": total_users}
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.log.info("Built Prediction Network, model params = %s", params)
        gc.collect()
        return prediction_artifacts

    def predict(self, user_item_pairs: List[Tuple[str, str]], clip=True) -> List[float]:
        from .gcn import get_score
        model = self.prediction_artifacts["model"]
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
                scores = model.forward(u, i)
                scores = list(scores.numpy())
                predictions.extend(scores)
        return predictions

    def __build_svd_model__(self, user_ids: List[str], item_ids: List[str],
                            user_item_affinities: List[Tuple[str, str, float]],
                            user_id_to_index: Dict[str, int], item_id_to_index: Dict[str, int],
                            rating_scale: Tuple[float, float], **svd_params):
        pass