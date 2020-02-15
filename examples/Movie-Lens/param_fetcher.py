import json
import os
import argparse
from pprint import pprint


def read_params(location, dataset, algo):
    import importlib
    pkg = importlib.import_module('best_params')
    loc = os.path.join(location, "%s_%s.json" %(algo, dataset))
    if dataset == "100K":
        if algo == "gcn":
            return pkg.params_gcn_100K
        if algo == "gcn_ncf":
            return pkg.params_gcn_ncf_100K
        if algo == "svdpp":
            return pkg.params_svdpp_100K


def fetch_content_params():
    return dict(n_dims=64, combining_factor=0.1, knn_params=knn_params)


def fetch_gcn_params(dataset, algo, conv_arch):
    p = read_params("best_params/", dataset, algo)
    p = p[conv_arch]
    p["knn_params"] = knn_params
    p["collaborative_params"]["user_item_params"]["conv_arch"] = conv_arch
    p["collaborative_params"]["prediction_network_params"]["conv_arch"] = conv_arch

    p["collaborative_params"]["user_item_params"]["verbose"] = verbose
    p["collaborative_params"]["prediction_network_params"]["verbose"] = verbose
    p["collaborative_params"]["prediction_network_params"]["use_content"] = use_content
    p["collaborative_params"]["user_item_params"]["enable_gcn"] = enable_gcn
    p["collaborative_params"]["user_item_params"]["enable_node2vec"] = enable_node2vec
    p["collaborative_params"]["user_item_params"]["enable_triplet_loss"] = enable_triplet_loss
    return p


def fetch_svdpp_params(dataset):
    p = read_params("best_params/", dataset, "svdpp")
    p["knn_params"] = knn_params
    p["collaborative_params"]["user_item_params"]["verbose"] = verbose
    p["collaborative_params"]["prediction_network_params"]["verbose"] = verbose
    p["collaborative_params"]["prediction_network_params"]["use_content"] = use_content
    return p


def get_best_params(dataset, gcn_conv_variant):

    hyperparameter_content = fetch_content_params()

    hyperparameters_svdpp = fetch_svdpp_params(dataset)

    hyperparameters_gcn = fetch_gcn_params(dataset, "gcn", gcn_conv_variant)

    hyperparameters_gcn_ncf = fetch_gcn_params(dataset, "gcn_ncf", gcn_conv_variant)

    hyperparameters_surprise = {"svdpp": {"n_factors": 20, "n_epochs": 20},
                                "svd": {"biased": True, "n_factors": 20},
                                "algos": ["svd"]}
    hyperparamters_dict = dict(gcn_hybrid=hyperparameters_gcn,
                               content_only=hyperparameter_content,
                               gcn_ncf=hyperparameters_gcn_ncf,
                               svdpp_hybrid=hyperparameters_svdpp, surprise=hyperparameters_surprise)
    return hyperparamters_dict


n_neighbors = 200
knn_params=dict(n_neighbors=n_neighbors, index_time_params={'M': 15, 'ef_construction': 200, })
enable_node2vec = True
enable_triplet_loss = True
use_content = True
enable_gcn = True
verbose = 2

datasets = ["100K", "1M", "20M"]
algos = ["gcn", "gcn_ncf", "svdpp"]

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="Dataset name")
    ap.add_argument("--algo", required=True,
                    help="Algorithm")
    args = vars(ap.parse_args())
    dataset = args['dataset']
    algo = args['algo']
    pprint(read_params("best_params/", dataset, algo))