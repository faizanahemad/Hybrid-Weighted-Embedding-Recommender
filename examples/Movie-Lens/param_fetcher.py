import json
import os
import argparse
from pprint import pprint


def read_params(location, dataset, algo):
    import importlib
    pkg = importlib.import_module('best_params')
    if dataset == "100K":
        if algo == "gcn":
            return pkg.params_gcn_100K
        if algo == "gcn_retriever":
            return pkg.params_gcn_retriever_100K
        if algo == "gcn_ncf":
            return pkg.params_gcn_ncf_100K
    if dataset == "1M":
        if algo == "gcn":
            return pkg.params_gcn_1M
        if algo == "gcn_retriever":
            return pkg.params_gcn_retriever_1M
        if algo == "gcn_ncf":
            return pkg.params_gcn_ncf_1M
    if dataset == "20M":
        if algo == "gcn":
            pass
        if algo == "gcn_retriever":
            return pkg.params_gcn_retriever_20M
        if algo == "gcn_ncf":
            return pkg.params_gcn_ncf_20M


def fetch_content_params():
    return dict(n_dims=64, combining_factor=0.1, knn_params=knn_params)


def fetch_ncf_params(dataset, algo):
    p = read_params("best_params/", dataset, algo)
    p["knn_params"] = knn_params

    p["collaborative_params"]["user_item_params"]["verbose"] = verbose
    p["collaborative_params"]["prediction_network_params"]["verbose"] = verbose
    return p


def fetch_gcn_params(dataset, algo):
    p = read_params("best_params/", dataset, algo)
    p["knn_params"] = knn_params

    p["collaborative_params"]["user_item_params"]["verbose"] = verbose
    p["collaborative_params"]["prediction_network_params"]["verbose"] = verbose

    p["collaborative_params"]["user_item_params"]["enable_gcn"] = enable_gcn if "enable_gcn" not in \
                                                                                     p["collaborative_params"][
                                                                                         "user_item_params"] else \
        p["collaborative_params"]["user_item_params"]["enable_gcn"]

    p["collaborative_params"]["user_item_params"]["enable_svd"] = enable_svd if "enable_svd" not in \
                                                                                p["collaborative_params"][
                                                                                    "user_item_params"] else \
        p["collaborative_params"]["user_item_params"]["enable_svd"]
    return p


def fetch_retriever_params(dataset, algo):
    p = read_params("best_params/", dataset, algo)
    p["knn_params"] = knn_params

    p["collaborative_params"]["user_item_params"]["verbose"] = verbose

    p["collaborative_params"]["user_item_params"]["enable_gcn"] = enable_gcn if "enable_gcn" not in \
                                                                                     p["collaborative_params"][
                                                                                         "user_item_params"] else \
        p["collaborative_params"]["user_item_params"]["enable_gcn"]

    p["collaborative_params"]["user_item_params"]["enable_svd"] = enable_svd if "enable_svd" not in \
                                                                                          p["collaborative_params"][
                                                                                              "user_item_params"] else \
        p["collaborative_params"]["user_item_params"]["enable_svd"]
    return p


def get_best_params(dataset):

    hyperparameter_content = fetch_content_params()

    hyperparameters_gcn = fetch_gcn_params(dataset, "gcn")

    hyperparameters_gcn_retriever = fetch_retriever_params(dataset, "gcn_retriever")

    hyperparameters_ncf_retriever = fetch_ncf_params(dataset, "gcn_ncf")

    hyperparameters_surprise = {"svdpp": {"n_factors": 20, "n_epochs": 20, "reg_all": 0.1},
                                "svd": {"biased": True, "n_factors": 20},
                                "algos": ["svdpp"]}
    hyperparamters_dict = dict(gcn_hybrid=hyperparameters_gcn, gcn_ncf=hyperparameters_ncf_retriever,
                               gcn_retriever=hyperparameters_gcn_retriever, content_only=hyperparameter_content,
                               surprise=hyperparameters_surprise)
    return hyperparamters_dict


n_neighbors = 500
knn_params = dict(n_neighbors=n_neighbors, index_time_params={'M': 15, 'ef_construction': 200, })
enable_gcn = True
enable_svd = True
verbose = 2


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="Dataset name")
    ap.add_argument("--algo", required=True,
                    help="Algorithm")
    args = vars(ap.parse_args())
    dataset = args['dataset']
    from hwer.utils import str2bool
    algo = args['algo']
    pprint(read_params("best_params/", dataset, algo))
