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
        if algo == "gcn_ncf":
            return pkg.params_gcn_ncf_100K
    if dataset == "1M":
        if algo == "gcn":
            return pkg.params_gcn_1M
        if algo == "gcn_ncf":
            return pkg.params_gcn_ncf_1M
    if dataset == "20M":
        if algo == "gcn":
            pass
        if algo == "gcn_ncf":
            return pkg.params_gcn_ncf_20M


def fetch_content_params():
    return dict(n_dims=64)


def fetch_ncf_params(dataset, algo):
    p = read_params("best_params/", dataset, algo)
    p["collaborative_params"]["verbose"] = verbose
    p["link_prediction_params"]["verbose"] = verbose
    return p


def fetch_gcn_params(dataset, algo):
    p = read_params("best_params/", dataset, algo)
    p["collaborative_params"]["verbose"] = verbose
    return p


def get_best_params(dataset):

    hyperparameter_content = fetch_content_params()

    hyperparameters_gcn = fetch_gcn_params(dataset, "gcn")

    hyperparameters_ncf_retriever = fetch_ncf_params(dataset, "gcn_ncf")

    hyperparamters_dict = dict(gcn=hyperparameters_gcn, gcn_ncf=hyperparameters_ncf_retriever,
                               content=hyperparameter_content)
    return hyperparamters_dict


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
