# python hybrid_recommender_example.py --dataset 100K --conv_arch 5 --algo gcn_ncf --enable_kfold False

from hwer.validation import *
import argparse

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.options.display.width = 0
import warnings

warnings.filterwarnings('ignore')
import numpy as np

from movielens_data_reader import build_dataset
from param_fetcher import get_best_params
from hwer.utils import str2bool

# TODO: Make test bench and HPO bench Re-usable


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--algo', type=str, metavar='N', nargs='+',
                    choices=["gcn_ncf", "content"],
                    help='')
    ap.add_argument('--dataset', type=str, default="100K", metavar='N',
                    choices=["100K", "1M", "20M"],
                    help='')
    ap.add_argument('--retrieved_node_type', type=str, required=True, metavar='N',
                    help='')
    ap.add_argument('--test_method', type=str, default="ncf", metavar='N',
                    choices=["ncf", "vae_cf", "stratified-split", "random-split"],
                    help='')
    args = vars(ap.parse_args())
    algos = args["algo"]
    dataset = args["dataset"]
    retrieved_node_type = args["retrieved_node_type"]
    test_method = args["test_method"]
    hyperparamters_dict = get_best_params(dataset)

    nodes, edges, node_types, prepare_data_mappers = build_dataset(dataset, fold=1, test_method=test_method)
    #
    verbose = 2  # if os.environ.get("LOGLEVEL") in ["DEBUG", "INFO"] else 0

    print("Total Nodes = %s, Edges = %s, |Node Types| = %s" % (len(nodes), len(edges), len(node_types)))
    run_models_for_testing(nodes, edges, node_types, retrieved_node_type,
                           prepare_data_mappers,
                           algos, hyperparamters_dict,
                           display=True)
    for algo in algos:
        print("algo = %s" % algo, hyperparamters_dict[algo])
