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

from movielens_data_reader import *
from param_fetcher import get_best_params
from hwer.utils import str2bool

# TODO: Make test bench and HPO bench Re-usable


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--algo', type=str, metavar='N', nargs='+',
                    choices=["gcn_hybrid", "gcn_ncf", "surprise", "gcn_retriever"],
                    help='')
    ap.add_argument('--dataset', type=str, default="100K", metavar='N',
                    choices=["100K", "1M", "20M"],
                    help='')
    ap.add_argument('--enable_baselines', type=str2bool, default=False, metavar='N',
                    help='')
    ap.add_argument('--test_method', type=str, default="ncf", metavar='N',
                    choices=["ncf", "vae_cf", "stratified-split", "random-split"],
                    help='')
    args = vars(ap.parse_args())
    algos = args["algo"]
    dataset = args["dataset"]
    enable_baselines = args["enable_baselines"]
    test_method = args["test_method"]
    hyperparamters_dict = get_best_params(dataset)

    df_user, df_item, user_item_affinities, prepare_data_mappers, rating_scale, ts = build_dataset(dataset, fold=1, test_method=test_method)
    #
    verbose = 2  # if os.environ.get("LOGLEVEL") in ["DEBUG", "INFO"] else 0

    print("Total Samples Taken = %s, |Users| = %s |Items| = %s, Rating scale = %s" % (
        len(user_item_affinities), len(df_user.user.values), len(df_item.item.values), rating_scale))
    run_models_for_testing(df_user, df_item, user_item_affinities,
                           prepare_data_mappers, rating_scale,
                           algos, hyperparamters_dict,
                           enable_error_analysis=False, enable_baselines=enable_baselines,
                           enable_kfold=False, display=True, provided_test_set=ts)
    for algo in algos:
        print("algo = %s" % algo, hyperparamters_dict[algo])
