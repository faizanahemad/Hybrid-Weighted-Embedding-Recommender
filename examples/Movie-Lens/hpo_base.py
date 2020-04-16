from hwer.validation import *
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
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

enable_kfold = False


def optimisation_objective(hyperparameters, algo, report, dataset, test_method):
    df_user, df_item, user_item_affinities, prepare_data_mappers, rating_scale, ts = build_dataset(dataset, test_method=test_method)
    rmse, ndcg, ncf_ndcg = run_model_for_hpo(df_user, df_item, user_item_affinities, prepare_data_mappers, rating_scale,
                                             hyperparameters, algo, report,
                                             enable_kfold=enable_kfold)
    print("Algo = %s, RMSE = %.4f, NDCG = %.4f, NCF_NDCG = %.4f" % (algo, rmse, ndcg, ncf_ndcg))
    return rmse, ndcg, ncf_ndcg


def init_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--objective', type=str, default="rmse", metavar='N', choices=["rmse", "ndcg", "ncf_ndcg"],
                    help='')
    ap.add_argument('--algo', type=str, default="gcn_hybrid", metavar='N',
                    choices=["gcn_hybrid", "gcn_ncf", "gcn_retriever"],
                    help='')
    ap.add_argument('--dataset', type=str, default="100K", metavar='N',
                    choices=["100K", "1M", "20M"],
                    help='')
    ap.add_argument('--test_method', type=str, default="ncf", metavar='N',
                    choices=["ncf", "vae_cf", "stratified-split", "random-split"],
                    help='')
    args = vars(ap.parse_args())
    algo = args["algo"]
    objective = args["objective"]
    dataset = args["dataset"]
    test_method = args["test_method"]
    hyperparamters_dict = get_best_params(dataset)
    params = copy.deepcopy(hyperparamters_dict[algo])
    return params, dataset, objective, algo, test_method
