from hwer.validation import *
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import argparse
from param_fetcher import get_best_params

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.options.display.width = 0
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import movielens_data_reader as mdr

enable_kfold = False


def optimisation_objective(hyperparameters, algo, report, objective, dataset):
    df_user, df_item, user_item_affinities, prepare_data_mappers, rating_scale = mdr.build_dataset(dataset)
    print("Total Samples Taken = %s, |Users| = %s |Items| = %s, Rating scale = %s" % (
        len(user_item_affinities), len(df_user.user.values), len(df_item.item.values), rating_scale))
    if not enable_kfold:
        train_affinities, validation_affinities = train_test_split(user_item_affinities, test_size=0.2,
                                                                   stratify=[u for u, i, r in user_item_affinities])

        _, results, _, _, _ = test_hybrid(train_affinities, validation_affinities, list(df_user.user.values),
                                          list(df_item.item.values), hyperparameters,
                                          prepare_data_mappers, rating_scale, algo,
                                          enable_error_analysis=False, enable_baselines=False)
        rmse, ndcg = results[0]['rmse'], results[0]['ndcg']
    else:
        X = np.array(user_item_affinities)
        y = np.array([u for u, i, r in user_item_affinities])
        skf = StratifiedKFold(n_splits=5)
        results = []
        step = 0
        for train_index, test_index in skf.split(X, y):
            train_affinities, validation_affinities = X[train_index], X[test_index]
            train_affinities = [(u, i, int(r)) for u, i, r in train_affinities]
            validation_affinities = [(u, i, int(r)) for u, i, r in validation_affinities]
            #
            _, res, _, _, _ = test_hybrid(train_affinities, validation_affinities, list(df_user.user.values),
                                          list(df_item.item.values), hyperparameters,
                                          prepare_data_mappers, rating_scale, algo,
                                          enable_error_analysis=False, enable_baselines=False)
            results.extend(res)
            rmse, ndcg = results[0]['rmse'], results[0]['ndcg']
            imv = rmse if objective == "rmse" else 1 - ndcg
            report(imv, step)
            step += 1

        results = pd.DataFrame.from_records(results)
        results = results.groupby(["algo"]).mean().reset_index()
        rmse, ndcg = results["rmse"].values[0], results["ndcg"].values[0]
    print("RMSE = %.4f, NDCG = %.4f" % (rmse, ndcg))
    return rmse, ndcg


def init_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--objective', type=str, default="rmse", metavar='N', choices=["rmse", "ndcg"],
                    help='')
    ap.add_argument('--algo', type=str, default="gcn_hybrid", metavar='N',
                    choices=["gcn_hybrid", "gcn_ncf", "svdpp_hybrid"],
                    help='')
    ap.add_argument('--dataset', type=str, default="100K", metavar='N',
                    choices=["100K", "1M", "20M"],
                    help='')
    ap.add_argument('--conv_arch', type=int, default=1, metavar='N',
                    choices=[1, 2, 3, 4, 5],
                    help='')
    args = vars(ap.parse_args())
    algo = args["algo"]
    objective = args["objective"]
    dataset = args["dataset"]
    conv_arch = int(args["conv_arch"])
    hyperparamters_dict = get_best_params(dataset, conv_arch)
    params = copy.deepcopy(hyperparamters_dict[algo])
    return params, dataset, objective, algo
