# python hybrid_recommender_example.py --dataset 100K --conv_arch 5 --algo gcn_ncf --enable_kfold False

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

import movielens_data_reader as mdr
from param_fetcher import get_best_params
from hwer.utils import str2bool

# TODO: Make test bench and HPO bench Re-usable


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--algo', type=str, metavar='N', nargs='+',
                    choices=["gcn_hybrid", "gcn_ncf", "svdpp_hybrid", "surprise"],
                    help='')
    ap.add_argument('--dataset', type=str, default="100K", metavar='N',
                    choices=["100K", "1M", "20M"],
                    help='')
    ap.add_argument('--conv_arch', type=int, default=1, metavar='N',
                    choices=[1, 2, 3, 4, 5],
                    help='')
    ap.add_argument('--enable_kfold', type=str2bool, default=False, metavar='N',
                    help='')
    ap.add_argument('--enable_baselines', type=str2bool, default=False, metavar='N',
                    help='')
    args = vars(ap.parse_args())
    algos = args["algo"]
    dataset = args["dataset"]
    conv_arch = int(args["conv_arch"])
    enable_kfold = args["enable_kfold"]
    enable_baselines = args["enable_baselines"]
    hyperparamters_dict = get_best_params(dataset, conv_arch)

    df_user, df_item, user_item_affinities, prepare_data_mappers, rating_scale = mdr.build_dataset(dataset)
    #
    enable_error_analysis = False
    verbose = 2  # if os.environ.get("LOGLEVEL") in ["DEBUG", "INFO"] else 0

    print("Total Samples Taken = %s, |Users| = %s |Items| = %s, Rating scale = %s" % (
        len(user_item_affinities), len(df_user.user.values), len(df_item.item.values), rating_scale))

    if not enable_kfold:
        train_affinities, validation_affinities = train_test_split(user_item_affinities, test_size=0.2,
                                                                   stratify=[u for u, i, r in user_item_affinities])
        print("Train Length = ", len(train_affinities))
        print("Validation Length =", len(validation_affinities))
        recs, results, user_rating_count_metrics = test_once(train_affinities, validation_affinities,
                                                             list(df_user.user.values),
                                                             list(df_item.item.values),
                                                             hyperparamters_dict,
                                                             prepare_data_mappers, rating_scale, algos,
                                                             enable_error_analysis=enable_error_analysis,
                                                             enable_baselines=enable_baselines)
        results = display_results(results)
        user_rating_count_metrics = user_rating_count_metrics.sort_values(["algo", "user_rating_count"])
        # print(user_rating_count_metrics)
        # user_rating_count_metrics.to_csv("algo_user_rating_count_%s.csv" % dataset, index=False)
        results.reset_index().to_csv("overall_results_%s.csv" % (dataset), index=False)
        visualize_results(results, user_rating_count_metrics, train_affinities, validation_affinities)
    else:
        X = np.array(user_item_affinities)
        y = np.array([u for u, i, r in user_item_affinities])
        skf = StratifiedKFold(n_splits=5)
        results = []
        user_rating_count_metrics = pd.DataFrame([],
                                                 columns=["algo", "user_rating_count", "rmse", "mae", "map",
                                                          "train_rmse",
                                                          "train_mae"])
        for train_index, test_index in skf.split(X, y):
            train_affinities, validation_affinities = X[train_index], X[test_index]
            train_affinities = [(u, i, int(r)) for u, i, r in train_affinities]
            validation_affinities = [(u, i, int(r)) for u, i, r in validation_affinities]
            #
            recs, res, ucrms = test_once(train_affinities, validation_affinities, list(df_user.user.values),
                                         list(df_item.item.values),
                                         hyperparamters_dict,
                                         prepare_data_mappers, rating_scale, algos,
                                         enable_error_analysis=False, enable_baselines=enable_baselines)

            user_rating_count_metrics = pd.concat((user_rating_count_metrics, ucrms))
            results.extend(res)

        user_rating_count_metrics = user_rating_count_metrics.groupby(
            ["algo", "user_rating_count"]).mean().reset_index()
        print(pd.DataFrame.from_records(results))
        print("#" * 80)
        display_results(results)
        visualize_results(results, user_rating_count_metrics, train_affinities, validation_affinities)
    for algo in algos:
        print("algo = %s" % algo, hyperparamters_dict[algo])
