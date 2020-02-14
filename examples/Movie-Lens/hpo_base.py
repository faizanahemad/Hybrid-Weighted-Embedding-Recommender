from hwer.validation import *

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.options.display.width = 0
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import movielens_data_reader as mdr
from param_fetcher import fetch_svdpp_params, fetch_gcn_params, fetch_content_params

dataset = "100K"
df_user, df_item, user_item_affinities, prepare_data_mappers, rating_scale = mdr.build_dataset(dataset)
print("Total Samples Taken = %s, |Users| = %s |Items| = %s, Rating scale = %s" % (
    len(user_item_affinities), len(df_user.user.values), len(df_item.item.values), rating_scale))

hyperparameter_content = fetch_content_params()

hyperparameters_svdpp = fetch_svdpp_params(dataset)

hyperparameters_gcn = fetch_gcn_params(dataset, "gcn", 2)

hyperparameters_gcn_ncf = fetch_gcn_params(dataset, "gcn_ncf", 2)

hyperparamters_dict = dict(gcn_hybrid=hyperparameters_gcn,
                           content_only=hyperparameter_content,
                           gcn_ncf=hyperparameters_gcn_ncf,
                           svdpp_hybrid=hyperparameters_svdpp,)

