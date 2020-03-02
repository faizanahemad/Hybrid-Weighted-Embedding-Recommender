from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials
import argparse
import copy
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from hpo_base import optimisation_objective, init_args
from param_fetcher import get_best_params
import numpy as np
from hwer.validation import *
import dill as pkl
import sys
import hyperopt

enable_kfold = False

TRIALS_FOLDER = 'hyperopt_trials'
NUMBER_TRIALS_PER_RUN = 1

# https://github.com/MilesCranmer/easy_distributed_hyperopt

def report(imv, step):
    pass


def build_params(args, objective, params):
    params = copy.deepcopy(params)
    if objective == "rmse":
        params["collaborative_params"]["prediction_network_params"]["lr"] = args["lr"]
        params["collaborative_params"]["prediction_network_params"]["epochs"] = int(args["epochs"])
        params["collaborative_params"]["prediction_network_params"]["kernel_l2"] = args["kernel_l2"]
        params["collaborative_params"]["prediction_network_params"]["batch_size"] = int(args["batch_size"])
        params["collaborative_params"]["prediction_network_params"]["conv_depth"] = int(args["conv_depth"])
        params["collaborative_params"]["prediction_network_params"]["gaussian_noise"] = args["gaussian_noise"]
        params["collaborative_params"]["prediction_network_params"]["network_depth"] = int(args["network_depth"])
        params["n_dims"] = int(args["n_dims"])
    else:
        params["collaborative_params"]["user_item_params"]["gcn_lr"] = args["gcn_lr"]
        params["collaborative_params"]["user_item_params"]["gcn_epochs"] = int(args["gcn_epochs"])
        params["collaborative_params"]["user_item_params"]["gaussian_noise"] = args["gaussian_noise"]
        params["collaborative_params"]["user_item_params"]["margin"] = args["margin"]

    return params


def run_trial(args):
    """Evaluate the model loss using the hyperparams in args
    :args: A dictionary containing all hyperparameters
    :returns: Dict with status and loss from cross-validation
    """

    hyperparams = build_params(args, objective, params)
    rmse, ndcg = optimisation_objective(hyperparams, algo, report, dataset)
    loss = rmse if objective == "rmse" else 1 - ndcg
    return {
        'status': 'fail' if np.isnan(loss) else 'ok',
        'loss': loss
    }


def define_search_space(objective, starting_params):
    prediction = starting_params["collaborative_params"]["prediction_network_params"]
    rmse_space = {
        'lr': hp.qlognormal("lr", np.log(prediction["lr"]),
                           0.5 * prediction["lr"],
                            0.05 * prediction["lr"]),
        'epochs': hp.quniform('epochs',
                              prediction["epochs"] - 20,
                              prediction["epochs"] + 20, 5),
        'kernel_l2': hp.choice('kernel_l2', [0.0, hp.qloguniform('kernel_l2_choice', np.log(1e-9), np.log(1e-6), 1e-9)]),
        'batch_size': hp.qloguniform('batch_size', np.log(512), np.log(2048), 512),
        'conv_depth': hp.quniform('conv_depth', 1, 6, 1),
        'gaussian_noise': hp.qlognormal('gaussian_noise', np.log(prediction["gaussian_noise"]),
                                       0.5 * prediction["gaussian_noise"], 0.005),
        'network_depth': hp.quniform('network_depth', 2, prediction['network_depth'] + 2, 1),
        'n_dims': hp.quniform('n_dims',
                              starting_params["n_dims"] - 32,
                              starting_params["n_dims"] + 64, 16),
    }

    embedding = starting_params["collaborative_params"]["user_item_params"]
    ndcg_space = {
        'gcn_lr': hp.qlognormal("gcn_lr", np.log(embedding["gcn_lr"]),
                           0.5 * embedding["gcn_lr"],
                                0.05 * embedding["gcn_lr"]),
        'gcn_epochs': hp.quniform('gcn_epochs',
                              embedding["gcn_epochs"] - 10,
                              embedding["gcn_epochs"] + 20, 5),
        'gaussian_noise': hp.qlognormal('gaussian_noise', np.log(embedding["gaussian_noise"]),
                                       0.5 * embedding["gaussian_noise"], 0.005),
        'margin': hp.quniform('margin', 0.8, 1.8, 0.2)
    }

    return rmse_space if objective == "rmse" else ndcg_space


def merge_trials(trials1, trials2_slice):
    """Merge two hyperopt trials objects
    :trials1: The primary trials object
    :trials2_slice: A slice of the trials object to be merged,
        obtained with, e.g., trials2.trials[:10]
    :returns: The merged trials object
    """
    max_tid = 0
    if len(trials1.trials) > 0:
        max_tid = max([trial['tid'] for trial in trials1.trials])

    for trial in trials1.trials:
        for key in trial['misc']['idxs'].keys():
            if len(trial['misc']['vals'][key]) == 0:
                trial['misc']['idxs'][key] = []

    for trial in trials2_slice:
        tid = trial['tid'] + max_tid + 1
        hyperopt_trial = Trials().new_trial_docs(
                tids=[None],
                specs=[None],
                results=[None],
                miscs=[None])
        hyperopt_trial[0] = trial
        hyperopt_trial[0]['tid'] = tid
        hyperopt_trial[0]['misc']['tid'] = tid
        for key in hyperopt_trial[0]['misc']['idxs'].keys():
            if len(hyperopt_trial[0]['misc']['vals'][key]) == 0:
                hyperopt_trial[0]['misc']['idxs'][key] = []
            else:
                hyperopt_trial[0]['misc']['idxs'][key] = [tid]
        trials1.insert_trial_docs(hyperopt_trial)
        trials1.refresh()
    return trials1


def load_trials(algo, dataset, objective, conv_arch):
    loaded_fnames = []
    trials = None
    path = TRIALS_FOLDER + '/%s_%s_%s_%s_*.pkl' % (algo, dataset, objective, conv_arch)
    for fname in glob.glob(path):
        trials_obj = pkl.load(open(fname, 'rb'))
        n_trials = trials_obj['n']
        trials_obj = trials_obj['trials']
        if len(loaded_fnames) == 0:
            trials = trials_obj
        else:
            print("Merging trials")
            trials = merge_trials(trials, trials_obj.trials[-n_trials:])

        loaded_fnames.append(fname)
    print("Loaded trials", len(loaded_fnames))
    return trials


def print_trial_details(trials):
    best_loss = np.inf
    best_trial = None
    vals = []
    for trial in trials:
        if trial['result']['status'] == 'ok':
            loss = trial['result']['loss']
            val = dict(copy.deepcopy(trial['misc']['vals']))
            val['loss'] = loss
            vals.append(val)
            if loss < best_loss:
                best_loss = loss
                best_trial = trial
    print(pd.DataFrame.from_records(vals))
    print("Best = ", best_loss, best_trial['misc']['vals'])


if __name__ == '__main__':
    params, dataset, objective, algo = init_args()
    conv_arch = params["collaborative_params"]["prediction_network_params"]["conv_arch"]
    # Run new hyperparameter trials until killed
    while True:
        np.random.seed()

        # Load up all runs:
        import glob

        trials = load_trials(algo, dataset, objective, conv_arch)
        if trials is None:
            trials = Trials()
        else:
            print_trial_details(trials)

        n = NUMBER_TRIALS_PER_RUN
        try:
            best = fmin(run_trial,
                        space=define_search_space(objective, params),
                        algo=tpe.suggest,
                        max_evals=n + len(trials.trials),
                        trials=trials,
                        verbose=1,
                        rstate=np.random.RandomState(np.random.randint(1, 10 ** 6))
                        )
        except hyperopt.exceptions.AllTrialsFailed:
            continue

        print('current best', best)
        hyperopt_trial = Trials()

        # Merge with empty trials dataset:
        save_trials = merge_trials(hyperopt_trial, trials.trials[-n:])
        new_fname = TRIALS_FOLDER + '/%s_%s_%s_%s_' % (algo, dataset, objective, conv_arch) + str(np.random.randint(0, sys.maxsize)) + '.pkl'
        pkl.dump({'trials': save_trials, 'n': n}, open(new_fname, 'wb'))




