import optuna
import copy
from hpo_base import optimisation_objective, init_args
from optuna.pruners import PercentilePruner
import numpy as np
from hwer.validation import *


def get_reporter(trial):
    def report(imv, step):
        trial.report(imv, step)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return report


def rmse_objective(trial):
    network_depth = trial.suggest_int('network_depth', 2, 3)
    conv_depth = trial.suggest_int('conv_depth', 1, 3)
    epochs = trial.suggest_discrete_uniform('epochs', 50, 100, 10)
    gaussian_noise = trial.suggest_discrete_uniform('gaussian_noise', 0.1, 0.35, 0.0125)
    lr = trial.suggest_loguniform('lr', 1e-2, 7e-2)
    kernel_l2 = trial.suggest_loguniform('kernel_l2', 1e-9, 1e-7)
    n_dims = trial.suggest_discrete_uniform('n_dims', 48, 96, 16)
    params = copy.deepcopy(parameters)

    params["n_dims"] = int(n_dims)
    params["collaborative_params"]["prediction_network_params"]["lr"] = lr
    params["collaborative_params"]["prediction_network_params"]["gaussian_noise"] = gaussian_noise
    params["collaborative_params"]["prediction_network_params"]["epochs"] = int(epochs)
    params["collaborative_params"]["prediction_network_params"]["network_depth"] = int(network_depth)
    params["collaborative_params"]["prediction_network_params"]["conv_depth"] = int(conv_depth)
    params["collaborative_params"]["prediction_network_params"]["kernel_l2"] = kernel_l2
    report = get_reporter(trial)

    rmse, _ = optimisation_objective(params, algo, report, objective, dataset)
    return rmse


def ndcg_objective(trial):
    gcn_lr = trial.suggest_loguniform('gcn_lr', 3e-4, 5e-3)
    gcn_epochs = trial.suggest_discrete_uniform('gcn_epochs', 10, 40, 5)
    gcn_layers = trial.suggest_int('gcn_layers', 1, 3)
    gcn_kernel_l2 = trial.suggest_loguniform('gcn_kernel_l2', 1e-9, 1e-6)
    conv_arch = trial.suggest_categorical('conv_arch', [1])
    gaussian_noise = trial.suggest_discrete_uniform('gaussian_noise', 0.05, 0.2, 0.0125)
    margin = trial.suggest_discrete_uniform('margin', 1.0, 2.0, 0.2)
    n_dims = trial.suggest_discrete_uniform('n_dims', 48, 96, 16)
    params = copy.deepcopy(parameters)
    params["n_dims"] = int(n_dims)
    params["collaborative_params"]["user_item_params"]["gcn_lr"] = gcn_lr
    params["collaborative_params"]["user_item_params"]["gcn_epochs"] = gcn_epochs
    params["collaborative_params"]["user_item_params"]["gcn_layers"] = gcn_layers
    params["collaborative_params"]["user_item_params"]["gcn_kernel_l2"] = gcn_kernel_l2
    params["collaborative_params"]["user_item_params"]["conv_arch"] = conv_arch
    params["collaborative_params"]["user_item_params"]["gaussian_noise"] = gaussian_noise
    params["collaborative_params"]["user_item_params"]["margin"] = margin
    report = get_reporter(trial)
    _, ndcg = optimisation_objective(params, algo, report, objective, dataset)
    return 1 - ndcg


if __name__ == '__main__':
    parameters, dataset, objective, algo = init_args()

    storage = 'sqlite:///%s_%s_%s.db' % (algo, dataset, objective)
    study = optuna.create_study(study_name=algo, storage=storage,
                                load_if_exists=True,
                                pruner=PercentilePruner(percentile=25.0, n_startup_trials=2, n_warmup_steps=1, interval_steps=1))
    print("Previous Trials...")
    df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
    print(df)

    if objective == "rmse":
        study.optimize(rmse_objective, n_trials=10)
    if objective == "ndcg":
        study.optimize(ndcg_objective, n_trials=10)

    print("Current Trial Results...")
    print(study.best_params, study.best_value)
    print(study.best_value)
    print(study.best_trial)
    df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
    print(df)
