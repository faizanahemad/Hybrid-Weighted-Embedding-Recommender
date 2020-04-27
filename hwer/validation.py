import pandas as pd
from bidict import bidict
import random

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.options.display.width = 0
import warnings
import copy
from collections import defaultdict
import operator
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, accuracy_score

warnings.filterwarnings('ignore')
from typing import List, Dict, Any, Tuple, Set
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import describe
from .recommendation_base import RecommendationBase, NodeType, Node, Edge, FeatureName

from .utils import average_precision, reciprocal_rank, ndcg, binary_ndcg, recall, binary_ndcg_v2

# Link Prediction accuracy with same proportion negatives as positives


def model_get_topk_knn(model, anchors: List[Node], node_type: NodeType) -> Dict[Node, List[Tuple[Node, float]]]:
    predictions = defaultdict(list)
    for u in anchors:
        p = model.find_closest_neighbours(node_type, u)
        predictions[u] = p
    return predictions


model_get_topk = model_get_topk_knn


def link_prediction_accuracy(model, nodes: List[Node], train_edges: List[Edge], validation_edges: List[Edge]):

    train_set = [(e.src, e.dst) for e in train_edges] + list(zip(random.choices(nodes, k=len(train_edges)), random.choices(nodes, k=len(train_edges))))
    train_labels = [1] * len(train_edges) + [0] * len(train_edges)

    validation_set = [(e.src, e.dst) for e in validation_edges] + list(zip(random.choices(nodes, k=len(validation_edges)), random.choices(nodes, k=len(validation_edges))))
    validation_labels = [1] * len(validation_edges) + [0] * len(validation_edges)

    train_predictions = np.array(model.predict(train_set))
    validation_predictions = np.array(model.predict(validation_set))

    lp_train_ap = average_precision_score(train_labels, train_predictions)
    lp_val_ap = average_precision_score(validation_labels, validation_predictions)

    lp_train_precision, lp_train_recall, _, _ = precision_recall_fscore_support(train_labels, train_predictions >= 0.5, average='binary')
    lp_val_precision, lp_val_recall, _, _ = precision_recall_fscore_support(validation_labels, validation_predictions >= 0.5,
                                                                                average='binary')
    lp_train_accuracy = accuracy_score(train_labels, train_predictions >= 0.5)
    lp_val_accuracy = accuracy_score(validation_labels, validation_predictions >= 0.5)

    results = dict(lp_train_ap=lp_train_ap, lp_val_ap=lp_val_ap, lp_train_precision=lp_train_precision,
                   lp_train_recall=lp_train_recall, lp_val_precision=lp_val_precision, lp_val_recall=lp_val_recall,
                   lp_train_accuracy=lp_train_accuracy, lp_val_accuracy=lp_val_accuracy)
    return results


def ncf_eval(model, train_edges: List[Edge], validation_edges: List[Edge], item_list: List[Node]):
    item_list = set(item_list)
    interactions = defaultdict(set)
    for u, i, _ in train_edges:
        interactions[u].add(i)
    for u, i, _ in validation_edges:
        interactions[u].add(i)

    def calc(edges, interactions, item_list):
        user_test_item = {}
        actual = {}
        for u, i, _ in edges:
            user_test_item[u] = [i, *random.sample(item_list - interactions[u], 100)]
            actual[u] = i

        for u, items in user_test_item.items():
            it = list(zip(items, model.predict([(u, i) for i in items])))
            it = list(sorted(it, key=operator.itemgetter(1), reverse=True))
            user_test_item[u], _ = zip(*it[:10])

        hr = []
        ndcg = []
        for u, i in actual.items():
            preds = user_test_item[u]
            hr.append(i in preds)
            ndcg.append(binary_ndcg_v2([i], preds))
        return np.mean(hr), np.mean(ndcg)
    ncf_hr, ncf_ndcg = calc(validation_edges, interactions, item_list)

    return {"ncf_hr": ncf_hr, "ncf_ndcg": ncf_ndcg}


def extraction_efficiency(model, train_edges: List[Edge], validation_edges: List[Edge], get_topk, node_type: NodeType):
    validation_users = list(set([u for u, i, r in validation_edges]))
    train_users = list(set([u for u, i, r in train_edges]))
    validation_items = list(set([i for u, i, r in validation_edges]))
    train_items = list(set([i for u, i, r in train_edges]))
    all_users = list(set(train_users + validation_users))
    all_items = list(set(validation_items + train_items))
    all_items = list(filter(lambda x: x.node_type == node_type, all_items))
    train_uid = defaultdict(set)
    items_extracted_length = []
    s = time.time()
    predictions = get_topk(model, all_users, node_type)
    e = time.time()
    pred_time = e - s
    for u, i, r in train_edges:
        train_uid[u].add(i)

    train_actuals = defaultdict(list)
    train_actuals_score_dict = defaultdict(dict)
    for u, i, r in train_edges:
        train_actuals[u].append((i, r))
        train_actuals_score_dict[u][i] = r

    for u, i in train_actuals.items():
        remaining_items = list(sorted(i, key=operator.itemgetter(1), reverse=True))
        remaining_items = [i for i, r in remaining_items]
        train_actuals[u] = remaining_items

    train_predictions = dict()
    predictions_10 = dict()
    predictions_20 = dict()
    predictions_50 = dict()
    predictions_100 = dict()
    for u, i in predictions.items():
        remaining_items = list(sorted(i, key=operator.itemgetter(1), reverse=True))
        remaining_items = [i for i, r in remaining_items]
        filtered_items = list(filter(lambda x: x not in train_uid[u], remaining_items))
        train_predictions[u] = remaining_items[:100]
        predictions_10[u] = filtered_items[:10]
        predictions_20[u] = filtered_items[:20]
        predictions_50[u] = filtered_items[:50]
        predictions_100[u] = filtered_items[:100]

    from more_itertools import flatten
    train_diversity = len(set(list(flatten(list(train_predictions.values())))))/len(all_items)
    diversity = len(set(list(flatten(list(predictions_100.values()))))) / len(all_items)

    train_mrr = np.mean([reciprocal_rank(train_actuals[u], train_predictions[u]) for u in train_users])
    train_ndcg = np.mean([ndcg(train_actuals_score_dict[u], train_predictions[u]) for u in train_users])
    train_binary_ndcg = np.mean([binary_ndcg(train_actuals_score_dict[u], train_predictions[u]) for u in train_users])
    train_recall = np.mean([recall(train_actuals_score_dict[u], train_predictions[u]) for u in train_users])

    validation_actuals = defaultdict(list)
    for u, i, r in validation_edges:
        validation_actuals[u].append((i, r))

    validation_actuals_score_dict = defaultdict(dict)
    for u, i in validation_actuals.items():
        remaining_items = list(sorted(i, key=operator.itemgetter(1), reverse=True))
        remaining_items = list(filter(lambda x: x[0] not in train_uid[u], remaining_items))
        validation_actuals_score_dict[u] = dict(remaining_items)
        remaining_items = [i for i, r in remaining_items]
        items_extracted_length.append(len(remaining_items))
        validation_actuals[u] = remaining_items

    mrr = np.mean([reciprocal_rank(validation_actuals[u], predictions_100[u]) for u in validation_users])
    val_ndcg = np.mean([ndcg(validation_actuals_score_dict[u], predictions_100[u]) for u in validation_users])
    val_binary_ndcg = np.mean([binary_ndcg(validation_actuals_score_dict[u], predictions_100[u]) for u in validation_users])
    val_recall = np.mean([recall(validation_actuals_score_dict[u], predictions_100[u]) for u in validation_users])

    val_ndcg_10 = np.mean([ndcg(validation_actuals_score_dict[u], predictions_10[u]) for u in validation_users])
    val_binary_ndcg_10 = np.mean([binary_ndcg(validation_actuals_score_dict[u], predictions_10[u]) for u in validation_users])
    val_recall_10 = np.mean([recall(validation_actuals_score_dict[u], predictions_10[u]) for u in validation_users])

    val_recall_20 = np.mean([recall(validation_actuals_score_dict[u], predictions_20[u]) for u in validation_users])

    ncf_metrics = ncf_eval(model, train_edges, validation_edges, all_items)

    metrics = {"retrieval_time": pred_time,
               "recall@100": val_recall,
               "ndcg_b@100": val_binary_ndcg,
               "ndcg_b@10": val_binary_ndcg_10,
               "recall@10": val_recall_10,
               "diversity": diversity, **ncf_metrics}
    return {"actuals": validation_actuals, "predictions": predictions_100,
            "train_actuals": train_actuals, "train_predictions": train_predictions,
            "train_actuals_score_dict": train_actuals_score_dict,
            "validation_actuals_score_dict": validation_actuals_score_dict, "metrics": metrics}


def test_algorithm(train_affinities: List[Edge], validation_affinities: List[Edge],
                   nodes: List[Node], node_types: Set[NodeType], hyperparameters,
                   get_data_mappers, algo, node_type: NodeType):
    from . import GcnNCF, ContentRecommendation
    embedding_mapper, node_data = get_data_mappers()
    kwargs = dict(hyperparameters=copy.deepcopy(hyperparameters))
    algo_map = dict(gcn_ncf=GcnNCF, content=ContentRecommendation)
    recsys = algo_map[algo](embedding_mapper=embedding_mapper,
                                node_types=node_types,
                                n_dims=hyperparameters["n_dims"])

    start = time.time()
    _ = recsys.fit(nodes, train_affinities, node_data, **kwargs)
    end = time.time()
    total_time = end - start

    rnode = Node(list(node_types)[0], "eifjcchchbniufclvfdugvhnftdvjculhjitjihuncce")
    rnode2 = Node(list(node_types)[0], "eifjcchchbnirdjknkrvtfkbfurvjdfjhllbddtbvicb")
    default_preds = recsys.predict([(train_affinities[0].src, rnode),
                                    (train_affinities[0].src, train_affinities[0].dst),
                                    (rnode, rnode2),
                                    (rnode2, train_affinities[0].src)])
    print("Default Preds = ", default_preds)
    assert np.sum(np.isnan(default_preds)) == 0

    res2 = {"algo": algo, "time": total_time}
    predictions, actuals, stats = get_prediction_details(recsys, nodes, train_affinities,
                                                         validation_affinities,
                                                         model_get_topk, node_type)
    res2.update(stats)
    results = [res2]

    return recsys, results, predictions, actuals


def test_multiple_algorithms(train_affinities, validation_affinities, nodes: List[Node], node_types: Set[NodeType],
                             hyperparamters_dict, get_data_mappers, algos, node_type: NodeType):
    results = []
    recs = []
    assert len(algos) > 0
    algos = set(algos)
    assert len(algos - {"content", "gcn_ncf"}) == 0
    for algo in algos:
        hyperparameters = hyperparamters_dict[algo]
        rec, res, _, _ = test_algorithm(train_affinities,
                                        validation_affinities, nodes, node_types, hyperparameters,
                                        get_data_mappers, algo, node_type)
        results.extend(res)
        recs.append(rec)

    return recs, results


def display_results(results: List[Dict[str, Any]]):
    df = pd.DataFrame.from_records(results)
    df = df.groupby(['algo']).mean()
    df['time'] = df['time'].apply(lambda s: str(datetime.timedelta(seconds=s)))
    time = df['retrieval_time']
    df['retrieval_time'] = df['retrieval_time'].apply(lambda s: str(datetime.timedelta(seconds=s)))
    from tabulate import tabulate
    from more_itertools import chunked
    col_lists = list(chunked(df.columns, 8))
    for c in col_lists:
        print(tabulate(df[c], headers='keys', tablefmt='psql'))
    df['retrieval_time'] = time
    return df


def get_prediction_details(recsys, nodes: List[Node], train_affinities: List[Edge], validation_affinities: List[Edge],
                           model_get_topk, node_type: NodeType):
    def get_details(recsys, affinities):
        predictions = np.array(recsys.predict([(u, i) for u, i, r in affinities]))
        if np.sum(np.isnan(predictions)) > 0:
            count = np.sum(np.isnan(predictions))
            raise AssertionError("Encountered Nan Predictions = %s" % count,
                                 np.array(affinities)[np.isnan(predictions)])

        actuals = np.array([r for u, i, r in affinities])
        return predictions, actuals

    predictions, actuals = get_details(recsys, validation_affinities)
    train_predictions, _ = get_details(recsys, train_affinities)
    ex_ee = extraction_efficiency(recsys, train_affinities, validation_affinities, model_get_topk, node_type)
    lp_res = link_prediction_accuracy(recsys, nodes, train_affinities, validation_affinities)
    lp_res.update(ex_ee["metrics"])
    return predictions, actuals, lp_res


def run_model_for_hpo(nodes: List[Node], edges: List[Tuple[Edge, bool]],
                      node_types: Set[NodeType], retrieved_node_type: NodeType,
                      prepare_data_mappers,
                      hyperparameters, algo,):
    ndcg, ncf_ndcg = run_models_for_testing(nodes, edges, node_types, retrieved_node_type,
                                                  prepare_data_mappers,
                                                  [algo], {algo: hyperparameters},
                                                  display=False)

    return ndcg, ncf_ndcg


def run_models_for_testing(nodes: List[Node], edges: List[Tuple[Edge, bool]],
                           node_types: Set[NodeType], retrieved_node_type: NodeType,
                           prepare_data_mappers,
                           algos, hyperparamters_dict,
                           display=True):
    train_affinities = [e for e, t in edges if not t]
    validation_affinities = [e for e, t in edges if t]

    recs, results = test_multiple_algorithms(train_affinities, validation_affinities, nodes, node_types,
                                             hyperparamters_dict, prepare_data_mappers, algos, retrieved_node_type)
    ndcg, ncf_ndcg = results[0]['ndcg_b@100'], results[0]['ncf_ndcg']

    if display:
        results = display_results(results)
        results.to_csv("overall_results.csv")
    else:
        results = pd.DataFrame.from_records(results)
        results = results.groupby(["algo"]).mean().reset_index()
        ndcg, ncf_ndcg = results["ndcg_b@100"].values[0], results["ncf_ndcg"].values[0]
    return ndcg, ncf_ndcg
