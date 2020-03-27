import numpy as np
from typing import List, Dict, Tuple


def ndcg(y_true: Dict[str, float], y_pred: List[str]):
    """
    True NDCG function with actual relevance scores.
    :param y_true:
    :param y_pred:
    :return:
    """
    y_true_sorted = sorted(y_true.values(), reverse=True)
    y_true_sorted = y_true_sorted[:len(y_pred)]
    idcg = np.sum((np.power(2, y_true_sorted) - 1)/(np.log2(np.arange(len(y_true_sorted)) + 2)))
    y_pred = [y_true[i] if i in y_true else 0 for i in y_pred]
    dcg = np.sum((np.power(2, y_pred) - 1) / (np.log2(np.arange(len(y_pred)) + 2)))
    return dcg/(idcg + 1e-8)


def binary_ndcg(y_true: List[str], y_pred: List[str]):
    return ndcg({k: 1 for k in y_true}, y_pred)


def recall(y_true: List[str], y_pred: List[str]):
    y_true = set(y_true)
    norm = min(len(y_pred), len(y_true))
    recall = sum([1 if i in y_true else 0 for i in y_pred])
    return recall/max(norm, 1.0)

## Example of Binary NDCG ##
# Positive list: Sellers who have gms > Threshold
actual_top_sellers = ["Seller1", "Seller2", "Seller3", "Seller4", "Seller5"]

# Ranked Sellers by Model
predicted_top_ranks = ["Seller6", "Seller1", "Seller3", "Seller2", "Seller5"]
print(binary_ndcg(actual_top_sellers, predicted_top_ranks))

# Note:
predicted_top_ranks = ["Seller1", "Seller3", "Seller2", "Seller5", "Seller6", ]
print(binary_ndcg(actual_top_sellers, predicted_top_ranks))

predicted_top_ranks = ["Seller1", "Seller2", "Seller3", "Seller5", "Seller6", ]
print(binary_ndcg(actual_top_sellers, predicted_top_ranks))
# end note: no difference in two cases since in binary ndcg we did not tell which seller is better than other, we just told which sellers are above threshold in y_true.

print("--" * 60)
## True NDCG ##
# Need to provide relevance scores for each Seller as well. Also note you can provide all sellers in y_true with their relevance score
# no need to give sellers above a threshold.
# A seller with higher relevance should be ranked higher. A seller with higher gms has higher relevance.
# Do not put gms directly as relevance (too big numbers raised as 2 power causes numerical issues). Use log10(gms)
# If gms also has -ve values, sigmoid(gms) or tanh(gms)

actual_seller_relevance = dict(Seller1=5.0, Seller2=4.8, Seller3=3.0, Seller4=4.1, Seller5=2.9, Seller6=0.9)
# Note we gave relevance score for Seller6 as well, no thresholding
# The ordering given here doesn't matter, the ndcg fucntion sorts internally

# Just provide your model's ordering here
predicted_top_ranks = ["Seller1", "Seller2", "Seller3", "Seller5", "Seller6",]
print(ndcg(actual_seller_relevance, predicted_top_ranks))

predicted_top_ranks = ["Seller5", "Seller6", "Seller1", "Seller2", "Seller3",]
print(ndcg(actual_seller_relevance, predicted_top_ranks))

# Perfect ranking, score ~1
predicted_top_ranks = ["Seller1", "Seller2", "Seller4", "Seller3", "Seller5", "Seller6"]
print(ndcg(actual_seller_relevance, predicted_top_ranks))

# Just an example gms preprocesser before it is used as relevance scores for true ndcg.
# This function is monotonically increasing. But more bounded on both sides.
def gms_preprocess(gms: np.array) -> np.array:
    gms = gms.astype(float)
    i1 = gms <= 1
    i2 = gms > 1
    gms[i1] = 1 + np.tanh(gms[i1])
    gms[i2] = np.log2(gms[i2])
    return gms


print(gms_preprocess(np.arange(-10, 10,1)))
print(gms_preprocess(np.arange(-2, 2,1)))



