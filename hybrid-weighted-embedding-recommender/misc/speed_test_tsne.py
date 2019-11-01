from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.manifold import TSNE as ScikitTSNE
from sklearn.decomposition import PCA, KernelPCA
from numpy.linalg import norm
from sklearn.utils import shuffle
import numpy as np
from umap import UMAP
import time
import numpy as np
import fitsne
from scipy.special import comb
from math import factorial

from tensorflow import keras
import tensorflow as tf



# from sklearn.datasets import load_digits
# digits = load_digits().data
# X = digits

(X, _), (_, _) = tf.keras.datasets.mnist.load_data() # [('PCA', '5.2', 0.3014303061728395), ('AutoEnc', '122.7', 0.2701841850617284)]
X = X.reshape(len(X),28*28)
print(X.shape)
# X = np.random.randn(100000, 128)


shuf_ind0 = np.array(shuffle(list(range(len(X)))))
shuf_ind1 = np.array(shuffle(list(range(len(X)))))
shuf_ind2 = np.array(shuffle(list(range(len(X)))))

def unit_length(a, axis=0):
    return a/np.expand_dims(norm(a, axis=axis), axis=axis)


def measure_array_dist_element_displacement(X1, X2):
    X1 = list(X1)
    X2 = list(X2)

    assert len(X1) == len(X2)
    diff = 0.
    elem_to_index = {e:i for i,e in enumerate(X1)}
    for index, element in enumerate(X2):
        actual_index = elem_to_index[element]
        diff += abs(index - actual_index)

    return diff / len(X1) ** 2 * 2


def measure_array_dist_inversions(X1, X2):
    def merge_sort_inv_counter(arr):
        return _merge_sort_counter(arr,[0]*len(arr),0,len(arr)-1)

    def _merge_sort_counter(arr, temp_arr, left, right):
        inv_count = 0
        if left < right:
            mid = (left + right) // 2
            inv_count = _merge_sort_counter(arr, temp_arr, left, mid)
            inv_count += _merge_sort_counter(arr, temp_arr, mid + 1, right)
            inv_count += merge(arr, temp_arr, left, mid, right)
        return inv_count

    def merge(arr, temp_arr, left, mid, right):
        i = left  # Starting index of left subarray
        j = mid + 1  # Starting index of right subarray
        k = left  # Starting index of to be sorted subarray
        inv_count = 0
        while i <= mid and j <= right:
            if arr[i] <= arr[j]:
                temp_arr[k] = arr[i]
                k += 1
                i += 1
            else:
                # Inversion will occur.
                temp_arr[k] = arr[j]
                inv_count += (mid - i + 1)
                k += 1
                j += 1

        while i <= mid:
            temp_arr[k] = arr[i]
            k += 1
            i += 1

        while j <= right:
            temp_arr[k] = arr[j]
            k += 1
            j += 1

        for loop_var in range(left, right + 1):
            arr[loop_var] = temp_arr[loop_var]

        return inv_count

    X1 = list(X1)
    X2 = list(X2)
    assert len(X1) == len(X2)
    elem_to_index_unsorted= {e: i for i, e in enumerate(X2)}
    unsorted = [elem_to_index_unsorted[e] for e in X1]
    return merge_sort_inv_counter(unsorted)/comb(len(X1),2)

# print(measure_array_dist_inversions([1,2,3,4,5,6],[2,3,4,5,6,1]))
# print(measure_array_dist_inversions([1,2,3,4,5,6],[6,1,2,3,4,5]))
#
# print(measure_array_dist_element_displacement([1,2,3,4,5,6],[2,3,4,5,6,1]))
# print(measure_array_dist_element_displacement([1,2,3,4,5,6],[6,1,2,3,4,5]))




def dist_compare(X1, X2):
    assert X1.shape[0] == X2.shape[0]

    def generate_distances(X, order):
        x_rev = X[order]
        distances = np.sum(np.square(X - x_rev), axis=1)
        return list(distances)

    def get_sorted_distances(X):
        distances = generate_distances(X, shuf_ind0) + generate_distances(X, shuf_ind1) + generate_distances(X,shuf_ind2)
        sd, si = zip(*list(sorted(zip(distances, range(len(distances))), key=lambda x:x[0])))
        si = np.array(si)
        return si
    sd1 = get_sorted_distances(X1)
    sd2 = get_sorted_distances(X2)
    score = measure_array_dist_element_displacement(sd1, sd2)
    score2 = measure_array_dist_inversions(sd1, sd2)
    return score, score2


timings = []
X = unit_length(X, axis=1)
dist_compare(X, X)


def auto_encoder_transform(X, n_dims=32,):
    loss = "mean_squared_error"
    initial_dims = X.shape[1]
    avg_value = 1.0/np.sqrt(initial_dims)
    print(avg_value)
    es = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0, patience=5, verbose=0, )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.2, patience=4, min_lr=0.0001)
    input_layer = tf.keras.Input(shape=(X.shape[1],))
    # encoded = tf.keras.layers.GaussianNoise(0.01)(input_layer)
    encoded = tf.keras.layers.Dense(n_dims * 4, activation='elu')(input_layer)
    encoded = tf.keras.layers.GaussianNoise(0.01*avg_value)(encoded)
    encoded = tf.keras.layers.Dense(n_dims * 2, activation='elu')(encoded)
    # encoded = tf.keras.layers.GaussianNoise(0.01)(encoded)
    encoded = tf.keras.layers.Dense(n_dims, activation='elu')(encoded)

    decoded = tf.keras.layers.Dense(n_dims * 2, activation='elu')(encoded)
    decoded = tf.keras.layers.GaussianNoise(0.01*avg_value)(decoded)
    decoded = tf.keras.layers.Dense(n_dims * 4, activation='elu')(decoded)
    decoded = tf.keras.layers.Dense(X.shape[1], activation='elu')(decoded)

    autoencoder = tf.keras.Model(input_layer, decoded)
    encoder = tf.keras.Model(input_layer, encoded)
    adam = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.05, amsgrad=False)
    autoencoder.compile(optimizer=adam, loss=loss)
    autoencoder.fit(X, X,
                    epochs=50,
                    batch_size=4096,
                    shuffle=True,
                    verbose=1,
                    callbacks=[es, reduce_lr])


    Z = encoder.predict(X)
    return Z


start = time.time()
ZPCA = PCA(n_components=32,).fit_transform(X)
end = time.time()
ZPCA = unit_length(ZPCA, axis=1)
score = dist_compare(X, ZPCA)
print("PCA time = %.1f" % (end-start))
timings.append(("PCA", "%.1f"%(end-start),score))

start = time.time()
Zenc = auto_encoder_transform(X, n_dims=32,)
end = time.time()
Zenc = unit_length(Zenc, axis=1)
score = dist_compare(X, Zenc)
print("AutoEnc time = %.1f" % (end-start))
timings.append(("AutoEnc", "%.1f"%(end-start),score))

#
# start = time.time()
# Z = fitsne.FItSNE(X, no_dims=32, nthreads=2, initialization=Zenc.copy(order='C').astype(np.float64), perplexity=5)
# end = time.time()
# score = dist_compare(X, Z)
# print("FitSNE time = %.1f" % (end-start))
# timings.append(("Initialized FitSNE","%.1f"%(end-start), score))
#
#
# start = time.time()
# Z = fitsne.FItSNE(X, no_dims=32, nthreads=2, perplexity=5)
# end = time.time()
# score = dist_compare(X, Z)
# print("FitSNE time = %.1f" % (end-start))
# timings.append(("FitSNE","%.1f"%(end-start), score))

#
# start = time.time()
# Z = UMAP(n_components=32,n_neighbors=10, metric="euclidean", init=ZPCA, min_dist=0.1,negative_sample_rate=10).fit_transform(X)
# end = time.time()
# score = dist_compare(X, Z)
# print("UMAP init time = %.1f" % (end-start))
# timings.append(("UMAP init","%.1f"%(end-start), score))
#
# start = time.time()
# Z = UMAP(n_components=32,n_neighbors=10, metric="euclidean", init="spectral",min_dist=0.1,negative_sample_rate=10).fit_transform(X)
# end = time.time()
# score = dist_compare(X, Z)
# print("UMAP time = %.1f" % (end-start))
# timings.append(("UMAP","%.1f"%(end-start), score))

# start = time.time()
# tsne_model = TSNE(2, n_jobs=2, perplexity=5.0)
# Z = tsne_model.fit_transform(X)
# end = time.time()
# score = dist_compare(X, Z)
# print("MulticoreTSNE time = %.1f" % (end-start))
# timings.append(("MulticoreTSNE", "%.1f"%(end-start),score))

# start = time.time()
# tsne_model = ScikitTSNE(32, method='exact')
# Z = tsne_model.fit_transform(X)
# end = time.time()
# print("Scikit TSNE time = %.1f" % (end-start))
# timings.append("%.1f"%(end-start))

#
# start = time.time()
# Z = KernelPCA(n_components=32,kernel="linear", n_jobs=2,).fit_transform(X)
# end = time.time()
# print("Linear Kernel PCA time = %.1f" % (end-start))
# timings.append(("Linear Kernel PCA","%.1f"%(end-start)))
#
# start = time.time()
# Z = KernelPCA(n_components=32,kernel="rbf", n_jobs=2).fit_transform(X)
# end = time.time()
# print("RBF PCA time = %.1f" % (end-start))
# timings.append(("RBF Kernel PCA","%.1f"%(end-start)))

print(timings)


