import hnswlib
import numpy as np

dim = 8
num_elements = 100

# Generating sample data
data = np.float32(np.random.random((num_elements, dim)))
data_labels = np.arange(num_elements)

# Declaring index
p = hnswlib.Index(space='l2', dim=dim)  # possible options are l2, cosine or ip

# Initing index - the maximum number of elements should be known beforehand
p.init_index(max_elements=num_elements, ef_construction=200, M=16)

# Element insertion (can be called several times):
p.add_items(data, data_labels)

# Controlling the recall by setting ef:
p.set_ef(50)  # ef should always be > k

# Query dataset, k - number of closest elements (returns 2 numpy arrays)
labels, distances = p.knn_query(data[0:5], k=5)

print(p.get_ids_list())

print(p.get_items([0,1]))
