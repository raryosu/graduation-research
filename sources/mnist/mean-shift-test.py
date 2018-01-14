import sys
import numpy as np
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.cluster import MeanShift, estimate_bandwidth

DIM = int(sys.argv[1])
K = int(sys.argv[2])
NUM = int(sys.argv[3])
n_samples = NUM * K
X, TrueLabels = make_blobs(n_samples=n_samples, centers=K, n_features=DIM)


bandwidth = estimate_bandwidth(X, quantile=0.2)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)

labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

purity = metrics.adjusted_rand_score(TrueLabels, labels)
nmi = metrics.normalized_mutual_info_score(TrueLabels, labels)
ari = metrics.adjusted_rand_score(TrueLabels, labels)

print("Estimated k = " + str(n_clusters_) + ", purity = " + str(purity) + ", NMI = " + str(nmi) + ", ARI = " + str(ari) + "\n")
