import sys
import tensorflow as tf
import numpy as np
from sklearn import datasets, metrics
import xmeans
from xmeans import plot_clusters, plot_clusters_3d

if len(sys.argv) == 5:
    DIM = int(sys.argv[1])
    K = int(sys.argv[2])
    NUM = int(sys.argv[3])
    name = str(sys.argv[4])
    X, TrueLabels = datasets.make_blobs(n_samples=NUM, centers=K, n_features=DIM)
else:
    # wine = np.loadtxt("../datasets/winequality-white.csv", delimiter=";", skiprows=1)
    # X = wine[:,:-2]
    # TrueLabels = wine[:,-1]
    # print(TrueLabels)
    iris = datasets.load_iris()
    X = iris.data
    TrueLabels = iris.target


X = tf.Variable(X)
TrueLabels = tf.Variable(TrueLabels)

model = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(model)
    X_values = session.run(X)
    TrueLabels_values = session.run(TrueLabels)

    xm = xmeans.XMeans(X_values)
    xm.fit()

    purity = metrics.adjusted_rand_score(TrueLabels_values, xm.labels)
    nmi = metrics.normalized_mutual_info_score(TrueLabels_values, xm.labels)
    ari = metrics.adjusted_rand_score(TrueLabels_values, xm.labels)

    # print("Estimated k = " + str(xm.k) + ", purity = " + str(purity) + ", NMI = " + str(nmi) + ", ARI = " + str(ari) + "\n")
    print(str(xm.k) + ", " + str(purity) + ", " + str(nmi) + ", " + str(ari))

    # plot_clusters(X_values, xm.labels, NUM, xm.k, name=name)
    plot_clusters_3d(X_values, xm.labels, NUM, xm.k, name=name)
