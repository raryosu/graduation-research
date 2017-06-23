import tensorflow as tf
import numpy as np

from functions import create_samples

n_features = 2
n_clusters = 3
n_samples_per_cluster = 500
seed = 700
embiggen_factor = 70

np.random.seed(seed)

centroids, samles = create_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed)

model = tf.global_variables_initializer()
with tf.Session() as session:
    sample_values = sess.run(samples)
    centroid_values = session.run(centroids)

plot_clusters(sample_values, centroid_values, n_samples_per_cluster)
