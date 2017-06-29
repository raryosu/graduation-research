# -*- coding: utf-8 -*-

"""
Generate and plot sample data
"""

import tensorflow as tf
import numpy as np
from functions import *

n_features = 2
n_clusters = 3
n_samples_per_cluster = 500
seed = 700
embiggen_factor = 70

data_centroids, samples = create_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed)
initial_centroids = choose_random_centroids(samples, n_clusters)
centroids = tf.concat(initial_centroids, 0, name='centroids')

model = tf.global_variables_initializer()

with tf.Session() as session:
    sample_values = session.run(samples)
    for i in range(10):
        centroids = session.run(update(samples, centroids, n_clusters))
        print(centroids)
        plot_clusters(sample_values, centroids, n_samples_per_cluster)

