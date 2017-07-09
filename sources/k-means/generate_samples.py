# -*- coding: utf-8 -*-

"""
Generate and plot sample data
"""

import tensorflow as tf
import numpy as np
from sklearn import datasets
from functions import *

n_features = 2
n_clusters = 3
n_samples_per_cluster = 1000
seed = 700
embiggen_factor = 70
threshold = 0.1e-9

centers = np.random.randint(-5, 5, (n_clusters, 2))
samples, labels = datasets.make_blobs(
        n_samples = n_samples_per_cluster,
        centers = centers,
        random_state=seed
        )
samples = tf.Variable(samples)
labels = tf.Variable(labels)
initial_centroids = choose_random_centroids(samples, n_clusters)
centroids = tf.concat(initial_centroids, 0, name='centroids')

model = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(model)
    sample_values = session.run(samples)
    label = session.run(labels)
    centroids = session.run(centroids)
    plot_samples(sample_values, save=True)
    count = 0
    while True:
        old_centroids = centroids
        centroids = session.run(update(samples, old_centroids, n_clusters))
        diff = old_centroids - centroids
        print(count, centroids)
        if len(np.where(np.array( diff.flatten() <= threshold ) )[0]) == n_clusters*2:
            break
        count += 1
    plot_clusters(sample_values, label, centroids, n_samples_per_cluster, save=True)

