# -*- coding: utf-8 -*-

"""
k-means algorithm

http://learningtensorflow.com/lesson6/
"""

import tensorflow as tf
import numpy as np

def create_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed):
    np.random.seed(seed)
    slices = []
    centroids = []
    # Create samples for each cluster
    for i in range(n_clusters):
        samples = tf.random_normal((n_samples_per_cluster, n_features),
                                   mean=0.0, stddev=5.0, dtype=tf.float32, seed=seed, name="cluster_{}".format(i))
        current_centroid = (np.random.random((1, n_features)) * embiggen_factor) - (embiggen_factor/2)
        centroids.append(current_centroid)
        samples += current_centroid
        slices.append(samples)
    # Create "samples" dataset
    samples = tf.concat(slices, 0, name='samples')
    centroids = tf.concat(centroids, 0, name='centroids')
    return centroids, samples

def plot_clusters(all_samples, centroids, n_samples_per_cluster):
    """
    Plot the clusters with different colour for each cluster
    """
    import matplotlib.pyplot as plt
    colour = plt.cm.rainbow(np.linspace(0,1,len(centroids)))
    for i, centroid in enumerate(centroids):
         samples = all_samples[i*n_samples_per_cluster:(i+1)*n_samples_per_cluster]
         plt.scatter(samples[:,0], samples[:,1], c=colour[i])
         plt.plot(centroid[0], centroid[1], markersize=35, marker="x", color='k', mew=10)
         plt.plot(centroid[0], centroid[1], markersize=30, marker="x", color='m', mew=5)
    plt.show()

def choose_random_centroids(samples, n_clusters, seed=None):
    """
    Initialisation
    - Select `n_clusters` numboer of random points
    """
    n_samples = tf.shape(samples)[0]
    random_indices = tf.random_shuffle(tf.range(0, n_samples), seed=seed)
    begin = [0,]
    size = [n_clusters,]
    size[0] = n_clusters
    centroid_indices = tf.slice(random_indices, begin, size)
    initial_centroids = tf.gather(samples, centroid_indices)
    return initial_centroids

def assign_to_nearest(samples, centroids):
    """
    Finds the nearest centroid for each samples
    - Compute distances from each sample to centroids
    """
    expanded_vectors = tf.expand_dims(samples, 0)
    expanded_centroids = tf.expand_dims(centroids, 1)
    distances = tf.reduce_sum( tf.square(
        tf.subtract(expanded_vectors, expanded_centroids)), 2)
    mins = tf.argmin(distances, 0)

    nearest_indices = mins
    return nearest_indices

def update_centroids(samples, nearest_indices, n_clusters):
    """
    Updates the centroid to be the mean of all samples associated with it.
    - Separate samples by "nearest_indices"
    - Compute the mean of each cluster and set centroid there
    """
    nearest_indices = tf.to_int32(nearest_indices)
    # Clustering the samples
    partitions = tf.dynamic_partition(samples, nearest_indices, n_clusters)
    # New centroids position is mean the samples
    new_centroids = tf.concat([tf.expand_dims(tf.reduce_mean(partition, 0), 0) for partition in partitions], 0)
    return new_centroids

def update(samples, centroids, n_clusters):
    nearest_indices = assign_to_nearest(samples, centroids)
    centroids = update_centroids(samples, nearest_indices, n_clusters)
    return centroids

