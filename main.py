import math
from functools import lru_cache

import matplotlib.image
import numpy as np
import sobol


def maximin(x, k):
    """
    Maximin initialization method (for batch/online k-means)

    For a comprehensive survey of k-means initialization methods, see
    M. E. Celebi, H. Kingravi, and P. A. Vela,
    A Comparative Study of Efficient Initialization Methods
    for the K-Means Clustering Algorithm,
    Expert Systems with Applications, 40(1): 200�210, 2013.

    x: Input data
    x: The number of clusters
    """
    centroids = [np.mean(x, axis=0)]
    distances = np.full(shape=x.shape[0], fill_value=np.inf)

    for _ in range(1, k):
        # Compute the distances to the latest centroid using broadcasting and vectorized operations
        dist_to_last_centroid = np.linalg.norm(x - centroids[-1], axis=1) ** 2

        # Update the minimum distances
        distances = np.minimum(distances, dist_to_last_centroid)

        # Find the point with the maximum distance
        max_dist_index = np.argmax(distances)

        # Point with maximum distance to its nearest center is chosen as a center
        centroids.append(x[max_dist_index])

    return np.array(centroids)


@lru_cache
def get_sobol_sequence(dimension, n_points):
    return sobol.sample(dimension=dimension, n_points=n_points)


def sobol_sequence(index):
    """
    Function to generate two quasi-random numbers from
    a Sobol sequence. Adapted from Numerical Recipies
    in C. Upon return, X and Y fall in [0,1).
    """
    sequence = get_sobol_sequence(dimension=3, n_points=2_000_000)
    return sequence[index][1], sequence[index][2]


def find_nearest_cluster(point, cluster):
    min_dist = np.inf
    min_dist_index = -np.inf

    for j in range(len(cluster)):
        delta = cluster[j] - point
        distance = np.dot(delta, delta)  # Calculate Euclidean distance

        if distance < min_dist:
            min_dist = distance
            min_dist_index = j

    return min_dist_index


def bkm(x, k, **kwargs):
    """
    Batch K-Means Algorithm:

    M. E. Celebi,
    Improving the Performance of K-Means for Color Quantization,
    Image and Vision Computing, 29(4): 260�271, 2011.

    :param x: The input data
    :param k: The number of clusters
    """
    cluster = kwargs.get('cluster', maximin(x=x, k=k))
    sizes = kwargs.get('sizes', np.zeros(shape=k))

    temp_cluster = np.zeros(shape=(k, x.shape[1]))
    member = np.zeros(shape=x.shape[0], dtype=np.int32)

    max_iters = 10_000
    num_iters = 0
    while True:
        num_iters += 1
        num_changes = 0

        # reset the new clusters
        temp_cluster[:, :] = 0
        sizes[:] = 0

        for i, point in enumerate(x):
            min_dist_index = find_nearest_cluster(point=point, cluster=cluster)

            if (num_iters == 1) or (member[i] != min_dist_index):
                # Update the membership of the point
                member[i] = min_dist_index
                num_changes += 1

            # Update the temporary center & size of the nearest cluster
            temp_cluster[min_dist_index] += point
            sizes[min_dist_index] += 1

        # update all centers
        for j in range(k):
            if sizes[j] != 0:
                cluster[j] = temp_cluster[j] / sizes[j]

        if num_changes <= 0 or num_iters >= max_iters:
            break

    return cluster, sizes, None


def ibkm(x, k, **kwargs):
    """
    Incremental Batch K-Means Algorithm:

    Y. Linde, A. Buzo, and R. Gray,
    An Algorithm for Vector Quantizer Design,
    IEEE Transactions on Communications, 28(1): 84-95, 1980.

    :param x: Input data
    :param k: Number of clusters
    """
    epsilon = 0.255  # small perturbation constant
    num_splits = int(math.log2(k) / math.log2(2) + 0.5)

    cluster = np.zeros(shape=(2 * k - 1, x.shape[1]))
    cluster[0] = np.mean(x, axis=0)
    sizes = np.zeros(shape=2 * k - 1)

    for t in range(num_splits):
        for n in range(pow(2, t) - 1, pow(2, t + 1) - 1):
            point = cluster[n]  # Split c[n] into c[2n + 1] and c[2n + 2]
            cluster[2 * n + 1] = point  # Left child
            cluster[2 * n + 2] = point + epsilon  # Right child

        # Refine the new centers using batch k-means
        bkm_index_start = pow(2, t + 1) - 1
        bkm_index_end = pow(2, t + 2) - 1
        cluster[bkm_index_start:bkm_index_end], sizes[bkm_index_start:bkm_index_end], _ = bkm(
            x=x,
            k=pow(2, t + 1),
            sizes=sizes[bkm_index_start:bkm_index_end],
            cluster=cluster[bkm_index_start:bkm_index_end],
        )

    cluster = cluster[-k:]  # last k centers are the final centers

    return cluster, None, None


def okm(x, k, lr_exp=0.5, sample_rate=1.0, **kwargs):
    """
    Online K-Means Algorithm:

    S. Thompson, M. E. Celebi, and K. H. Buck,
    Fast Color Quantization Using MacQueen�s K-Means Algorithm,
    Journal of Real-Time Image Processing,
    17(5): 1609-1624, 2020.

    :param x: Input data
    :param k: Number of clusters
    :param lr_exp: Learning rate exponent (must be in [0.5, 1])
    :param sample_rate: Fraction of the input (must be in (0, 1])
    """
    cluster = kwargs.get('cluster', maximin(x=x, k=k))
    sizes = kwargs.get('sizes', np.zeros(shape=k))

    image_height = kwargs.get('image').shape[0]
    image_width = kwargs.get('image').shape[1]

    num_samples = int(sample_rate * x.shape[0] + 0.5)

    sobel_index = 0
    for sobel_index in range(num_samples):
        sob_x, sob_y = sobol_sequence(index=sobel_index + kwargs.get('sobel_index', 0))
        row_idx = min(int(sob_y * image_height + 0.5), image_height - 1)
        col_idx = min(int(sob_x * image_width + 0.5), image_width - 1)
        rand_x = x[row_idx * image_width + col_idx]

        min_dist_index = find_nearest_cluster(point=rand_x, cluster=cluster)

        sizes[min_dist_index] += 1
        learn_rate = pow(sizes[min_dist_index], -lr_exp)

        # Update the cluster with the learning rate and difference
        cluster[min_dist_index] += learn_rate * (rand_x - cluster[min_dist_index])

    return cluster, sizes, sobel_index + kwargs.get('sobel_index', 0) + 1


def iokm(x, k, lr_exp=0.5, sample_rate=0.5, **kwargs):
    """
    Incremental Online K-Means Algorithm:

    A. D. Abernathy and M. E. Celebi,
    The Incremental Online K-Means Clustering Algorithm
    and Its Application to Color Quantization,
    Expert Systems with Applications,
    accepted for publication, 2022.

    :param x: Input data
    :param k: Number of clusters
    :param lr_exp: Learning rate exponent (must be in [0.5, 1])
    :param sample_rate: Fraction of the input x (must be in (0, 1])
    """
    num_splits = int(math.log2(k) / math.log2(2) + 0.5)
    cluster = np.zeros(shape=(2 * k - 1, x.shape[1]), dtype=np.float64)
    cluster[0] = np.mean(x, axis=0)
    sizes = np.zeros(shape=2 * k - 1)

    sobel_index = 0
    for t in range(num_splits):
        for n in range(pow(2, t) - 1, pow(2, t + 1) - 1):
            point = cluster[n]  # Split c[n] into c[2n + 1] and c[2n + 2]
            cluster[2 * n + 1] = point  # Left child
            cluster[2 * n + 2] = point  # Right child

        # Refine the new centers using online k-means
        okm_index_start = pow(2, t + 1) - 1
        okm_index_end = pow(2, t + 2) - 1
        cluster[okm_index_start:okm_index_end], sizes[okm_index_start:okm_index_end], sobel_index = okm(
            x=x,
            lr_exp=lr_exp,
            k=pow(2, t + 1),
            sample_rate=sample_rate,
            sobel_index=sobel_index,
            sizes=sizes[okm_index_start:okm_index_end],
            cluster=cluster[okm_index_start:okm_index_end],
            **kwargs,
        )

    cluster = cluster[-k:]  # last k centers are the final centers

    return cluster, None, None


def mse(x, cluster, k):
    """ Compute the Mean Squared Error of a given partition """
    min_dists = np.inf * np.ones(x.shape[0])
    for j in range(k):
        dists = np.sum((x - cluster[j]) ** 2, axis=1)
        min_dists = np.minimum(min_dists, dists)
    sse = np.sum(min_dists)
    return sse / x.shape[0]


def main():
    image = matplotlib.image.imread('fish.ppm')  # Load the image

    k = 8  # Number of colors to quantize to
    pixels = image.reshape(-1, 3)  # Reshape the image to be a list of RGB colors.
    pixels = pixels.astype(np.float64)

    for algorithm in ['bkm', 'ibkm', 'okm', 'iokm']:
        kwargs = {'k': k, 'image': image, 'x': pixels}
        cluster, _, _ = globals()[algorithm](**kwargs)

        # Assign each pixel to the nearest cluster centroid
        labels = np.argmin(np.linalg.norm(pixels[:, None] - cluster, axis=-1), axis=-1)

        # compute the MSE of quantized image
        error = mse(x=pixels, cluster=cluster, k=k)
        print(f'MSE for {algorithm} with {k} colors: {error}')

        # Replace each pixel with its corresponding cluster centroid
        quantized_image = cluster[labels]

        # Reshape the quantized image back to its original shape
        quantized_image = quantized_image.astype(np.uint8)
        quantized_image = quantized_image.reshape(image.shape)

        matplotlib.image.imsave(f'out/{algorithm}_{int(math.log2(k))}bit_image.png', quantized_image)


if __name__ == '__main__':
    main()
