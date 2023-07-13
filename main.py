import math

import matplotlib.image
import numpy as np
from scipy.stats import qmc


class SequenceIterator:
    def __init__(self):
        engine = qmc.Sobol(d=1, scramble=False)
        self.sequence = engine.random_base2(m=23)  # 2^m points (m=23 is 8,388,608 points)
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.sequence):
            raise StopIteration
        value = self.sequence[self.index]
        self.index += 1
        return value[0]

    def reset(self):
        self.index = 0


sequence = SequenceIterator()


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


def is_power_of_two(number):
    return False if number <= 0 else (number & (number - 1)) == 0


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

    return cluster, sizes


def ibkm(x, k, epsilon=0.255, **kwargs):
    """
    Incremental Batch K-Means Algorithm:

    Y. Linde, A. Buzo, and R. Gray,
    An Algorithm for Vector Quantizer Design,
    IEEE Transactions on Communications, 28(1): 84-95, 1980.

    :param x: Input data
    :param k: Number of clusters
    :param epsilon: Small perturbation constant
    """
    num_splits = math.ceil(math.log2(k))
    cluster = np.zeros(shape=(2 * k - 1, x.shape[1]))
    cluster[0] = np.mean(x, axis=0)
    sizes = np.zeros(shape=2 * k - 1)

    for t in range(num_splits):
        split_start, split_end = pow(2, t) - 1, pow(2, t + 1) - 1
        bkm_start, bkm_end = pow(2, t + 1) - 1, pow(2, t + 2) - 1
        if not is_power_of_two(k) and t == (num_splits - 1):
            split_end = split_start + (k - pow(2, int(math.log2(k))))
            bkm_start = split_end
            bkm_end = bkm_start + pow(2, t + 1)
        for n in range(split_start, split_end):
            point = cluster[n]  # Split c[n] into c[2n + 1] and c[2n + 2]
            cluster[2 * n + 1] = point  # Left child
            cluster[2 * n + 2] = point + epsilon  # Right child

        # Refine the new centers using batch k-means
        cluster[bkm_start:bkm_end], sizes[bkm_start:bkm_end] = bkm(
            x=x,
            sizes=sizes[bkm_start:bkm_end],
            k=len(cluster[bkm_start:bkm_end]),
            cluster=cluster[bkm_start:bkm_end],
        )

    cluster = cluster[-k:]  # last k centers are the final centers

    return cluster, None


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

    num_samples = int(sample_rate * x.shape[0] + 0.5)
    for _ in range(num_samples):
        sob_x = next(sequence)
        rand_x = x[int(sob_x * len(x))]
        min_dist_index = find_nearest_cluster(point=rand_x, cluster=cluster)
        sizes[min_dist_index] += 1
        learn_rate = pow(sizes[min_dist_index], -lr_exp)

        # Update the cluster with the learning rate and difference
        cluster[min_dist_index] += learn_rate * (rand_x - cluster[min_dist_index])

    return cluster, sizes


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
    num_splits = math.ceil(math.log2(k))
    cluster = np.zeros(shape=(2 * k - 1, x.shape[1]), dtype=np.float64)
    cluster[0] = np.mean(x, axis=0)
    sizes = np.zeros(shape=2 * k - 1)

    for t in range(num_splits):
        split_start, split_end = pow(2, t) - 1, pow(2, t + 1) - 1
        okm_start, okm_end = pow(2, t + 1) - 1, pow(2, t + 2) - 1
        if not is_power_of_two(k) and t == (num_splits - 1):
            split_end = split_start + (k - pow(2, int(math.log2(k))))
            okm_start = split_end
            okm_end = okm_start + pow(2, t + 1)

        for n in range(split_start, split_end):
            point = cluster[n]  # Split c[n] into c[2n + 1] and c[2n + 2]
            cluster[2 * n + 1] = point  # Left child
            cluster[2 * n + 2] = point  # Right child

        # Refine the new centers using online k-means
        cluster[okm_start:okm_end], sizes[okm_start:okm_end] = okm(
            x=x,
            lr_exp=lr_exp,
            sample_rate=sample_rate,
            sizes=sizes[okm_start:okm_end],
            k=len(cluster[okm_start:okm_end]),
            cluster=cluster[okm_start:okm_end],
            **kwargs,
        )

    cluster = cluster[-k:]  # last k centers are the final centers

    return cluster, None


def mse(x, cluster, k):
    """ Compute the Mean Squared Error of a given partition """
    min_dists = np.inf * np.ones(x.shape[0])
    for j in range(k):
        dists = np.sum((x - cluster[j]) ** 2, axis=1)
        min_dists = np.minimum(min_dists, dists)
    sse = np.sum(min_dists)
    return sse / x.shape[0]


def main():
    filename = 'fish'
    image = matplotlib.image.imread(f"{filename}.ppm")  # Load the image
    matplotlib.image.imsave(f"out/{filename}_original.png", image)

    k = 7  # Number of colors to quantize to
    pixels = image.reshape(-1, 3)  # Reshape the image to be a list of RGB colors.
    pixels = pixels.astype(np.float64)

    for algorithm in ['bkm', 'ibkm', 'okm', 'iokm']:
        sequence.reset()
        cluster, _ = globals()[algorithm](x=pixels, k=k)

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

        matplotlib.image.imsave(f'out/{filename}_{algorithm}_{k}K_image.png', quantized_image)


if __name__ == '__main__':
    main()
