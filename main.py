import math

import matplotlib.image
import numpy as np

np.random.seed(42)  # for reproducibility


def maximin(pixels, k):
    """
    Maximin initialization method (for batch/online k-means)
    """
    centroids = [np.mean(pixels, axis=0)]
    distances = np.full(shape=pixels.shape[0], fill_value=np.inf)

    for _ in range(1, k):
        # Compute the distances to the latest centroid using broadcasting and vectorized operations
        dist_to_last_centroid = np.linalg.norm(pixels - centroids[-1], axis=1) ** 2

        # Update the minimum distances
        distances = np.minimum(distances, dist_to_last_centroid)

        # Find the pixel with the maximum distance
        max_dist_index = np.argmax(distances)

        # max_dist = -np.inf
        # max_dist_index = -np.inf
        # for j, pixel in enumerate(pixels):
        #     # Compute this pixel's distance to the previously chosen center
        #     del_red = centroids[i - 1][0] - pixel[0]
        #     del_green = centroids[i - 1][1] - pixel[1]
        #     del_blue = centroids[i - 1][2] - pixel[2]
        #     distance = (del_red * del_red) + (del_green * del_green) + (del_blue * del_blue)
        #
        #     # Update the nearest-center-distance for this pixel
        #     if distance < distances[j]:
        #         distances[j] = distance
        #
        #     if max_dist < distances[j]:
        #         max_dist = distances[j]
        #         max_dist_index = j

        # Pixel with maximum distance to its nearest center is chosen as a center
        centroids.append(pixels[max_dist_index])

    return np.array(centroids)


def okm(pixels, k, lr_exp=0.5, sample_rate=1.0, cluster: np.array = None):
    """
    Online K-Means Algorithm:
    S. Thompson, M. E. Celebi, and K. H. Buck,
    Fast Color Quantization Using MacQueen�s K-Means Algorithm,
    Journal of Real-Time Image Processing,
    17(5): 1609-1624, 2020.

    :param lr_exp: Learning rate exponent (must be in [0.5, 1])
    :param sample_rate: Fraction of the input pixels (must be in (0, 1])
    """
    if cluster is None:
        cluster = maximin(pixels=pixels, k=k)
    sizes = np.full(shape=k, fill_value=0)

    num_samples = int(sample_rate * pixels.shape[0] + 0.5)

    for _ in range(num_samples):
        rand_pixel = pixels[np.random.choice(pixels.shape[0])]

        min_dist = np.inf
        min_dist_index = -np.inf

        for j in range(k):
            del_red = cluster[j][0] - rand_pixel[0]
            del_green = cluster[j][1] - rand_pixel[1]
            del_blue = cluster[j][2] - rand_pixel[2]
            distance = (del_red * del_red) + (del_green * del_green) + (del_blue * del_blue)
            if distance < min_dist:
                min_dist = distance
                min_dist_index = j

        sizes[min_dist_index] += 1
        learn_rate = pow(sizes[min_dist_index], -lr_exp)

        cluster[min_dist_index][0] += learn_rate * (rand_pixel[0] - cluster[min_dist_index][0])
        cluster[min_dist_index][1] += learn_rate * (rand_pixel[1] - cluster[min_dist_index][1])
        cluster[min_dist_index][2] += learn_rate * (rand_pixel[2] - cluster[min_dist_index][2])

    # Assign each pixel to the nearest cluster centroid
    labels = np.argmin(np.linalg.norm(pixels[:, None] - cluster, axis=-1), axis=-1)
    return cluster, labels


def iokm(pixels, k, lr_exp=0.5, sample_rate=0.5):
    """
    Incremental Online K-Means Algorithm:

    A. D. Abernathy and M. E. Celebi,
    The Incremental Online K-Means Clustering Algorithm
    and Its Application to Color Quantization,
    Expert Systems with Applications,
    accepted for publication, 2022.

    :param lr_exp: Learning rate exponent (must be in [0.5, 1])
    :param sample_rate: Fraction of the input pixels (must be in (0, 1])
    """
    num_splits = int(math.log2(k) / math.log2(2) + 0.5)

    cluster = np.full(shape=(k, pixels.shape[1]), fill_value=0, dtype=np.float64)
    temp_cluster = np.full(shape=(2 * k - 1, pixels.shape[1]), fill_value=0, dtype=np.float64)
    temp_cluster[0] = np.mean(pixels, axis=0)

    for t in range(num_splits):
        for n in range(pow(2, t) - 1, pow(2, t + 1) - 1):
            pixel = temp_cluster[n]

            index = 2 * n + 1
            temp_cluster[index] = pixel

            index += 1
            temp_cluster[index] = pixel

        # Refine the new centers using online k-means
        temp_cluster[pow(2, t + 1) - 1:], _ = okm(
            pixels=pixels,
            k=pow(2, t + 1),
            lr_exp=lr_exp,
            sample_rate=sample_rate,
            cluster=temp_cluster[pow(2, t + 1) - 1:],
        )

    # last k centers are the final centers
    for j in range(k):
        cluster[j] = temp_cluster[j + k - 1]

    # Assign each pixel to the nearest cluster centroid
    labels = np.argmin(np.linalg.norm(pixels[:, None] - cluster, axis=-1), axis=-1)

    return cluster, labels


def mse(pixels, cluster, k):
    """
    Compute the Mean Squared Error of a given partition
    """
    min_dists = np.inf * np.ones(pixels.shape[0])
    for j in range(k):
        dists = np.sum((pixels - cluster[j]) ** 2, axis=1)
        min_dists = np.minimum(min_dists, dists)
    sse = np.sum(min_dists)
    return sse / pixels.shape[0]


def main():
    image = matplotlib.image.imread('fish.ppm')  # Load the image

    k = 32  # Number of colors to quantize to
    pixels = image.reshape(-1, 3)  # Reshape the image to be a list of RGB colors.
    pixels = pixels.astype(np.float64)

    for algorithm in ['okm', 'iokm']:
        cluster, labels = globals()[algorithm](pixels=pixels, k=k)

        # compute the MSE of quantized image
        error = mse(pixels=pixels, cluster=cluster, k=k)
        print(f'MSE for {algorithm} with {k} colors: {error}')

        # Replace each pixel with its corresponding cluster centroid
        quantized_image = cluster[labels]

        # Reshape the quantized image back to its original shape
        quantized_image = quantized_image.astype(np.uint8)
        quantized_image = quantized_image.reshape(image.shape)

        matplotlib.image.imsave(f'out/{algorithm}_{int(math.log2(k))}bit_image.png', quantized_image)


if __name__ == '__main__':
    main()
