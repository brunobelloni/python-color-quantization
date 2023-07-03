import time

import matplotlib.image
import numpy as np
import pytest

from main import maximin, okm, iokm, bkm, ibkm


@pytest.fixture(scope='function')
def test_data():
    image = matplotlib.image.imread('fish.ppm')
    pixels = image.astype(np.float64).reshape(-1, 3)
    k = 8
    return image, pixels, k


expected_maximin_centroids = np.array([
    [82.91608333, 60.7278, 73.73255],
    [255., 255., 255.],
    [219., 124., 156.],
    [0., 1., 0.],
    [106., 162., 151.],
    [183., 209., 198.],
    [180., 96., 70.],
    [145., 63., 139.],
])


def test_maximin(test_data):
    _, pixels, k = test_data
    start = time.time()
    centroids = maximin(x=pixels, k=k)
    end = time.time()
    print(f"maximin took {end - start} seconds")
    assert np.allclose(centroids, expected_maximin_centroids)


expected_bkm_centroids = np.array([
    [64.11864275, 38.24235569, 59.44134165],
    [229.03501629, 214.45521173, 208.78338762],
    [157.83429264, 99.49197213, 113.16449561],
    [26.52414371, 24.15842673, 24.87278726],
    [131.7322239, 129.37760968, 122.39092284],
    [173.80428888, 159.39733793, 154.06778408],
    [102.15318066, 80.48329092, 70.40627651],
    [106.94699029, 53.55621359, 108.00504854],
])


def test_bkm(test_data):
    image, pixels, k = test_data
    start = time.time()
    centroids, _, _ = bkm(x=pixels, k=k, image=image)
    end = time.time()
    print(f"bkm took {end - start} seconds")
    assert np.allclose(centroids, expected_bkm_centroids)


expected_ibkm_centroids = np.array([
    [23.72077476, 22.25610747, 22.68528585],
    [53.07582164, 36.72112243, 44.68152989],
    [77.3754144, 38.93608275, 83.80506564],
    [97.7086588, 76.52966595, 66.14442413],
    [114.77046263, 57.18022369, 113.35244026],
    [140.56482968, 110.00107009, 110.3015873],
    [167.13524186, 152.58065153, 148.66594274],
    [223.94701543, 209.08920188, 202.97987928],
])


def test_ibkm(test_data):
    image, pixels, k = test_data
    start = time.time()
    centroids, _, _ = ibkm(x=pixels, k=k, image=image)
    end = time.time()
    print(f"ibkm took {end - start} seconds")
    assert np.allclose(centroids, expected_ibkm_centroids)


expected_okm_centroids = np.array([
    [77.79575141, 38.75060744, 84.88395499],
    [223.61492797, 209.97182455, 203.93884006],
    [141.83313473, 111.68958113, 111.86575699],
    [23.990431, 22.1595759, 23.00231426],
    [56.07926268, 38.00606496, 46.84634357],
    [168.63963331, 153.97110142, 149.06101547],
    [100.11549186, 78.26882546, 68.17848677],
    [115.62866898, 57.56336025, 114.37223609],
])


def test_okm(test_data):
    image, pixels, k = test_data
    start = time.time()
    centroids, _, _ = okm(x=pixels, k=k, image=image)
    end = time.time()
    print(f"okm took {end - start} seconds")
    assert np.allclose(centroids, expected_okm_centroids)


expected_iokm_centroids = np.array([
    [225.27615183, 212.46241702, 207.30913485],
    [141.67805674, 113.13264214, 114.69950929],
    [102.69846672, 79.90448436, 66.74217395],
    [169.15305493, 153.58308369, 150.51332023],
    [22.11004311, 21.31228701, 21.73024129],
    [77.45016404, 39.21223411, 83.11241404],
    [53.31490344, 36.44048851, 44.88693621],
    [114.27203737, 56.66921303, 113.14759349],
])


def test_iokm(test_data):
    image, pixels, k = test_data
    start = time.time()
    centroids, _, _ = iokm(x=pixels, k=k, image=image)
    end = time.time()
    print(f"iokm took {end - start} seconds")
    assert np.allclose(centroids, expected_iokm_centroids)
