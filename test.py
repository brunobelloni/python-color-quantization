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
    centroids, _ = bkm(x=pixels, k=k, image=image)
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
    centroids, _ = ibkm(x=pixels, k=k, image=image)
    end = time.time()
    print(f"ibkm took {end - start} seconds")
    assert np.allclose(centroids, expected_ibkm_centroids)


expected_okm_centroids = np.array([
    [77.74637033, 39.28060205, 84.33717076],
    [226.50314107, 213.44474807, 206.69082079],
    [142.10996277, 114.43215636, 115.43885513],
    [25.02656539, 23.18651181, 23.71655805],
    [99.76898795, 77.64559736, 68.35826365],
    [167.75841188, 155.71355499, 150.73420829],
    [115.35460949, 58.45529955, 112.64026363],
    [56.38523941, 38.12788488, 45.57434503],
])


def test_okm(test_data):
    image, pixels, k = test_data
    start = time.time()
    centroids, _ = okm(x=pixels, k=k, image=image)
    end = time.time()
    print(f"okm took {end - start} seconds")
    assert np.allclose(centroids, expected_okm_centroids)


expected_iokm_centroids = np.array([
    [114.93314634, 57.14793879, 112.93016151],
    [52.16297328, 35.45762003, 45.11185502],
    [22.86995512, 21.9065144, 22.10624803],
    [77.54122084, 39.47154599, 84.26477269],
    [101.07824149, 79.29842017, 68.1426873],
    [142.72568028, 112.50033983, 112.70591883],
    [230.09942453, 216.34020199, 212.95416125],
    [171.68705579, 157.69065765, 152.32808518],
])


def test_iokm(test_data):
    image, pixels, k = test_data
    start = time.time()
    centroids, _ = iokm(x=pixels, k=k, image=image)
    end = time.time()
    print(f"iokm took {end - start} seconds")
    assert np.allclose(centroids, expected_iokm_centroids)
