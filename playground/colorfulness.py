import time

import matplotlib

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

from main import okm


# Code extracted from Colorfy: https://github.com/davidkrantz/Colorfy/blob/master/spotify_background_color.py
class SpotifyBackgroundColor:
    """Analyzes an image and finds a fitting background color.

    Main use is to analyze album artwork and calculate the background
    color Spotify sets when playing on a Chromecast.

    Attributes:
        img (ndarray): The image to analyze.

    """

    K = 8
    COLOR_TOL = 0
    SIZE = (100, 100)

    def __init__(self, img, format='RGB', image_processing_size=SIZE, _type='kmeans'):
        """Prepare the image for analyzation.

        Args:
            img (ndarray): The image to analyze.
            format (str): Format of `img`, either RGB or BGR.
            image_processing_size: (tuple): Process image or not.
                tuple as (width, height) of the output image (must be integers)

        Raises:
            ValueError: If `format` is not RGB or BGR.

        """
        self._type = _type
        if format == 'RGB':
            self.img = img
        elif format == 'BGR':
            self.img = self.img[..., ::-1]
        else:
            raise ValueError('Invalid format. Only RGB and BGR image format supported.')

        if image_processing_size:
            img = Image.fromarray(self.img)
            self.img = np.asarray(img.resize(image_processing_size, Image.Resampling.BILINEAR))

    def best_color(self, k=K, color_tol=COLOR_TOL):
        """Returns a suitable background color for the given image.

        Uses k-means clustering to find `k` distinct colors in
        the image. A colorfulness index is then calculated for each
        of these colors. The color with the highest colorfulness
        index is returned if it is greater than or equal to the
        colorfulness tolerance `color_tol`. If no color is colorful
        enough, a gray color will be returned. Returns more or less
        the same color as Spotify in 80 % of the cases.

        Args:
            k (int): Number of clusters to form.
            color_tol (float): Tolerance for a colorful color.
                Colorfulness is defined as described by Hasler and
                Süsstrunk (2003) in https://infoscience.epfl.ch/
                record/33994/files/HaslerS03.pdf.

        Returns:
            tuple: (R, G, B). The calculated background color.

        """
        self.img = self.img.reshape((self.img.shape[0] * self.img.shape[1], 3))

        if self._type == 'okm':
            centroids, _ = okm(x=self.img, k=k)
        else:
            clt = KMeans(n_clusters=k, n_init=10)
            clt.fit(self.img)
            centroids = clt.cluster_centers_

        colorfulness = [self.colorfulness(color[0], color[1], color[2]) for color in centroids]
        max_colorful = np.max(colorfulness)

        if max_colorful < color_tol:
            best_color = [230, 230, 230]  # If not colorful, set to gray
        else:
            best_color = centroids[np.argmax(colorfulness)]  # Pick the most colorful color

        return int(best_color[0]), int(best_color[1]), int(best_color[2])

    def colorfulness(self, r, g, b):
        """Returns a colorfulness index of given RGB combination.

        Implementation of the colorfulness metric proposed by
        Hasler and Süsstrunk (2003) in https://infoscience.epfl.ch/
        record/33994/files/HaslerS03.pdf.

        Args:
            r (int): Red component.
            g (int): Green component.
            b (int): Blue component.

        Returns:
            float: Colorfulness metric.

        """
        rg = np.absolute(r - g)
        yb = np.absolute(0.5 * (r + g) - b)

        # Compute the mean and standard deviation of both `rg` and `yb`.
        rg_mean, rg_std = (np.mean(rg), np.std(rg))
        yb_mean, yb_std = (np.mean(yb), np.std(yb))

        # Combine the mean and standard deviations.
        std_root = np.sqrt((rg_std ** 2) + (yb_std ** 2))
        mean_root = np.sqrt((rg_mean ** 2) + (yb_mean ** 2))

        return std_root + (0.3 * mean_root)


def main():
    filename = '../fish'
    image = matplotlib.image.imread(f"{filename}.ppm")  # Load the image

    start = time.time()
    colorfulness = SpotifyBackgroundColor(img=image)
    best_color = colorfulness.best_color()
    print(best_color)
    end = time.time()
    print(f"colorfulness kmeans took {(end - start):.4f} seconds")

    start = time.time()
    colorfulness = SpotifyBackgroundColor(img=image, _type='okm')
    best_color = colorfulness.best_color()
    print(best_color)
    end = time.time()
    print(f"colorfulness okm took {(end - start):.4f} seconds")


if __name__ == '__main__':
    main()
