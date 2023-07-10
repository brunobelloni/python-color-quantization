import imageio
import matplotlib.pyplot as plt
import numpy as np

images = []


def plot_clusters_3d(x, cluster, step=0):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set the labels for X, Y, and Z axes
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')

    # Assign colors to points based on their nearest centroid
    distances = np.linalg.norm(x[:, np.newaxis] - cluster, axis=2)
    nearest_centroids = np.argmin(distances, axis=1)
    colors = cluster[nearest_centroids]

    ax.set_xlim([0, 255])
    ax.set_ylim([0, 255])
    ax.set_zlim([0, 255])

    colors_normalized = colors / 255.0
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=colors_normalized, alpha=0.025, zorder=2, marker='o')

    ax.scatter(
        cluster[:, 0], cluster[:, 1], cluster[:, 2],
        c='red',
        marker='x',
        s=100,
        linewidth=5,
        alpha=1.0,
        zorder=10,
    )

    # Add legend
    legend_elements = []
    for i, centroid in enumerate(cluster):
        legend_elements.append(plt.Line2D(
            [0], [0],
            color='w',
            marker='o',
            markersize=10,
            markerfacecolor=centroid / 255.0,
            label=f'Centroid {i} - ({int(centroid[0])}, {int(centroid[1])}, {int(centroid[2])})',
        ))

    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(-0.375, 1.15))

    # Add step information at the bottom of the image
    ax.text2D(0.5, -0.1, f'Step: {step}', transform=ax.transAxes, ha='center')

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)

    images.append(image)


def save_gif():
    imageio.mimsave('3d_clusters_iokm.gif', images)
