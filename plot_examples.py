from helpers import make_data, _make_box_pts
import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import line


fig, ax = plt.subplots(1, 3, figsize=(12, 4))


def plot(ax, img, label, title):
    ax.imshow(img, cmap="gray")
    ax.set_title(title)

    if label.size > 0:
        x, y, _, _, _ = label
        ax.scatter(x, y, c="r")

        xy = _make_box_pts(*label)
        ax.plot(xy[:, 0], xy[:, 1], c="r")


img, label = make_data(has_spaceship=True)
plot(ax[0], img, label, "example (with spaceship)")

img, label = make_data(has_spaceship=False)
plot(ax[1], img, label, "example (without spaceship)")


img, label = make_data(has_spaceship=True, no_lines=0, noise_level=0)
plot(ax[2], img, label, "example (without noise)")

fig.savefig("example.png")

