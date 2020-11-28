import numpy as np
from skimage.draw import polygon_perimeter, line

from helpers import (
    _get_pos,
    _get_yaw,
    _get_size,
    _get_l2w,
    _get_t2l,
    _make_spaceship,
    _make_box_pts,
)


def plot(ax, img, label, title, make_dot=True, make_box=True):
    ax.imshow(img, cmap="gray")
    ax.set_title(title)

    if label.size > 0:
        x, y, _, _, _ = label

        if make_dot:
            ax.scatter(x, y, c="r")

        if make_box:
            xy = _make_box_pts(*label)
            ax.plot(xy[:, 0], xy[:, 1], c="r")


def convert_to_uint8(img):
    return np.array(img * 255, dtype=np.uint8)


def convert_to_float32(img):
    return np.array(img, dtype=np.float32) / 255


def make_data_custom(
    has_spaceship: bool = None,
    noise_level: float = 0.8,
    no_lines: int = 6,
    image_size: int = 200,
):
    if has_spaceship is None:
        has_spaceship = np.random.choice([True, False], p=(0.8, 0.2))

    img = np.zeros(shape=(image_size, image_size))
    label = np.full(5, np.nan)

    # draw ship
    if has_spaceship:
        params = (_get_pos(image_size), _get_yaw(), _get_size(), _get_l2w(), _get_t2l())
        pts, label = _make_spaceship(*params)

        rr, cc = polygon_perimeter(pts[:, 0], pts[:, 1])
        valid = (rr >= 0) & (rr < image_size) & (cc >= 0) & (cc < image_size)

        img[rr[valid], cc[valid]] = np.random.rand(np.sum(valid))

    # noise lines
    line_noise = np.zeros(shape=(image_size, image_size))
    for _ in range(no_lines):
        rr, cc = line(*np.random.randint(0, 200, size=4))
        line_noise[rr, cc] = np.random.rand(rr.size)

    # static noise
    noise = noise_level * np.random.rand(image_size, image_size)

    # compile different image combinations
    images = {
        "spaceship": img.T,
        "noise": noise.T,
        "lines": line_noise.T,
        "noise+lines": np.stack([noise, line_noise], axis=0).max(axis=0).T,
        "spaceship+lines": np.stack([img, line_noise], axis=0).max(axis=0).T,
        "spaceship+noise": np.stack([img, noise], axis=0).max(axis=0).T,
        "img": np.stack([img, noise, line_noise], axis=0).max(axis=0).T,
    }

    return images, label
