import numpy as np
from skimage.draw import polygon_perimeter, line
from shapely.geometry import Polygon
from typing import Tuple, Optional


def _rotation(pts: np.ndarray, theta: float) -> np.ndarray:
    r = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pts = pts @ r
    return pts


def _make_box_pts(
    pos_x: float, pos_y: float, yaw: float, dim_x: float, dim_y: float
) -> np.ndarray:

    hx = dim_x / 2
    hy = dim_y / 2

    pts = np.asarray([(-hx, -hy), (-hx, hy), (hx, hy), (hx, -hy)])
    pts = _rotation(pts, yaw)
    pts += (pos_x, pos_y)
    return pts


def _make_spaceship(
    pos: np.asarray, yaw: float, scale: float, l2w: float, t2l: float
) -> Tuple[np.ndarray, np.ndarray]:

    dim_x = scale
    dim_y = scale * l2w

    # spaceship
    x1 = (0, dim_y)
    x2 = (-dim_x / 2, 0)
    x3 = (0, dim_y * t2l)
    x4 = (dim_x / 2, 0)
    pts = np.asarray([x1, x2, x3, x4])
    pts[:, 1] -= dim_y / 2

    # rotation + translation
    pts = _rotation(pts, yaw)
    pts += pos

    # label
    # pos_y, pos_x, yaw, dim_x, dim_y
    params = np.asarray([*pos, yaw, dim_x, dim_y])

    return pts, params


def _get_pos(s: float) -> np.ndarray:
    return np.random.randint(10, s - 10, size=2)


def _get_yaw() -> float:
    return np.random.rand() * 2 * np.pi


def _get_size() -> int:
    return np.random.randint(18, 37)


def _get_l2w() -> float:
    return abs(np.random.normal(3 / 2, 0.2))


def _get_t2l() -> float:
    return abs(np.random.normal(1 / 3, 0.1))


def make_data(
    has_spaceship: bool = None,
    noise_level: float = 0.8,
    no_lines: int = 6,
    image_size: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """ Data generator

    Args:
        has_spaceship (bool, optional): Whether a spaceship is included. Defaults to None (randomly sampled).
        noise_level (float, optional): Level of the background noise. Defaults to 0.8.
        no_lines (int, optional): No. of lines for line noise. Defaults to 6.
        image_size (int, optional): Size of generated image. Defaults to 200.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Generated Image and the corresponding label
        The label parameters are x, y, yaw, x size, and y size respectively
        An empty array is returned when a spaceship is not included.
    """

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

    # combined noise
    noise = noise_level * np.random.rand(image_size, image_size)
    img = np.stack([img, noise, line_noise], axis=0).max(axis=0)

    img = img.T  # ensure image space matches with coordinate space

    return img, label


def score_iou(ypred: np.ndarray, ytrue: np.ndarray) -> Optional[float]:

    assert (
        ypred.size == ytrue.size == 5
    ), "Inputs should have 5 parameters, use null array for empty predictions/labels."

    no_pred = np.any(np.isnan(ypred))
    no_label = np.any(np.isnan(ytrue))

    if no_label and no_pred:
        # true negative
        return None
    elif no_label and not no_pred:
        # false positive
        return 0
    elif not no_label and not no_pred:
        # true positive
        t = Polygon(_make_box_pts(*ytrue))
        p = Polygon(_make_box_pts(*ypred))
        iou = t.intersection(p).area / t.union(p).area
        return iou
    elif not no_label and no_pred:
        # false negative
        return 0
    else:
        raise NotImplementedError
