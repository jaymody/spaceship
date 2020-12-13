# Computer Vision Takehome - Decloaking Spaceship V2

**Problem:**
The goal is to detect spaceships which have been fitted with a cloaking device that makes them less visible. You are expected to use a deep learning model to complete this task. The model will take a single channel image as input and detects the spaceship (if it exists). Not all image will contain a spaceship, but they will contain no more than 1. For any spaceship, the model should predict their bounding box and heading. This can be described using five parameters:

* X and Y position (centre of the bounding box)
* Yaw (direction of heading)
* Width (size tangential to the direction of yaw)
* Height (size along the direct of yaw)

We have supplied a base model as a reference which performs poorly and has some serious limitations. You can extend the existing model or reimplement from scratch in any framework of your choice.

The metric for the model is AP at an IOU threshold of 0.7, for at least 1000 random samples, with the default generation parameters (see `main.py`). Please do not modify any of the generation code directly.

**Evaluation Criteria:**
* Model metric, score as high as you can while being under 2 million trainable parameters. Please streamline the parameters where possible
* Model architecture
* Loss function
* Code readability and maintainability, please follow general python conventions

**Deliverables**
1. Report a final score
1. A summary of the model architecture. E.g. `model.summary()` or `torchsummary`
1. A `train.py` script that allows the same model to be reproduced
1. The final model weights
1. A `requirements.txt` file that includes all python dependencies and their versions
1. A `main.py` file that reproduces the reported score


**Tips:**
* Carefully consider how the loss function should be formulated (especially yaw)
* Sanity check how trainable parameters are distributed in the network
* You may use as many training examples as you want. Both train and test used the same generation function
* You may use existing a codebase but please reference the source
* Submitted solutions achieve 0.5 score on average, but it is possible to achieve near perfect score.
* Any pre/post-processing that can be reproduced at inference is fair game.


**requirements.txt**
```
scikit-image==0.16.2
Shapely==1.7.0
numpy==1.18.1
tqdm==4.48.2
```


**helpers.py**
```python
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
```


**train.py**
```python
import numpy as np
from helpers import make_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPool2D,
    BatchNormalization,
    Activation,
    Reshape,
    Flatten,
    Dense,
)


def gen_model():
    IMAGE_SIZE = 200
    NFILTERS = 8
    CONV_PARAMS = {"kernel_size": 3, "use_bias": False, "padding": "same"}

    model = Sequential()
    model.add(
        Reshape((IMAGE_SIZE, IMAGE_SIZE, 1), input_shape=(IMAGE_SIZE, IMAGE_SIZE))
    )
    for i in [1, 2, 4, 8, 16, 32, 64]:
        model.add(Conv2D(NFILTERS * i, **CONV_PARAMS))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(MaxPool2D())
    model.add(Flatten())
    model.add(Dense(5))
    return model


def make_batch(batch_size):
    # this model can only train on data where a spaceship is guaranteed, this is not true when testing
    imgs, labels = zip(*[make_data(has_spaceship=True) for _ in range(batch_size)])
    imgs = np.stack(imgs)
    labels = np.stack(labels)
    return imgs, labels


def main():
    BATCH_SIZE = 64

    model = gen_model()
    model.compile(loss="mse", optimizer="adam")
    model.summary()

    model.fit_generator(
        iter(lambda: make_batch(BATCH_SIZE), None), steps_per_epoch=500, epochs=30,
    )
    model.save("model.hdf5")
    print("Done")


if __name__ == "__main__":
    main()
```


**main.py**
```python
from helpers import make_data, score_iou
import numpy as np
from tensorflow import keras
from tqdm import tqdm


def eval():
    model = keras.models.load_model("model.hdf5")

    ious = []
    for _ in tqdm(range(1000)):
        img, label = make_data()
        pred = model.predict(img[None])
        pred = np.squeeze(pred)
        ious.append(score_iou(label, pred))

    ious = np.asarray(ious, dtype="float")
    ious = ious[~np.isnan(ious)]  # remove true negatives
    print((ious > 0.7).mean())
    # 0.004


if __name__ == "__main__":
    eval()
```


**plot_examples.py**
```python
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
```
