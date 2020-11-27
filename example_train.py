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
