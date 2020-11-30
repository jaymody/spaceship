import os
import argparse

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
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

from helpers import make_data, score_iou

IMAGE_SIZE = 200
NOISE_LEVEL = 0.8
MAX_WIDTH = 37
MAX_HEIGHT = MAX_WIDTH * 2
SCALE_VECTOR = [IMAGE_SIZE, IMAGE_SIZE, 2 * np.pi, MAX_WIDTH, MAX_HEIGHT]


def preprocess_image(img):
    return np.where(img >= NOISE_LEVEL, img, 0.0)


class RescaleLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scale_vector = None

    def build(self, input_shape):
        self.scale_vector = tf.constant(SCALE_VECTOR)

    def call(self, inputs):
        return tf.keras.backend.map_fn(
            lambda x: tf.keras.layers.Multiply()([x, self.scale_vector]), inputs,
        )


def generate_model():
    model = Sequential()
    model.add(
        Reshape((IMAGE_SIZE, IMAGE_SIZE, 1), input_shape=(IMAGE_SIZE, IMAGE_SIZE))
    )

    for nfilters, kernel_size in [
        (8, 11),
        (16, 9),
        (32, 7),
        (64, 5),
        (128, 3),
        (256, 3),
        (512, 3),
    ]:
        model.add(
            Conv2D(
                nfilters,
                kernel_size=kernel_size,
                use_bias=False,
                padding="same",
                activation="relu",
            )
        )
        model.add(BatchNormalization())
        model.add(MaxPool2D())

    model.add(Flatten())
    model.add(Dense(5))
    return model


def make_batch(batch_size, task="regression"):
    # this model can only train on data where a spaceship is guaranteed, this is not true when testing
    imgs, labels = zip(*[make_data(has_spaceship=True) for _ in range(batch_size)])
    imgs = [preprocess_image(img) for img in imgs]

    if task == "classification":
        labels = [0 if np.any(np.isnan(label)) else 1 for label in labels]
    elif task != "regression":
        raise ValueError("task must be one of classification or regression")

    imgs = np.stack(imgs)
    labels = np.stack(labels)
    return imgs, labels


class Metrics(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.mean_iou = []
        self.score = []
        self.loss = []

    def on_epoch_end(self, epoch, logs=None):
        images, labels = self.model.validation_data
        preds = np.asarray(self.model.predict(images))
        ious = np.array([score_iou(pred, label) for pred, label in zip(preds, labels)])
        self.mean_iou.append(ious.mean())
        self.score.append((ious > 0.7).mean())
        self.loss.append(logs["loss"])


def train(batch_size, steps_per_epoch, epochs, n_val_examples, lr, save_dir, **kwargs):
    # model
    model = generate_model()
    model.compile(loss="mse", optimizer=tf.optimizers.Adam(lr=lr))
    model.summary()

    # validation data
    model.validation_data = make_batch(n_val_examples)

    # fit
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=save_dir)
    metrics = Metrics()
    model.fit(
        iter(lambda: make_batch(batch_size), None),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[tb_callback, metrics],
    )
    return model, metrics


if __name__ == "__main__":
    # cli
    parser = argparse.ArgumentParser("Train script to reproduce model.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--steps_per_epoch", type=int, default=250)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--n_val_examples", type=int, default=1000)
    parser.add_argument("--lr", type=int, default=1e-3)
    parser.add_argument("--save_dir", type=str, default="models")
    args = parser.parse_args()

    # save locations
    os.makedirs(args.save_dir, exist_ok=True)
    model_save_file = os.path.join(args.save_dir, "model.hdf5")
    model_summary_file = os.path.join(args.save_dir, "summary.txt")
    train_metrics_file = os.path.join(args.save_dir, "metrics.png")

    # train
    trained_model, metrics = train(**args.__dict__)
    print()
    print("--- done training ---")
    print(f"             loss = {metrics.loss[-1]:.3f}")
    print(f"         mean_iou = {metrics.mean_iou[-1]:.3f}")
    print(f"            score = {metrics.score[-1]:.3f}")
    print()
    print(f"        best_loss = {min(metrics.loss):.3f}")
    print(f"    best_mean_iou = {max(metrics.mean_iou):.3f}")
    print(f"       best_score = {max(metrics.score):.3f}")
    print()

    # save model and summary
    trained_model.save(model_save_file)
    with open(model_summary_file, "w") as fo:
        trained_model.summary(print_fn=lambda x: fo.write(x + "\n"))

    # save metrics plot
    fig, ax = plt.subplots(1, 3, figsize=(32, 8))
    t = list(range(args.epochs))
    ax[0].plot(t, metrics.loss)
    ax[0].set_title(f"loss ({metrics.loss[-1]:.3f})")
    ax[1].plot(t, metrics.mean_iou)
    ax[1].set_title(f"mean_iou ({metrics.mean_iou[-1]:.3f})")
    ax[2].plot(t, metrics.score)
    ax[2].set_title(f"score ({metrics.score[-1]:.3f})")
    plt.savefig(train_metrics_file)

    print("--- done ---")
    print(f"    model saved to {model_save_file}")
    print(f"    model summary saved to {model_summary_file}")
    print(f"    train metrics plot saved to {train_metrics_file}")
