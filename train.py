import os
import argparse

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPool2D,
    BatchNormalization,
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


def generate_model(task):
    model = Sequential()
    model.add(
        Reshape((IMAGE_SIZE, IMAGE_SIZE, 1), input_shape=(IMAGE_SIZE, IMAGE_SIZE))
    )

    # 200
    # 100
    # 50
    # 25
    # 12
    # 6
    # 3
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
                use_bias=True,
                padding="same",
                activation="relu",
            )
        )
        model.add(BatchNormalization())
        model.add(MaxPool2D())

    model.add(Flatten())

    if task == "classification":
        model.add(Dense(1, activation="sigmoid"))
    elif task == "regression":
        model.add(Dense(5, activation="sigmoid"))
    else:
        raise ValueError("task must be one of classification or regression")

    return model


def make_batch(batch_size, task):
    # uses 50/50 split (i%2==0) if in classification mode, otherwise we'll use
    # always create a spaceship for the regression task
    imgs, labels = zip(
        *[
            make_data(has_spaceship=i % 2 == 0 if task == "classification" else True)
            for i in range(batch_size)
        ]
    )
    imgs = [preprocess_image(img) for img in imgs]

    if task == "classification":
        labels = [0 if np.any(np.isnan(label)) else 1 for label in labels]
    elif task == "regression":
        labels = [label / SCALE_VECTOR for label in labels]  # normalize labels
    else:
        raise ValueError("task must be one of classification or regression")

    imgs = np.stack(imgs)
    labels = np.stack(labels)
    return imgs, labels


class BBRMetrics(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.mean_iou = []
        self.score = []
        self.loss = []

    def on_epoch_end(self, epoch, logs=None):
        images, labels = self.model.validation_data
        preds = np.asarray(self.model.predict(images))
        preds = [pred * SCALE_VECTOR for pred in preds]  # denormalize preds
        ious = np.array([score_iou(pred, label) for pred, label in zip(preds, labels)])

        mean_iou = ious.mean()
        score = (ious > 0.7).mean()

        print(f" -- mean_iou: {mean_iou:.3f} -- score: {score:.3f}")
        if self.score and score > max(self.score):
            self.model.save(self.model.best_model_path)
            print(f"\n\tnew best model save to {self.model.best_model_path}\n")

        self.mean_iou.append(mean_iou)
        self.score.append(score)
        self.loss.append(logs["loss"])


class BBRModel:
    def __init__(self, model=None):
        self.task = "regression"
        self.model = model

    def train(
        self,
        batch_size,
        lr,
        steps_per_epoch,
        epochs,
        n_val_examples,
        save_dir,
        **kwargs,
    ):
        # model
        model = generate_model(self.task)
        model.compile(loss="mse", optimizer=tf.optimizers.Adam(lr=lr))
        model.summary()

        # validation data
        val_images, val_labels = make_batch(n_val_examples, self.task)
        val_labels = [val_label * SCALE_VECTOR for val_label in val_labels]
        model.validation_data = (val_images, val_labels)
        model.best_model_path = os.path.join(save_dir, "best_model.hdf5")

        # fit
        metrics = BBRMetrics()
        history = model.fit(
            iter(lambda: make_batch(batch_size, self.task), None),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=[metrics],
        )

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

        # save metrics plot
        history.history["loss"] = metrics.loss
        history.history["mean_iou"] = metrics.mean_iou
        history.history["score"] = metrics.score

        self.model = model
        return history

    def predict(self, image):
        image = preprocess_image(image)
        pred = self.model.predict(image[None])
        pred = np.squeeze(pred)
        pred *= SCALE_VECTOR  # denormalize predictions
        return pred

    @classmethod
    def load_model(cls, filename):
        model = tf.keras.models.load_model(filename)
        return cls(model)


class ClassificationModel:
    def __init__(self, model=None):
        self.task = "classification"
        self.model = model

    def train(self, batch_size, lr, steps_per_epoch, epochs, save_dir, **kwargs):
        # model
        model = generate_model(self.task)
        model.compile(
            loss="binary_crossentropy",
            optimizer=tf.optimizers.Adam(lr=lr),
            metrics="accuracy",
        )
        model.summary()

        # fit
        best_model_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(save_dir, "best_model.hdf5"),
            save_weights_only=False,
            monitor="accuracy",
            mode="max",
            save_best_only=True,
        )
        history = model.fit(
            iter(lambda: make_batch(batch_size, self.task), None),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=[best_model_callback],
        )
        self.model = model
        return history

    def predict(self, image):
        image = preprocess_image(image)
        pred = self.model.predict(image[None])
        pred = round(float(np.squeeze(pred)))
        return pred

    @classmethod
    def load_model(cls, filename):
        model = tf.keras.models.load_model(filename)
        return cls(model)


class CombinedModel:
    def __init__(self, clf_model, reg_model):
        self.clf_model = clf_model
        self.reg_model = reg_model

    def predict(self, image):
        image = preprocess_image(image)
        if self.clf_model.predict(image) == 1:
            return self.reg_model.predict(image)
        else:
            return np.full(5, np.nan)


def train(model, args):
    # save locations
    os.makedirs(args.save_dir, exist_ok=True)
    model_save_file = os.path.join(args.save_dir, "model.hdf5")
    model_summary_file = os.path.join(args.save_dir, "summary.txt")
    train_metrics_file = os.path.join(args.save_dir, "metrics.png")

    # train
    history = model.train(**args.__dict__)

    # save model and summary
    model.model.save(model_save_file)
    with open(model_summary_file, "w") as fo:
        model.model.summary(print_fn=lambda x: fo.write(x + "\n"))

    # train metrics
    tb_writer = SummaryWriter(log_dir=args.save_dir)
    nplots = len(history.history)
    t = list(range(args.epochs))
    _, ax = plt.subplots(1, nplots, figsize=(nplots * 6, 8))
    for i, (k, v) in enumerate(history.history.items()):
        ax[i].plot(t, v)
        ax[i].set_title(f"{k} ({v[-1]:.3f})")
        for epoch, val in enumerate(v):
            tb_writer.add_scalar(k, val, epoch * args.batch_size * args.steps_per_epoch)
    plt.savefig(train_metrics_file)
    tb_writer.close()

    # done
    print("--- done ---")
    print(f"    model saved to {model_save_file}")
    print(f"    model summary saved to {model_summary_file}")
    print(f"    train metrics plot saved to {train_metrics_file}")

    # save
    os.makedirs(args.save_dir, exist_ok=True)
    model_save_file = os.path.join(args.save_dir, "model.hdf5")
    model_summary_file = os.path.join(args.save_dir, "summary.txt")
    train_metrics_file = os.path.join(args.save_dir, "metrics.png")


if __name__ == "__main__":
    # cli
    parser = argparse.ArgumentParser("Train script to reproduce model.")
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        help=(
            "Task type, either classification, regression, or all. classification "
            "will train a binary model that predicts if a spaceship exists in "
            "an image. Regression predicts a bounding box of the spaceship in "
            "an image. all will train both models."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Train batch size.")
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=250,
        help="Number of training steps (batches) per epoch",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs."
    )
    parser.add_argument(
        "--n_val_examples",
        type=int,
        default=1000,
        help="Number of examples to evaluate at every epoch. Only applicable to regression",
    )
    parser.add_argument("--lr", type=int, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="models",
        help="Save directory for model, metrics, summary, and tensorboard logs.",
    )
    args = parser.parse_args()

    if args.task == "all":
        train(BBRModel(), args)
        train(ClassificationModel(), args)
    elif args.task == "classification":
        train(BBRModel(), args)
    elif args.task == "regression":
        train(ClassificationModel(), args)
    else:
        raise ValueError("task must be one of classification, regression, or all")

