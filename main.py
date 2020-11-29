import argparse

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from train import load_model, predict
from helpers import make_data, score_iou


def evaluate(model_file):
    model = tf.keras.models.load_model(model_file)

    ious = []
    for _ in tqdm(range(1000)):
        img, label = make_data()
        pred = model.predict(img[None])  # batch size 1
        ious.append(score_iou(label, pred))

    ious = np.asarray(ious, dtype="float")
    ious = ious[~np.isnan(ious)]  # remove true negatives

    print()
    print("score:", (ious > 0.7).mean())


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Reproduce model score.")
    parser.add_argument("--model_file", type=str, default="model.hdf5")
    args = parser.parse_args()

    evaluate(**args.__dict__)
