import argparse

import numpy as np
from tqdm import tqdm

from helpers import make_data, score_iou
from train import ClassificationModel, BBRModel, CombinedModel, preprocess_image


def evaluate(model, batch_size, num_samples):
    images, labels = zip(
        *[make_data() for _ in tqdm(range(num_samples), desc="creating data")]
    )
    images = [preprocess_image(img) for img in tqdm(images, desc="preprocessing")]
    images = np.stack(images)

    predictions, _, _ = model.predict_batch(images, batch_size)

    ious = [score_iou(label, pred) for pred, label in zip(predictions, labels)]
    ious = np.asarray(ious, dtype="float")
    ious = ious[~np.isnan(ious)]  # remove true negatives
    score = (ious > 0.7).mean()

    print()
    print(f"score: {score:.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Reproduce model score.")
    parser.add_argument(
        "--clf_model_file", type=str, default="models/clf/best_model.hdf5"
    )
    parser.add_argument(
        "--reg_model_file", type=str, default="models/clf/best_model.hdf5"
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_samples", type=int, default=1000)
    args = parser.parse_args()

    clf_model = ClassificationModel.load_model(args.clf_model_file)
    reg_model = BBRModel.load_model(args.reg_model_file)
    model = CombinedModel(clf_model, reg_model)

    evaluate(model, args.batch_size, args.num_samples)
