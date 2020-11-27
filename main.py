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
