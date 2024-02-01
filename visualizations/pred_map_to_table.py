import os
from hashlib import sha1

import numpy as np
import pandas as pd

from atoms_detection.dl_detection import DLDetection
from atoms_detection.dataset import CoordinatesDataset
from utils.constants import Split, ModelArgs
from utils.paths import PT_DATASET, PREDS_PATH, DETECTION_PATH, PRED_MAP_TABLE_LOGS


threshold = 0.89
extension_name = "replicate"
detections_path = os.path.join(DETECTION_PATH, f"dl_detection_{extension_name}_{threshold}")
inference_cache_path = os.path.join(PREDS_PATH, os.path.basename(detections_path))


def get_pred_map(img_filename: str) -> np.ndarray:
    img_hash = sha1(img_filename.encode()).hexdigest()
    prediciton_cache = os.path.join(inference_cache_path, f"{img_hash}.npy")
    if not os.path.exists(prediciton_cache):
        detection = DLDetection(
            model_name=ModelArgs.BASICCNN,
            ckpt_filename="/home/fpares/PycharmProjects/stem_atoms/models/basic_replicate.ckpt",
            dataset_csv="/home/fpares/PycharmProjects/stem_atoms/dataset/Coordinate_image_pairs.csv",
            threshold=threshold,
            detections_path=detections_path
        )
        img = DLDetection.open_image(image_path)
        pred_map = detection.image_to_pred_map(img)
        np.save(prediciton_cache, pred_map)
    else:
        pred_map = np.load(prediciton_cache)
    return pred_map


if not os.path.exists(PRED_MAP_TABLE_LOGS):
    os.makedirs(PRED_MAP_TABLE_LOGS)

coordinates_dataset = CoordinatesDataset(PT_DATASET)
for image_path, coordinates_path in coordinates_dataset.iterate_data(Split.TEST):
    pred_map = get_pred_map(image_path)

    pred_table = {'X': [], 'Y': [], 'Z': []}
    for index, likelihood in np.ndenumerate(pred_map):
        pred_table['X'].append(index[0])
        pred_table['Y'].append(index[1])
        pred_table['Z'].append(likelihood)

    pred_df = pd.DataFrame(pred_table)

    img_name = os.path.splitext(os.path.basename(image_path))[0]
    pred_table_output_path = os.path.join(PRED_MAP_TABLE_LOGS, f"{img_name}_likelihood_{extension_name}_{threshold}.csv")
    pred_df.to_csv(pred_table_output_path, index=False)
