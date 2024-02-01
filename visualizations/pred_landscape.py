import os
from hashlib import sha1

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

from atoms_detection.dl_detection import DLDetection
from atoms_detection.dataset import CoordinatesDataset
from utils.constants import Split, ModelArgs
from utils.paths import PT_DATASET, PREDS_PATH, DETECTION_PATH,LANDS_VIS_PATH


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


def short_proj():
    return np.dot(Axes3D.get_proj(ax), scale)


if not os.path.exists(LANDS_VIS_PATH):
    os.makedirs(LANDS_VIS_PATH)

coordinates_dataset = CoordinatesDataset(PT_DATASET)
for image_path, coordinates_path in coordinates_dataset.iterate_data(Split.TEST):
    pred_map = get_pred_map(image_path)

    """
    Scaling is done from here...
    """
    x_scale = 1
    y_scale = 1
    z_scale = 0.1

    scale = np.diag([x_scale, y_scale, z_scale, 1.0])
    scale = scale * (1.0 / scale.max())
    scale[3, 3] = 1.0

    X = np.arange(0, 512, 1)
    Y = np.arange(0, 512, 1)
    X, Y = np.meshgrid(X, Y)

    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection='3d')
    ax.get_proj = short_proj
    surf = ax.plot_surface(X, Y, pred_map, cmap=cm.coolwarm,
                           rstride=2, cstride=2,
                           linewidth=0.2, antialiased=True)

    ax.set_axis_off()

    img_name = os.path.splitext(os.path.basename(image_path))[0]
    landscape_output_path = os.path.join(LANDS_VIS_PATH, f"{img_name}_landscape_{extension_name}_{threshold}.png")
    plt.savefig(landscape_output_path, bbox_inches='tight', pad_inches=0.0, transparent=True)
    # plt.show()
