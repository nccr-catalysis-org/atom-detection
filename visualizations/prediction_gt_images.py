import argparse
import os
from enum import Enum

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from atoms_detection.dataset import CoordinatesDataset
from utils.constants import Split, Catalyst, Method
from utils.paths import DETECTION_LOGS, IMG_PATH, PRED_GT_VIS_PATH, PT_DATASET, FE_DATASET, DETECTION_PATH
from visualizations.utils import plot_gt_pred_on_img


def main(args):
    catalyst = args.catalyst
    method = args.method
    if not os.path.exists(PRED_GT_VIS_PATH):
        os.makedirs(PRED_GT_VIS_PATH)
    if catalyst == Catalyst.Pt:
        coordinates_dataset = CoordinatesDataset(PT_DATASET)
        if method == Method.DL:
            detection_path = "data/detection_data/dl_detection_sac_cnn/dl_detection_sac_cnn_0.89"
        elif method == Method.CV:
            detection_path = os.path.join(DETECTION_PATH, "cv_detection_trial_0.18")
        elif method == Method.TEM:
            detection_path = os.path.join(DETECTION_PATH, "tem_imagenet_pt",
                                          "tem_imagenet_pt_denoise-bg_Gen1GaussianMask")
        else:
            raise NotImplementedError

    elif catalyst == Catalyst.Fe:
        coordinates_dataset = CoordinatesDataset(FE_DATASET)
        if method == Method.DL:
            detection_path = os.path.join(DETECTION_PATH, f"dl_fe_detection_trial",
                                          f"dl_fe_detection_trial_0.97")
        elif method == Method.CV:
            detection_path = os.path.join(DETECTION_PATH, "cv_fe_detection_trial",
                                          "cv_fe_detection_trial_0.21")
        elif method == Method.TEM:
            detection_path = os.path.join(DETECTION_PATH, "tem_imagenet_fe",
                                          "tem_imagenet_fe_denoise-bg_Gen1GaussianMask")
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError

    gt_coords_dict = get_gt_coords(coordinates_dataset)
    for name_file in os.listdir(detection_path):
        image_name = os.path.splitext(name_file)[0] + ".tif"
        print(image_name)
        if image_name not in gt_coords_dict:
            continue

        filepath = os.path.join(detection_path, name_file)
        image_filename = os.path.join(IMG_PATH, image_name)
        img = Image.open(image_filename)

        gt_coords = gt_coords_dict[image_name]
        df_predicted = pd.read_csv(filepath)
        pred_coords = [(row['x'], row['y']) for _, row in df_predicted.iterrows()]

        img_arr = np.array(img).astype(np.float32)
        img_normed = (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min())

        plot_gt_pred_on_img(img_normed, gt_coords, pred_coords)

        vis_folder = os.path.join(PRED_GT_VIS_PATH, f"{catalyst}-Catalyst_{method}-Method")
        if not os.path.exists(vis_folder):
            os.makedirs(vis_folder)

        clean_image_name = os.path.splitext(image_name)[0]
        vis_path = os.path.join(vis_folder, f'{clean_image_name}.png')
        plt.savefig(vis_path, bbox_inches='tight', pad_inches=0.0, transparent=True)
        plt.close()



def get_gt_coords(coordinates_dataset):
    gt_coords_dict = {}
    for image_path, coordinates_path in coordinates_dataset.iterate_data(Split.TEST):
        # orig . image_name = os.path.splitext(os.path.basename(image_path))[0] + ".tif"
        image_name = os.path.basename(image_path)
        gt_coords = coordinates_dataset.load_coordinates(coordinates_path)
        gt_coords_dict[image_name] = gt_coords
    return gt_coords_dict


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "catalyst",
        type=Catalyst,
        choices=Catalyst,
        help="Select data by catalyst"
    )
    parser.add_argument(
        "method",
        type=Method,
        choices=Method,
        help="Select method"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
