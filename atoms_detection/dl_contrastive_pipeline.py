from typing import List

import argparse
import os

from atoms_detection.create_crop_dataset import create_contrastive_crops_dataset
from atoms_detection.dl_detection import DLDetection
from atoms_detection.dl_detection_with_gmm import DLGMMdetection
from atoms_detection.evaluation import Evaluation
from atoms_detection.training_model import train_model
from utils.paths import (
    CROPS_PATH,
    CROPS_DATASET,
    MODELS_PATH,
    LOGS_PATH,
    DETECTION_PATH,
    PREDS_PATH,
    PRED_GT_VIS_PATH,
)
from utils.constants import ModelArgs, Split
from matplotlib import pyplot as plt
import pandas as pd
from PIL import Image
import numpy as np

from visualizations.prediction_gt_images import get_gt_coords
from visualizations.utils import plot_gt_pred_on_img


def dl_full_pipeline(
    extension_name: str,
    architecture: ModelArgs,
    coords_csv: str,
    thresholds_list: List[float],
    force_create_dataset: bool = False,
    force_evaluation: bool = False,
    show_sampling_image: bool = False,
    train: bool = False,
    visualise: bool = False,
    upsample: bool = False,
    upsample_neg_amount: float = 0,
    clip_max: float = 1,
    negative_dist: float = 1.1,
):
    # Create crops data
    crops_folder = CROPS_PATH + f"_{extension_name}"
    crops_dataset = CROPS_DATASET.replace(".csv", f"_{extension_name}.csv")
    print(os.path.exists(crops_dataset))
    if force_create_dataset or not os.path.exists(crops_dataset):
        print("Creating crops dataset...")
        create_contrastive_crops_dataset(
            crops_folder,
            coords_csv,
            crops_dataset,
            show_sampling_result=show_sampling_image,
            pos_data_upsampling=upsample,
            neg_upsample_multiplier=upsample_neg_amount,
            contrastive_distance_multiplier=negative_dist,
        )  # , clip=clip_max
        # training DL model
    ckpt_filename = os.path.join(MODELS_PATH, f"model_{extension_name}.ckpt")
    if train or not os.path.exists(ckpt_filename):
        print("Training DL crops model...")
        train_model(architecture, crops_dataset, crops_folder, ckpt_filename)

    for threshold in thresholds_list:
        inference_cache_path = os.path.join(
            PREDS_PATH, f"dl_detection_{extension_name}"
        )
        detections_path = os.path.join(
            DETECTION_PATH,
            f"dl_detection_{extension_name}",
            f"dl_detection_{extension_name}_{threshold}",
        )
        if force_evaluation or visualise or not os.path.exists(detections_path):
            print(f"Detecting atoms on test data with threshold={threshold}...")
            if args.run_gmm_for_multimers:
                detection_pipeline = DLGMMdetection
            else:
                detection_pipeline = DLDetection

            detection = detection_pipeline(
                model_name=architecture,
                ckpt_filename=ckpt_filename,
                dataset_csv=coords_csv,
                threshold=threshold,
                detections_path=detections_path,
                inference_cache_path=inference_cache_path,
            )
            detection.run()

        logging_filename = os.path.join(
            LOGS_PATH,
            f"dl_evaluation_{extension_name}",
            f"dl_evaluation_{extension_name}_{threshold}.csv",
        )
        if force_evaluation or visualise or not os.path.exists(logging_filename):
            evaluation = Evaluation(
                coords_csv=coords_csv,
                predictions_path=detections_path,
                logging_filename=logging_filename,
            )
            evaluation.run()
        if visualise:
            vis_folder = os.path.join(
                PRED_GT_VIS_PATH, f"dl_detection_{extension_name}"
            )
            if not os.path.exists(vis_folder):
                os.makedirs(vis_folder)

            vis_folder = os.path.join(
                vis_folder, f"dl_detection_{extension_name}_{threshold}"
            )
            if not os.path.exists(vis_folder):
                os.makedirs(vis_folder)
            is_evaluation = True
            if is_evaluation:
                gt_coords_dict = get_gt_coords(evaluation.coordinates_dataset)

            for image_path in detection.image_dataset.iterate_data(Split.TEST):
                img_name = os.path.split(image_path)[-1]
                gt_coords = gt_coords_dict[img_name] if is_evaluation else None
                pred_df_path = os.path.join(
                    detections_path, os.path.splitext(img_name)[0] + ".csv"
                )
                df_predicted = pd.read_csv(pred_df_path)
                pred_coords = [
                    (row["x"], row["y"]) for _, row in df_predicted.iterrows()
                ]
                img = Image.open(image_path)
                img_arr = np.array(img).astype(np.float32)
                img_normed = (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min())

                plot_gt_pred_on_img(img_normed, gt_coords, pred_coords)
                clean_image_name = os.path.splitext(img_name)[0]
                vis_path = os.path.join(vis_folder, f"{clean_image_name}.png")
                plt.savefig(
                    vis_path, bbox_inches="tight", pad_inches=0.0, transparent=True
                )
                plt.close()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("extension_name", type=str, help="Experiment extension name")
    parser.add_argument(
        "architecture", type=ModelArgs, choices=ModelArgs, help="Architecture name"
    )
    parser.add_argument(
        "coords_csv", type=str, help="Coordinates CSV file to use as input"
    )
    parser.add_argument(
        "-t" "--thresholds", nargs="+", type=float, help="Threshold value"
    )
    parser.add_argument(
        "-c", type=float, default=1, help="Clipping quantile (0..1]. CURRENTLY USELESS!"
    )
    parser.add_argument(
        "-nd", type=float, default=1.1, help="Negative contrastive crop distance"
    )
    parser.add_argument("--force_create_dataset", action="store_true")
    parser.add_argument("--force_evaluation", action="store_true")
    parser.add_argument("--show_sampling_result", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--visualise", action="store_true")
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument(
        "--run_gmm_for_multimers",
        action="store_true",
        help="If selected, a postprocessing will be run to split large atoms (possible multimers) with a GMM",
    )
    parser.add_argument(
        "--upsample_neg",
        type=float,
        default=0,
        help="Upsample amount for negative crops during training",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    print(args)
    dl_full_pipeline(
        args.extension_name,
        args.architecture,
        args.coords_csv,
        args.t__thresholds,
        args.force_create_dataset,
        args.force_evaluation,
        args.show_sampling_result,
        args.train,
        args.visualise,
        args.upsample,
        args.upsample_neg,
        args.c,
        args.nd,
    )
