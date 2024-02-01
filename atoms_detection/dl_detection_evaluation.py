import argparse
import os
import random
import numpy as np
from matplotlib import pyplot as plt

from PIL import Image
from networkx.tests.test_convert_pandas import pd

from atoms_detection.dl_detection import DLDetection
from atoms_detection.dl_detection_scaled import DLScaled
from atoms_detection.evaluation import Evaluation
from utils.paths import MODELS_PATH, LOGS_PATH, DETECTION_PATH, PREDS_PATH, FE_DATASET, PRED_GT_VIS_PATH
from utils.constants import ModelArgs, Split
from visualizations.prediction_gt_images import plot_gt_pred_on_img, get_gt_coords


def detection_pipeline(args):
    extension_name = args.extension_name
    print(f"Storing at {extension_name}")
    architecture = ModelArgs.BASICCNN
    ckpt_filename = os.path.join(MODELS_PATH, "model_sac_cnn.ckpt")

    inference_cache_path = os.path.join(PREDS_PATH, f"dl_detection_{extension_name}")

    testing_thresholds = [0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    testing_thresholds = [0.8, 0.85, 0.9, 0.95]
    for threshold in testing_thresholds:
        detections_path = os.path.join(DETECTION_PATH, f"dl_detection_{extension_name}",
                                       f"dl_detection_{extension_name}_{threshold}")
        print(f"Detecting atoms on test data with threshold={threshold}...")
        if args.experimental_rescale:
            print("Using experimental ruler rescaling")
            detection = DLScaled(
                model_name=architecture,
                ckpt_filename=ckpt_filename,
                dataset_csv=args.dataset,
                threshold=threshold,
                detections_path=detections_path,
                inference_cache_path=inference_cache_path
            )
        else:
            detection = DLDetection(
                model_name=architecture,
                ckpt_filename=ckpt_filename,
                dataset_csv=args.dataset,
                threshold=threshold,
                detections_path=detections_path,
                inference_cache_path=inference_cache_path
            )
        detection.run()
        if args.eval:
            logging_filename = os.path.join(LOGS_PATH, f"dl_detection_{extension_name}",
                                            f"dl_detection_{extension_name}_{threshold}.csv")
            evaluation = Evaluation(
                coords_csv=args.dataset,
                predictions_path=detections_path,
                logging_filename=logging_filename
            )
            evaluation.run()
        if args.visualise:

            vis_folder = os.path.join(PRED_GT_VIS_PATH, f"dl_detection_{extension_name}")
            if not os.path.exists(vis_folder):
                os.makedirs(vis_folder)

            vis_folder = os.path.join(vis_folder, f"dl_detection_{extension_name}_{threshold}")
            if not os.path.exists(vis_folder):
                os.makedirs(vis_folder)

            if args.eval:
                gt_coords_dict = get_gt_coords(evaluation.coordinates_dataset)

            for image_path in detection.image_dataset.iterate_data(Split.TEST):
                print(image_path)
                img_name = os.path.split(image_path)[-1]
                gt_coords = gt_coords_dict[img_name] if args.eval else None
                pred_df_path = os.path.join(detections_path, os.path.splitext(img_name)[0]+'.csv')
                df_predicted = pd.read_csv(pred_df_path)
                pred_coords = [(row['x'], row['y']) for _, row in df_predicted.iterrows()]
                img = Image.open(image_path)
                img_arr = np.array(img).astype(np.float32)
                img_normed = (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min())

                plot_gt_pred_on_img(img_normed, gt_coords, pred_coords)
                clean_image_name = os.path.splitext(img_name)[0]
                vis_path = os.path.join(vis_folder, f'{clean_image_name}.png')
                plt.savefig(vis_path, bbox_inches='tight', pad_inches=0.0, transparent=True)
                plt.close()

    print(f"Experiment {extension_name} completed")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "extension_name",
        type=str,
        help="Experiment extension name"
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Dataset file upon which to do inference"
    )
    parser.add_argument(
        "--eval",
        action='store_true',
        help="Whether to perform evaluation after inference",
        default=False
    )
    parser.add_argument(
        "--visualise",
        action='store_true',
        help="Whether to store inference results as visual png images",
        default=False
    )
    parser.add_argument(
        "--experimental_rescale",
        action='store_true',
        help="Whether to rescale inputs based on the ruler of the image as preprocess",
        default=False
    )
    parser.add_argument('--feature', )
    return parser.parse_args()


if __name__=='__main__':
    args = get_args()
    detection_pipeline(args)
