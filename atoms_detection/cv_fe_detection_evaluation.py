import os

from atoms_detection.cv_detection import CVDetection
from atoms_detection.evaluation import Evaluation
from utils.paths import CROPS_PATH, CROPS_DATASET, MODELS_PATH, LOGS_PATH, DETECTION_PATH, PREDS_PATH, DATASET_PATH
from utils.constants import ModelArgs


extension_name = "trial"
threshold = 0.21
architecture = ModelArgs.BASICCNN
ckpt_filename = os.path.join(MODELS_PATH, "basic_replicate.ckpt")
dataset_csv = os.path.join(DATASET_PATH, "Fe_dataset.csv")


inference_cache_path = os.path.join(PREDS_PATH, f"cv_fe_detection_{extension_name}")

for threshold in [0.1, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25]:
    detections_path = os.path.join(DETECTION_PATH, f"cv_fe_detection_{extension_name}",
                                   f"cv_fe_detection_{extension_name}_{threshold}")
    print(f"Detecting atoms on test data with threshold={threshold}...")
    detection = CVDetection(
        dataset_csv=dataset_csv,
        threshold=threshold,
        detections_path=detections_path,
        inference_cache_path=inference_cache_path
    )
    detection.run()

    logging_filename = os.path.join(LOGS_PATH, f"cv_fe_evaluation_{extension_name}",
                                    f"cv_fe_evaluation_{extension_name}_{threshold}.csv")
    evaluation = Evaluation(
        coords_csv=dataset_csv,
        predictions_path=detections_path,
        logging_filename=logging_filename
    )
    evaluation.run()
