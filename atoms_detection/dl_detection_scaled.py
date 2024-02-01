import os
from _sha1 import sha1
from typing import Tuple, List

from PIL import Image

from atoms_detection.dl_detection import DLDetection
from atoms_detection.image_preprocessing import dl_prepro_image
from utils.constants import ModelArgs
import numpy as np

class DLScaled(DLDetection):
    # Should take into account for the resize:
    #  Ruler of the image (pixelsxnm)
    #  Covalent radius
    #  beam size/voltage/exposure? (can create larger distortions) (Metadata should be in dm3 files, if it can be read)
    def __init__(self,
                 model_name: ModelArgs,
                 ckpt_filename: str,
                 dataset_csv: str,
                 threshold: float,
                 detections_path: str,
                 inference_cache_path: str):
        super().__init__(model_name, ckpt_filename,dataset_csv,threshold,detections_path, inference_cache_path)

    def image_to_pred_map(self, img: np.ndarray) -> np.ndarray:
        ruler_units = self.image_dataset.get_ruler_units_by_img_name(self.currently_processing)
        preprocessed_img, scale_factor = dl_prepro_image(img, ruler_units=ruler_units)
        padded_image = self.padding_image(preprocessed_img)
        pred_map = self.get_prediction_map(padded_image)

        new_dimensions = img.shape[0], img.shape[1]
        pred_map = np.array(Image.fromarray(pred_map).resize(new_dimensions))
        return pred_map

    def cache_image_identifier(self, img_filename):
        x = sha1((img_filename+'scaled').encode()).hexdigest()
        print(x)
        return x



