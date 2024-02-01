#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2023 March 17, 10:56:06
@last modified : 2023 July 18, 10:25:32
"""

# Naive import of atomdetection, maybe should make a package out of it
from functools import lru_cache
import sys

from .tiff_utils import tiff_to_png
if ".." not in sys.path:
    sys.path.append("..")

import os
import torch
import numpy as np
from PIL import Image
from PIL.Image import Image as PILImage
from typing import Union
from utils.constants import ModelArgs
from utils.paths import MODELS_PATH, DATASET_PATH
from atoms_detection.dl_detection import DLDetection
from atoms_detection.evaluation import Evaluation

LOGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
VOID_DS = os.path.join(DATASET_PATH, "void.csv")
DET_PATH = os.path.join(LOGS_PATH, "detections")
INF_PATH = os.path.join(LOGS_PATH, "inference_cache")

from atoms_detection.create_crop_dataset import create_crop
from atoms_detection.vae_svi_train import SVItrainer, init_dataloader
from atoms_detection.vae_model import rVAE
from sklearn.mixture import GaussianMixture


@lru_cache(maxsize=100)
def get_vae_model(
    in_dim: tuple = (21, 21),
    latent_dim: int = 50,
    coord: int = 3,
    seed: int = 42,
):
    return rVAE(in_dim=in_dim, latent_dim=latent_dim, coord=coord, seed=seed)


def multimers_classification(
    img,
    coords,
    likelihood,
    n_species,
    latent_dim: int = 50,
    coord: int = 3,
    reg_covar: float = 0.0001,
    seed: int = 42,
    epochs: int = 20,
    scale_factor: float = 3.0,
    batch_size: int = 100,
):
    def get_crops(img, coords):
        """Get crops from image and coords"""
        crops = np.array(
            [np.array(create_crop(Image.fromarray(img), x, y)) for x, y in coords]
        )  # TODO : can be optimized if computationally heavy (multithreading)
        return crops

    # Get crops to train VAE on
    crops = get_crops(img, coords)
    # Initialize VAE
    rvae = rVAE(in_dim=(21, 21), latent_dim=latent_dim, coord=coord, seed=seed)

    # Train VAE to reconstruct crops
    torch_crops = torch.tensor(crops).float()
    train_loader = init_dataloader(torch_crops, batch_size=batch_size)
    trainer = SVItrainer(rvae)
    for e in range(epochs):
        trainer.step(train_loader, scale_factor=scale_factor)
        trainer.print_statistics()
    
    # Extract latent space (only mean) from VAE
    z_mean, _ = rvae.encode(torch_crops)

    # Cluster latent space with GMM
    gmm = GaussianMixture(
        n_components=n_species, reg_covar=reg_covar, random_state=seed
    )
    preds = gmm.fit_predict(z_mean)
    pred_proba = gmm.predict_proba(z_mean)
    pred_proba = np.array([pred_proba[i, pred] for i, pred in enumerate(preds)])

    # To order clusters, signal-to-noise ratio OR median (across crops) of some intensity quality (eg mean top-5% int)
    cluster_median_values = list()
    for k in range(n_species):
        relevant_crops = crops[preds == k]
        crop_95_percentile = np.percentile(relevant_crops, q=95, axis=0)
        img_means = []
        for crop, q in zip(relevant_crops, crop_95_percentile):
            if (crop >= q).any():
                img_means.append(crop.mean())
        cluster_median_value = np.median(np.array(img_means))
        cluster_median_values.append(cluster_median_value)
    # Sort clusters by median value
    sorted_clusters = sorted(
        [(mval, c_id) for c_id, mval in enumerate(cluster_median_values)]
    )

    # Return results in a dict with cluster id as key
    results = {}
    for _, c_id in sorted_clusters:
        c_idd = np.array([c_id])
        results[c_id] = {
            "coords": coords[preds == c_idd],
            "likelihood": likelihood[preds == c_idd],
            "confidence": pred_proba[preds == c_idd],
        }
    return results


def inference_fn(
    architecture: ModelArgs,
    image: Union[str, PILImage],
    threshold: float,
    n_species: int,
):
    if architecture != ModelArgs.BASICCNN:
        raise ValueError(f"Architecture {architecture} not supported yet")
    ckpt_filename = os.path.join(
        MODELS_PATH,
        {
            ModelArgs.BASICCNN: "model_C_NT_CLIP.ckpt",
            # ModelArgs.BASICCNN: "model_replicate20.ckpt",
            # ModelArgs.RESNET18 "inference_resnet.ckpt",
        }[architecture],
    )
    detection = DLDetection(
        model_name=architecture,
        ckpt_filename=ckpt_filename,
        dataset_csv=VOID_DS,
        threshold=threshold,
        detections_path=DET_PATH,
        inference_cache_path=INF_PATH,
        batch_size=512,
    )
    # Force the image to be in float32 because otherwise it will output wrong results (probably due to the median filter)
    if type(image) == str:
        image = Image.open(image)
    img = np.asarray(image, dtype=np.float32)
    # if img.max() <= 1:
    #     raise ValueError("Gradio seems to preprocess badly the tiff images. Did you adapt the preprocessing function as mentionned in the app.py file comments?")
    prepro_img, _, pred_map = detection.image_to_pred_map(img, return_intermediate=True)
    center_coords_list, likelihood_list = (np.array(x) for x in detection.pred_map_to_atoms(pred_map))
    results = (
        multimers_classification(
            img=prepro_img,
            coords=center_coords_list,
            likelihood=likelihood_list,
            n_species=n_species,
        )
        if n_species > 1
        else {
            0: {
                
                "coords": center_coords_list,
                "likelihood": likelihood_list,
                "confidence": np.ones(len(center_coords_list)),
            }
        }
    )
    for k, v in results.items():
        results[k]["atoms_bbs"] = [
            Evaluation.center_coords_to_bbox(center_coords)
            for center_coords in v["coords"]
        ]
    return tiff_to_png(image), {
        "image": tiff_to_png(image),
        "pred_map": pred_map,
        "species": results,
    }


if __name__ == "__main__":
    from utils.paths import IMG_PATH

    img_path = os.path.join(IMG_PATH, "091_HAADF_15nm_Sample_PtNC_21Oct20.tif")
    _ = inference_fn(ModelArgs.BASICCNN, Image.open(img_path), 0.8)
