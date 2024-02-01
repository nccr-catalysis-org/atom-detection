# run VAE + GMM assignement
import argparse
from typing import List

import numpy as np
# import rasterio
import torch
import warnings
import os
import re
import pandas as pd
from PIL import Image

from sklearn.mixture import GaussianMixture

from atoms_detection.create_crop_dataset import create_crop
from atoms_detection.vae_utilities.vae_model import rVAE
from atoms_detection.vae_utilities.vae_svi_train import init_dataloader, SVItrainer
from atoms_detection.image_preprocessing import dl_prepro_image

"""
Code sourced from:
https://colab.research.google.com/github/ziatdinovmax/notebooks_for_medium/blob/main/pyroVAE_MNIST_medium.ipynb

"""

numbers = re.compile(r'(\d+)')

def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

warnings.filterwarnings("ignore", module="torchvision.datasets")


def get_crops_from_prediction_csvs(pred_crop_file):
    data = pd.read_csv(pred_crop_file)
    xx = data['x'].values
    yy = data['y'].values
    coords = zip(xx,yy)  

    img_file = data['Filename'][0]
    likelihood = data['Likelihood'].values
    img_path = os.path.join('data/tif_data', img_file)

    img = Image.open(img_path)
    np_img = np.asarray(img).astype(np.float64)
    np_img = dl_prepro_image(np_img)
    img = Image.fromarray(np_img)

    crops = list()
    coords_list = []
    for x, y in coords:
        coords_list.append([x,y])
        new_crop = create_crop(img, x, y)
        crops.append(new_crop)
   
    print(coords_list[0])
    print(np_img[0])

    coords_array = np.array(coords_list)

    return crops, coords_array, likelihood, img_file


def classify_crop_species(args):
    # crop_list = get_crops_from_folder(crops_source_folder='./Ni')
    crop_list, crop_coords, likelihood, img_filename = get_crops_from_prediction_csvs(args.pred_crop_file)
    crop_tensor = np.array(crop_list)
    
    # Assuming crop_tensor is a list or array of Image objects
    processed_images = []
    for image in crop_tensor:
        # Convert the Image to a NumPy array
        image_array = np.array(image)
        # Append the processed image array to the list
        processed_images.append(image_array)
    # Convert the processed images list to a NumPy array
    processed_images = np.array(processed_images)
    # Convert the processed_images array to float32
    processed_images = processed_images.astype(np.float32)

    #print(processed_images.shape)

    rvae = rVAE(in_dim=(21, 21), latent_dim=args.latent_dim, coord=args.coord, seed=args.seed)

    train_data = torch.from_numpy(processed_images).float()
    # train_data = torch.from_numpy(crop_tensor).float()
    train_loader = init_dataloader(train_data, batch_size=args.batchsize)
    latent_crop_tensor = train_vae(rvae, train_data, train_loader, args)

    gmm = GaussianMixture(n_components=args.n_species, reg_covar=args.GMMcovar, random_state=args.seed).fit(
        latent_crop_tensor)
    preds = gmm.predict(latent_crop_tensor)
    print(preds)
    pred_proba = gmm.predict_proba(latent_crop_tensor)
    pred_proba = [pred_proba[i, pred] for i, pred in enumerate(preds)]
    
    # To order clusters, signal-to-noise ratio OR median (across crops) of some intensity quality (eg mean top-5% int)
    cluster_median_values = list()
    for k in range(args.n_species):
        print(k)
        relevant_crops = processed_images[preds == k]
        crop_95_percentile = np.percentile(relevant_crops, q=95, axis=0)
        img_means = []
        for crop, q in zip(relevant_crops, crop_95_percentile):
            if (crop >= q).any():
                print(crop.mean())
                img_means.append(crop.mean())
            #img_means.append(crop.mean(axis=0, where=crop >= q))
        cluster_median_value = np.median(np.array(img_means))
        cluster_median_values.append(cluster_median_value)
    sorted_clusters = sorted([(mval, c_id) for c_id, mval in enumerate(cluster_median_values)])

    with open(f"data/detection_data/Multimetallic_{img_filename}.csv", "a") as f:
        f.write("Filename,x,y,Likelihood,cluster,cluster_confidence\n")
        for _, c_id in sorted_clusters:
            c_idd = np.array([c_id])
            pred_proba = np.array(pred_proba)
            relevant_crops_coords = crop_coords[preds == c_idd]
            relevant_crops_likelihood = likelihood[preds == c_idd]
            relevant_crops_confidence = pred_proba[preds == c_idd]
            #print(relevant_crops_confidence)
            for coords, l, c in zip(relevant_crops_coords, relevant_crops_likelihood, relevant_crops_confidence):
                x, y = coords
                f.write(f"{img_filename},{x},{y},{l},{c_id},{c}\n")



def train_vae(rvae, train_data, train_loader, args):
    # Initialize SVI trainer
    trainer = SVItrainer(rvae)
    for e in range(args.epochs):
        trainer.step(train_loader, scale_factor=args.scale_factor)
        trainer.print_statistics()
    z_mean, z_sd = rvae.encode(train_data)
    latent_crop_tensor = z_mean
    return latent_crop_tensor


def get_crops_from_folder(crops_source_folder) -> List[np.ndarray]:
    ffiles = []
    files = []
    for dirname, dirnames, filenames in os.walk(crops_source_folder):
        # print path to all subdirectories first.
        for subdirname in dirnames:
            files.append(os.path.join(dirname, subdirname))

        # print path to all filenames.
        for filename in filenames:
            files.append(os.path.join(dirname, filename))

        for filename in sorted((filenames), key=numericalSort):
            ffiles.append(os.path.join(filename))
    crops = ffiles
    # print(len(crops))
    path_crops = './Ni/'
    all_img = []
    for i in range(0, len(crops)):
        src_path = path_crops + crops[i]
        img = rasterio.open(src_path)
        test = np.reshape(img.read([1]), (21, 21))
        all_img.append(np.array(test))
    return all_img


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'pred_crop_file',
        type=str,
        help="Path to the CSV of predicted crop locations (eg in data/detection_data/X/Y.csv)"
    )
    parser.add_argument(
        "-latent_dim",
        type=int,
        default=50,
        help="Experiment extension name"
    )
    parser.add_argument(
        "-seed",
        type=int,
        default=444,
        help="Random seed"
    )
    parser.add_argument(
        "-coord",
        type=int,
        default=3,
        help="Amount of equivariances, 0: None,1: Rotational, 2: Translational, 3:Rotational and Translational"
    )
    parser.add_argument(
        "-batchsize",
        type=int,
        default=100,
        help="Batch size for the VAE model"
    )
    parser.add_argument(
        "-epochs",
        type=int,
        default=20,
        help="Number of training epochs for the VAE"
    )
    parser.add_argument(
        "-scale_factor",
        type=int,
        default=3,
        help="Number of training epochs for the VAE"
    )
    parser.add_argument(
        "-n_species",
        type=int,
        default=2,
        help="Number of chemical species expected in the sample."
    )
    parser.add_argument(
        "-GMMcovar",
        type=float,
        default=0.0001,
        help="Regcovar for the training of the GMM clustering algorithm."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    print(args)
    classify_crop_species(args)
