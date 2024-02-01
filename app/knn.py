#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2023 May 16, 16:18:43
@last modified : 2023 August 07, 11:54:19
"""

from typing import List, Tuple

from PIL import Image
from collections import defaultdict
from tempfile import mktemp
import matplotlib
import numpy as np
import os
import seaborn as sns
from .logger import logger


matplotlib.use("agg")
import matplotlib.pyplot as plt
from scipy.stats import rayleigh
from sklearn.neighbors import NearestNeighbors


def segment_image(filename, alpha=5):
    # Get a random image png file
    filename = filename.replace(" ", "\ ")
    png_img = mktemp(suffix=".png")
    segmented_img = mktemp(suffix=".png")
    logger.debug(f"Segmenting image {filename}...")
    logger.debug(f"Saving image to {png_img}...")
    logger.debug(f"Saving segmented image to {segmented_img}...")
    try:
        # Segment with image magic in the terminal
        ret = os.system(f"convert {filename} {png_img}")
        if ret != 0:
            raise RuntimeError(f"PNG conversion failed with return code {ret}")
        ret = os.system(
            f"convert {png_img} -alpha on -fill none -fuzz {alpha}% -draw 'color 0,0 replace' {segmented_img}"
        )
        if ret != 0:
            raise RuntimeError(f"Segmentation failed with return code {ret}")
        # Load the image
        img = Image.open(segmented_img)
        # Get mask from image
        mask = np.array(img) == 0
    finally:
        # Delete the temporary files
        if os.path.exists(png_img):
            os.remove(png_img)
        if os.path.exists(segmented_img):
            os.remove(segmented_img)
    return mask


condition = lambda x: x < 0.23


def knn(coords: List[Tuple[int, int]], scale: float, factor: float, edge: float):
    coords = np.array(coords)  # B, 2
    x, y = coords.T * scale

    print(f"edge: {edge}, scale: {scale}, factor: {factor}, edge*scale: {edge*scale}")
    # edge = edge * scale

    data = np.c_[x, y]

    neighbors = NearestNeighbors(n_neighbors=2, algorithm="ball_tree").fit(data)
    distances = neighbors.kneighbors(data)[0][:, 1]

    # loc, scale = rayleigh.fit(distances, floc=0)
    # r_KNN = scale * np.sqrt(np.pi / 2.0)

    lamda_RNN = len(x) / (edge * edge * factor)
    r_RNN = 1 / (2 * np.sqrt(lamda_RNN))

    n_samples = len(distances)
    n_multimers = sum(condition(x) for x in distances)
    percentage_multimers = 100.0 * n_multimers / n_samples
    density = n_samples / (factor * edge**2)
    min_dist = min(distances)
    μ_dist = np.mean(distances)

    return {
        "n_samples": {
            "description": "Number of atoms detected (units = #atoms)",
            "value": n_samples,
        },
        "number of atoms in multimers": {
            "description": "Number of atoms detected to belong to a multimer (units = #atoms)",
            "value": n_multimers,
        },
        "share of multimers": {
            "description": "Percentage of atoms that belong to a multimer (units = %)",
            "value": percentage_multimers,
        },
        "density": {
            "description": "Number of atoms / area in the micrograph (units = #atoms/nm²)",
            "value": density,
        },
        "min_dist": {
            "description": "Lowest first nearest neighbour distance detected (units = nm)",
            "value": min_dist,
        },
        "<NND>": {
            "description": "Mean first nearest neighbour distance (units = nm)",
            "value": μ_dist,
        },
        "r_RNN": {
            "description": "First neighbour distance expected from a purely random distribution (units = nm)",
            "value": r_RNN,
        },
        "distances": distances,
    }


from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure
from scipy.stats import gaussian_kde
from collections import defaultdict

color_palette = sns.color_palette("Set3")[2:]


def bokeh_plot_knn(distances, with_cumulative=False):
    """
    Plot the KNN distances for the different images with the possibility to zoom in and out and toggle the lines
    """
    p = figure(title="K=1 NN distances", background_fill_color="#fafafa")
    p.xaxis.axis_label = "Distances (nm)"
    p.yaxis.axis_label = "Density"
    p.x_range.start = 0

    if with_cumulative:
        cum_dists = defaultdict(list)
        for _, dists in distances:
            for specie, dist in dists.items():
                cum_dists[specie].extend(dist)
        cum_dists = {specie: np.array(dist) for specie, dist in cum_dists.items()}
        distances.append(("Cumulative", cum_dists))

    plot_dict = defaultdict(dict)
    base_colors = color_palette
    for (image_name, species_distances), base_color in zip(distances, base_colors):
        palette = (
            sns.light_palette(
                base_color, n_colors=len(species_distances) + 1, reverse=True
            )[1::-1]
            if len(species_distances) > 1
            else [base_color]
        )
        colors = [
            f"#{int(255*r):02x}{int(255*g):02x}{int(255*b):02x}" for r, g, b in palette
        ]
        for (specie, dists), color in zip(species_distances.items(), colors):
            kde = gaussian_kde(dists)
            # Reduce smoothing
            kde.set_bandwidth(bw_method=0.1)
            x = np.linspace(-0.5, 1.2 * dists.max(), 100)
            source = ColumnDataSource(
                dict(
                    x=x,
                    y=kde(x),
                    species=[specie] * len(x),
                    p_below=[np.mean(dists < 0.22)] * len(x),
                    mean=[np.mean(dists)] * len(x),
                    filename=[image_name] * len(x),
                )
            )
            plot_dict[image_name][specie] = [
                p.line(
                    line_width=2,
                    alpha=0.8,
                    legend_label=image_name,
                    line_color=color,
                    source=source,
                ),
                p.varea(
                    y1="y",
                    y2=0,
                    alpha=0.3,
                    legend_label=image_name,
                    source=source,
                    fill_color=color,
                ),
            ]
    p.add_tools(
        HoverTool(
            show_arrow=False,
            line_policy="next",
            tooltips=[
                ("First NN distances < 0.22nm", "@p_below{0.00%}"),
                ("<NND>", "@mean{0.00} nm"),
                ("species", "@species"),
                ("filename", "@filename"),
            ],
        )
    )
    p.legend.click_policy = "hide"
    return p
