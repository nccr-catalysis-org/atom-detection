#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2023 April 25, 14:39:03
@last modified : 2023 September 20, 15:35:23
"""

# TODO : add the training of the vae
# TODO : add the description of the settings

import sys

import numpy as np
from PIL import Image, ImageDraw
import gradio as gr
from tiff_utils import extract_physical_metadata
from dl_inference import inference_fn
from knn import knn, segment_image, bokeh_plot_knn, color_palette

import tempfile
import shutil
import json
from zipfile import ZipFile
from datetime import datetime

from collections import namedtuple

block_state_entry = namedtuple(
    "block_state", ["results", "knn_results", "physical_metadata"]
)

if ".." not in sys.path:
    sys.path.append("..")

from utils.constants import ModelArgs


def inf(img, n_species, threshold, architecture):
    # Get the coordinates of the atoms
    img, results = inference_fn(architecture, img, threshold, n_species=n_species)
    draw = ImageDraw.Draw(img)
    for (k, v), color in zip(results["species"].items(), color_palette):
        color = "#" + "".join([f"{int(255 * x):02x}" for x in color])
        draw.text((5, 5 + 15 * k), f"species {k}", fill=color)
        for x, y in v["coords"]:
            draw.ellipse(
                [x - 5, y - 5, x + 5, y + 5],
                outline=color,
                width=2,
            )
    return img, results


def batch_fn(files, n_species, threshold, architecture, block_state):
    block_state = {}
    if not files:
        raise ValueError("No files were uploaded")

    gallery = []
    for file in files:
        error_physical_metadata = None
        try:
            physical_metadata = extract_physical_metadata(file.name)
            if physical_metadata.unit != "nm":
                raise ValueError(f"Unit of {file.name} is not nm, cannot process it")
        except Exception as e:
            error_physical_metadata = e
            physical_metadata = None

        original_file_name = file.name.split("/")[-1]
        img, results = inf(file.name, n_species, threshold, architecture)
        mask = segment_image(file.name)
        gallery.append((img, original_file_name))

        if physical_metadata is not None:
            factor = 1.0 - np.mean(mask)
            scale = physical_metadata.pixel_width
            edge = physical_metadata.pixel_width * physical_metadata.width
            knn_results = {
                k: knn(results["species"][k]["coords"], scale, factor, edge)
                for k in results["species"]
            }
        else:
            knn_results = None

        block_state[original_file_name] = block_state_entry(
            results, knn_results, physical_metadata
        )

    knn_args = [
        (
            original_file_name,
            {
                k: block_state[original_file_name].knn_results[k]["distances"]
                for k in block_state[original_file_name].knn_results
            },
        )
        for original_file_name in block_state
        if block_state[original_file_name].knn_results is not None
    ]
    if len(knn_args) > 0:
        bokeh_plot = gr.update(
            value=bokeh_plot_knn(knn_args, with_cumulative=True), visible=True
        )
    else:
        bokeh_plot = gr.update(visible=False)
    return (
        gallery,
        block_state,
        gr.update(visible=True),
        bokeh_plot,
        gr.HTML.update(
            value=f"<p style='width:fit-content; background-color:rgba(255, 0, 0, 0.75); border-radius:5px; padding:5px; color:white;'>{error_physical_metadata}</p>",
            visible=bool(error_physical_metadata),
        ),
    )


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def batch_export_files(gallery, block_state):
    # Return images, coords as csv and a zip containing everything
    files = []
    tmpdir = tempfile.mkdtemp()
    with ZipFile(
        f"{tmpdir}/all_results_{datetime.now().isoformat()}.zip", "w"
    ) as zipObj:
        # Add all metatada
        for data_dict, original_file_name in gallery:
            file_name = original_file_name.split(".")[0]

            # Save the image
            pred_map_path = f"{tmpdir}/pred_map_{file_name}.png"
            file_path = data_dict["name"]
            shutil.copy(file_path, pred_map_path)
            zipObj.write(pred_map_path, arcname=f"{file_name}/pred_map.png")
            files.append(pred_map_path)

            # Save the coords
            results = block_state[original_file_name].results
            coords_path = f"{tmpdir}/coords_{file_name}.csv"
            with open(coords_path, "w") as f:
                f.write("x,y,likelihood,specie,confidence\n")
                for k, v in results["species"].items():
                    for (x, y), likelihood, confidence in zip(
                        v["coords"], v["likelihood"], v["confidence"]
                    ):
                        f.write(f"{x},{y},{likelihood},{k},{confidence}\n")
            zipObj.write(coords_path, arcname=f"{file_name}/coords.csv")
            files.append(coords_path)

            # Save the knn results
            if block_state[original_file_name].knn_results is not None:
                knn_results = block_state[original_file_name].knn_results
                knn_path = f"{tmpdir}/knn_results_{file_name}.json"
                with open(knn_path, "w") as f:
                    json.dump(knn_results, f, cls=NumpyEncoder)
                zipObj.write(knn_path, arcname=f"{file_name}/knn_results.json")
                files.append(knn_path)

            # Save the physical metadata
            if block_state[original_file_name].physical_metadata is not None:
                physical_metadata = block_state[original_file_name].physical_metadata
                metadata_path = f"{tmpdir}/physical_metadata_{file_name}.json"
                with open(metadata_path, "w") as f:
                    json.dump(physical_metadata._asdict(), f, cls=NumpyEncoder)
                zipObj.write(
                    metadata_path, arcname=f"{file_name}/physical_metadata.json"
                )
                files.append(metadata_path)

    files.append(zipObj.filename)
    return gr.update(value=files, visible=True)


CSS = """
        .header {
            display: flex;
            justify-content: center;
            align-items: center;
            padding: var(--block-padding);
            border-radius: var(--block-radius);
            background: var(--button-secondary-background-hover);
        }

        img {
            width: 150px;
            margin-right: 40px;
        }

        .title {
            text-align: left;
        }

        h1 {
            font-size: 36px;
            margin-bottom: 10px;
        }

        p {
            font-size: 18px;
        }

        input {
            width: 70px;
        }

        @media (max-width: 600px) {
            h1 {
                font-size: 24px;
            }

            p {
                font-size: 14px;
            }
        }

"""


with gr.Blocks(css=CSS) as block:
    block_state = gr.State({})
    gr.HTML(
        """
        <div class="header">
            <a href="https://www.nccr-catalysis.ch/" target="_blank">
                <img src="https://www.nccr-catalysis.ch/site/assets/files/1/nccr_catalysis_logo.svg" alt="NCCR Catalysis">
            </a>
            <div class="title">
                <h1>Atom Detection</h1>
                <p>Quantitative description of metal center organization in single-atom catalysts</p>
            </div>
        </div>
        """
    )
    with gr.Row():
        with gr.Column():
            with gr.Row():
                n_species = gr.Number(
                    label="Number of species",
                    min=1,
                    max=10,
                    value=1,
                    step=1,
                    precision=0,
                    visible=True,
                )
                threshold = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.8,
                    label="Threshold",
                    visible=True,
                )
                architecture = gr.Dropdown(
                    label="Architecture",
                    choices=[
                        ModelArgs.BASICCNN,
                        # ModelArgs.RESNET18,
                    ],
                    value=ModelArgs.BASICCNN,
                    visible=False,
                )
            files = gr.Files(
                label="Images",
                file_types=[".tif", ".tiff"],
                type="file",
                interactive=True,
            )
            button = gr.Button(value="Run")
        with gr.Column():
            with gr.Tab("Masked prediction") as masked_tab:
                masked_prediction_gallery = gr.Gallery(
                    label="Masked predictions"
                ).style(columns=3)
            with gr.Tab("Nearest neighbors") as nn_tab:
                bokeh_plot = gr.Plot(show_label=False)
                error_html = gr.HTML(visible=False)
            export_btn = gr.Button(value="Export files", visible=False)
            exported_files = gr.File(
                label="Exported files",
                file_count="multiple",
                type="file",
                interactive=False,
                visible=False,
            )
    button.click(
        batch_fn,
        inputs=[files, n_species, threshold, architecture, block_state],
        outputs=[
            masked_prediction_gallery,
            block_state,
            export_btn,
            bokeh_plot,
            error_html,
        ],
    )
    export_btn.click(
        batch_export_files, [masked_prediction_gallery, block_state], [exported_files]
    )
    with gr.Accordion(label="How to ✨", open=True):
        gr.HTML(
            """
            <div style="font-size: 14px;">
            <ol>
                <li>Select one or multiple microscopy images as <b>.tiff files</b> 📷🔬</li>
                <li>Upload individual or multiple .tif images for processing 📤🔢</li>
                <li>Export the output files. The generated zip archive will contain:
                    <ul>
                        <li>An image with overlayed atomic positions 🌟🔍</li>
                        <li>A table of atomic positions (in px) along with their probability 📊💎</li>
                        <li>Physical metadata of the respective images 📄🔍</li>
                        <li>JSON-formatted plot data 📊📝</li>
                    </ul>
                </li>
            </ol>
            <details style="padding: 5px; border-radius: 5px; background: var(--button-secondary-background-hover); font-size: 14px;">
            <summary>Note</summary>
            <ul style="padding-left: 10px;">
            <li>
            Structural descriptors beyond pixel-wise atom detections are available as outputs only if images present an embedded real-space calibration (e.g., in <a href="https://imagej.nih.gov/ij/docs/guide/146-30.html#sub:Set-Scale...">nm px-1</a>) 📷🔬
            </li>
            <li>
            32-bit images will be processed correctly, but appear as mostly white in the image preview window
            </li>
            </ul>
            </details>
            </div>
     """
        )
    with gr.Accordion(label="Disclaimer and License", open=False):
        gr.HTML(
            """
            <div class="acknowledgments">
                <h3>Disclaimer</h3>
                <p>NCCR licenses the Atom Detection Web-App utilisation “as is” with no express or implied warranty of any kind. NCCR specifically disclaims all express or implied warranties to the fullest extent allowed by applicable law, including without limitation all implied warranties of merchantability, title or fitness for any particular purpose or non-infringement. No oral or written information or advice given by the authors shall create or form the basis of any warranty of any kind.</p>
                <h3>License</h3>
                <p>Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
<br>
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
<br>
The software is provided “as is”, without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the software.</p>
            </div>
            """
        )
    gr.HTML(
        """
            <div style="background-color: var(--secondary-100); border-radius: 5px; padding: 10px;">
                <p style='font-size: 14px; color: black'>To reference the use of this web app in a publication, please refer to the Atom Detection web app and the development described in this publication: K. Rossi et al. Adv. Mater. 2023, <a href="https://doi.org/10.1002/adma.202307991">doi:10.1002/adma.202307991</a>.</p>
            </div>
            """
    )


block.launch(
    share=False,
    show_error=True,
    server_name="0.0.0.0",
    server_port=9003,
    enable_queue=True,
)
