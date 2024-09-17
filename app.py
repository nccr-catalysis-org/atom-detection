#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2023 April 25, 14:39:03
@last modified : 2024 February 01, 15:59:37
"""

# TODO : add the training of the vae
# TODO : add the description of the settings


import os
import gradio as gr
import json
import numpy as np
import shutil
import sys
import tempfile
import torch

from PIL import ImageDraw
from app.dl_inference import inference_fn
from app.knn import knn, segment_image, bokeh_plot_knn, color_palette
from app.tiff_utils import extract_physical_metadata
from collections import namedtuple
from datetime import datetime
from zipfile import ZipFile

block_state_entry = namedtuple(
    "block_state", ["results", "knn_results", "physical_metadata"]
)

if torch_availbale := torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
print(f"Is CUDA available: {torch_availbale}")

if ".." not in sys.path:
    sys.path.append("..")

from utils.constants import ModelArgs


def inf(img, n_species, threshold, architecture):
    # Get the coordinates of the atoms
    img, results = inference_fn(architecture, img, threshold, n_species=n_species)
    draw = ImageDraw.Draw(img)
    for (k, v), color in zip(results["species"].items(), color_palette):
        color = "#{:02x}{:02x}{:02x}".format(*[int(255 * x) for x in color])
        draw.text((5, 5 + 15 * k), f"species {k}", fill=color)
        for x, y in v["coords"]:
            draw.ellipse(
                [x - 5, y - 5, x + 5, y + 5],
                outline=color,
                width=2,
            )
    return img, results


def batch_fn(
    files, n_species, threshold, architecture, block_state, progress=gr.Progress()
):
    progress(0, desc="Starting...")
    block_state = {}
    if not files:
        raise gr.Error("No files were uploaded")

    if any(not file.name.lower().endswith((".tif", ".tiff")) for file in files):
        raise gr.Error("Only TIFF images are supported")

    gallery = []
    error_messages = []

    for file_idx, file in enumerate(files):
        base_progress = file_idx / len(files)

        def display_progress(value, text=None):
            progress(
                base_progress + (1 / len(files)) * value,
                desc=f"Processing image {file_idx+1}/{len(files)}{' - ' + text if text else '...'}",
            )

        display_progress(0.1, "Extracting metadata...")
        physical_metadata = None
        try:
            physical_metadata = extract_physical_metadata(file.name)
            if physical_metadata.unit != "nm":
                raise gr.Error(f"Unit of {file.name} is not nm, cannot process it")
        except Exception as e:
            error_messages.append(f"Error processing {file.name}: {str(e)}")
            raise gr.Error(f"Error processing {file.name}: {str(e)}")

        original_file_name = os.path.basename(file.name)
        sanitized_file_name = original_file_name.replace(" ", "_")
        temp_file_path = os.path.join(tempfile.gettempdir(), sanitized_file_name)

        try:
            shutil.copy2(file.name, temp_file_path)

            display_progress(0.2, "Inference...")
            img, results = inf(temp_file_path, n_species, threshold, architecture)

            display_progress(0.8, "Segmentation...")
            mask = segment_image(temp_file_path)
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
            display_progress(1, "Done")
        except Exception as e:
            error_messages.append(f"Error processing {file.name}: {str(e)}")
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

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

    bokeh_plot = gr.update(
        value=bokeh_plot_knn(knn_args, with_cumulative=True) if knn_args else None,
        visible=bool(knn_args),
    )

    error_html = gr.update(
        value="<br>".join(
            [
                f"<p style='width:fit-content; background-color:rgba(255, 0, 0, 0.75); border-radius:5px; padding:5px; color:white;'>{msg}</p>"
                for msg in error_messages
            ]
        ),
        visible=bool(error_messages),
    )

    return (
        gallery,
        block_state,
        gr.update(
            visible=bool(gallery)
        ),  # Show export button only if there are results
        bokeh_plot,
        error_html,
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
        # Add all metadata
        for img_path, original_file_name in gallery:
            file_name = original_file_name.split(".")[0]

            # Copy the image
            pred_map_path = f"{tmpdir}/pred_map_{file_name}.png"
            shutil.copy2(img_path, pred_map_path)
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
    return gr.update(value=files[::-1], visible=True)


CSS = """
        .header {
            display: flex;
            flex-direction: row;
            align-items: center;
            justify-content: flex-start;
            padding: 12px;
            gap: 12px;
            border-radius: 4px;
            background: var(--block-background-fill);
        }

        .header img {
            width: 150px;
            height: auto;
        }

        .title {
            text-align: left;
        }

        .title h1 {
            font-size: 28px;
            margin-bottom: 5px;
        }

        .title h2 {
            font-size: 18px;
            font-weight: normal;
            margin-top: 0;
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


COLORS = {
    "primary": {
        "name": "nccr-catalysis-primary",
        "c50": "#faecef",
        "c100": "#f4d9df",
        "c200": "#e9b3bf",
        "c300": "#de8d9f",
        "c400": "#d3677f",
        "c500": "#c8415f",
        "c600": "#a0344c",
        "c700": "#782739",
        "c800": "#501a26",
        "c900": "#280d13",
        "c950": "#14060a",
    },
    "secondary": {
        "name": "nccr-catalysis-secondary",
        "c50": "#fff8ed",
        "c100": "#fef0da",
        "c200": "#fde1b5",
        "c300": "#fcd290",
        "c400": "#fbc36b",
        "c500": "#fab446",
        "c600": "#c89038",
        "c700": "#966c2a",
        "c800": "#64481c",
        "c900": "#32240e",
        "c950": "#191207",
    },
}

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue=gr.themes.colors.Color(**COLORS["primary"]),
        secondary_hue=gr.themes.colors.Color(**COLORS["secondary"]),
        spacing_size=gr.themes.sizes.spacing_sm,
        radius_size=gr.themes.sizes.radius_sm,
        text_size=gr.themes.sizes.text_sm,
    ),
    css=CSS,
) as block:
    block_state = gr.State({})

    gr.Markdown(
        """
        <div class="header">
            <img src="https://www.nccr-catalysis.ch/site/assets/files/1/nccr_catalysis_logo.svg" alt="NCCR Catalysis">
            <div class="title">
                <h1>Atom Detection</h1>
                <h2>Quantitative description of metal center organization in single-atom catalysts</h2>
            </div>
            
        </div>
        """
    )

    with gr.Row():
        with gr.Column():
            n_species = gr.Number(label="Number of species", value=1, precision=0)
            threshold = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.8,
                label="Threshold",
                info="Threshold for the confidence of the prediction",
            )
            architecture = gr.Dropdown(
                label="Architecture",
                choices=[ModelArgs.BASICCNN],
                value=ModelArgs.BASICCNN,
                visible=False,
            )
            files = gr.File(
                label="Images", file_types=[".tif", ".tiff"], file_count="multiple"
            )
            run_button = gr.Button("Run", variant="primary")

        with gr.Column():
            with gr.Tabs():
                with gr.TabItem("Masked prediction"):
                    masked_prediction_gallery = gr.Gallery(label="Masked predictions")
                with gr.TabItem("Nearest neighbors"):
                    bokeh_plot = gr.Plot(show_label=False)
                    error_html = gr.HTML(visible=False)

            export_btn = gr.Button("Export files", visible=False)
            exported_files = gr.File(
                label="Exported files",
                file_count="multiple",
                type="filepath",
                interactive=False,
                visible=False,
            )

    run_button.click(
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
        batch_export_files,
        inputs=[masked_prediction_gallery, block_state],
        outputs=[exported_files],
    )

    with gr.Accordion("How to ✨", open=True):
        gr.Markdown(
            """
            1. Select one or multiple microscopy images as **.tiff files** 📷🔬
            2. Upload individual or multiple .tif images for processing 📤🔢
            3. Export the output files. The generated zip archive will contain:
                - An image with overlayed atomic positions 🌟🔍
                - A table of atomic positions (in px) along with their probability 📊💎
                - Physical metadata of the respective images 📄🔍
                - JSON-formatted plot data 📊📝
            
            > **Note:**
            > - Structural descriptors beyond pixel-wise atom detections are available as outputs only if images present an embedded real-space calibration (e.g., in [nm px-1](https://imagej.nih.gov/ij/docs/guide/146-30.html#sub:Set-Scale...)) 📷🔬
            > - 32-bit images will be processed correctly, but appear as mostly white in the image preview window
            """
        )

    with gr.Accordion("Disclaimer and License", open=False):
        gr.Markdown(
            """
            ### Disclaimer
            NCCR licenses the Atom Detection Web-App utilisation "as is" with no express or implied warranty of any kind. NCCR specifically disclaims all express or implied warranties to the fullest extent allowed by applicable law, including without limitation all implied warranties of merchantability, title or fitness for any particular purpose or non-infringement. No oral or written information or advice given by the authors shall create or form the basis of any warranty of any kind.

            ### License
            Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

            The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

            The software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the software.
            """
        )

    gr.Markdown(
        """
        > To reference the use of this web app in a publication, please refer to the Atom Detection web app and the development described in this publication: K. Rossi et al. Adv. Mater. 2023, [doi:10.1002/adma.202307991](https://doi.org/10.1002/adma.202307991).
        """
    )

if __name__ == "__main__":
    block.launch()
