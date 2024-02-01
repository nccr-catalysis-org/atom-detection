from typing import Tuple, List

from atoms_detection.dl_detection import DLDetection
from utils.constants import ModelArgs
from sklearn.mixture import GaussianMixture
from scipy.ndimage import label
import math
import numpy as np


class DLGMMdetection(DLDetection):
    MAX_SINGLE_ATOM_AREA = 200
    MAX_ATOMS_PER_AREA = 3
    COVARIANCE_TYPE = "full"

    def __init__(
        self,
        model_name: ModelArgs,
        ckpt_filename: str,
        dataset_csv: str,
        threshold: float,
        detections_path: str,
        inference_cache_path: str,
        covariance_penalisation: float = 0.03,
        n_clusters_penalisation: float = 0.33,
        distance_penalisation: float = 0.11,
        n_samples_per_gmm: int = 600,
    ):
        super(DLGMMdetection, self).__init__(
            model_name,
            ckpt_filename,
            dataset_csv,
            threshold,
            detections_path,
            inference_cache_path,
        )
        self.covariance_penalisation = covariance_penalisation
        self.n_clusters_penalisation = n_clusters_penalisation
        self.distance_penalisation = distance_penalisation
        self.n_samples_per_gmm = n_samples_per_gmm

    def pred_map_to_atoms(
        self, pred_map: np.ndarray
    ) -> Tuple[List[Tuple[int, int]], List[float]]:
        pred_mask = pred_map > self.threshold
        labeled_array, num_features = label(pred_mask)
        self.current_pred_map = pred_map

        # Convert labelled_array to indexes
        center_coords_list = []
        likelihood_list = []
        for label_idx in range(num_features + 1):
            if label_idx == 0:
                continue
            label_mask = np.where(labeled_array == label_idx)
            likelihood = np.max(pred_map[label_mask])
            # label_size = len(label_mask[0])
            # print(f"\t\tAtom {label_idx}: {label_size}")
            atom_bbox = (
                label_mask[1].min(),
                label_mask[1].max(),
                label_mask[0].min(),
                label_mask[0].max(),
            )
            center_coord = self.bbox_to_center_coords(atom_bbox)
            center_coords_list += center_coord
            pixel_area = (atom_bbox[1] - atom_bbox[0]) * (atom_bbox[3] - atom_bbox[2])
            if pixel_area < self.MAX_SINGLE_ATOM_AREA:
                likelihood_list.append(likelihood)
            else:
                for i in range(0, len(center_coord)):
                    likelihood_list.append(likelihood)
        self.current_pred_map = None
        print(f"number for atoms {len(center_coords_list)}")
        return center_coords_list, likelihood_list

    def bbox_to_center_coords(
        self, bbox: Tuple[int, int, int, int]
    ) -> List[Tuple[int, int]]:
        pixel_area = (bbox[1] - bbox[0]) * (bbox[3] - bbox[2])
        if pixel_area < self.MAX_SINGLE_ATOM_AREA:
            return super().bbox_to_center_coords(bbox)
        else:
            pmap = self.get_current_prediction_map_region(bbox)
            local_atom_center_list = self.run_gmm_pipeline(pmap)
            atom_center_list = [
                (x + bbox[0], y + bbox[2]) for x, y in local_atom_center_list
            ]
            return atom_center_list

    def sample_img_hist(self, img_region):
        x_bin_midpoints = list(range(img_region.shape[1]))
        y_bin_midpoints = list(range(img_region.shape[0]))
        # noinspection PyUnresolvedReferences
        cdf = np.cumsum(img_region.ravel())
        cdf = cdf / cdf[-1]
        values = np.random.rand(self.n_samples_per_gmm)
        # noinspection PyUnresolvedReferences
        value_bins = np.searchsorted(cdf, values)
        x_idx, y_idx = np.unravel_index(
            value_bins, (len(x_bin_midpoints), len(y_bin_midpoints))
        )
        random_from_cdf = np.column_stack((x_idx, y_idx))
        new_x, new_y = random_from_cdf.T
        return new_x, new_y

    def run_gmm_pipeline(self, prediction_map: np.ndarray) -> List[Tuple[int, int]]:
        retries = 2
        new_x, new_y = self.sample_img_hist(prediction_map)
        best_gmm, best_score = None, np.NINF
        obs = np.array((new_x, new_y)).T
        for k in range(1, self.MAX_ATOMS_PER_AREA + 1):
            for i in range(retries):
                gmm = GaussianMixture(
                    n_components=k, covariance_type=self.COVARIANCE_TYPE
                )
                gmm.fit(obs)
                logLike = gmm.score(obs)
                covar = np.linalg.norm(gmm.covariances_)
                if k == 1:
                    score = (
                        logLike
                        - covar * self.covariance_penalisation
                        - k * self.n_clusters_penalisation
                    )
                    print(k, score)
                else:
                    distances = [
                        math.dist(p1, p2)
                        for i, p1 in enumerate(gmm.means_[:-1])
                        for p2 in gmm.means_[i + 1 :]
                    ]
                    dist_penalisation = sum([max(12 - d, 0) ** 2 for d in distances])
                    score = (
                        logLike
                        - covar * self.covariance_penalisation
                        - k * self.n_clusters_penalisation
                        - dist_penalisation * self.distance_penalisation
                    )
                    print(
                        k,
                        score,
                        logLike,
                        covar * self.covariance_penalisation,
                        k * self.n_clusters_penalisation,
                        dist_penalisation * self.distance_penalisation,
                    )
                if score > best_score:
                    best_gmm, best_score = gmm, score
        # print(best_gmm.means_)
        return [(x, y) for y, x in best_gmm.means_.tolist()]
