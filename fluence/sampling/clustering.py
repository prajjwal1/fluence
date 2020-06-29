__all__ = ["Clustering_Arguments", "Clustering_Processor"]

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import sklearn
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min


@dataclass
class Clustering_Arguments:
    batch_size: int = field(metadata={"help": "Batch size to use for MiniBatchKMeans"})
    num_clusters: int = field(metadata={"help": "number of clusters to obtain"})
    embedding_path: str = field(
        metadata={"help": "Path from where embeddings will be loaded"}
    )
    data_pct: Optional[float] = field(
        default=None, metadata={"help": "specifies how much data will be used"}
    )
    num_clusters_elements: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "specifies the number of clusters that will be used. If this"
                " is enabled, `data_pct` should be set to None"
            )
        },
    )
    cluster_output_path: str = field(
        default=None, metadata={"help": "Path where embedding will be stored"}
    )
    cluster_only: bool = field(default=False, metadata={"help": "Run only clustering"})
    random_state: int = field(
        default=0,
        metadata={"help": "for producing deterministic results with MiniBatchKMeans"},
    )
    cluster_input_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path from there clustering labels will be loaded"},
    )
    cluster_n_jobs: Optional[int] = field(
        default=-1,
        metadata={"help": "Number of parallel processes to run for clustering"},
    )
    centroid_elements_only: bool = field(
        default=False,
        metadata={"help": "Specify to use cluster centroid elements for training"},
    )


class Clustering_Processor:
    """
    A processor class that makes it easy to obtain indices from clusters with
    various methods
    """

    labels: np.array
    data_pct: float
    num_clusters: int
    cluster_num: int

    def __init__(self, cluster):
        self.labels = cluster["labels_"]
        self.kmeans_cluster_centers = cluster["cluster_centers_"]

    def get_cluster_indices(self, cluster_num: int):
        return np.where(self.labels == cluster_num)[0]

    def get_cluster_indices_by_pct(self, data_pct: float, original_len: int) -> List:
        """
        Input:
            data_pct: specify how many elements are required from clusters
            original_len: length of the dataset
        Output:
            cluster_indices: cluster indices

        This method return concatenated cluster indices whose propotion equals
        len(dataset)*data_percentage
        """
        current_len, cluster_indices = 0, []
        for i in set(self.labels):
            curr_cluster_indices = self.get_cluster_indices(i)
            current_len += len(curr_cluster_indices)
            if current_len < int(original_len * data_pct):
                cluster_indices.extend(curr_cluster_indices)
            else:
                return cluster_indices

    def get_cluster_indices_by_num(self, num_clusters: int) -> List:
        """
        Input:
            num_clusters: specify how many clusters to return
        Output:
            cluster_indices: cluster indices

        This method returns concatenated cluster indices whose propotion equals
        to that of number of elements in specified number of cluster
        """
        indices = []
        for i in range(num_clusters):
            indices.extend(self.get_cluster_indices(i))
        return indices

    def get_cluster_indices_from_centroid(self, embeddings: torch.tensor) -> np.array:
        return pairwise_distances_argmin_min(self.kmeans_cluster_centers, embeddings)[0]
