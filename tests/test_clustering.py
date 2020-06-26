import os
import unittest
import urllib

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min
from transformers import AutoTokenizer, GlueDataset
from transformers import GlueDataTrainingArguments as DataTrainingArguments

from fluence.sampling import Clustering_Arguments, Clustering_Processor


def get_embeddings(embedding_path):
    if not os.path.isfile(embedding_path):
        url = "https://s3.amazonaws.com/models.huggingface.co/bert/prajjwal1/albert-base-v2-mnli/cls_embeddings_mnli.pth"
        embedding_path = "cls_embeddings_mnli.pth"
        urllib.request.urlretreive(url, embedding_path)
    embeddings = torch.load(embedding_path)
    embeddings = np.concatenate(embeddings)  # (392702, 768)
    return embeddings


def get_clustering_obj(embeddings):
    clustering = MiniBatchKMeans(n_clusters=512, batch_size=256,).fit(embeddings)
    return clustering


class Test_Clustering(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Test_Clustering, self).__init__(*args, **kwargs)
        self.embedding_path = "/home/nlp/experiments/cls_embeddings_mnli.pth"
        self.cluster_output_path = "/home/nlp/experiments/tmp/c.pth"
        # self.data_dir = "/home/nlp/data/glue_data/MNLI"
        self.embeddings = get_embeddings(self.embedding_path)
        self.clustering_obj = get_clustering_obj(self.embeddings)

    def test_data_pct(self):
        clustering_args = Clustering_Arguments(
            batch_size=32,
            num_clusters=32,
            embedding_path=self.embedding_path,
            data_pct=0.2,
            cluster_output_path=self.cluster_output_path,
        )
        clustering_proc = Clustering_Processor(vars(self.clustering_obj))
        cluster_indices = clustering_proc.get_cluster_indices_by_pct(
            clustering_args.data_pct, self.embeddings.shape[0]
        )
        self.assertTrue(len(cluster_indices) > 70000)

    def test_cluster_indices(self):
        clustering_args = Clustering_Arguments(
            batch_size=32,
            num_clusters_elements=32,
            embedding_path=self.embedding_path,
            num_clusters=8,
            cluster_output_path=self.cluster_output_path,
        )
        clustering_proc = Clustering_Processor(vars(self.clustering_obj))
        cluster_indices = clustering_proc.get_cluster_indices_by_num(
            clustering_args.num_clusters_elements
        )
        self.assertTrue(len(cluster_indices) > 10000)

    def test_cluster_centroids(self):
        clustering_proc = Clustering_Processor(vars(self.clustering_obj))
        cluster_indices = clustering_proc.get_cluster_indices_from_centroid(
            self.embeddings
        )
        self.assertEqual(len(cluster_indices), 512)


#    def test_torch_hf_dataset(self):
#        data_args = DataTrainingArguments(task_name="MNLI", data_dir=self.data_dir)
#        tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
#        train_dataset = GlueDataset(data_args, tokenizer)
#        train_dataset = torch.utils.data.Subset(train_dataset, self.cluster_indices)
#        self.assertEqual(len(train_dataset[0].input_ids), 128)


if __name__ == "__main__":
    unittest.main()
