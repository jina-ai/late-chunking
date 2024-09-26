# overlap size and soft boundary

import numpy as np

import matplotlib.pyplot as plt

import json


import os










if __name__ == "__main__":

    task_name = "LEMBWikimQARetrievalChunked"
    soft_boundary_dir = "results-soft-boundary"
    hard_boundary_dir = "results-hard-boundary"
    truncate_dir = "results-truncation"

    embed_sizes = [1024, 2048, 4096, 8192]
    overlap_sizes = [32, 64, 128, 256, 512]

    soft_boundary_ndcgs = np.empty((len(embed_sizes), len(overlap_sizes)))
    hard_boundary_ndcgs = np.empty(len(embed_sizes))

    soft_boundary_maps = np.empty((len(embed_sizes), len(overlap_sizes)))
    hard_boundary_maps = np.empty(len(embed_sizes))

    for embed_i, embed_size in enumerate(embed_sizes):

        for overlap_i, overlap_size in enumerate(overlap_sizes):
            
            soft_dir = os.path.join(soft_boundary_dir, f"embed_size_{embed_size}", f"overlap_{overlap_size}", "no_model_name_available", "no_revision_available", f"{task_name}.json")
            with open(soft_dir, "r") as f:
                results = json.load(f)

            soft_boundary_ndcgs[embed_i, overlap_i] = results["scores"]["test"][0]["ndcg_at_10"]
            soft_boundary_maps[embed_i, overlap_i]  = results["scores"]["test"][0]["map_at_10"]

        soft_dir = os.path.join(hard_boundary_dir, f"embed_size_{embed_size}", "no_model_name_available", "no_revision_available", f"{task_name}.json")
        with open(soft_dir, "r") as f:
            results = json.load(f)
        
        hard_boundary_ndcgs[embed_i] = results["scores"]["test"][0]["ndcg_at_10"]
        hard_boundary_maps[embed_i]  = results["scores"]["test"][0]["map_at_10"]


    # NDCG
    fig, ax = plt.subplots(len(embed_sizes), 1, figsize=(12, 9))

    for embed_i, embed_size in enumerate(embed_sizes):

        ax[embed_i].plot(overlap_sizes, soft_boundary_ndcgs[embed_i], label="Soft Boundary")
        ax[embed_i].scatter(overlap_sizes, soft_boundary_ndcgs[embed_i])
        ax[embed_i].axhline(hard_boundary_ndcgs[embed_i], color="red", linestyle="--", label="Hard Boundary")

        ax[embed_i].set_title(f"Embed Size: {embed_size}")
        ax[embed_i].set_xlabel("Overlap Size")
        ax[embed_i].set_ylabel("nCDG@10")

    ax[embed_i].legend()
    fig.tight_layout()

    # MAP
    fig, ax = plt.subplots(len(embed_sizes), 1, figsize=(12, 9))
    for embed_i, embed_size in enumerate(embed_sizes):

        ax[embed_i].plot(overlap_sizes, soft_boundary_maps[embed_i], label="Soft Boundary")
        ax[embed_i].scatter(overlap_sizes, soft_boundary_maps[embed_i])
        ax[embed_i].axhline(hard_boundary_maps[embed_i], color="red", linestyle="--", label="Hard Boundary")

        ax[embed_i].set_title(f"Embed Size: {embed_size}")
        ax[embed_i].set_xlabel("Overlap Size")
        ax[embed_i].set_ylabel("mAP@10")

    ax[embed_i].legend()
    fig.tight_layout()