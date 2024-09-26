# overlap size and soft boundary

import numpy as np

import matplotlib.pyplot as plt

import json


import os




if __name__ == "__main__":

    task_name = "LEMBWikimQARetrievalChunked"
    soft_boundary_dir = "results-soft-boundary"
    hard_boundary_dir = "results-hard-boundary"
    truncated_boundary_dir = "results-truncation"

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

    truncation_dir = os.path.join(truncated_boundary_dir, "no_model_name_available", "no_revision_available", f"{task_name}.json")
    with open(truncation_dir, "r") as f:
        results = json.load(f)
    
    truncated_boundary_ndcg = results["scores"]["test"][0]["ndcg_at_10"]
    truncated_boundary_map  = results["scores"]["test"][0]["map_at_10"]

    # NDCG
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.size': 14, 'font.family': 'serif'})

    fig, ax = plt.subplots(1, 1, figsize=(12, 9), sharex=True)

    colors = plt.cm.get_cmap('plasma', len(embed_sizes))(np.linspace(0, 1, len(embed_sizes), endpoint=False))
    colors = ["red", "blue", "green", "orange"]

    for embed_i, embed_size in enumerate(embed_sizes[:-1]):

        ax.plot([o - embed_i*5 for o in overlap_sizes], soft_boundary_ndcgs[embed_i], label=f"Soft Boundary ({embed_size})", color=colors[embed_i], marker='o', linestyle='-', linewidth=2, markersize=6)
        ax.axhline(hard_boundary_ndcgs[embed_i], linestyle="--", label=f"Hard Boundary ({embed_size})", linewidth=2, color=colors[embed_i])

        ax.set_title(f"Embed Size: {embed_size}", fontsize=16)
        ax.set_ylabel("nCDG@10", fontsize=14)
        ax.legend(loc='best', fontsize=12)

    # ax.axhline(truncated_boundary_ndcg, label=f"Truncated Boundary (at 8192)", color="k", linestyle='-.', linewidth=2)
    ax.set_xlabel("Overlap Size", fontsize=14)
    fig.suptitle("NDCG@10 for Different Embed Sizes and Overlap Sizes", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # MAP
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.size': 14, 'font.family': 'serif'})

    fig, ax = plt.subplots(1, 1, figsize=(12, 9), sharex=True)

    colors = plt.cm.get_cmap('plasma', len(embed_sizes))(np.linspace(0, 1, len(embed_sizes), endpoint=False))
    colors = ["red", "blue", "green", "orange"]

    for embed_i, embed_size in enumerate(embed_sizes[:-1]):

        ax.plot([o - embed_i*5 for o in overlap_sizes], soft_boundary_maps[embed_i], label=f"Soft Boundary ({embed_size})", color=colors[embed_i], marker='o', linestyle='-', linewidth=2, markersize=6)
        ax.axhline(hard_boundary_maps[embed_i], linestyle="--", label=f"Hard Boundary ({embed_size})", linewidth=2, color=colors[embed_i])

        ax.set_title(f"Embed Size: {embed_size}", fontsize=16)
        ax.set_ylabel("mAP@10", fontsize=14)
        ax.legend(loc='best', fontsize=12)


    # ax.axhline(truncated_boundary_map, label=f"Truncated Boundary (at 8192)", color="k", linestyle='-.', linewidth=2)
    ax.set_xlabel("Overlap Size", fontsize=14)
    fig.suptitle("mAP@10 for Different Embed Sizes and Overlap Sizes", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.96])



    from datasets import load_dataset
    name = "narrativeqa"
    dataset = load_dataset(path="dwzhu/LongEmbed", name=name, split="corpus")
    print(dataset.info.download_checksums)

    name = "summ_screen_fd"
    dataset = load_dataset(path="dwzhu/LongEmbed", name=name, split="corpus")
    print(dataset.info.download_checksums)

    name = "qmsum"
    dataset = load_dataset(path="dwzhu/LongEmbed", name=name, split="corpus")
    print(dataset.info.download_checksums)