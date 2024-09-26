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

    # == Load results

    embed_sizes = [1024, 2048, 4096, 8192]
    overlap_sizes = [32, 64, 128, 256, 512]
    task_names = ["QMSum", "SummScreenFD", "WikimQA"]

    soft_boundary_ndcgs = np.empty((len(task_names), len(embed_sizes), len(overlap_sizes)))
    hard_boundary_ndcgs = np.empty((len(task_names), len(embed_sizes)))

    soft_boundary_maps = np.empty((len(task_names), len(embed_sizes), len(overlap_sizes)))
    hard_boundary_maps = np.empty((len(task_names), len(embed_sizes)))

    for task_i, task_name in enumerate(task_names):

        task_name = f"LEMB{task_name}RetrievalChunked"

        for embed_i, embed_size in enumerate(embed_sizes):

            for overlap_i, overlap_size in enumerate(overlap_sizes):
                
                soft_dir = os.path.join(soft_boundary_dir, f"embed_size_{embed_size}", f"overlap_{overlap_size}", "no_model_name_available", "no_revision_available", f"{task_name}.json")
                with open(soft_dir, "r") as f:
                    results = json.load(f)

                soft_boundary_ndcgs[task_i, embed_i, overlap_i] = results["scores"]["test"][0]["ndcg_at_10"]
                soft_boundary_maps[task_i, embed_i, overlap_i]  = results["scores"]["test"][0]["map_at_10"]

            soft_dir = os.path.join(hard_boundary_dir, f"embed_size_{embed_size}", "no_model_name_available", "no_revision_available", f"{task_name}.json")
            with open(soft_dir, "r") as f:
                results = json.load(f)
            
            hard_boundary_ndcgs[task_i, embed_i] = results["scores"]["test"][0]["ndcg_at_10"]
            hard_boundary_maps[task_i, embed_i]  = results["scores"]["test"][0]["map_at_10"]

    # truncation_dir = os.path.join(truncated_boundary_dir, "no_model_name_available", "no_revision_available", f"{task_name}.json")
    # with open(truncation_dir, "r") as f:
    #     results = json.load(f)
    
    # truncated_boundary_ndcg = results["scores"]["test"][0]["ndcg_at_10"]
    # truncated_boundary_map  = results["scores"]["test"][0]["map_at_10"]


    # == Plot (by task separately)

    # NDCG
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.size': 14, 'font.family': 'serif'})

    fig, ax = plt.subplots(len(task_names), 1, figsize=(12, 3*len(task_names)), sharex=True)

    # colors = plt.cm.get_cmap('plasma', len(embed_sizes))(np.linspace(0, 1, len(embed_sizes), endpoint=False))
    colors = ["red", "blue", "green", "orange"]

    for task_i in range(len(task_names)):
        for embed_i, embed_size in enumerate(embed_sizes[:-1]):

            ax[task_i].plot([o - embed_i*5 for o in overlap_sizes], soft_boundary_ndcgs[task_i, embed_i], label=f"Soft Boundary ({embed_size})", color=colors[embed_i], marker='o', linestyle='-', linewidth=2, markersize=6)
            ax[task_i].axhline(hard_boundary_ndcgs[task_i, embed_i], linestyle="--", label=f"Hard Boundary ({embed_size})", linewidth=2, color=colors[embed_i])

            ax[task_i].set_title(f"Embed Size: {embed_size}", fontsize=16)
            ax[task_i].set_ylabel("nCDG@10", fontsize=14)
            ax[task_i].legend(loc='best', fontsize=12)
        
        ax[task_i].set_title("Task: " + task_names[task_i], fontsize=16)

    # ax.axhline(truncated_boundary_ndcg, label=f"Truncated Boundary (at 8192)", color="k", linestyle='-.', linewidth=2)
    ax[task_i].set_xlabel("Overlap Size", fontsize=14)

    fig.suptitle("NDCG@10 for Different Embed Sizes and Overlap Sizes", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # MAP
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.size': 14, 'font.family': 'serif'})

    fig, ax = plt.subplots(len(task_names), 1, figsize=(12, 3*len(task_names)), sharex=True)

    # colors = plt.cm.get_cmap('plasma', len(embed_sizes))(np.linspace(0, 1, len(embed_sizes), endpoint=False))
    colors = ["red", "blue", "green", "orange"]

    for task_i in range(len(task_names)):
        for embed_i, embed_size in enumerate(embed_sizes[:-1]):

            ax[task_i].plot([o - embed_i*5 for o in overlap_sizes], soft_boundary_maps[task_i, embed_i], label=f"Soft Boundary ({embed_size})", color=colors[embed_i], marker='o', linestyle='-', linewidth=2, markersize=6)
            ax[task_i].axhline(hard_boundary_maps[task_i, embed_i], linestyle="--", label=f"Hard Boundary ({embed_size})", linewidth=2, color=colors[embed_i])

            ax[task_i].set_title(f"Embed Size: {embed_size}", fontsize=16)
            ax[task_i].set_ylabel("nCDG@10", fontsize=14)
            ax[task_i].legend(loc='best', fontsize=12)
        
        ax[task_i].set_title("Task: " + task_names[task_i], fontsize=16)

    # ax.axhline(truncated_boundary_map, label=f"Truncated Boundary (at 8192)", color="k", linestyle='-.', linewidth=2)
    ax[task_i].set_xlabel("Overlap Size", fontsize=14)

    fig.suptitle("mAP@10 for Different Embed Sizes and Overlap Sizes", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.96])


    # ==  plot with normalized ncdg for all tasks
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.size': 14, 'font.family': 'serif'})

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # colors = plt.cm.get_cmap('plasma', len(embed_sizes))(np.linspace(0, 1, len(embed_sizes), endpoint=False))
    colors = ["red", "blue", "green", "orange"]

    for task_i in range(len(task_names)):
        for embed_i, embed_size in enumerate(embed_sizes[:-1]):
            ax.plot([o - embed_i*5 for o in overlap_sizes], soft_boundary_ndcgs[task_i, embed_i] / np.mean(soft_boundary_ndcgs[task_i, embed_i]), color=colors[embed_i], marker='o', linestyle='-', linewidth=1, markersize=4)
            ax.axhline(hard_boundary_ndcgs[task_i, embed_i] / np.mean(hard_boundary_ndcgs[task_i, embed_i]), linestyle="--", linewidth=1, color=colors[embed_i])

    for embed_i, embed_size in enumerate(embed_sizes[:-1]):
        ax.plot([], [], label=f"Embed Size: {embed_size}", color=colors[embed_i], linestyle='-', linewidth=2)
 


    ax.set_title("NDCG@10 for Different Embed Sizes and Overlap Sizes Across Tasks (Normalized)", fontsize=16)
    ax.set_ylabel("Normalized nCDG@10", fontsize=14)
    ax.legend(loc='best', fontsize=12)
    ax.set_xlabel("Overlap Size", fontsize=14)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # ==


    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.size': 14, 'font.family': 'serif'})

    fig, ax = plt.subplots(2, 1, figsize=(12, 10))

    colors = ["red", "blue", "green", "orange"]
    markers = ["o", "s", "D", "X"]

    for task_i in range(len(task_names)):
        for embed_i, embed_size in enumerate(embed_sizes):
            diff_ndcgs = soft_boundary_ndcgs[task_i, embed_i] - hard_boundary_ndcgs[task_i, embed_i]
            ax[0].plot([o - embed_i*5 for o in overlap_sizes], diff_ndcgs, color=colors[embed_i], marker=markers[task_i], linestyle='-', linewidth=1.25, markersize=6)

    for embed_i, embed_size in enumerate(embed_sizes):
        ax[0].plot([], [], label=f"Num Tokens: {embed_size}", color=colors[embed_i], linestyle='-', linewidth=2)
    
    for task_i in range(len(task_names)):
        ax[0].scatter([], [], label=f"Task: {task_names[task_i]}", marker=markers[task_i],  color = "k")

    ax[0].axhline(0, color="black", linestyle="--", linewidth=2)
    ax[0].set_title("Difference in nDCG@10 Between Soft and Hard Boundaries Across Tasks", fontsize=16)
    ax[0].set_ylabel("Difference in nDCG@10", fontsize=14)
    ax[0].set_xlabel("Overlap Size", fontsize=14)

    for task_i in range(len(task_names)):
        for embed_i, embed_size in enumerate(embed_sizes):
            diff_maps = soft_boundary_maps[task_i, embed_i] - hard_boundary_maps[task_i, embed_i]
            ax[1].plot([o - embed_i*5 for o in overlap_sizes], diff_maps, color=colors[embed_i], marker=markers[task_i], linestyle='-', linewidth=1.25, markersize=6)

    for embed_i, embed_size in enumerate(embed_sizes):
        ax[1].plot([], [], label=f"Num Tokens: {embed_size}", color=colors[embed_i], linestyle='-', linewidth=2)
    
    for task_i in range(len(task_names)):
        ax[1].scatter([], [], label=f"Task: {task_names[task_i]}", marker=markers[task_i],  color = "k")

    ax[1].axhline(0, color="black", linestyle="--", linewidth=2)
    ax[1].set_title("Difference in mAP@10 Between Soft and Hard Boundaries Across Tasks", fontsize=16)
    ax[1].set_ylabel("Difference in mAP@10", fontsize=14)
    ax[1].set_xlabel("Overlap Size", fontsize=14)


    ax[-1].legend(loc='best', fontsize=12, frameon=True)

    fig.tight_layout(rect=[0, 0, 1, 0.96])


    