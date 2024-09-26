import numpy as np
import matplotlib.pyplot as plt
import json
import os

if __name__ == "__main__":

    lc_dir = "results-chunked-pooling"
    nc_dir = "results-normal-pooling"

    # == Load results

    chunk_sizes = [8, 16, 32, 64, 128, 256, 512, 1024]
    task_names = ["SummScreenFD", "WikimQA"] # "QMSum",

    lc_ndcgs = np.empty((len(task_names), len(chunk_sizes)))
    nc_ndcgs = np.empty((len(task_names), len(chunk_sizes)))

    lc_maps = np.empty((len(task_names), len(chunk_sizes)))
    nc_maps = np.empty((len(task_names), len(chunk_sizes)))

    for task_i, task_name in enumerate(task_names):

        task_name = f"LEMB{task_name}RetrievalChunked"

        for chunk_i, chunk_size in enumerate(chunk_sizes):
                
            lc_dir_chunk_i = os.path.join(lc_dir, f"chunk_size_{chunk_size}", "no_model_name_available", "no_revision_available", f"{task_name}.json")
            with open(lc_dir_chunk_i, "r") as f:
                results = json.load(f)

            lc_ndcgs[task_i, chunk_i] = results["scores"]["test"][0]["ndcg_at_10"]
            lc_maps[task_i, chunk_i]  = results["scores"]["test"][0]["map_at_10"]

            nc_dir_chunk_i = os.path.join(nc_dir, f"chunk_size_{chunk_size}", "no_model_name_available", "no_revision_available", f"{task_name}.json")
            with open(nc_dir_chunk_i, "r") as f:
                results = json.load(f)
            
            nc_ndcgs[task_i, chunk_i] = results["scores"]["test"][0]["ndcg_at_10"]
            nc_maps[task_i, chunk_i]  = results["scores"]["test"][0]["map_at_10"]

    # == Plot
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.size': 14, 'font.family': 'serif'})

    # -- NDCG
    fig, ax = plt.subplots(len(task_names), 1, figsize=(12, 3*len(task_names)), sharex=True)

    for task_i in range(len(task_names)):
        ax[task_i].plot(chunk_sizes, lc_ndcgs[task_i], label="Late Chunking", color="blue", marker='o', linestyle='-', linewidth=2, markersize=6)
        ax[task_i].plot(chunk_sizes, nc_ndcgs[task_i], label="Naive Chunking", color="red", marker='o', linestyle='-', linewidth=2, markersize=6)
        ax[task_i].set_title(f"Task: {task_names[task_i]}", fontsize=16)
        ax[task_i].set_ylabel("nCDG@10", fontsize=14)
    ax[task_i].legend(loc='best', fontsize=12, frameon=True)
    
    ax[task_i].set_xlabel("Chunk Size", fontsize=14)

    # log scale
    ax[task_i].set_xscale('log')

    # set specific ticks
    ax[task_i].set_xticks([8, 16, 32, 64, 128, 256, 512, 1024])
    ax[task_i].set_xticklabels([8, 16, 32, 64, 128, 256, 512, 1024])
    fig.suptitle("nCDG@10 for Different Chunk Sizes", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # -- mAP
    fig, ax = plt.subplots(len(task_names), 1, figsize=(12, 3*len(task_names)), sharex=True)

    for task_i in range(len(task_names)):
        ax[task_i].plot(chunk_sizes, lc_maps[task_i], label="Late Chunking", color="blue", marker='o', linestyle='-', linewidth=2, markersize=6)
        ax[task_i].plot(chunk_sizes, nc_maps[task_i], label="Naive Chunking", color="red", marker='o', linestyle='-', linewidth=2, markersize=6)
        ax[task_i].set_title(f"Task: {task_names[task_i]}", fontsize=16)
        ax[task_i].set_ylabel("mAP@10", fontsize=14)
    ax[task_i].legend(loc='best', fontsize=12, frameon=True)
    
    ax[task_i].set_xlabel("Chunk Size", fontsize=14)

    # log scale
    ax[task_i].set_xscale('log')

    # set specific ticks
    ax[task_i].set_xticks([8, 16, 32, 64, 128, 256, 512, 1024])
    ax[task_i].set_xticklabels([8, 16, 32, 64, 128, 256, 512, 1024])
    fig.suptitle("mAP@10 for Different Chunk Sizes", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()