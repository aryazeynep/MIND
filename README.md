# Molecular Intelligence for Novel Discovery (MIND)

## About The Project

[cite_start]MIND is a universal foundation model for molecular representation learning, designed to handle diverse biological molecules like DNA, RNA, proteins, and small molecules within a single, unified framework[cite: 1]. [cite_start]The model is built upon a Graph Transformer backbone and is trained using a self-supervised strategy to comprehend the universal principles of 3D molecular structure[cite: 1].

This guide explains how to set up the environment and run the complete data processing and pre-training pipeline for the protein dataset.

## Getting Started

Follow these steps to set up your local environment for development and training.

### Prerequisites

-   A Linux-based system with NVIDIA GPUs and CUDA installed.
-   **Conda** package manager.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd MIND
    ```

2.  **Create and activate the Conda environment:**
    The project dependencies are specified in the `env.yml` file.
    ```bash
    # Create the conda environment from the provided file
    conda env create -f GET/env.yml

    # Activate the environment
    conda activate mind_esa3d
    ```

## Training Pipeline: Step-by-Step Guide

This guide will walk you through the entire process, from downloading raw data to starting the pre-training job. The pipeline is designed to be memory-efficient and scalable to millions of proteins by processing data in chunks.

### Step 1: Download Initial Metadata

First, we need to download the metadata file from the AlphaFold DB clusters. This file contains information about all protein structures, which we will filter in the next step.

```bash
python data/download_protein_clusters.py
```
This will download `representatives_metadata.tsv.gz` into the `data/proteins/afdb_clusters/` directory.

### Step 2: Create a Filtered Manifest

Next, we create a "manifest" file. This is a CSV file that lists the specific proteins we want to use for training, based on quality (pLDDT score) and size (amino acid length).

```bash
# This example creates a manifest for 40,000 proteins with a pLDDT score > 70 and fewer than 512 amino acids.
python data/protein_pipeline/1_filter_and_create_manifest.py \
    --mode manifest \
    --metadata-file data/proteins/afdb_clusters/representatives_metadata.tsv.gz \
    --target-count 40000 \
    --output data/proteins/afdb_clusters/manifest_hq_40k_len512.csv \
    --existing-structures-dir data/proteins/raw_structures_hq_40k \
    --plddt 70 \
    --max-len 512
```

### Step 3: Download Protein Structures

Using the manifest file created in the previous step, this script downloads the actual 3D structure files (`.pdb` or `.cif`) for the selected proteins.

```bash
python data/protein_pipeline/2_download_pdbs_from_manifest.py \
    --manifest-file data/proteins/afdb_clusters/manifest_hq_40k_len512.csv \
    --structures-outdir data/proteins/raw_structures_hq_40k \
    --workers 16
```

### Step 4: Process Data into Scalable Chunks

This is the main data processing step. A new automated script, `process_chunked_dataset.py`, handles the entire memory-intensive conversion. [cite_start]It splits the manifest into smaller parts ("chunks") and processes each one independently to avoid running out of RAM[cite: 14].

This script performs two stages for each chunk:
1.  Converts raw `.pdb` files to the universal `.pkl` format.
2.  Converts the `.pkl` file to the final, optimized PyTorch Geometric `.pt` format.

```bash
# This command processes the 40k manifest into 2 chunks (20k proteins each).
# For 1M proteins, you can increase --num-chunks to 200.
python data_loading/process_chunked_dataset.py \
    --config-yaml-path core/pretraining_config_protein.yaml \
    --data-path data/proteins/raw_structures_hq_40k \
    --manifest-file data/proteins/afdb_clusters/manifest_hq_40k_len512.csv \
    --num-chunks 2
```
After this step, your processed data will be ready in separate chunk directories (e.g., `../data/proteins/processed_graphs_40k_chunk_0/`, `..._chunk_1/`).

### Step 5: Start Pre-training

Finally, start the training. [cite_start]The training script (`core/train_pretrain.py`) is designed to automatically detect the chunked dataset directories and load the data efficiently using the `LazyUniversalDataset`[cite: 18].

We use `nohup` to ensure the training continues even if your SSH connection is lost.

```bash
# Make sure you are in the project's root directory (MIND/)
# This command starts training on GPU 6 and saves all logs to 'mind_training.log'.
CUDA_VISIBLE_DEVICES=6 nohup python -m core.train_pretrain \
    --config-yaml-path core/pretraining_config_protein.yaml \
    > mind_training.log 2>&1 &
```
