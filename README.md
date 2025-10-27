# Yarrowia Language Modelling

This project explores **genome-scale language modeling** applied to _Yarrowia lipolytica_.  
It begins by recreating and adapting key notebooks from the [Arc Institute’s Evo 2 repository](https://github.com/ArcInstitute/evo2) using the **NVIDIA API**, then extends those methods to Yarrowia genomic databases.

## Goals

1. Reproduce Evo 2 notebooks for embedding, sparse autoencoder, and interpretability workflows.
2. Build an API-based workflow for running forward passes and training lightweight classifiers.
3. Prepare Yarrowia genome data (FASTA + GFF3) for future modeling and analysis.

## Status

- ✅ Notebook recreation and API setup (in progress)
- ⏳ Application to Yarrowia (not yet implemented)

## Requirements

- Python 3.10+
- Access to NVIDIA API (Evo 2 endpoints)
- Standard packages: `numpy`, `pandas`, `requests`, `torch`

## API setup

Make a .env file with this structure:
NVIDIA_API_KEY="nvapi-XXXXXXXXXXX-XXX-XXXXXXXXXXXXX-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX-XX"
