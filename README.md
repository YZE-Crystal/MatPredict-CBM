# MatPredict-CBM

## Overview
MatPredict-CBM is a multi-branch deep learning pipeline that predicts the Conduction Band Minimum (CBM) of crystal materials from `.cif` files. It processes three distinct information sources in parallel to make highly accurate predictions within the [-10, 10] eV range.

## Core Architecture
* **GNN Branch:** A 6-layer Crystal Graph Convolutional Network (CGConv) using 16 Gaussian RBF encodings for edge features.
* **TDA Branch:** Uses topological data analysis (Vietoris-Rips complex) to extract 0th and 1st dimension persistent homology features.
* **Fusion & CPPN Head:** Combines the branches into a 384-dimensional vector and passes it through a Compositional Pattern Producing Network (CPPN) with linear, quadratic, and cubic branches.

## Key Features
* **Efficient Caching:** Extracts and saves node, edge, and TDA features as `.pt` files for fast reuse.
* **Robust Training:** Utilizes Huber loss, gradient norm penalty, AdamW optimizer, and OneCycleLR scheduling. Mixed precision (AMP) is supported.
* **Smart Filtering:** Automatically handles NaN/Inf values and filters out predictions for invalid items.

## Inputs & Outputs
* **Inputs Required:** `labels.csv` (with filename and cbm columns) and a directory of `.cif` files.
* **Outputs Generated:** `predictions.csv` (true vs. predicted values), `training_log_v2.csv`, and model weights (`model_v2.pt`, `model_best_v2.pt`).
