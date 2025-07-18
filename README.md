# Oil Mine Presence Prediction using LSGAT and Graph Neural Networks

## Project Overview

This project focuses on predicting the presence of oil mines using geological and environmental data points. It leverages Graph Neural Networks (GNNs), specifically a custom Layer-wise Skip-connection Graph Attention Network (LSGAT), to model the spatial relationships and features of the data. The solution includes a comprehensive training pipeline, detailed evaluation, hyperparameter sensitivity analysis, and an interactive web application built with Streamlit for live predictions.

## Features

* **Data Preprocessing**: Handles raw geological and environmental data, including categorical feature encoding and numerical feature scaling.
* **Graph Construction**: Builds a K-Nearest Neighbors (KNN) graph using `scipy.spatial.KDTree` to represent spatial relationships between data points.
* **Custom LSGAT Model**: Implements a novel Graph Attention Network layer with skip connections, learnable gamma parameters, and feature-level attention for enhanced performance and stability.
* **Robust Training Pipeline**: Incorporates advanced techniques like DropEdge regularization, `ReduceLROnPlateau` learning rate scheduling, and early stopping to optimize model training and prevent overfitting.
* **Comprehensive Evaluation**: Provides a wide array of classification metrics, including Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, MCC, and Cohen's Kappa, for a thorough performance assessment.
* **Hyperparameter Sensitivity Analysis**: Systematically explores the impact of key hyperparameters on model performance, generating insightful plots.
* **Interactive Web Application (Streamlit)**: A user-friendly frontend that allows live input of parameters and real-time oil mine presence prediction, visualizing results on a Folium map.
* **Geographical Visualization**: Generates Folium maps to visualize actual vs. predicted oil mine locations and prediction correctness for the test set.

## Setup and Installation

### Prerequisites

* Python 3.8+ (preferably 3.9 or 3.10 for broader library compatibility)
* Git (for version control)

### Steps

1.  **Clone the Repository (or create it locally):**
    If you're starting a new repository, initialize it locally. If you've already created it on GitHub, clone it:
    ```bash
    git clone https://github.com/paraspatil11/Oil-Mine-Presence-Prediction.git
    cd Oil-Mine-Presence-Prediction
    ```

2.  **Create a Virtual Environment (Recommended):**
    It's good practice to use a virtual environment to manage dependencies.
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    Install all required Python packages using `pip` and the provided `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```
    **Note on PyTorch:** The `requirements.txt` will install the CPU version of PyTorch by default. If you have a compatible NVIDIA GPU and want to use it, you might need to install PyTorch separately from their official website (https://pytorch.org/get-started/locally/) based on your CUDA version, then install `torch-geometric` as specified.

4.  **Obtain Data:**
    Ensure `preprocessed_mining_data.csv` is placed in the root directory of your project. This file is essential for both training and running the application.

## Usage

### 1. Train the Model and Generate Resources

Run the `train.py` script to train the LSGAT model, evaluate its performance, perform hyperparameter sensitivity analysis, and save the necessary model weights and preprocessing objects (`.pth` and `.pkl` files).
This script will also generate several output files including evaluation metrics, plots, and Folium maps.

```bash
python train.py

```

### 2. Run the Streamlit Web Application
Once train.py has successfully run and generated all the required files, you can launch the interactive Streamlit application.

```bash
streamlit run app.py
