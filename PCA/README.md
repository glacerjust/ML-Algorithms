# PCA Project

This project implements **Principal Component Analysis (PCA)** from scratch in Python. It demonstrates dimensionality reduction by projecting high-dimensional data into a lower-dimensional subspace while preserving as much variance as possible.

---

## Dataset

The dataset used in this project is `wine-clustering.csv`, located in the root directory.

### Dataset Description

- **Source**: UCI Wine Dataset (commonly used for clustering & classification tasks)  
- **Format**: CSV  
- **Columns**: Chemical composition features of wines (e.g., alcohol, malic acid, ash, alcalinity, magnesium, flavanoids, etc.)  
- **Size**: 178 rows × 13 columns  

The task is unsupervised: reducing the dataset into principal components for visualization and further analysis.

---

## Usage

To run the PCA algorithm:

```bash
# 1. Clone the repository:
git clone https://github.com/glacerjust/ML-Algorithms
# 2. Navigate to the PCA directory:
cd ML-Algorithms/PCA
# 3. Run the script:
python PCA.py