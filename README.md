# KNN Algorithm Project

This project implements a **K-Nearest Neighbors (KNN) algorithm** in Python. It demonstrates the basics of KNN on the **Breast Cancer Wisconsin dataset** from Kaggle.

---

## Dataset

The dataset used in this project is located in the `data/` folder. It is used to train and evaluate the KNN algorithm implemented in `knn.py`.

### Dataset Description

- **Source**: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
- **Format**: CSV
- **Columns**: "id","diagnosis","radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"
- **Size**: 570x32

### Usage

To run the KNN algorithm:


```bash
#1. Clone the repository:
git clone https://github.com/glacerjust/hello-github.git
#2. Navigate to the project directory:
cd hello-github
#3. Run the KNN script:
python knn.py
```

Make sure the dataset (**dataset.csv**) is located in the data/ folder.

### Notes

This implementation is a from-scratch KNN algorithm for educational purposes.

You can extend it by adding data preprocessing, scaling, or evaluation metrics.