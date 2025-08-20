# Logistic Regression Project

This project implements a **Logistic Regression algorithm** from scratch in python. It demonstrate the classification of binary output by optimizing the Binary Cross-Entropy Loss function with gradient descent on **Breast Cancer Wisconsin dataset** from Kaggle.

---

## Dataset

The dataset used in this project is located in the `data/` folder. It is used to train and evaluate the Logistic Regression model implemented in `LogisticRegression.py`.

### Dataset Description

- **Source**: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
- **Format**: CSV
- **Columns**: "id","diagnosis","radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"
- **Size**: 570x32

### Usage

To run the Logistic Regression algorithm:

```bash
# 1. Clone the repository:
git clone https://github.com/glacerjust/ML-Algorithms
# 2. Navigate to the LogisticRegression directory:
cd ML-Algorithms/LogisticRegression
# 3. Run the script:
python LogisticRegression.py