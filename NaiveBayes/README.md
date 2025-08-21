# Naive Bayes Project

This project implements a **Naive Bayes classifier** from scratch in Python. It demonstrates probabilistic classification using Bayes’ Theorem with Gaussian likelihoods on the **Adult Income dataset** from Kaggle.

---

## Dataset

The dataset used in this project is located in the `data/` folder. It is used to train and evaluate the Naive Bayes model implemented in `NaiveBayes.py`.

### Dataset Description

- **Source**: https://www.kaggle.com/datasets/uciml/adult-census-income  
- **Format**: CSV  
- **Columns** (simplified): `age`, `workclass`, `fnlwgt`, `education`, `education-num`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `capital-gain`, `capital-loss`, `hours-per-week`, `native-country`, `income`  
- **Size**: ~48,000 rows × 15 columns  

The task is to predict whether a person’s income exceeds **$50K/year**.

---

## Usage

To run the Naive Bayes algorithm:

```bash
# 1. Clone the repository:
git clone https://github.com/glacerjust/ML-Algorithms
# 2. Navigate to the NaiveBayes directory:
cd ML-Algorithms/NaiveBayes
# 3. Run the script:
python NaiveBayes.py