# 🟢Malicious URLs Classification

This project aims to classify URLs as **benign** or **malicious** (including phishing, defacement, and malware) using classic Machine Learning models.

It was developed as part of a Machine Learning course for the academic year 2024/2025.


##  Dataset

- Source: [Kaggle – Malicious URLs Dataset](https://www.kaggle.com/datasets/naveenbhadouria/malicious)


## Project Structure

- `malicious_phish1.csv` – Raw dataset (URLs + labels)
- `features_dataset.csv` – Preprocessed dataset with extracted features
- `X_train_scaled.npy`, `X_test_scaled.npy`, `y_train.npy`, `y_test.npy` – Scaled datasets ready for training
- `Data_exploration.ipynb` – Dataset analysis and feature extraction
- `PreProcessing.ipynb` – Data cleaning, scaling, and preparation
- `Model_training.ipynb` – Model training, evaluation, and hyperparameter tuning



## Models Used

Three models were implemented and compared:

- **Logistic Regression**
- **k-Nearest Neighbors (KNN)**
- **Random Forest**

Each model was trained, evaluated, and **tuned using `GridSearchCV`** with cross-validation.



## Feature Engineering

The following features were extracted from raw URLs:

- URL length
- Number of digits
- Number of special characters
- Number of dots (`.`)
- Number of subdirectories (`/`)
- Presence of IP address
- Use of HTTPS



## Installation

1. Clone or download this repository.
2. (Recommended) Create a virtual environment.
3. Install the required packages:

```bash
pip install -r requirements.txt
```
## Authors

- **Yousra Houacine**
- **Ikram Kadri**


