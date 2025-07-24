# 🟡Malicious URL Detection Web App

This project is a lightweight and interactive FastAPI web application for detecting malicious URLs using two machine learning models:

- MLP (Multi-Layer Perceptron)
- Random Forest

## Features

- URL analysis with extracted features
- Model choice (MLP or Random Forest)
- Confusion matrix and accuracy display
- Interactive web interface (FastAPI + Jinja2)
- Dynamic chart of prediction scores

## Setup

### 1. Clone the repository

```
git clone --filter=blob:none --sparse https://github.com/ikramkdr/Malicious-URLs-Classification.git
cd Malicious-URLs-Classification
git sparse-checkout set fastapi_version

```
### 2. How to Run the project

### a. Install dependencies

```
pip install -r requirements.txt
```
### b. Generate the Features Dataset

```
python utils/generate_dataset.py

```
### c. Train the models

```
python train.py
```

This will train both the MLP and Random Forest models and save them.

### d. Launch the app

```
uvicorn app:app --reload
```

Then open your browser at: [http://127.0.0.1:8000](http://127.0.0.1:8000)

## Project Structure

```
.
├── app.py                  # FastAPI backend
├── train.py                # Training script
├── models/                 # Contains MLP model class
├── utils/                  # Feature extraction & dataset scripts
├── templates/              # Jinja2 HTML templates
├── static/                 # Images, CSS, output PNGs
├── data/                   # features_dataset.csv
├── test_urls.txt           # Sample test URLs
├── mlp_model.pt            # Trained PyTorch model
├── rf_model.pkl            # Trained Random Forest model
├── scaler.pkl              # Scaler for features
├── requirements.txt
└── README.md
```

## Authors

- **Ikram KADRI**
- **Yousra HOUACINE**
