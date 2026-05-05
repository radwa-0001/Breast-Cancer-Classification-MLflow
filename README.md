🩺 Breast Cancer Classification with MLOps (MLflow)

This project implements a robust machine learning pipeline for the classification of breast cancer tumors as Malignant or Benign. It demonstrates a professional MLOps workflow by integrating MLflow for experiment tracking and Flask for model deployment.

🌟 Key Features

1- High Accuracy: Achieved a model accuracy of 96.49% using Random Forest.

2- Experiment Tracking: Comprehensive logging of hyperparameters (n_estimators, random_state) and metrics using MLflow.

3- Persistent Storage: Uses a SQLite backend (mlflow.db) to ensure all training history is saved.

4- Modular Architecture: Separate scripts for data preprocessing, training, and API deployment.

5- REST API: A Flask-based inference server for real-time predictions.

🛠️ Tech Stack

1- Language: Python 3.x

2- ML Libraries: Scikit-Learn, Pandas, NumPy

3- MLOps: MLflow

4- API Framework: Flask

5- Environment: Virtual Environment (venv)

📁 Project Structure

MLflow Project/
├── myenv/                # Virtual environment
├── data_preprocessing.py # Data scaling and splitting logic
├── train_model.py        # MLflow training script
├── app.py                # Flask API for deployment
├── mlflow.db             # SQLite database for experiments
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation


🚀 Getting Started

1. Installation

Clone the repository and install the required packages:

.\myenv\Scripts\activate
pip install -r requirements.txt


2. Training the Model

To execute the training pipeline and log results to MLflow:

python train_model.py


3. Visualizing Experiments

Start the MLflow UI to view the dashboard and compare runs:

python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 8080


Then, visit http://127.0.0.1:8080 in your browser.

4. Running the API

To start the prediction server:

python app.py


The API will be available at http://127.0.0.1:5000/predict.

📊 Results

The experiments tracked via MLflow revealed that the Random Forest classifier with optimal scaling provides highly reliable diagnostic predictions. All model artifacts (model.pkl and scaler.pkl) are versioned and stored for production readiness.

🎓 Academic Context

This project was developed as part of the Field Training at King Salman International University (KSIU). It reflects advanced skills in AI Engineering and the application of industry-standard MLOps practices.

Developed by Radwa Yahia | Aspiring AI Engineer
Supervised by | Dr.Saeed Mohesn