import mlflow
import mlflow.sklearn
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from data_preprocessing import preprocess_data

# Set the Tracking URI to use a local SQLite database for experiment logs 
mlflow.set_tracking_uri("sqlite:///mlflow.db")
# Create or set the experiment name in MLflow 
mlflow.set_experiment("Breast_Cancer_Experiment")

def train():
    # Load and preprocess the Breast Cancer Wisconsin Dataset 
    X_train, X_test, y_train, y_test, scaler = preprocess_data()

    # Start an MLflow run to track this experiment [cite: 29]
    with mlflow.start_run():
        # Initialize and train the Random Forest model as required 
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Make predictions and evaluate accuracy 
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        # Log hyperparameters and evaluation metrics to MLflow UI 
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", acc)
        
        # Log and register the model for versioning and registry control 
        mlflow.sklearn.log_model(model, "model")

        # Save the best trained model as an artifact (pickle file) 
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        # Save the scaler to ensure consistent preprocessing during inference 
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

        print(f"Model trained with accuracy: {acc}")
        print("Model and Scaler saved as .pkl files.")

if __name__ == "__main__":
    train()