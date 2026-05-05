from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the saved model and scaler produced during the training phase
# These files are necessary for real-time predictions
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Define the endpoint for real-time predictions via HTTP POST requests 
# @app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receive input data in JSON format
        data = request.get_json(force=True)
        
        # Convert the input features into a NumPy array and reshape for the model
        features = np.array(data['features']).reshape(1, -1)
        
        # Apply the same scaling transformation used during model training
        features_scaled = scaler.transform(features)
        
        # Perform the prediction using the loaded model
        prediction = model.predict(features_scaled)
        
        # Map the model output to the corresponding diagnosis 
        # 0 usually represents Malignant and 1 represents Benign in this dataset
        result = "Malignant" if prediction[0] == 0 else "Benign"
        
        # Return the prediction result as a JSON response
        return jsonify({'prediction': result})
    
    except Exception as e:
        # Return the error message if something goes wrong during the request
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    # Start the Flask development server on port 5000 
    app.run(port=5000, debug=True)