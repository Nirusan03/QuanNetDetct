from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import joblib
import pandas as pd

# Define QuantumLayer
class QuantumLayer(tf.keras.layers.Layer):
    def __init__(self, num_qubits, **kwargs):
        super(QuantumLayer, self).__init__(**kwargs)
        self.num_qubits = num_qubits
        self.q_weights = self.add_weight(name="q_weights", shape=(1, num_qubits), initializer="glorot_uniform", trainable=True)

    def call(self, inputs):
        return tf.random.uniform((tf.shape(inputs)[0], self.num_qubits))

    def get_config(self):
        config = super().get_config()
        config.update({"num_qubits": self.num_qubits})
        return config

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Load trained model with QuantumLayer registered
model_path = "E:\\Studies\\IIT\\4 - Forth Year\\Final Year Project\\QuanNetDetct\\Model\\hybrid_qnn_model.h5"
hybrid_model = tf.keras.models.load_model(model_path, custom_objects={'QuantumLayer': QuantumLayer})

# Load preprocessing components
scaler = joblib.load("E:\\Studies\\IIT\\4 - Forth Year\\Final Year Project\\QuanNetDetct\\Model\\scaler.pkl")
pca = joblib.load("E:\\Studies\\IIT\\4 - Forth Year\\Final Year Project\\QuanNetDetct\\Model\\pca.pkl")
label_encoder = joblib.load("E:\\Studies\\IIT\\4 - Forth Year\\Final Year Project\\QuanNetDetct\\Model\\label_encoder.pkl")
selected_features = joblib.load("E:\\Studies\\IIT\\4 - Forth Year\\Final Year Project\\QuanNetDetct\\Model\\selected_features.pkl")
print("Expected Features:", selected_features)

num_qubits = 3  # Number of quantum features expected

def preprocess_input(data):
    """Preprocess input data for model prediction."""
    try:
        # Convert input into DataFrame
        df = pd.DataFrame([data], columns=selected_features)

        # Validate: Ensure correct feature names
        missing_features = set(selected_features) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing Features: {missing_features}")

        # Apply preprocessing
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = label_encoder.transform(df[col])

        data_scaled = scaler.transform(df)
        data_pca = pca.transform(data_scaled)

        return data_pca[:, :num_qubits], data_pca[:, num_qubits:]

    except Exception as e:
        raise ValueError(f"Preprocessing Error: {str(e)}")

@app.route("/predict", methods=["POST"])
def predict():
    """Predict if the TLS traffic is malicious or non-malicious."""
    try:
        input_data = request.json.get("features")
        
        # Validate input format
        if not isinstance(input_data, list) or len(input_data) != len(selected_features):
            return jsonify({"error": "Invalid input format. Ensure correct number of features."}), 400
        
        # Convert input values to numbers and validate
        try:
            input_data = [float(x) for x in input_data]
        except ValueError:
            return jsonify({"error": "Invalid input! Ensure all values are numeric."}), 400

        # Preprocess input
        quantum_features, classical_features = preprocess_input(input_data)

        # Get prediction
        pred_prob = hybrid_model.predict([quantum_features, classical_features])
        pred_label = np.argmax(pred_prob, axis=1)[0]

        # Define Labels
        labels_dict = {0: "Malicious", 1: "Non-Malicious", 2: "Uncertain"}

        return jsonify({"prediction": labels_dict[pred_label], "probabilities": pred_prob.tolist()})

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400

    except Exception as e:
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
