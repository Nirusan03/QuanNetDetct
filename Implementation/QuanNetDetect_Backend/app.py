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
scaler_feature_names = scaler.feature_names_in_.tolist()

# Extract expected number of features
num_features_before_pca = scaler.n_features_in_
num_features_after_pca = pca.n_components_

print(f"Expected features before PCA: {num_features_before_pca}, after PCA: {num_features_after_pca}")
print("Expected feature names:", scaler_feature_names)

num_qubits = 3  # Number of quantum features expected

def preprocess_input(data):
    """Preprocess input data for model prediction."""
    try:
        if len(data) != num_features_before_pca:
            return None, None, f"Invalid input: Expected {num_features_before_pca} features, but received {len(data)}"

        df = pd.DataFrame([data], columns=scaler_feature_names)

        # Apply preprocessing
        data_scaled = scaler.transform(df)
        data_scaled_selected = data_scaled[:, :12]  # Select only first 12 features
        data_pca = pca.transform(data_scaled_selected)

        return data_pca[:, :num_qubits], data_pca[:, num_qubits:], None

    except Exception as e:
        return None, None, f"Preprocessing Error: {str(e)}"

@app.route("/predict", methods=["POST"])
def predict():
    """Predict if the TLS traffic is malicious or non-malicious."""
    try:
        input_data = request.json.get("features")
        
        if not isinstance(input_data, list):
            return jsonify({"error": "Invalid input format: Expected list of numerical values."}), 400
        
        quantum_features, classical_features, error = preprocess_input(input_data)
        if error:
            return jsonify({"error": error}), 400

        pred_prob = hybrid_model.predict([quantum_features, classical_features])
        pred_label = np.argmax(pred_prob, axis=1)[0]

        labels_dict = {0: "Malicious", 1: "Non-Malicious", 2: "Uncertain"}

        return jsonify({"prediction": labels_dict.get(pred_label, "Unknown"), "probabilities": pred_prob.tolist()})

    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
