from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import joblib
import pandas as pd
import pennylane as qml
from imblearn.over_sampling import SMOTE

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
hybrid_model = tf.keras.models.load_model(
    "E:\\Studies\\IIT\\4 - Forth Year\\Final Year Project\\QuanNetDetct\\Model\\hybrid_qnn_model.h5",
    custom_objects={'QuantumLayer': QuantumLayer}
)

# Load preprocessing components
scaler = joblib.load("E:\\Studies\\IIT\\4 - Forth Year\\Final Year Project\\QuanNetDetct\\Model\\scaler.pkl")
pca = joblib.load("E:\\Studies\\IIT\\4 - Forth Year\\Final Year Project\\QuanNetDetct\\Model\\pca.pkl")
ohe = joblib.load("E:\\Studies\\IIT\\4 - Forth Year\\Final Year Project\\QuanNetDetct\\Model\\ohe.pkl")
label_encoder = joblib.load("E:\\Studies\\IIT\\4 - Forth Year\\Final Year Project\\QuanNetDetct\\Model\\label_encoder.pkl")
selected_features = joblib.load("E:\\Studies\\IIT\\4 - Forth Year\\Final Year Project\\QuanNetDetct\\Model\\selected_features.pkl")
smote = joblib.load("E:\\Studies\\IIT\\4 - Forth Year\\Final Year Project\\QuanNetDetct\\Model\\smote.pkl")

num_qubits = 3

def preprocess_input(data):
    """Preprocess input data for model prediction."""
    df = pd.DataFrame([data], columns=selected_features)
    
    # Apply Label Encoding (if categorical features exist)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = label_encoder.transform(df[col])
    
    # Scale data
    data_scaled = scaler.transform(df)
    
    # Apply PCA
    data_pca = pca.transform(data_scaled)
    
    # Apply SMOTE to balance data
    data_pca, _ = smote.fit_resample(data_pca, np.zeros((data_pca.shape[0],)))
    
    # Split Quantum and Classical features
    quantum_features = data_pca[:, :num_qubits]
    classical_features = data_pca[:, num_qubits:]
    return quantum_features, classical_features

@app.route("/predict", methods=["POST"])
def predict():
    """Predict if the TLS traffic is malicious or non-malicious."""
    try:
        input_data = request.json.get("features")
        if input_data is None:
            return jsonify({"error": "No features provided"}), 400
        
        # Preprocess input
        quantum_features, classical_features = preprocess_input(input_data)
        
        # Get prediction
        pred_prob = hybrid_model.predict([quantum_features, classical_features])
        pred_label = np.argmax(pred_prob, axis=1)[0]
        
        # Define Labels
        labels_dict = {0: "Malicious", 1: "Non-Malicious", 2: "Uncertain"}
        
        return jsonify({"prediction": labels_dict[pred_label], "probabilities": pred_prob.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
