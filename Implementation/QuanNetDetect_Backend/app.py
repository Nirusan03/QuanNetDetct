from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
import io

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
CORS(app)

# Load trained model with QuantumLayer registered
model_path = "E:\\Studies\\IIT\\4 - Forth Year\\Final Year Project\\QuanNetDetct\\Model\\hybrid_qnn_model.h5"
hybrid_model = tf.keras.models.load_model(model_path, custom_objects={'QuantumLayer': QuantumLayer})

# Load preprocessing components
scaler = joblib.load("E:\\Studies\\IIT\\4 - Forth Year\\Final Year Project\\QuanNetDetct\\Model\\scaler.pkl")
pca = joblib.load("E:\\Studies\\IIT\\4 - Forth Year\\Final Year Project\\QuanNetDetct\\Model\\pca.pkl")
scaler_feature_names = scaler.feature_names_in_.tolist()

num_features_before_pca = scaler.n_features_in_
num_features_after_pca = pca.n_components_
num_qubits = 3  # Number of quantum features expected

def validate_csv(file):
    """Check if the uploaded file is a valid CSV."""
    try:
        df = pd.read_csv(io.StringIO(file.read().decode("utf-8")))
        
        # Ensure it contains the required number of columns
        if len(df.columns) != num_features_before_pca:
            return None, f"Invalid CSV format. Expected {num_features_before_pca} features, but received {len(df.columns)}."
        
        # Check for missing values
        if df.isnull().values.any():
            return None, "CSV contains missing values. Please remove or fill missing data."

        # Ensure all values are numeric
        if not all(df.applymap(lambda x: isinstance(x, (int, float))).all()):
            return None, "CSV contains non-numeric values. Please upload a properly formatted dataset."
        
        return df, None
    except Exception as e:
        return None, f"Error reading CSV file: {str(e)}"

def preprocess_input(dataframe):
    """Preprocess input DataFrame for model prediction."""
    try:
        df = dataframe.copy()
        df = df[scaler_feature_names]  # Ensure correct feature order

        # Apply preprocessing
        data_scaled = scaler.transform(df)

        selected_feature_names = [
            'Total Bwd packets', 'Bwd Packet Length Min', 'Fwd Header Length', 'Bwd Header Length', 'Bwd Packets/s',
            'SYN Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'FWD Init Win Bytes', 'Bwd Init Win Bytes', 'Fwd Seg Size Min', 'Bwd Packet Length Mean'
        ]
        
        feature_indices = [scaler_feature_names.index(f) for f in selected_feature_names if f in scaler_feature_names]
        data_scaled_selected = data_scaled[:, feature_indices]
        data_pca = pca.transform(data_scaled_selected)

        return data_pca[:, :num_qubits], data_pca[:, num_qubits:], None
    except Exception as e:
        return None, f"Preprocessing Error: {str(e)}"

@app.route("/predict", methods=["POST"])
def predict():
    """Handle both JSON feature input and CSV file upload."""
    try:
        if "file" in request.files:
            file = request.files["file"]

            # Validate file existence
            if file.filename == "":
                return jsonify({"error": "Empty file uploaded."}), 400
            # Validate and process CSV
            df, error = validate_csv(file)
            if error:
                return jsonify({"error": error}), 400

            quantum_features, classical_features, error = preprocess_input(df)
            if error:
                return jsonify({"error": error}), 400

        elif request.is_json:
            json_data = request.get_json()
            features = json_data.get("features", None)

            if features is None or not isinstance(features, list):
                return jsonify({"error": "Invalid JSON format. Expected a list of numerical features."}), 400

            if len(features) != num_features_before_pca:
                return jsonify({"error": f"Invalid input length. Expected {num_features_before_pca} features, received {len(features)}"}), 400

            df = pd.DataFrame([features], columns=scaler_feature_names)
            
            quantum_features, classical_features, error = preprocess_input(df)
            if error:
                return jsonify({"error": error}), 400

        else:
            return jsonify({"error": "No valid input provided. Please upload a CSV file or send JSON features."}), 400

        # Make predictions
        pred_probs = hybrid_model.predict([quantum_features, classical_features])
        pred_labels = np.argmax(pred_probs, axis=1)

        # Define TLS-specific label descriptions
        # Define TLS-specific label descriptions
        labels_dict = {
            0: {
                "label": "Malicious TLS Traffic",
                "description": "This TLS session has characteristics commonly associated with cyber threats such as phishing, malware communication, or unauthorized access.",
                "potential_risks": [
                    "Encrypted communication with a suspicious server.",
                    "Possible exfiltration of sensitive data.",
                    "TLS handshake anomalies indicating an attack."
                ],
                "recommended_actions": [
                    "Investigate the destination server and its reputation.",
                    "Analyze packet payloads (if possible) for malicious patterns.",
                    "Consider blocking the IP address if it appears in threat databases."
                ]
            },
            1: {
                "label": "Non-Malicious TLS Traffic",
                "description": "This TLS flow is classified as safe and does not exhibit signs of an attack or anomaly.",
                "potential_risks": [
                    "Minimal risks detected, as the communication pattern aligns with normal TLS usage."
                ],
                "recommended_actions": [
                    "No immediate action required.",
                    "Continue monitoring for any unusual patterns over time."
                ]
            },
            2: {
                "label": "Uncertain TLS Traffic",
                "description": "The model is unable to confidently classify this traffic. It may contain unusual characteristics but lacks a strong indication of malicious activity.",
                "potential_risks": [
                    "TLS traffic does not clearly match either malicious or safe patterns.",
                    "Anomalies such as inconsistent handshake messages, timing irregularities, or unexpected packet sizes."
                ],
                "recommended_actions": [
                    "Conduct a deeper forensic analysis using additional network indicators.",
                    "Check historical logs to see if this IP or TLS pattern has been observed before.",
                    "Consider sandboxing the communication to analyze behavior."
                ]
            },
            3: {
                "label": "Benign TLS Traffic",
                "description": "This TLS communication is completely normal and aligns with expected secure traffic behavior.",
                "potential_risks": [
                    "No security concerns detected.",
                    "Consistent with routine encrypted traffic such as web browsing, email exchange, or API communication."
                ],
                "recommended_actions": [
                    "No action required.",
                    "Ensure general security policies are in place (e.g., TLS version enforcement, certificate validation)."
                ]
            }
        }

        predictions = []
        for idx, (label, prob) in enumerate(zip(pred_labels, pred_probs)):
            confidence = round(float(max(prob)) * 100, 2)  # Convert confidence to percentage
            predictions.append({
                "index": idx,
                "prediction": labels_dict[label]["label"],
                "description": labels_dict[label]["description"],
                "confidence": f"{confidence}%",
                "potential_risks": labels_dict[label]["potential_risks"],
                "recommended_actions": labels_dict[label]["recommended_actions"],
                "probabilities": prob.tolist()
            })

        return jsonify({"predictions": predictions})

    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
