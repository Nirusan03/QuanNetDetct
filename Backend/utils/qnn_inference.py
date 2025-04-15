import os
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
import logging

# Suppress logs and force CPU mode
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
warnings.filterwarnings("ignore")
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('pennylane').setLevel(logging.ERROR)

import tensorflow as tf
import pennylane as qml

# Model Configuration
class ModelConfig:
    def __init__(self):
        self.quantum_feature_count = 30
        self.num_classes = 5
        self.num_qubits = 5
        self.embedding_dim = 2 ** self.num_qubits
        self.num_layers = 4

config = ModelConfig()

# Quantum Circuit Definition
dev = qml.device("default.qubit.tf", wires=config.num_qubits)

@qml.qnode(dev, interface="tf", diff_method="backprop")
def quantum_circuit(inputs, weights):
    inputs = tf.cast(inputs, tf.float32)
    weights = tf.cast(weights, tf.float32)

    for i in range(config.num_qubits):
        qml.Hadamard(wires=i)

    qml.AmplitudeEmbedding(inputs, wires=range(config.num_qubits), normalize=True, pad_with=0.0)
    qml.StronglyEntanglingLayers(weights, wires=range(config.num_qubits))

    return [qml.expval(qml.PauliZ(i)) for i in range(config.num_qubits)]

# Quantum Layer for Keras
class QuantumLayer(tf.keras.layers.Layer):
    def __init__(self, num_qubits, num_layers, **kwargs):
        super().__init__(**kwargs)
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.quantum_weights = self.add_weight(
            shape=(num_layers, num_qubits, 3),
            initializer='glorot_uniform',
            trainable=True,
            dtype=tf.float32
        )

    def call(self, inputs):
        padded_inputs = tf.pad(inputs, [[0, 0], [0, config.embedding_dim - tf.shape(inputs)[1]]])
        padded_inputs = tf.cast(padded_inputs, tf.float32)

        def process_sample(x):
            measurements = quantum_circuit(x, self.quantum_weights)
            return tf.convert_to_tensor(measurements, dtype=tf.float32)

        return tf.map_fn(
            process_sample,
            padded_inputs,
            fn_output_signature=tf.TensorSpec(shape=(config.num_qubits,), dtype=tf.float32)
        )

    def get_config(self):
        return {'num_qubits': self.num_qubits, 'num_layers': self.num_layers}

# Load Trained Model
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/QNN_DDos2019.h5'))
model = tf.keras.models.load_model(
    model_path,
    custom_objects={"QuantumLayer": QuantumLayer},
    compile=False
)

# Class Names for TLS Classification
class_names = ['DrDoS_DNS', 'DrDoS_LDAP', 'Syn', 'LDAP', 'BENIGN']

# Prediction + Quantum Log Report
def run_qnn_prediction(csv_path, file_id=None):
    print(f"[+] Running QNN inference on: {csv_path}")
    start_time = time.time()

    df = pd.read_csv(csv_path)
    X_test = df.values.astype(np.float32)

    # Quantum circuit output: expectation values
    all_expvals = model.predict([X_test[:, :30], X_test[:, :30]])
    inference_time = round(time.time() - start_time, 4)

    predicted_indices = np.argmax(all_expvals, axis=1)
    predicted_labels = [class_names[i] for i in predicted_indices]

    result = []
    for i, (probs, label) in enumerate(zip(all_expvals, predicted_labels)):
        record = {"id": i + 1, "predicted_class": label}
        for j, prob in enumerate(probs):
            record[class_names[j]] = float(round(prob, 4))
        result.append(record)

    print(f"[+] Prediction complete: {len(result)} flows analyzed.")

    # Quantum Log Enhancements
    avg_expval_per_wire = np.mean(all_expvals, axis=0).tolist()
    var_expval_per_wire = np.var(all_expvals, axis=0).tolist()

    expval_flat = all_expvals.flatten()
    measurement_stats = {
        "min": float(np.min(expval_flat)),
        "max": float(np.max(expval_flat)),
        "mean": float(np.mean(expval_flat)),
        "std": float(np.std(expval_flat))
    }

    expval_samples = all_expvals[:min(5, len(all_expvals))].tolist()

    quantum_log = {
        "file_id": file_id or "N/A",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "quantum_config": {
            "num_qubits": config.num_qubits,
            "embedding_dimension": config.embedding_dim,
            "entangling_layers": config.num_layers,
            "circuit_structure": "AmplitudeEmbedding + StronglyEntanglingLayers"
        },
        "num_flows_predicted": len(X_test),
        "inference_time_seconds": inference_time,
        "avg_expectation_per_qubit": avg_expval_per_wire,
        "var_expectation_per_qubit": var_expval_per_wire,
        "measurement_distribution": measurement_stats,
        "sample_expectation_matrix": expval_samples
    }

    # Save as JSON
    log_path = os.path.join(os.path.dirname(csv_path), f"{file_id}_quantum_log.json")
    with open(log_path, "w") as f:
        json.dump(quantum_log, f, indent=4)

    return result, quantum_log, log_path
