# utils/qnn_inference.py

import os
import warnings
import logging
import numpy as np
import pandas as pd

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

# Attack Labels
class_names = ['DrDoS_DNS', 'DrDoS_LDAP', 'Syn', 'LDAP', 'BENIGN']

def run_qnn_prediction(csv_path):
    print(f"[+] Running QNN inference on: {csv_path}")

    df = pd.read_csv(csv_path)
    X_test = df.values.astype(np.float32)

    # Quantum layer expects inputs twice (due to model structure)
    predictions = model.predict([X_test[:, :30], X_test[:, :30]])

    predicted_indices = np.argmax(predictions, axis=1)
    predicted_labels = [class_names[i] for i in predicted_indices]

    result = []
    for i, (probs, label) in enumerate(zip(predictions, predicted_labels)):
        record = {"id": i + 1, "predicted_class": label}
        for j, prob in enumerate(probs):
            record[class_names[j]] = float(round(prob, 4))
        result.append(record)

    print(f"[+] Prediction complete: {len(result)} flows analyzed.")
    return result
