import os
import warnings
import logging
import pandas as pd
import numpy as np


# Suppress ALL logs before importing TensorFlow 
# Only show ERRORs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Force disable GPU & CUDA logs
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Suppress all Python & TensorFlow warnings
warnings.filterwarnings("ignore")

# Suppress TensorFlow logging
tf_logger = logging.getLogger('tensorflow')
tf_logger.setLevel(logging.ERROR)

# Suppress PennyLane logging
qml_logger = logging.getLogger("pennylane")
qml_logger.setLevel(logging.ERROR)

# Now import the actual libraries
import tensorflow as tf
import pennylane as qml

# Suppress Autograph logs
tf.autograph.set_verbosity(0)

# Model Configuration
class ModelConfig:
    def __init__(self):
        self.quantum_feature_count = 30
        self.num_classes = 5 
        self.num_qubits = 5
        self.embedding_dim = 2 ** self.num_qubits
        self.num_layers = 4

config = ModelConfig()

# Quantum Circuit
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

# Custom Quantum Layer
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

        quantum_output = tf.map_fn(
            process_sample,
            padded_inputs,
            fn_output_signature=tf.TensorSpec(shape=(config.num_qubits,), dtype=tf.float32)
        )
        return quantum_output

    def get_config(self):
        return {'num_qubits': self.num_qubits, 'num_layers': self.num_layers}

# Load Trained Model
model_path = r"e:\\Studies\\IIT\\4 - Forth Year\\Final Year Project\\QuanNetDetct\\Model\\Quantum_Model\DoS2019\\trained_models\\QNN_DDos2019.h5"

model = tf.keras.models.load_model(
    model_path,
    custom_objects={"QuantumLayer": QuantumLayer},
    compile=False
)

# Model Summary and Info
model.summary()

print("\nModel Inputs:")
for i, inp in enumerate(model.inputs):
    print(f"Input {i+1}: Name = {inp.name}, Shape = {inp.shape}, Dtype = {inp.dtype}")

print("\nModel Output:")
print(f"Name = {model.output.name}, Shape = {model.output.shape}, Dtype = {model.output.dtype}")

# Load the test CSV
test_df = pd.read_csv(r"e:\Studies\IIT\4 - Forth Year\Final Year Project\QuanNetDetct\Model\Quantum_Model\DoS2019\trained_models\Model_Input_1.csv")

# Convert to NumPy
X_test = test_df.values.astype(np.float32)

# Run prediction
predictions = model.predict([X_test[:, :30], X_test[:, :30]])
predicted_indices = np.argmax(predictions, axis=1)

# Correct Class Label Mapping 
class_names = ['DrDoS_DNS', 'DrDoS_LDAP', 'Syn', 'LDAP', 'BENIGN']
predicted_labels = [class_names[i] for i in predicted_indices]

# Display the Results 
for i, (probs, label) in enumerate(zip(predictions, predicted_labels)):
    print(f"\nSample {i+1}")
    for j, prob in enumerate(probs):
        print(f"   {class_names[j]:<12}: {prob:.4f}")
    print(f"Predicted Class: {label}")