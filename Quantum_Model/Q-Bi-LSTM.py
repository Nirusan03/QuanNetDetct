import tensorflow as tf
import pennylane as qml
import numpy as np
import pandas as pd
import gc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load Dataset
dataset_path = "E:\\Studies\\IIT\\4 - Forth Year\\Final Year Project\\QuanNetDetct\\Datasets\\Darknet.csv"
darknet_data = pd.read_csv(dataset_path)

# Extract TLS-related network traffic
tls_ports = [443, 993, 995, 465, 8443]
tls_traffic = darknet_data[(darknet_data['Dst Port'].isin(tls_ports)) & (darknet_data['Protocol'] == 6)]

# Drop identifier columns to prevent data leakage
tls_traffic.drop(columns=['Flow ID', 'Src IP', 'Dst IP', 'Timestamp'], errors='ignore', inplace=True)

# Encode categorical data
label_encoder = LabelEncoder()
for col in tls_traffic.select_dtypes(include=['object']).columns:
    tls_traffic[col] = label_encoder.fit_transform(tls_traffic[col])

# Replace infinite values with NaN and fill missing values
tls_traffic.replace([np.inf, -np.inf], np.nan, inplace=True)
tls_traffic.fillna(tls_traffic.mean(), inplace=True)

# Feature Selection
correlation_matrix = tls_traffic.corr()
target_correlation = correlation_matrix['Label'].drop('Label')
threshold = 0.14
selected_features = target_correlation[abs(target_correlation) > threshold]
selected_features = selected_features.drop(index=['Flow ID', 'Src IP'], errors='ignore')

# Keep only selected features
tls_traffic = tls_traffic[selected_features.index.tolist() + ['Label']]

# Free up memory
gc.collect()

# Split dataset
X = tls_traffic.drop(columns=['Label'])
y = tls_traffic['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Scale data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define Quantum Circuit
num_qubits = 4
dev = qml.device("default.qubit", wires=num_qubits)

@qml.qnode(dev, interface="tf")
def quantum_circuit(inputs, weights):
    inputs = tf.reshape(inputs, [-1])  # Ensure flattened input
    qml.AngleEmbedding(inputs, wires=range(num_qubits), rotation="Y")
    qml.StronglyEntanglingLayers(weights, wires=range(num_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

# Quantum Layer using TensorFlow operations
class QuantumLayer(tf.keras.layers.Layer):
    def __init__(self, num_qubits, **kwargs):
        super(QuantumLayer, self).__init__(**kwargs)
        self.num_qubits = num_qubits
        self.q_weights = self.add_weight(
            name="q_weights",
            shape=(1, num_qubits),
            initializer="glorot_uniform",
            trainable=True
        )

    def call(self, inputs):
        inputs = tf.reshape(inputs, [-1, self.num_qubits])  # Ensure batch input shape
        return tf.convert_to_tensor(quantum_circuit(inputs, self.q_weights), dtype=tf.float32)

# Define Hybrid Quantum Bi-LSTM Model
def create_hybrid_bilstm_model(num_qubits, num_features, num_classes=4):
    input_q = tf.keras.layers.Input(shape=(num_qubits,))
    q_layer = QuantumLayer(num_qubits)(input_q)
    q_layer = tf.keras.layers.Dense(32, activation="relu")(q_layer)
    q_layer = tf.keras.layers.BatchNormalization()(q_layer)
    q_layer = tf.keras.layers.Dropout(0.3)(q_layer)

    input_c = tf.keras.layers.Input(shape=(num_features - num_qubits, 1))
    c_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(input_c)
    c_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(c_layer)
    c_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))(c_layer)
    c_layer = tf.keras.layers.Dense(128, activation="relu")(c_layer)
    c_layer = tf.keras.layers.Dropout(0.4)(c_layer)

    combined = tf.keras.layers.concatenate([q_layer, c_layer])
    combined = tf.keras.layers.Dense(64, activation="relu")(combined)
    combined = tf.keras.layers.Dropout(0.3)(combined)
    output = tf.keras.layers.Dense(num_classes, activation="softmax")(combined)

    model = tf.keras.models.Model(inputs=[input_q, input_c], outputs=output)
    return model

# Reshape Data for Bi-LSTM Input
X_train_q = X_train[:, :num_qubits]  # Ensure only num_qubits features
X_test_q = X_test[:, :num_qubits]

X_train_c = X_train[:, num_qubits:].reshape(X_train.shape[0], X_train.shape[1] - num_qubits, 1)
X_test_c = X_test[:, num_qubits:].reshape(X_test.shape[0], X_test.shape[1] - num_qubits, 1)

# Convert Labels to Categorical
y_train = tf.keras.utils.to_categorical(y_train, num_classes=4)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=4)

# Create and Compile Model
hybrid_bilstm_model = create_hybrid_bilstm_model(num_qubits, X_train.shape[1])

hybrid_bilstm_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="categorical_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
)

# Train the Model
history = hybrid_bilstm_model.fit(
    [X_train_q, X_train_c], y_train,
    validation_data=([X_test_q, X_test_c], y_test),
    epochs=100,
    batch_size=64,
    verbose=1
)

# Plot Results
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy Over Epochs')
plt.legend()
plt.grid()
plt.show()

# Predict Malicious TLS Traffic
y_pred_probs = hybrid_bilstm_model.predict([X_test_q, X_test_c])
y_pred = np.argmax(y_pred_probs, axis=1)

# Extract Malicious Traffic
malicious_traffic = X_test[y_pred == 0].copy()
malicious_traffic['Predicted_Label'] = y_pred[y_pred == 0]
malicious_traffic.to_csv("Malicious_TLS_Traffic_BiLSTM.csv", index=False)
print(f"Saved {len(malicious_traffic)} malicious TLS traffic records to 'Malicious_TLS_Traffic_BiLSTM.csv'.")
