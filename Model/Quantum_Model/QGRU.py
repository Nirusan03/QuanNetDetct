import pennylane as qml
import tensorflow as tf
import numpy as np
import pandas as pd
import gc
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from tensorflow.keras.utils import to_categorical

# Set random seed
np.random.seed(42)

# Load the TLS network traffic dataset
dataset_path = "E:\\Studies\\IIT\\4 - Forth Year\\Final Year Project\\QuanNetDetct\\Model\\Datasets\\Darknet.csv"
darknet_data = pd.read_csv(dataset_path)

# Filter only TLS traffic
tls_ports = [443, 993, 995, 465, 8443]
tls_traffic = darknet_data[(darknet_data['Dst Port'].isin(tls_ports)) & (darknet_data['Protocol'] == 6)].copy()

# Encode categorical columns
label_encoder = LabelEncoder()
for col in tls_traffic.select_dtypes(include=['object']).columns:
    tls_traffic[col] = label_encoder.fit_transform(tls_traffic[col])

# Select numeric columns and scale
tls_traffic = tls_traffic.drop(columns=['Flow ID', 'Src IP', 'Dst IP', 'Timestamp'], errors='ignore')
numeric_columns = tls_traffic.select_dtypes(include=['float64', 'int64']).columns

# Handle Missing & Extreme Values
tls_traffic[numeric_columns] = tls_traffic[numeric_columns].replace([np.inf, -np.inf], np.nan)
tls_traffic[numeric_columns] = tls_traffic[numeric_columns].fillna(tls_traffic[numeric_columns].mean())

# Clip extreme values (1st and 99th percentiles)
tls_traffic[numeric_columns] = tls_traffic[numeric_columns].clip(
    lower=tls_traffic[numeric_columns].quantile(0.01),
    upper=tls_traffic[numeric_columns].quantile(0.99),
    axis=1
)

# Apply MinMax Scaling
tls_traffic[numeric_columns] = MinMaxScaler().fit_transform(tls_traffic[numeric_columns])

gc.collect()

# Apply SMOTE to handle class imbalance
X = tls_traffic.drop('Label', axis=1)
y = tls_traffic['Label']
smote = SMOTE(sampling_strategy='auto', random_state=42)
X, y = smote.fit_resample(X, y)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42, stratify=y)

tf.keras.backend.clear_session()

# Define Quantum Circuit
dev = qml.device("default.qubit", wires=3)

@qml.qnode(dev, interface="tf")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(3), rotation="Y")
    qml.StronglyEntanglingLayers(weights, wires=range(3))
    return [qml.expval(qml.PauliZ(i)) for i in range(3)]

# Define Quantum Layer
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

def create_hybrid_gru_model(num_features, num_classes=4):
    input_layer = tf.keras.layers.Input(shape=(num_features, 1))
    
    # Quantum Layer
    quantum_layer = QuantumLayer(3)(input_layer)
    quantum_layer = tf.keras.layers.Dense(32, activation="relu")(quantum_layer)
    
    # GRU Layer
    gru_layer = tf.keras.layers.GRU(64, return_sequences=True, dropout=0.3)(input_layer)
    gru_layer = tf.keras.layers.BatchNormalization()(gru_layer)
    gru_layer = tf.keras.layers.GRU(32, return_sequences=False, dropout=0.3)(gru_layer)
    gru_layer = tf.keras.layers.BatchNormalization()(gru_layer)
    
    # Combine Quantum and Classical Features
    combined = tf.keras.layers.concatenate([quantum_layer, gru_layer])
    
    # Fully Connected Layers
    dense_layer = tf.keras.layers.Dense(64, activation='relu')(combined)
    dense_layer = tf.keras.layers.Dropout(0.3)(dense_layer)
    dense_layer = tf.keras.layers.Dense(32, activation='relu')(dense_layer)
    
    # Output Layer
    output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')(dense_layer)
    
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
    return model

# Reshape input for GRU
X_train_reshaped = np.expand_dims(X_train, axis=-1)
X_test_reshaped = np.expand_dims(X_test, axis=-1)

# Convert labels to categorical format
y_train_categorical = to_categorical(y_train, num_classes=4)
y_test_categorical = to_categorical(y_test, num_classes=4)

# Create and compile the model
gru_model = create_hybrid_gru_model(X_train.shape[1])
gru_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss="categorical_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")])

# Train the Model
history = gru_model.fit(
    X_train_reshaped, y_train_categorical,
    validation_data=(X_test_reshaped, y_test_categorical),
    epochs=10,
    batch_size=32,
    verbose=1
)

# Save the trained model
gru_model.save("E:\\Studies\\IIT\\4 - Forth Year\\Final Year Project\\QuanNetDetct\\Model\\hybrid_gru_model.h5")
print("Hybrid Quantum GRU Model saved successfully.")
