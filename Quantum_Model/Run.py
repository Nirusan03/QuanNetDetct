# Importing the required libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import SMOTE
import pennylane as qml
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import gc

# Set random seed for reproducibility
np.random.seed(42)

# Load Dataset
file_path = "E:\\Studies\\IIT\\4 - Forth Year\\Final Year Project\\QuanNetDetct\\Datasets\\Darknet.csv"
darknet_data = pd.read_csv(file_path)

# Filter TLS-related traffic
tls_ports = [443, 993, 995, 465, 8443]
tls_traffic = darknet_data[(darknet_data['Dst Port'].isin(tls_ports)) & (darknet_data['Protocol'] == 6)]

# Encode categorical data
label_encoder = LabelEncoder()
for column in tls_traffic.select_dtypes(include=['object']).columns:
    tls_traffic[column] = label_encoder.fit_transform(tls_traffic[column])

# Select Numeric Columns
columns_to_exclude = ['Protocol', 'Dst Port', 'Label']
numeric_columns = tls_traffic.select_dtypes(include=['float64', 'int64']).columns.difference(columns_to_exclude)

# Optimize Memory
tls_traffic[numeric_columns] = tls_traffic[numeric_columns].astype(np.float32)
gc.collect()

# Handle Missing & Extreme Values
tls_traffic[numeric_columns] = tls_traffic[numeric_columns].replace([np.inf, -np.inf], np.nan)
tls_traffic[numeric_columns] = tls_traffic[numeric_columns].fillna(tls_traffic[numeric_columns].mean())
tls_traffic[numeric_columns] = tls_traffic[numeric_columns].clip(lower=tls_traffic[numeric_columns].quantile(0.01),
                                                                 upper=tls_traffic[numeric_columns].quantile(0.99), axis=1)

# Apply MinMax Scaling
tls_traffic[numeric_columns] = MinMaxScaler().fit_transform(tls_traffic[numeric_columns])

# Feature Selection Process (Newly Added)
correlation_matrix = tls_traffic.corr()
target_correlation = correlation_matrix['Label'].drop('Label')
threshold = 0.14  # Selecting features with absolute correlation > 0.15
selected_features = target_correlation[abs(target_correlation) > threshold]

# Remove identifier columns if they exist
identifiers = ['Flow ID', 'Src IP']
selected_features = selected_features.drop(index=identifiers, errors='ignore')

print("Selected Features Based on Correlation with 'Label':")
print(selected_features)

# Keep only the selected features and the Label
tls_traffic = tls_traffic[selected_features.index.tolist() + ['Label']]
if 'Timestamp' in tls_traffic.columns:
    tls_traffic = tls_traffic.drop(columns=['Timestamp'])
    print("Timestamp feature removed after feature selection!")

# Apply SMOTE to handle class imbalance
X = tls_traffic.drop('Label', axis=1)
y = tls_traffic['Label']
smote = SMOTE(sampling_strategy='auto', random_state=42)
X, y = smote.fit_resample(X, y)
tls_traffic = pd.DataFrame(X, columns=X.columns)
tls_traffic['Label'] = y
gc.collect()

# Feature Selection with PCA
pca = PCA(n_components=10)  # Select the top 10 features
X_pca = pca.fit_transform(X)
X_pca = pd.DataFrame(X_pca)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42, stratify=y)

# Split for quantum & classical inputs
num_qubits = 3
quantum_features = X_train.iloc[:, :num_qubits]  # First 3 features for Quantum
classical_features = X_train.iloc[:, num_qubits:]  # Remaining for Classical

# Define Quantum Circuit
dev = qml.device("default.qubit", wires=num_qubits)

@qml.qnode(dev, interface="tf")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(num_qubits), rotation="Y")
    qml.StronglyEntanglingLayers(weights, wires=range(num_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

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

# Define Hybrid Model
def create_hybrid_model(num_qubits, num_features, num_classes=4):
    input_q = tf.keras.layers.Input(shape=(num_qubits,))
    input_c = tf.keras.layers.Input(shape=(num_features - num_qubits,))
    
    # Quantum Path
    q_layer = QuantumLayer(num_qubits)(input_q)
    q_layer = tf.keras.layers.Dense(32, activation="relu")(q_layer)

    # Classical Path
    c_layer = tf.keras.layers.Dense(128, activation="relu")(input_c)
    c_layer = tf.keras.layers.BatchNormalization()(c_layer)
    c_layer = tf.keras.layers.Dropout(0.2)(c_layer)
    c_layer = tf.keras.layers.Dense(64, activation="relu")(c_layer)
    c_layer = tf.keras.layers.BatchNormalization()(c_layer)
    c_layer = tf.keras.layers.Dropout(0.2)(c_layer)
    
    # Fusion of Quantum & Classical Features
    combined = tf.keras.layers.concatenate([q_layer, c_layer])
    output = tf.keras.layers.Dense(num_classes, activation="softmax")(combined)

    model = tf.keras.models.Model(inputs=[input_q, input_c], outputs=output)
    return model

hybrid_model = create_hybrid_model(num_qubits, X_train.shape[1])
hybrid_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                     loss="categorical_crossentropy",
                     metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")])

# Split Training & Validation Data
X_train_q, X_val_q, X_train_c, X_val_c, y_train_split, y_val_split = train_test_split(
    quantum_features, classical_features, to_categorical(y_train, num_classes=4),
    test_size=0.2, random_state=42, stratify=y_train
)

# Define Custom Data Generator
class CustomBatchGenerator(tf.keras.utils.Sequence):
    def __init__(self, X_q, X_c, y, batch_size):
        self.X_q = X_q
        self.X_c = X_c
        self.y = y
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.y) / self.batch_size))

    def __getitem__(self, idx):
        batch_X_q = self.X_q[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_X_c = self.X_c[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return [batch_X_q, batch_X_c], batch_y

train_generator = CustomBatchGenerator(X_train_q, X_train_c, y_train_split, batch_size=32)
val_generator = CustomBatchGenerator(X_val_q, X_val_c, y_val_split, batch_size=32)

# Train the Model for 100 Epochs
history = hybrid_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=100,  # Updated epoch count
    verbose=1
)

# Predict Malicious TLS Traffic
y_pred_probs = hybrid_model.predict([X_test.iloc[:, :num_qubits], X_test.iloc[:, num_qubits:]])
y_pred = np.argmax(y_pred_probs, axis=1)

# Extract Malicious Traffic
malicious_traffic = X_test.iloc[np.where(y_pred == 0)].copy()
malicious_traffic['Predicted_Label'] = y_pred[np.where(y_pred == 0)]

# Save Malicious TLS Traffic to CSV
malicious_traffic.to_csv("Malicious_TLS_Traffic.csv", index=False)
print(f"Saved {len(malicious_traffic)} malicious TLS traffic records to 'Malicious_TLS_Traffic.csv'.")

# Plot Training and Validation Accuracy
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy', linestyle='solid')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linestyle='dashed')

# Calculate Testing Accuracy
test_accuracy = np.mean(y_pred == y_test)
plt.axhline(y=test_accuracy, color='r', linestyle='dotted', label=f'Testing Accuracy: {test_accuracy:.4f}')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training, Validation, and Testing Accuracy Over Epochs')
plt.legend()
plt.grid()
plt.show()