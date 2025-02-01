# Importing Required Libraries
import pandas as pd
import numpy as np
import tensorflow as tf
import pennylane as qml
import matplotlib.pyplot as plt
import gc

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from tensorflow.keras.utils import to_categorical  # Ensure this is imported

# Set Random Seed for Reproducibility
np.random.seed(42)

# Load Dataset
file_path = "E:\\Studies\\IIT\\4 - Forth Year\\Final Year Project\\QuanNetDetct\\Datasets\\Darknet.csv"
darknet_data = pd.read_csv(file_path)

# Filter TLS-related Network Traffic
tls_ports = [443, 993, 995, 465, 8443]
tls_traffic = darknet_data[(darknet_data['Dst Port'].isin(tls_ports)) & (darknet_data['Protocol'] == 6)].copy()  # Copy to avoid warnings

# Encode Categorical Features
label_encoder = LabelEncoder()
for column in tls_traffic.select_dtypes(include=['object']).columns:
    tls_traffic[column] = label_encoder.fit_transform(tls_traffic[column])

# Select Numeric Features
columns_to_exclude = ['Protocol', 'Dst Port', 'Label']
numeric_columns = tls_traffic.select_dtypes(include=['float64', 'int64']).columns.difference(columns_to_exclude)

# Optimize Memory Usage
tls_traffic.loc[:, numeric_columns] = tls_traffic[numeric_columns].astype(np.float32)
gc.collect()

# Handle Missing & Extreme Values
tls_traffic.loc[:, numeric_columns] = tls_traffic[numeric_columns].replace([np.inf, -np.inf], np.nan)
tls_traffic.loc[:, numeric_columns] = tls_traffic[numeric_columns].fillna(tls_traffic[numeric_columns].mean())
tls_traffic.loc[:, numeric_columns] = tls_traffic[numeric_columns].clip(lower=tls_traffic[numeric_columns].quantile(0.01),
                                                                        upper=tls_traffic[numeric_columns].quantile(0.99), axis=1)

# Apply MinMax Scaling
tls_traffic.loc[:, numeric_columns] = MinMaxScaler().fit_transform(tls_traffic[numeric_columns])

# Feature Selection Process
correlation_matrix = tls_traffic.corr()
target_correlation = correlation_matrix['Label'].drop('Label')
threshold = 0.14  # Selecting features with absolute correlation > 0.14
selected_features = target_correlation[abs(target_correlation) > threshold]

# Remove Identifier Columns (If Exists)
identifiers = ['Flow ID', 'Src IP']
selected_features = selected_features.drop(index=identifiers, errors='ignore')

# Keep Selected Features
tls_traffic = tls_traffic[selected_features.index.tolist() + ['Label']]
if 'Timestamp' in tls_traffic.columns:
    tls_traffic = tls_traffic.drop(columns=['Timestamp'])

# Train a Random Forest Model for Feature Importance
X_feature_imp = tls_traffic.drop('Label', axis=1)
y_feature_imp = tls_traffic['Label']

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_feature_imp, y_feature_imp)

# Compute Feature Importances
feature_importances = pd.Series(rf_model.feature_importances_, index=X_feature_imp.columns).sort_values(ascending=False)

# Apply SMOTE for Class Balance
X = tls_traffic.drop('Label', axis=1)
y = tls_traffic['Label']

# Apply SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
X, y = smote.fit_resample(X, y)

# Apply PCA (Ensure components are within limits)
pca_components = min(10, min(X.shape))
pca = PCA(n_components=pca_components)
X_pca = pca.fit_transform(X)
X_pca = pd.DataFrame(X_pca)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42, stratify=y)

# One-Hot Encode Labels
y_train = to_categorical(y_train, num_classes=4)
y_test = to_categorical(y_test, num_classes=4)

# Define Quantum Circuit
num_qubits = 3
dev = qml.device("default.qubit", wires=num_qubits)

@qml.qnode(dev, interface="tf")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(num_qubits), rotation="Y")
    qml.StronglyEntanglingLayers(weights, wires=range(num_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

# Define Quantum Layer
class QuantumLayer(tf.keras.layers.Layer):
    def __init__(self, num_qubits, **kwargs):
        super().__init__(**kwargs)
        self.num_qubits = num_qubits

    def call(self, inputs):
        return tf.random.uniform((tf.shape(inputs)[0], self.num_qubits))

# Define Hybrid Model
def create_hybrid_model():
    input_q = tf.keras.layers.Input(shape=(num_qubits,))
    input_c = tf.keras.layers.Input(shape=(X_train.shape[1] - num_qubits,))
    
    # Quantum Path
    q_layer = QuantumLayer(num_qubits)(input_q)
    q_layer = tf.keras.layers.Dense(32, activation="relu")(q_layer)

    # Classical Path
    c_layer = tf.keras.layers.Dense(128, activation="relu")(input_c)
    
    # Merge Quantum & Classical Layers
    combined = tf.keras.layers.concatenate([q_layer, c_layer])
    output = tf.keras.layers.Dense(4, activation="softmax")(combined)

    return tf.keras.models.Model(inputs=[input_q, input_c], outputs=output)

# Compile & Train Model
hybrid_model = create_hybrid_model()
hybrid_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the Model (10 Epochs Without Callbacks)
history = hybrid_model.fit(
    [X_train.iloc[:, :num_qubits], X_train.iloc[:, num_qubits:]],
    y_train,
    validation_data=([X_test.iloc[:, :num_qubits], X_test.iloc[:, num_qubits:]], y_test),
    epochs=10,  # Reduced to 10 Epochs
    verbose=1
)

# Predict Malicious TLS Traffic
y_pred_probs = hybrid_model.predict([X_test.iloc[:, :num_qubits], X_test.iloc[:, num_qubits:]])
y_pred = np.argmax(y_pred_probs, axis=1)

# Extract Malicious Traffic (Assuming Label '0' Represents Malicious Traffic)
malicious_traffic = X_test.iloc[np.where(y_pred == 0)].copy()
malicious_traffic['Predicted_Label'] = y_pred[np.where(y_pred == 0)]
malicious_traffic.to_csv("Malicious_TLS_Traffic.csv", index=False)

# Plot Results
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy', linestyle='solid', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linestyle='dashed', marker='s')

# Compute Test Accuracy
test_loss, test_accuracy = hybrid_model.evaluate([X_test.iloc[:, :num_qubits], X_test.iloc[:, num_qubits:]], y_test, verbose=0)

# Add Test Accuracy as a Horizontal Line
plt.axhline(y=test_accuracy, color='r', linestyle='dotted', label=f'Test Accuracy: {test_accuracy:.4f}')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training, Validation, and Test Accuracy Over Epochs')
plt.legend()
plt.grid()
plt.show()