# Importing the required libraries
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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
import seaborn as sns
import gc

# Set random seed for reproducibility
np.random.seed(42)

# Load Dataset
file_path = "E:\\Studies\\IIT\\4 - Forth Year\\Final Year Project\\QuanNetDetct\\Model\\Datasets\\Darknet.csv"
darknet_data = pd.read_csv(file_path)

# Filter TLS-related traffic
tls_ports = [443, 993, 995, 465, 8443]
tls_traffic = darknet_data[(darknet_data['Dst Port'].isin(tls_ports)) & (darknet_data['Protocol'] == 6)]

# Encode categorical data
# Ensure tls_traffic is a separate DataFrame to avoid SettingWithCopyWarning
tls_traffic = tls_traffic.copy()

# Encode categorical data safely
label_encoder = LabelEncoder()
for column in tls_traffic.select_dtypes(include=['object']).columns:
    tls_traffic[column] = label_encoder.fit_transform(tls_traffic[column])

# Select Numeric Columns
# Ensure tls_traffic is a separate DataFrame
tls_traffic = tls_traffic.copy()

# Select Numeric Columns
columns_to_exclude = ['Protocol', 'Dst Port', 'Label']
numeric_columns = tls_traffic.select_dtypes(include=['float64', 'int64']).columns.difference(columns_to_exclude)

# Optimize Memory
tls_traffic[numeric_columns] = tls_traffic[numeric_columns].astype(np.float32)
gc.collect()

# Handle Missing & Extreme Values
# Ensure tls_traffic is a separate DataFrame to avoid modifying a view
tls_traffic = tls_traffic.copy()

# Handle Missing & Extreme Values
tls_traffic[numeric_columns] = tls_traffic[numeric_columns].replace([np.inf, -np.inf], np.nan)
tls_traffic[numeric_columns] = tls_traffic[numeric_columns].fillna(tls_traffic[numeric_columns].mean())
tls_traffic[numeric_columns] = tls_traffic[numeric_columns].clip(
    lower=tls_traffic[numeric_columns].quantile(0.01),
    upper=tls_traffic[numeric_columns].quantile(0.99),
    axis=1
)

# Apply MinMax Scaling
tls_traffic[numeric_columns] = MinMaxScaler().fit_transform(tls_traffic[numeric_columns])

# Feature Selection Process (Newly Added)
correlation_matrix = tls_traffic.corr(numeric_only=True)
target_correlation = correlation_matrix['Label'].drop('Label')
threshold = 0.14  # Selecting features with absolute correlation > 0.14
selected_features = target_correlation[abs(target_correlation) > threshold]

# Remove identifier columns if they exist
identifiers = ['Flow ID', 'Src IP']
selected_features = selected_features.drop(index=identifiers, errors='ignore')

# Keep only the selected features and the Label
tls_traffic = tls_traffic[selected_features.index.tolist() + ['Label']]
if 'Timestamp' in tls_traffic.columns:
    tls_traffic = tls_traffic.drop(columns=['Timestamp'])
    print("Timestamp feature removed after feature selection!")

print("Selected Features Based on Correlation with 'Label':")
print(selected_features)

# Generate correlation heatmap for the selected features
plt.figure(figsize=(12, 8))
sns.heatmap(tls_traffic.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Selected Features")
plt.show()


# Apply SMOTE to handle class imbalance
X = tls_traffic.drop('Label', axis=1)
y = tls_traffic['Label']
smote = SMOTE(sampling_strategy='auto', random_state=42)
X, y = smote.fit_resample(X, y)
tls_traffic = pd.DataFrame(X, columns=X.columns)
tls_traffic['Label'] = y
gc.collect()

# One-Hot Encode the 'Label' column
ohe = OneHotEncoder(sparse_output=False)  # Use dense output instead of sparse matrix
y_encoded = ohe.fit_transform(tls_traffic[['Label']])  # Encode the label column

# Convert to DataFrame with proper column names
ohe_columns = [f"Class_{i}" for i in range(y_encoded.shape[1])]
y_encoded_df = pd.DataFrame(y_encoded, columns=ohe_columns)

# Keep original 'Label' column while adding one-hot encoded versions
tls_traffic = pd.concat([tls_traffic, y_encoded_df], axis=1)

# Apply PCA
pca = PCA(n_components=10)  # Select the top 10 principal components
X_pca = pca.fit_transform(X)  # Fit PCA on the dataset

# Convert to DataFrame with proper column names
X_pca = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(10)])

# Generate the explained variance ratio plot
plt.figure(figsize=(10, 6))
plt.bar(range(1, 11), pca.explained_variance_ratio_, alpha=0.7, color="blue")
plt.xticks(range(1, 11), [f"PC{i}" for i in range(1, 11)])
plt.ylabel("Explained Variance Ratio")
plt.xlabel("Principal Component")
plt.title("Explained Variance Ratio of Top 10 Principal Components")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42, stratify=y)

# Split for quantum & classical inputs
num_qubits = 3

# First 3 features for Quantum
quantum_features = X_train.iloc[:, :num_qubits]  

# Remaining for Classical features alone
classical_features = X_train.iloc[:, num_qubits:]  

# Define Quantum Circuit
dev = qml.device("default.qubit", wires=num_qubits)

@qml.qnode(dev, interface="tf")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(num_qubits), rotation="Y")
    qml.StronglyEntanglingLayers(weights, wires=range(num_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

# Run the circuit to get expectation values
expectation_values = quantum_circuit(sample_input, sample_weights)

# Plot expectation values
plt.figure(figsize=(8, 5))
plt.bar(range(len(expectation_values)), expectation_values, color="blue", alpha=0.7)
plt.xlabel("Qubit")
plt.ylabel("Pauli-Z Expectation Value")
plt.title("Qubit Measurement Outputs")
plt.xticks(range(len(expectation_values)), [f"Qubit {i+1}" for i in range(len(expectation_values))])
plt.grid()
plt.show()

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
    # Reduce neurons & increase dropout
    c_layer = tf.keras.layers.Dense(64, activation="relu")(input_c)
    c_layer = tf.keras.layers.BatchNormalization()(c_layer)
    c_layer = tf.keras.layers.Dropout(0.3)(c_layer)
    c_layer = tf.keras.layers.Dense(32, activation="relu")(c_layer)
    c_layer = tf.keras.layers.BatchNormalization()(c_layer)
    c_layer = tf.keras.layers.Dropout(0.3)(c_layer)

    
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

# Train the Model
# Train the Model
history = hybrid_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
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

# Predict labels for the test set
y_pred = hybrid_model.predict([X_test.iloc[:, :num_qubits], X_test.iloc[:, num_qubits:]])
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels
y_true_classes = np.array(y_test)  # True labels

# Compute confusion matrix
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)

# Display confusion matrix
disp = ConfusionMatrixDisplay(conf_matrix)
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix")
plt.show()

# Shuffle the dataset before selecting a random sample
tls_traffic = tls_traffic.sample(frac=1).reset_index(drop=True)  # Ensures different selections each run

# Identify label columns (Both One-Hot Encoded and Original)
label_columns = [col for col in tls_traffic.columns if "Class_" in col]  # OHE labels
original_label_column = "Label" if "Label" in tls_traffic.columns else None

# Load a Truly Random TLS Traffic Sample (No Fixed Random State)
random_sample = tls_traffic.sample(1)

# Extract the true label
true_label = random_sample[original_label_column].values[0]  # Keep the original label

# Drop One-Hot Encoded label columns but KEEP original Label
random_sample = random_sample.drop(columns=label_columns)  

# Ensure Feature Alignment - Reorder Columns as in Training
random_sample = random_sample.reindex(columns=X.columns, fill_value=0)  # Ensures all features match PCA input

# Apply the Same Preprocessing as Training Data
scaler = MinMaxScaler().fit(X)  # Fit scaler on full training data
random_sample_scaled = scaler.transform(random_sample)  # Normalize

# Apply PCA
random_sample_pca = pca.transform(random_sample_scaled)  # Apply PCA

# Split into Quantum & Classical Inputs
random_sample_q = random_sample_pca[:, :num_qubits]  # First 3 PCA components
random_sample_c = random_sample_pca[:, num_qubits:]  # Remaining components

# Reshape for Prediction
random_sample_q = random_sample_q.reshape(1, -1)
random_sample_c = random_sample_c.reshape(1, -1)

# Display the Random TLS Traffic Sample Before Prediction
print("\n===== Random TLS Traffic Sample (Before Processing) =====")
display(pd.DataFrame(random_sample))  # Display the sample in a readable table

# Display the Scaled Data
print("\n===== Random TLS Traffic Sample (After Scaling) =====")
display(pd.DataFrame(random_sample_scaled, columns=X.columns))

# Display the PCA Transformed Data
print("\n===== Random TLS Traffic Sample (After PCA) =====")
display(pd.DataFrame(random_sample_pca, columns=[f"PCA_{i+1}" for i in range(random_sample_pca.shape[1])]))

# Display Quantum and Classical Features Separately
print("\n===== Quantum Features Passed to Model =====")
display(pd.DataFrame(random_sample_q, columns=[f"Q_{i+1}" for i in range(num_qubits)]))

print("\n===== Classical Features Passed to Model =====")
display(pd.DataFrame(random_sample_c, columns=[f"C_{i+1}" for i in range(random_sample_c.shape[1])]))

# Predict Using the Model
pred_prob = hybrid_model.predict([random_sample_q, random_sample_c])
pred_label = np.argmax(pred_prob, axis=1)[0]

# Define Labels (Assuming 0 = Malicious, 1 = Non-Malicious, 2 & 3 = Uncertain)
labels_dict = {0: "Malicious", 1: "Non-Malicious", 2: "Uncertain"}

# Display Prediction
print("\n===== Random TLS Traffic Prediction =====")
print(f"Predicted Label: {labels_dict[pred_label]}")
print(f"True Label: {'Malicious' if true_label == 0 else 'Non-Malicious'}")