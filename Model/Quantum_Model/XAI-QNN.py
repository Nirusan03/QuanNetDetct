import shap
import lime
import lime.lime_tabular
from tensorflow.keras.models import load_model
import pennylane as qml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns

def explain_model_xai(hybrid_model, X_sample, y_sample, feature_names):
    """
    Apply SHAP, LIME, and Quantum Interpretability to the hybrid model.
    """
    # Generate SHAP Explainers for Feature Importance
    explainer = shap.Explainer(hybrid_model, X_sample)
    shap_values = explainer(X_sample)
    
    # Plot SHAP Feature Importance
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names)
    
    # Apply LIME for Local Explanation
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_sample),
        feature_names=feature_names,
        class_names=['Malicious', 'Non-Malicious', 'Uncertain'],
        mode='classification'
    )
    
    # Select a single instance for LIME explanation
    idx = np.random.randint(0, len(X_sample))
    lime_exp = lime_explainer.explain_instance(
        X_sample.iloc[idx].values, hybrid_model.predict, num_features=5
    )
    
    # Show LIME Explanation
    lime_exp.show_in_notebook()
    
    # Quantum Feature Interpretability
    sample_input = tf.convert_to_tensor(X_sample.iloc[idx, :num_qubits], dtype=tf.float32)
    sample_weights = tf.random.normal(shape=(1, num_qubits))
    expectation_values = quantum_circuit(sample_input, sample_weights)
    
    # Plot Quantum Expectation Values
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(expectation_values)), expectation_values, color="blue", alpha=0.7)
    plt.xlabel("Qubit")
    plt.ylabel("Pauli-Z Expectation Value")
    plt.title("Quantum Circuit Measurement Outputs")
    plt.xticks(range(len(expectation_values)), [f"Qubit {i+1}" for i in range(len(expectation_values))])
    plt.grid()
    plt.show()

# Load the trained hybrid model
loaded_hybrid_model = load_model("E:\\Studies\\IIT\\4 - Forth Year\\Final Year Project\\QuanNetDetct\\Model\\hybrid_qnn_model.h5", custom_objects={"QuantumLayer": QuantumLayer})

# Apply XAI techniques on test data
explain_model_xai(loaded_hybrid_model, X_test, y_test, feature_names=X_test.columns)
