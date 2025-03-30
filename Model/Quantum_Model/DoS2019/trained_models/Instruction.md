# Quantum-Based DDoS Simulation & Detection - Instruction Guide

This guide outlines the **complete workflow** for simulating, transforming, and classifying network traffic using a hybrid Quantum Neural Network (QNN) model. It also includes **traffic validation**, **theoretical context**, and **step-by-step execution** of all components.

---

## Overview
- **Goal**: Detect DDoS-style malicious TLS flows using a quantum-classical hybrid model.
- **Dataset**: CIC-DDoS-2019 (pre-processed with 85 features, 30 used).
- **Frameworks**: TensorFlow, PennyLane, PyShark, Scapy, Wireshark.
- **Simulation**: Real network traffic + DDoS pattern embedding.
- **Model**: Quantum Neural Network (QNN) trained on 30 selected features.

---

## Theoretical Background

### Need for DDoS Simulation
In real-world cybersecurity research, purely synthetic traffic fails to represent realistic attack patterns. Hence, we use captured benign flows and inject real DDoS-style characteristics from CIC-DDoS2019.

This enhances:
- Model generalisability
- Evaluation against diverse flows

### Quantum Neural Network Justification
QNNs combine classical deep learning layers with quantum circuits:
- **Encoding**: Amplitude Embedding maps classical features into quantum states.
- **Quantum Circuit**: Uses Hadamard gates and StronglyEntanglingLayers.
- **Measurement**: Outputs expectation values of Pauli-Z observables.

This allows the model to learn patterns difficult for classical-only architectures.

---

## Step-by-Step Instructions

### 1. Capture Real TLS Traffic with Wireshark
1. Launch **Wireshark**.
2. Select active interface (e.g., `Wi-Fi`, `Ethernet`).
3. Start packet capture.
4. Visit websites (e.g., `https://example.com`) to generate TLS traffic.
5. Capture 30–60 seconds of traffic.
6. Save as `real_traffic_test.pcap`.

---

### 2. Extract Real Flow Features
**Script**: `QNN_DDOS2019_ExtractKey.py`

- Parses `.pcap` using PyShark.
- Aggregates into 5-tuple flows (src IP, dst IP, sport, dport, protocol).
- Extracts 3 fields:
  - Flow Duration
  - Source Port
  - Total Length of Fwd Packets

**Output**: Internal DataFrame `real_df` with extracted flow statistics.

---

### 3. Simulate DDoS Flows Based on Extracted Stats
**Script**: `QNN_DDOS2019_ExtractKey.py`

- Loads preprocessed attack samples from `TLS_OneHotEncoded.csv` and `sample_all_attacks_test_data.csv`.
- Filters only **DDoS** types (Label_0, Label_1, Label_2).
- For each real flow:
  - Randomly selects a DDoS feature row.
  - Injects the real flow duration, port, and byte size.
- Removes BENIGN traffic embedding.

**Output**: `Model_Input.csv` — pure DDoS-style flows.

---

### 4. Convert Model_Input.csv to Synthetic PCAP
**Script**: `QNN_DDOS2019_ExtractNetwork_PCAP_Validation.py`

- Reads `Model_Input.csv` row by row.
- Uses Scapy to:
  - Randomise source/destination IPs.
  - Set source port from CSV.
  - Set fixed destination port 443.
  - Use Flow Duration to generate timestamps.
  - Construct TCP/IP packets with dummy payload.
- Saves `.pcap` to `SimulatedDDoSOutput.pcap`.

---

### 5. Validate PCAP Output
**Also in**: `QNN_DDOS2019_ExtractNetwork_PCAP_Validation.py`

- Reads `.pcap` file using `rdpcap()`.
- Checks packet headers, flags, and sizes.
- Displays first 100 flows.

Optional manual validation:
- Open `SimulatedDDoSOutput.pcap` in **Wireshark**.
- Filter: `tcp.port == 443`
- Confirm IP variation, payload, flags, timestamps.

---

### 6. Interpret Results Using QNN Model
**Script**: `QNN_DDOS2019_Model_Interpret.py`

- Loads trained model `QNN_DDos2019.h5`.
- Uses QuantumLayer (AmplitudeEmbedding + Entanglement).
- Reads 30 features from `Model_Input.csv`.
- Predicts softmax probabilities across:
  - DrDoS_DNS
  - DrDoS_LDAP
  - Syn
  - LDAP
  - BENIGN

Prints each sample’s prediction probabilities and final label.

---

## Directory Overview
```
QuanNetDetect/
├── trained_models/
│   ├── QNN_DDos2019.h5
│   ├── real_traffic_test.pcap
│   ├── Model_Input.csv ← (generated)
│   ├── SimulatedDDoSOutput.pcap ← (generated)
├── TLS_OneHotEncoded.csv
├── sample_all_attacks_test_data.csv
├── QNN_DDOS2019_ExtractKey.py
├── QNN_DDOS2019_ExtractNetwork_PCAP_Validation.py
├── QNN_DDOS2019_Model_Interpret.py
```

---

## Execution Order Summary
```bash
# Step 1: Use Wireshark manually to create real_traffic_test.pcap

# Step 2: Extract features
python QNN_DDOS2019_ExtractKey.py

# Step 3: Simulate attack flows (outputs Model_Input.csv)
# (Already part of QNN_DDOS2019_ExtractKey.py)

# Step 4: Convert to PCAP and validate traffic
python QNN_DDOS2019_ExtractNetwork_PCAP_Validation.py

# Step 5: Predict using Quantum Neural Network
python QNN_DDOS2019_Model_Interpret.py
```

---

## Final Notes
- `Model_Input.csv` should contain **30 model-compatible features** only.
- All labels (Label_0 to Label_4) and `Timestamp` columns are removed.
- The generated `.pcap` is fully synthetic but structurally realistic.
- Wireshark visual validation is recommended for authenticity.

---

## Acknowledgements
- Quantum Circuit Design: PennyLane
- Classical Framework: TensorFlow 2.x
- Traffic Tools: Scapy, PyShark, Wireshark
- Dataset: CIC-DDoS-2019 (balanced using Cluster-SMOTE)
- Developer: Nirusan03

