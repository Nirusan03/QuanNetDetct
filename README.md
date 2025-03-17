# QuanNetDetect

This project implements a **Hybrid Quantum-Classical Neural Network (QNN)** for detecting **malicious TLS traffic** using the **Darknet 2020 dataset** and **CSE-CIC-IDS2018 dataset**. The model is served using a Flask API, and a React frontend provides an interface for inputting network traffic features and receiving predictions.

---

## **🚀 Features**

- **Hybrid Quantum-Classical Model**: Uses a combination of quantum and classical deep learning techniques.
- **Flask API Backend**: Serves the trained model for real-time inference.
- **React Frontend**: User-friendly web interface for making predictions.
- **Dataset Preprocessing & Merging**: Darknet 2020 and CSE-CIC-IDS2018 datasets merged for robust learning.
- **Cluster-Based SMOTE for Class Balancing**: Addresses class imbalance using clustering techniques.
- **PCA for Dimensionality Reduction**: Reduces feature complexity for efficient computation.
- **Network Traffic Validation**: Ensures realistic network traffic using **Wireshark, Snort, and Scapy**.
- **PCAP Generation & Validation**: Converts dataset to PCAP format and validates authenticity.
- **NPCAP for Packet Capture**: Utilizes NPCAP for real-time packet analysis.

---

## **🛠️ Installation and Setup**

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/your-username/qnn-tls-detection.git
cd qnn-tls-detection
```

### **2️⃣ Set Up the Backend (Flask API)**
#### **📌 Prerequisites**
- Python 3.10
- Virtual Environment (Recommended)

#### **📌 Install Dependencies**
```bash
cd backend
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

#### **📌 Start the Flask API**
```bash
python app.py
```

🚀 The API will be running at: **http://127.0.0.1:5000**

---

### **3️⃣ Set Up the Frontend (React App)**
#### **📌 Prerequisites**
- Node.js (v16+)
- npm or yarn

#### **📌 Install Dependencies**
```bash
cd frontend
npm install  # or yarn install
```

#### **📌 Start the React Application**
```bash
npm start  # or yarn start
```

🚀 The frontend will be running at: **http://localhost:3000**

---

### **4️⃣ Set Up Wireshark and Snort**
#### **📌 Install Wireshark**
- Download and install from [Wireshark](https://www.wireshark.org/).
- Ensure **NPCAP** is installed for packet capture.

#### **📌 Install Snort**
- Download and install Snort from [Snort Official Site](https://www.snort.org/).
- Configure Snort rules and set up `snort.conf`.
- Test configuration using:
```bash
snort -T -c C:\Snort\etc\snort.conf
```
- Run Snort in live traffic mode:
```bash
snort -i <interface_number> -c C:\Snort\etc\snort.conf -l C:\Snort\log
```

#### **📌 Filtering Traffic in Wireshark**
Use the following filters to analyze traffic:
```bash
tcp.flags.syn==1 and tcp.flags.ack==1  # Filter TCP handshake packets
tcp.flags.ack==1  # Filter TCP ACK packets
udp  # Show only UDP traffic
http  # Show only HTTP traffic
dns  # Show DNS queries and responses
tls  # Display TLS-encrypted traffic
```

---

## **🔧 API Endpoints**

| Method | Endpoint        | Description                                      |
|--------|----------------|--------------------------------------------------|
| `POST` | `/predict`     | Predicts if TLS traffic is malicious or benign |
| `GET`  | `/health`      | Checks if the API is running                    |
| `POST` | `/upload_pcap` | Uploads and validates PCAP files                |

### **Example Request (cURL)**
```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [1.2, 3.5, 0.7, 5.1, 2.9]}'
```

### **Example Response**
```json
{
  "prediction": "Malicious",
  "probabilities": [[0.85, 0.10, 0.05]]
}
```

---

## **🧪 Running Tests**

### **Backend Tests**
```bash
cd backend
pytest
```

### **Frontend Tests**
```bash
cd frontend
npm test
```

---

## **📂 Project Structure**
```
qnn-tls-detection/
│── backend/               # Flask API
│   ├── app.py             # Main API script
│   ├── model/             # Trained models and preprocessors
│   ├── pcap_processing/   # PCAP validation and traffic analysis
│   ├── requirements.txt   # Backend dependencies
│── frontend/              # React Frontend
│   ├── src/               # React components
│   ├── package.json       # Frontend dependencies
│── data/                  # Preprocessed datasets
│── PCAP/                  # Generated PCAP files
│── README.md              # Project documentation
```

---

## **📊 PCAP Validation & Traffic Analysis**

The project includes automated validation of network traffic using:
- **Wireshark**: Packet inspection and traffic validation.
- **Snort**: Intrusion detection and anomaly detection.
- **Scapy**: Scripted packet analysis and validation.
- **NPCAP**: Real-time packet capture.

### **✅ Network Traffic Validation Includes:**
- Bidirectional Traffic Confirmation
- Protocol Distribution (TCP, UDP, ICMP)
- Valid Source & Destination IPs
- Presence of Common Protocols (HTTP, DNS, TLS, etc.)
- PCAP File Analysis and Automated Reports

🚀 **QuanNetDetect ensures realistic network traffic validation for accurate threat detection!**
