# QuanNetDetect

This project implements a **Hybrid Quantum-Classical Neural Network (QNN)** for detecting **malicious TLS traffic** using the **CIC-DDOS-2019**. The model is served using a Flask API, and a React frontend provides an interface for interacting with PCAP data and getting real-time predictions.

---

## Features

- **Quantum-Classical Hybrid Model** using Pennylane and TensorFlow
- **Flask API Backend** for real-time model inference and feature simulation
- **React Frontend** with clean MUI-based dashboard
- **Cluster-Based SMOTE** for balancing CIC-DDoS2019 data
- **Feature Engineering with PCA & Filtering**
- **Realistic Network Traffic Simulation** using PyShark, Scapy
- **PCAP Validation** with auto preview (IP, Port, Flags, etc.)
- **MongoDB Storage** for reports and predictions

---

## Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Nirusan03/QuanNetDetct.git
cd quannetdetect
```

### 2. Backend Setup (Flask API)
```bash
cd backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python app.py
```

API is now live at: **http://localhost:5000**

### 3. Frontend Setup (React App)
```bash
cd frontend
npm install
npm start
```

Frontend is now live at: **http://localhost:3000**

---

## API Endpoints Summary

| Method | Endpoint                  | Description                             |
|--------|---------------------------|-----------------------------------------|
| POST   | `/upload-pcap`            | Upload a PCAP and simulate features     |
| POST   | `/generate-pcap`          | Generate simulated PCAP from features   |
| POST   | `/validate-pcap`          | View packet-level preview (100 packets) |
| POST   | `/predict`                | Perform QNN inference                   |
| GET    | `/get-report/<file_id>`   | Retrieve past prediction results        |
| GET    | `/list-reports`           | List all reports from MongoDB           |
| GET    | `/visualize-upload/<id>`  | Visualize uploaded traffic              |
| GET    | `/visualize-simulated/<id>`| Visualize simulated traffic            |

---

## Tools Used

- **Pennylane** for quantum circuits
- **TensorFlow/Keras** for classical layers
- **PyShark** and **Scapy** for traffic parsing
- **MUI (Material UI)** for frontend
- **MongoDB** for prediction logging

---

## Usage Flow

1. **Upload a `.pcap`** file via UI or API
2. Select **TLS version**, **mode** (auto/custom), and record limit
3. Backend extracts features, stores file, and returns a file ID
4. Use file ID to **simulate PCAP**, **predict with model**, or **validate packets**
5. **Download results** as CSV or preview in UI
6. **Visualize traffic** via built-in bar/pie charts

---

## Project Structure

```
quannetdetect/
├── backend/
│   ├── app.py                # Flask API
│   ├── model/                # QNN weights + PCA object
│   ├── utils/                # Feature engineering + visualization helpers
│   ├── outputs/              # CSVs and simulated PCAPs
│   ├── uploads/              # Uploaded PCAPs
├── frontend/
│   ├── src/pages/            # Upload, Simulate, Reports, Visualizations
│   ├── src/components/       # Sidebar, Footer, Wrapper, Cards
│   ├── src/theme/            # MUI Custom Theme
│   ├── package.json          # React dependencies
├── Model/                    # CIC-DDoS-2019 Dataset (Processed)
├── Network_Traffic_setup/    # Network traffic set up guide
```

---

## Report Generation

Each prediction run saves:
- Prediction results (.csv)
- Flow-level classification (class + probabilities)
- Stored in MongoDB with timestamp
- Downloadable and viewable from Reports tab

---

## Author & Attribution

**Nirusan Hariharan | 20200094 | W1867405**  
© 2025 Quan Net Detect