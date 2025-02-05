# QuanNetDetect

This project implements a **Hybrid Quantum-Classical Neural Network (QNN)** for detecting **malicious TLS traffic** using the **Darknet 2020 dataset**. The model is served using a Flask API, and a React frontend provides an interface for inputting network traffic features and receiving predictions.

---

## **🚀 Features**

- **Hybrid Quantum-Classical Model**: Uses a combination of quantum and classical deep learning techniques.
- **Flask API Backend**: Serves the trained model for real-time inference.
- **React Frontend**: User-friendly web interface for making predictions.
- **SMOTE for Class Balancing**: Handles class imbalance in the dataset.
- **PCA for Dimensionality Reduction**: Reduces feature complexity for efficient computation.

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

## **🔧 API Endpoints**

| Method | Endpoint        | Description                                      |
|--------|----------------|--------------------------------------------------|
| `POST` | `/predict`     | Predicts if TLS traffic is malicious or benign |
| `GET`  | `/health`      | Checks if the API is running                    |

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
│   ├── requirements.txt   # Backend dependencies
│── frontend/              # React Frontend
│   ├── src/               # React components
│   ├── package.json       # Frontend dependencies
│── README.md              # Project documentation
```