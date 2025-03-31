# QuanNetDetect: Backend API Guide (Flask)

This markdown file provides a complete guide to testing the QuanNetDetect backend API using Postman or any HTTP client. Each endpoint is explained with its purpose, method, expected input, and expected output. Follow the steps in the given order for a complete flow.

---

## **API Execution Order**

1. **`/upload-pcap`**  
2. **`/generate-pcap`**  
3. **`/validate-pcap`** (Optional)  
4. **`/predict`**  
5. **`/download-report/<file_id>`**  
6. **`/get-report/<file_id>`**  
7. **`/list-reports`**

---

## 1. **Upload PCAP and Simulate Features**

### **Endpoint**: `POST /upload-pcap`

### **Description**:

### URL : http://localhost:5000/upload-pcap
Uploads a `.pcap` file and extracts flow-level features. You can simulate attacks using either predefined DDoS vectors or custom user-defined feature values.

### **Body Type (Customer values)**: `form-data`
- `file`: (type = File) Your `.pcap` file
- `metadata`: (type = Text)
```json
{
  "tls_version": "3",     // 1: TLSv1.2, 2: TLSv1.3, 3: Both
  "mode": "custom",       // "auto" or "custom"
  "record_limit": 100,     // Optional: Limit flows
  "custom_features": {    // Required if mode = custom
    "Flow Duration": 0.1,
    "Source Port": 443,
    "Total Length of Fwd Packets": 1200
  }
}
```

### **Body Type (Automate value)**: `form-data`
- `file`: (type = File) Your `.pcap` file
- `metadata`: (type = Text)
```json
{
  "tls_version": "1", // 1: TLSv1.2, 2: TLSv1.3, 3: Both
  "mode": "auto",
  "record_limit": 100
}
```

### **Returns**:
```json
{
  "file_id": "<unique_id>",
  "message": "Upload and simulation complete"
}
```

### **Output Files**:
- CSV: `<file_id>_Model_Input.csv` (inside `outputs/`)
- PCAP: original upload saved in `uploads/`

---

## 2. **Generate Simulated PCAP File**

### **Endpoint**: `POST /generate-pcap`
### URL : http://localhost:5000/generate-pcap

### **Description**:
Generates synthetic `.pcap` traffic based on the simulated attack features.

### **Payload**:
```json
{
  "file_id": "<file_id>"
}
```

### **Returns**:
```json
{
  "message": "X packets written to PCAP",
  "pcap_path": "outputs/<file_id>_Simulated.pcap"
}
```

### **Output Files**:
- PCAP: `<file_id>_Simulated.pcap`

---

## 3. **Validate Generated PCAP (Optional)**

### **Endpoint**: `POST /validate-pcap`
### URL : http://localhost:5000/validate-pcap
### **Description**:
Displays the first 100 packets of the generated `.pcap` for quick inspection (IP, ports, flags, etc).

### **Payload**:
```json
{
  "file_id": "<file_id>"
}
```

### **Returns**:
```json
{
  "packets": [
    {
      "index": 1,
      "src_ip": "192.168.X.X",
      "dst_ip": "10.0.X.X",
      "sport": 443,
      "dport": 443,
      "flags": "S",
      "size": 1440
    },
    ...
  ]
}
```

---

## 4. **Run QNN Model Inference**

### **Endpoint**: `POST /predict`
### URL : http://localhost:5000/predict
### **Description**:
Performs inference on the simulated features using the trained QNN model and returns prediction results.

### **Payload**:
```json
{
  "file_id": "<file_id>"
}
```

### **Returns**:
```json
{
  "predictions": [
    {
      "id": 1,
      "predicted_class": "LDAP",
      "DrDoS_DNS": 0.01,
      "DrDoS_LDAP": 0.02,
      "Syn": 0.03,
      "LDAP": 0.88,
      "BENIGN": 0.06
    },
    ...
  ],
  "report_path": "outputs/<file_id>_report.csv"
}
```

### **Stored in DB**:
### **MongoDB** stores predictions with timestamp

---

## 5. **Download Detection Report**
### URL : http://localhost:5000/download-report/<file_id>

### **Description**:
Downloads the full `.csv` report containing prediction results.

### **Returns**:
- Content-Type: `text/csv`
- File: `outputs/<file_id>_report.csv`

---

## 6. **Get Prediction Details from DB**

### **Endpoint**: `GET /get-report/<file_id>`
### URL : http://localhost:5000/get-report/<file_id>
### **Description**:
Fetches prediction results from MongoDB based on `file_id`.

### **Returns**:
```json
{
  "file_id": "<file_id>",
  "created_at": "2025-03-31T12:00:00Z",
  "predictions": [... same as /predict ...]
}
```

---

## 7. **List All Scanned Traffic Reports**

### **Endpoint**: `GET /list-reports`
### URL : http://localhost:5000/list-reports
### **Description**:
Returns a list of all past scan reports stored in MongoDB.

### **Returns**:
```json
[
  {
    "file_id": "abc123",
    "created_at": "2025-03-30T18:30:00Z"
  },
  {
    "file_id": "def456",
    "created_at": "2025-03-29T11:05:00Z"
  }
]
```

Can be used to populate a dropdown/history page in the frontend.

---

## Summary of Outputs by Endpoint

| Endpoint              | Output Type | Output Description                            |
|-----------------------|-------------|-----------------------------------------------|
| `/upload-pcap`        | `.csv`      | Simulated features                            |
| `/generate-pcap`      | `.pcap`     | Synthetic PCAP                                |
| `/validate-pcap`      | `JSON`      | Preview of packets                            |
| `/predict`            | `JSON + .csv`| Model results and report                     |
| `/download-report`    | `.csv`      | Downloadable report                           |
| `/get-report`         | `JSON`      | Full result with probabilities                |
| `/list-reports`       | `JSON`      | Scan history summary                          |

---

## Tip:
Use the returned `file_id` from `/upload-pcap` to pass to all other endpoints as input.

---

You're now ready to fully test and integrate QuanNetDetect's backend APIs!

