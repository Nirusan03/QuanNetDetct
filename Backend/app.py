from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from pymongo import MongoClient
from datetime import datetime
import os
import uuid
import pandas as pd
import json

from utils.feature_extraction import process_pcap_and_simulate
from utils.pcap_generation import generate_pcap_from_csv, validate_pcap
from utils.qnn_inference import run_qnn_prediction
from utils.visualization_utils import summarize_pcap_for_visualization

# Base path of the directory - QuanNetDetect
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'outputs')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)

client = MongoClient("mongodb://localhost:27017/")
db = client['quan_net_detect']
reports_col = db['detection_reports']
quantum_log_col = db['quantum_logs']

# URL (POST) - Body (form-data): http://localhost:5000/upload-pcap
# (key) file - (type file) - PCAp file (value)
# (key) metadata - (type text) - { "tls_version": "1", "mode": "auto", "record_limit": 10 } (Sample value)
@app.route('/upload-pcap', methods=['POST'])
def upload_pcap():
    if 'file' not in request.files:
        return jsonify({'error': 'No pcap file found in request'}), 400

    file = request.files['file']
    metadata = request.form.get('metadata')
    if not metadata:
        return jsonify({'error': 'Missing metadata'}), 400

    try:
        params = json.loads(metadata)
    except json.JSONDecodeError:
        return jsonify({'error': 'Invalid metadata format. Must be valid JSON.'}), 400

    tls_version = str(params.get('tls_version', '3')).strip()
    mode = str(params.get('mode', 'auto')).strip().lower()
    custom_features = params.get('custom_features')
    record_limit = params.get('record_limit')

    if record_limit:
        try:
            record_limit = int(record_limit)
        except ValueError:
            return jsonify({'error': 'record_limit must be an integer.'}), 400

    if mode not in ['auto', 'custom']:
        return jsonify({'error': 'Invalid mode. Must be "auto" or "custom".'}), 400

    file_id = uuid.uuid4().hex
    filename = f"{file_id}.pcap"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    csv_path = os.path.join(OUTPUT_FOLDER, f"{file_id}_Model_Input.csv")

    try:
        process_pcap_and_simulate(
            pcap_path=filepath,
            save_csv_path=csv_path,
            tls_version=tls_version,
            mode=mode,
            custom_features=custom_features,
            record_limit=record_limit
        )
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

    return jsonify({'message': 'Upload and simulation complete', 'file_id': file_id})

# URL (POST) : http://localhost:5000/generate-pcap
@app.route('/generate-pcap', methods=['POST'])
def generate_simulated_pcap():
    data = request.get_json()
    if not data or 'file_id' not in data:
        return jsonify({'error': 'Missing file_id in request body'}), 400

    file_id = data['file_id']
    csv_path = os.path.join(OUTPUT_FOLDER, f"{file_id}_Model_Input.csv")
    pcap_out = os.path.join(OUTPUT_FOLDER, f"{file_id}_Simulated.pcap")

    if not os.path.exists(csv_path):
        return jsonify({'error': f'Model input CSV not found: {csv_path}'}), 404

    try:
        count = generate_pcap_from_csv(csv_path, pcap_out)
        return jsonify({'message': f'{count} packets written to PCAP', 'pcap_path': pcap_out})
    except Exception as e:
        return jsonify({'error': f'PCAP generation failed: {str(e)}'}), 500

# URL (POST) : http://localhost:5000/validate-pcap
@app.route('/validate-pcap', methods=['POST'])
def validate_simulated_pcap():
    data = request.get_json()
    if not data or 'file_id' not in data:
        return jsonify({'error': 'Missing file_id in request body'}), 400

    file_id = data['file_id']
    pcap_path = os.path.join(OUTPUT_FOLDER, f"{file_id}_Simulated.pcap")

    if not os.path.exists(pcap_path):
        return jsonify({'error': f'Simulated PCAP not found: {pcap_path}'}), 404

    try:
        packets = validate_pcap(pcap_path)
        return jsonify({'packets': packets})
    except Exception as e:
        return jsonify({'error': f'Validation failed: {str(e)}'}), 500

# URL (POST): http://localhost:5000/predict
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'file_id' not in data:
        return jsonify({'error': 'Missing file_id in request body'}), 400

    file_id = data['file_id']
    csv_path = os.path.join(OUTPUT_FOLDER, f"{file_id}_Model_Input.csv")
    report_path = os.path.join(OUTPUT_FOLDER, f"{file_id}_report.csv")

    if not os.path.exists(csv_path):
        return jsonify({'error': f'Model input CSV not found: {csv_path}'}), 404

    try:
        predictions, quantum_log, quantum_log_path = run_qnn_prediction(csv_path, file_id=file_id)
        pd.DataFrame(predictions).to_csv(report_path, index=False)

        reports_col.insert_one({
            'file_id': file_id,
            'report_path': report_path,
            'pcap_file': f'{file_id}.pcap',
            'created_at': datetime.utcnow(),
            'predictions': predictions
        })

        quantum_log_col.insert_one(quantum_log)

        return jsonify({
            'predictions': predictions,
            'report_path': report_path,
            'quantum_log_path': quantum_log_path
        })
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

# URL (GET) - http://localhost:5000/get-report/195e332758af400b97e9e8e50cfe8160  <file_id>
@app.route('/get-report/<file_id>', methods=['GET'])
def get_report(file_id):
    record = reports_col.find_one({'file_id': file_id}, {'_id': 0})
    if not record:
        return jsonify({'error': 'Report not found'}), 404
    return jsonify(record)

# URL (GET) - http://localhost:5000/list-reports
@app.route('/list-reports', methods=['GET'])
def list_reports():
    try:
        items = list(reports_col.find({}, {'_id': 0, 'file_id': 1, 'created_at': 1}))
        return jsonify(items)
    except Exception as e:
        return jsonify({'error': f'Failed to list reports: {str(e)}'}), 500

# URL (GET) - http://localhost:5000/download-report/195e332758af400b97e9e8e50cfe8160 <file_id>
@app.route('/download-report/<file_id>', methods=['GET'])
def download_report(file_id):
    report_path = os.path.normpath(os.path.join(OUTPUT_FOLDER, f"{file_id}_report.csv"))
    if os.path.exists(report_path):
        return send_file(report_path, as_attachment=True)
    return jsonify({'error': 'Report not found'}), 404

# URL (GET) - http://localhost:5000/visualize-upload/195e332758af400b97e9e8e50cfe8160 <file_id>
@app.route('/visualize-upload/<file_id>', methods=['GET'])
def visualize_uploaded_pcap(file_id):
    pcap_path = os.path.join(UPLOAD_FOLDER, f"{file_id}.pcap")
    if not os.path.exists(pcap_path):
        return jsonify({'error': 'Uploaded pcap not found'}), 404

    summary = summarize_pcap_for_visualization(pcap_path)
    return jsonify(summary)

# URL (GET) - http://localhost:5000/visualize-simulated/195e332758af400b97e9e8e50cfe8160 <file_id>
@app.route('/visualize-simulated/<file_id>', methods=['GET'])
def visualize_simulated_pcap(file_id):
    pcap_path = os.path.join(OUTPUT_FOLDER, f"{file_id}_Simulated.pcap")
    if not os.path.exists(pcap_path):
        return jsonify({'error': 'Simulated pcap not found'}), 404

    summary = summarize_pcap_for_visualization(pcap_path)
    return jsonify(summary)

if __name__ == '__main__':
    app.run(debug=True)
