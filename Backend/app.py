# backend/app.py

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

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)

client = MongoClient("mongodb://localhost:27017/")
db = client['quan_net_detect']
reports_col = db['detection_reports']

@app.route('/upload-pcap', methods=['POST'])
def upload_pcap():
    if 'file' not in request.files:
        return jsonify({'error': 'No pcap file found in request'}), 400

    file = request.files['file']
    metadata = request.form.get('metadata')

    if metadata is None:
        return jsonify({'error': 'Missing metadata'}), 400

    try:
        params = json.loads(metadata)
    except Exception as e:
        return jsonify({'error': 'Invalid metadata format'}), 400

    tls_version = str(params.get('tls_version', '3')).strip()
    mode = params.get('mode', 'auto').strip().lower()
    custom_features = params.get('custom_features')
    record_limit = params.get('record_limit')

    file_id = uuid.uuid4().hex
    filename = file_id + '.pcap'
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    csv_path = os.path.join(OUTPUT_FOLDER, f'{file_id}_Model_Input.csv')

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

@app.route('/generate-pcap', methods=['POST'])
def generate_simulated_pcap():
    file_id = request.json.get('file_id')
    if not file_id:
        return jsonify({'error': 'Missing file_id'}), 400

    csv_path = os.path.join(OUTPUT_FOLDER, f'{file_id}_Model_Input.csv')
    pcap_out = os.path.join(OUTPUT_FOLDER, f'{file_id}_Simulated.pcap')

    count = generate_pcap_from_csv(csv_path, pcap_out)
    return jsonify({'message': f'{count} packets written to PCAP', 'pcap_path': pcap_out})

@app.route('/validate-pcap', methods=['POST'])
def validate_simulated_pcap():
    file_id = request.json.get('file_id')
    if not file_id:
        return jsonify({'error': 'Missing file_id'}), 400

    pcap_path = os.path.join(OUTPUT_FOLDER, f'{file_id}_Simulated.pcap')
    packets = validate_pcap(pcap_path)
    return jsonify({'packets': packets})

@app.route('/predict', methods=['POST'])
def predict():
    file_id = request.json.get('file_id')
    if not file_id:
        return jsonify({'error': 'Missing file_id'}), 400

    csv_path = os.path.join(OUTPUT_FOLDER, f'{file_id}_Model_Input.csv')
    report_path = os.path.join(OUTPUT_FOLDER, f'{file_id}_report.csv')

    try:
        predictions = run_qnn_prediction(csv_path)
        pd.DataFrame(predictions).to_csv(report_path, index=False)

        reports_col.insert_one({
            'file_id': file_id,
            'report_path': report_path,
            'pcap_file': f'{file_id}.pcap',
            'created_at': datetime.utcnow(),
            'predictions': predictions
        })

        return jsonify({'predictions': predictions, 'report_path': report_path})
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/get-report/<file_id>', methods=['GET'])
def get_report(file_id):
    record = reports_col.find_one({'file_id': file_id}, {'_id': 0})
    if record is None:
        return jsonify({'error': 'Report not found'}), 404
    return jsonify(record)

@app.route('/list-reports', methods=['GET'])
def list_reports():
    items = list(reports_col.find({}, {'_id': 0, 'file_id': 1, 'created_at': 1}))
    return jsonify(items)

@app.route('/download-report/<file_id>', methods=['GET'])
def download_report(file_id):
    report_path = os.path.join(OUTPUT_FOLDER, f'{file_id}_report.csv')
    if os.path.exists(report_path):
        return send_file(report_path, as_attachment=True)
    return jsonify({'error': 'Report not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)
