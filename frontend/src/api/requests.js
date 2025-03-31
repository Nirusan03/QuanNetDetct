import axios from 'axios';

const BASE_URL = 'http://localhost:5000';

// Upload PCAP
export const uploadPcap = (formData) => {
  return axios.post(`${BASE_URL}/upload-pcap`, formData);
};

// Simulate PCAP
export const generatePcap = (file_id) => {
  return axios.post(`${BASE_URL}/generate-pcap`, { file_id });
};

// Validate PCAP
export const validatePcap = (file_id) => {
  return axios.post(`${BASE_URL}/validate-pcap`, { file_id });
};

// Predict with Quantum Model
export const predictTraffic = (file_id) => {
  return axios.post(`${BASE_URL}/predict`, { file_id });
};

// Visualize Uploaded PCAP
export const visualizeUpload = (file_id) => {
  return axios.get(`${BASE_URL}/visualize-upload/${file_id}`);
};

// Visualize Simulated PCAP
export const visualizeSimulated = (file_id) => {
  return axios.get(`${BASE_URL}/visualize-simulated/${file_id}`);
};

// List All Reports
export const listReports = () => {
  return axios.get(`${BASE_URL}/list-reports`);
};

// Get Detailed Report
export const getReport = (file_id) => {
  return axios.get(`${BASE_URL}/get-report/${file_id}`);
};

// Download Report (just use this as href in a button/link)
export const getDownloadUrl = (file_id) => {
  return `${BASE_URL}/download-report/${file_id}`;
};
