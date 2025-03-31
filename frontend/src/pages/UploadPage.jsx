import React, { useState } from 'react';
import {
  Paper, TextField, MenuItem, Button, Typography, FormControl, InputLabel, Select,
} from '@mui/material';
import axios from 'axios';

const UploadPage = () => {
  const [file, setFile] = useState(null);
  const [tlsVersion, setTlsVersion] = useState('1'); // Default: TLS 1.2
  const [mode, setMode] = useState('auto');
  const [customFeatures, setCustomFeatures] = useState('');
  const [recordLimit, setRecordLimit] = useState(10);
  const [fileId, setFileId] = useState('');
  const [message, setMessage] = useState('');

  const handleUpload = async () => {
    if (!file) {
      setMessage('Please select a .pcap file first.');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    const payload = {
      tls_version: tlsVersion,
      mode,
      record_limit: Number(recordLimit),
      ...(mode === 'custom' && { custom_features: customFeatures })
    };

    // Key must be "metadata" as backend expects
    formData.append('metadata', JSON.stringify(payload));

    try {
      const response = await axios.post('http://localhost:5000/upload-pcap', formData);
      setFileId(response.data.file_id);
      setMessage(response.data.message || 'Upload successful!');
    } catch (error) {
      console.error(error);
      setMessage('Upload failed.');
    }
  };

  return (
    <Paper elevation={3} style={{ padding: '2rem' }}>
      <Typography variant="h5" gutterBottom>Upload PCAP File</Typography>

      <input type="file" accept=".pcap" onChange={(e) => setFile(e.target.files[0])} />

      <FormControl fullWidth margin="normal">
        <InputLabel>TLS Version</InputLabel>
        <Select value={tlsVersion} onChange={(e) => setTlsVersion(e.target.value)} label="TLS Version">
          <MenuItem value="1">TLS 1.2</MenuItem>
          <MenuItem value="2">TLS 1.3</MenuItem>
          <MenuItem value="3">All TLS Versions (1.2 + 1.3)</MenuItem>
        </Select>
      </FormControl>

      <FormControl fullWidth margin="normal">
        <InputLabel>Mode</InputLabel>
        <Select value={mode} onChange={(e) => setMode(e.target.value)} label="Mode">
          <MenuItem value="auto">Auto</MenuItem>
          <MenuItem value="custom">Custom</MenuItem>
        </Select>
      </FormControl>

      {mode === 'custom' && (
        <TextField
          label="Custom Features"
          fullWidth
          margin="normal"
          value={customFeatures}
          onChange={(e) => setCustomFeatures(e.target.value)}
        />
      )}

      <TextField
        label="Record Limit"
        type="number"
        fullWidth
        margin="normal"
        value={recordLimit}
        onChange={(e) => setRecordLimit(e.target.value)}
      />

      <Button variant="contained" color="primary" onClick={handleUpload} style={{ marginTop: '1rem' }}>
        Upload & Simulate
      </Button>

      {fileId && (
        <Typography style={{ marginTop: '1rem' }}>
          ✅ File ID: <strong>{fileId}</strong>
        </Typography>
      )}

      {message && (
        <Typography style={{ marginTop: '0.5rem' }}>{message}</Typography>
      )}
    </Paper>
  );
};

export default UploadPage;
