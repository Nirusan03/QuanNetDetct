import React, { useState } from 'react';
import {
  Paper, TextField, MenuItem, Button, Typography, FormControl, InputLabel, Select, Box
} from '@mui/material';
import axios from 'axios';
import PageWrapper from '../components/PageWrapper';
import Footer from '../components/Footer';


const UploadPage = () => {
  const [file, setFile] = useState(null);
  const [tlsVersion, setTlsVersion] = useState('1');
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
    <PageWrapper>
      <Box sx={{ maxWidth: '1700px', mx: 'auto' }}>
        <Paper elevation={3} sx={{ padding: '2.5rem', backgroundColor: '#1e1e1e' }}>
          <Typography variant="h5" gutterBottom sx={{ color: '#90caf9' }}>
            Upload PCAP File
          </Typography>

          <input
            type="file"
            accept=".pcap"
            onChange={(e) => setFile(e.target.files[0])}
            style={{ marginTop: '1rem', marginBottom: '1.5rem', color: '#e0e0e0' }}
          />

          <FormControl fullWidth margin="normal">
            <InputLabel sx={{ color: '#ccc' }}>TLS Version</InputLabel>
            <Select
              value={tlsVersion}
              onChange={(e) => setTlsVersion(e.target.value)}
              label="TLS Version"
              sx={{ color: '#e0e0e0' }}
            >
              <MenuItem value="1">TLS 1.2</MenuItem>
              <MenuItem value="2">TLS 1.3</MenuItem>
              <MenuItem value="3">All TLS Versions (1.2 + 1.3)</MenuItem>
            </Select>
          </FormControl>

          <FormControl fullWidth margin="normal">
            <InputLabel sx={{ color: '#ccc' }}>Mode</InputLabel>
            <Select
              value={mode}
              onChange={(e) => setMode(e.target.value)}
              label="Mode"
              sx={{ color: '#e0e0e0' }}
            >
              <MenuItem value="auto">Auto</MenuItem>
              <MenuItem value="custom">Custom</MenuItem>
            </Select>
          </FormControl>

          {mode === 'custom' && (
            <TextField
              label="Custom Features (JSON)"
              placeholder='{"Source Port": 443, "Flow Duration": 0.2}'
              fullWidth
              margin="normal"
              multiline
              minRows={3}
              value={customFeatures}
              onChange={(e) => setCustomFeatures(e.target.value)}
              sx={{ input: { color: '#e0e0e0' } }}
            />
          )}

          <TextField
            label="Record Limit"
            type="number"
            fullWidth
            margin="normal"
            value={recordLimit}
            onChange={(e) => setRecordLimit(e.target.value)}
            sx={{ input: { color: '#e0e0e0' } }}
          />

          <Button
            variant="contained"
            color="primary"
            onClick={handleUpload}
            sx={{ mt: 3 }}
          >
            Upload & Simulate
          </Button>

          {fileId && (
            <Typography sx={{ mt: 3, color: '#66bb6a' }}>
              File ID: <strong>{fileId}</strong>
            </Typography>
          )}

          {message && (
            <Typography sx={{ mt: 1.5, color: '#ccc' }}>{message}</Typography>
          )}
        </Paper>
      </Box>
      <Footer />
    </PageWrapper>
  );
};

export default UploadPage;
