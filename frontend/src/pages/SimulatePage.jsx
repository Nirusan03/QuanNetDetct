import React, { useState } from 'react';
import {
  Button,
  Paper,
  Typography,
  Table,
  TableHead,
  TableRow,
  TableCell,
  TableBody,
  TextField
} from '@mui/material';
import axios from 'axios';

const SimulatePage = () => {
  const [fileId, setFileId] = useState('');
  const [packets, setPackets] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [reportPath, setReportPath] = useState('');

  const handleSimulate = async () => {
    try {
      const res = await axios.post('http://localhost:5000/generate-pcap', { file_id: fileId });
      console.log(res.data.message);
    } catch (err) {
      console.error('Simulation error:', err);
    }
  };

  const handleValidate = async () => {
    try {
      const res = await axios.post('http://localhost:5000/validate-pcap', { file_id: fileId });
      setPackets(res.data.packets);
    } catch (err) {
      console.error('Validation error:', err);
    }
  };

  const handlePredict = async () => {
    try {
      const res = await axios.post('http://localhost:5000/predict', { file_id: fileId });
      setPredictions(res.data.predictions);
      setReportPath(res.data.report_path);
    } catch (err) {
      console.error('Prediction error:', err);
    }
  };

  return (
    <Paper elevation={3} style={{ padding: '2rem' }}>
      <Typography variant="h5" gutterBottom>Simulate & Predict</Typography>

      <TextField
        label="Enter File ID"
        fullWidth
        margin="normal"
        value={fileId}
        onChange={(e) => setFileId(e.target.value)}
      />

      <Button variant="contained" color="primary" onClick={handleSimulate} style={{ marginRight: 10 }}>
        Generate Simulated PCAP
      </Button>

      <Button variant="contained" color="secondary" onClick={handleValidate} style={{ marginRight: 10 }}>
        Preview Packets
      </Button>

      <Button variant="contained" color="success" onClick={handlePredict}>
        Predict
      </Button>

      {/* Packet Table */}
      {packets.length > 0 && (
        <>
          <Typography variant="h6" style={{ marginTop: '2rem' }}>Packet Preview</Typography>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>#</TableCell>
                <TableCell>Src IP</TableCell>
                <TableCell>Dst IP</TableCell>
                <TableCell>Src Port</TableCell>
                <TableCell>Dst Port</TableCell>
                <TableCell>Size</TableCell>
                <TableCell>Flags</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {packets.map((pkt, idx) => (
                <TableRow key={idx}>
                  <TableCell>{pkt.index}</TableCell>
                  <TableCell>{pkt.src_ip}</TableCell>
                  <TableCell>{pkt.dst_ip}</TableCell>
                  <TableCell>{pkt.sport}</TableCell>
                  <TableCell>{pkt.dport}</TableCell>
                  <TableCell>{pkt.size}</TableCell>
                  <TableCell>{pkt.flags}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </>
      )}

      {/* Prediction Table */}
      {predictions.length > 0 && (
        <>
          <Typography variant="h6" style={{ marginTop: '2rem' }}>Prediction Results</Typography>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>#</TableCell>
                <TableCell>Predicted Class</TableCell>
                <TableCell>BENIGN</TableCell>
                <TableCell>DrDoS_DNS</TableCell>
                <TableCell>DrDoS_LDAP</TableCell>
                <TableCell>LDAP</TableCell>
                <TableCell>Syn</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {predictions.map((p) => (
                <TableRow key={p.id}>
                  <TableCell>{p.id}</TableCell>
                  <TableCell>{p.predicted_class}</TableCell>
                  <TableCell>{(p.BENIGN * 100).toFixed(2)}%</TableCell>
                  <TableCell>{(p.DrDoS_DNS * 100).toFixed(2)}%</TableCell>
                  <TableCell>{(p.DrDoS_LDAP * 100).toFixed(2)}%</TableCell>
                  <TableCell>{(p.LDAP * 100).toFixed(2)}%</TableCell>
                  <TableCell>{(p.Syn * 100).toFixed(2)}%</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>

          {reportPath && (
            <Button
              variant="outlined"
              color="info"
              href={`http://localhost:5000/download-report/${fileId}`}
              target="_blank"
              rel="noopener noreferrer"
              style={{ marginTop: '1rem' }}
            >
              Download Report
            </Button>
          )}
        </>
      )}
    </Paper>
  );
};

export default SimulatePage;
