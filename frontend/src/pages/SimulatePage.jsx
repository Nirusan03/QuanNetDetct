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
  TextField,
  Box,
  Alert
} from '@mui/material';
import axios from 'axios';
import PageWrapper from '../components/PageWrapper';
import Footer from '../components/Footer';

const SimulatePage = () => {
  const [fileId, setFileId] = useState('');
  const [packets, setPackets] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [reportPath, setReportPath] = useState('');
  const [simulationMessage, setSimulationMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [fileIdError, setFileIdError] = useState('');
  const [alertMsg, setAlertMsg] = useState('');

  const isValidFileId = (id) => /^[a-zA-Z0-9_-]{6,100}$/.test(id);

  const validateInput = () => {
    if (!fileId.trim()) {
      setFileIdError('File ID is required.');
      return false;
    }
    if (!isValidFileId(fileId.trim())) {
      setFileIdError('Invalid File ID format.');
      return false;
    }
    setFileIdError('');
    return true;
  };

  const handleSimulate = async () => {
    if (!validateInput()) return;
    try {
      setLoading(true);
      const res = await axios.post('http://localhost:5000/generate-pcap', { file_id: fileId.trim() });
      setSimulationMessage(res.data.message);
      setAlertMsg('');
    } catch (err) {
      setSimulationMessage('Simulation failed.');
      setAlertMsg('Failed to generate simulated PCAP.');
    } finally {
      setLoading(false);
    }
  };

  const handleValidate = async () => {
    if (!validateInput()) return;
    try {
      setLoading(true);
      const res = await axios.post('http://localhost:5000/validate-pcap', { file_id: fileId.trim() });
      setPackets(res.data.packets);
      setAlertMsg('');
    } catch (err) {
      setAlertMsg('Failed to preview packets.');
    } finally {
      setLoading(false);
    }
  };

  const handlePredict = async () => {
    if (!validateInput()) return;
    try {
      setLoading(true);
      const res = await axios.post('http://localhost:5000/predict', { file_id: fileId.trim() });
      setPredictions(res.data.predictions);
      setReportPath(res.data.report_path);
      setAlertMsg('');
    } catch (err) {
      setAlertMsg('Prediction failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <PageWrapper>
      <Box sx={{ maxWidth: '1900px', mx: 'auto' }}>
        <Paper elevation={3} sx={{ padding: '2.5rem', backgroundColor: '#1e1e1e' }}>
          <Typography variant="h5" gutterBottom sx={{ color: '#90caf9' }}>
            Simulate & Predict
          </Typography>

          <TextField
            label="Enter File ID"
            fullWidth
            margin="normal"
            value={fileId}
            error={!!fileIdError}
            helperText={fileIdError}
            onChange={(e) => setFileId(e.target.value)}
            sx={{ input: { color: '#e0e0e0' }, label: { color: '#e0e0e0' } }}
            onBlur={validateInput}
          />

          <Box sx={{ mt: 2, display: 'flex', flexWrap: 'wrap', gap: 2 }}>
            <Button
              variant="contained"
              onClick={handleSimulate}
              sx={{ backgroundColor: '#42a5f5' }}
              disabled={loading || !fileId.trim()}
            >
              Generate Simulated PCAP
            </Button>
            <Button
              variant="contained"
              onClick={handleValidate}
              sx={{ backgroundColor: '#ba68c8' }}
              disabled={loading || !fileId.trim()}
            >
              Preview Packets
            </Button>
            <Button
              variant="contained"
              onClick={handlePredict}
              sx={{ backgroundColor: '#66bb6a' }}
              disabled={loading || !fileId.trim()}
            >
              Predict
            </Button>
          </Box>

          {alertMsg && (
            <Alert severity="error" sx={{ mt: 3 }}>
              {alertMsg}
            </Alert>
          )}

          {simulationMessage && (
            <Typography sx={{ mt: 3, color: '#81c784' }}>{simulationMessage}</Typography>
          )}

          {packets.length > 0 && (
            <>
              <Typography variant="h6" sx={{ mt: 5, mb: 1, color: '#64b5f6' }}>
                Packet Preview
              </Typography>

              <Paper elevation={1} sx={{ borderRadius: 2, overflow: 'auto', backgroundColor: '#1e1e1e' }}>
                <Table sx={{ minWidth: 1000 }} size="medium">
                  <TableHead>
                    <TableRow sx={{ backgroundColor: '#263238' }}>
                      {['#', 'Src IP', 'Dst IP', 'Src Port', 'Dst Port', 'Size', 'Flags'].map((head, idx) => (
                        <TableCell key={idx} sx={{ color: '#ffffff', fontWeight: 600, py: 1.5 }}>{head}</TableCell>
                      ))}
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {packets.map((pkt, idx) => (
                      <TableRow
                        key={idx}
                        hover
                        sx={{
                          backgroundColor: idx % 2 === 0 ? '#2c2c2c' : '#252525',
                          '&:hover': { backgroundColor: '#37474f' }
                        }}
                      >
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
              </Paper>
            </>
          )}

          {predictions.length > 0 && (
            <>
              <Typography variant="h6" sx={{ mt: 5, mb: 1, color: '#64b5f6' }}>
                Prediction Results
              </Typography>

              <Paper elevation={1} sx={{ borderRadius: 2, overflow: 'auto', backgroundColor: '#1e1e1e' }}>
                <Table sx={{ minWidth: 1200 }} size="medium">
                  <TableHead>
                    <TableRow sx={{ backgroundColor: '#263238' }}>
                      {['#', 'Predicted Class', 'BENIGN', 'DrDoS_DNS', 'DrDoS_LDAP', 'LDAP', 'Syn'].map((head, idx) => (
                        <TableCell key={idx} sx={{ color: '#ffffff', fontWeight: 600, py: 1.5 }}>{head}</TableCell>
                      ))}
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {predictions.map((p) => (
                      <TableRow
                        key={p.id}
                        hover
                        sx={{
                          backgroundColor: p.id % 2 === 0 ? '#2c2c2c' : '#252525',
                          '&:hover': { backgroundColor: '#37474f' }
                        }}
                      >
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
              </Paper>

              {reportPath && (
                <Button
                  variant="outlined"
                  color="info"
                  href={`http://localhost:5000/download-report/${fileId.trim()}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  sx={{ mt: 3 }}
                >
                  Download Report
                </Button>
              )}
            </>
          )}
        </Paper>
      </Box>
      <Footer />
    </PageWrapper>
  );
};

export default SimulatePage;
