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
  Box
} from '@mui/material';
import axios from 'axios';
import PageWrapper from '../components/PageWrapper';

const SimulatePage = () => {
  const [fileId, setFileId] = useState('');
  const [packets, setPackets] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [reportPath, setReportPath] = useState('');
  const [simulationMessage, setSimulationMessage] = useState('');

  const handleSimulate = async () => {
    try {
      const res = await axios.post('http://localhost:5000/generate-pcap', { file_id: fileId });
      setSimulationMessage(res.data.message);
    } catch (err) {
      console.error('Simulation error:', err);
      setSimulationMessage('Simulation failed.');
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
            onChange={(e) => setFileId(e.target.value)}
            sx={{ input: { color: '#e0e0e0' } }}
          />

          <Box sx={{ mt: 2, display: 'flex', flexWrap: 'wrap', gap: 2 }}>
            <Button variant="contained" onClick={handleSimulate} sx={{ backgroundColor: '#42a5f5' }}>
              Generate Simulated PCAP
            </Button>
            <Button variant="contained" onClick={handleValidate} sx={{ backgroundColor: '#ba68c8' }}>
              Preview Packets
            </Button>
            <Button variant="contained" onClick={handlePredict} sx={{ backgroundColor: '#66bb6a' }}>
              Predict
            </Button>
          </Box>

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
                  href={`http://localhost:5000/download-report/${fileId}`}
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
    </PageWrapper>
  );
};

export default SimulatePage;
