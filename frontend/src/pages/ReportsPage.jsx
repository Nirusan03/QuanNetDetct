import React, { useState, useEffect } from 'react';
import {
  Paper,
  Typography,
  Table,
  TableHead,
  TableBody,
  TableRow,
  TableCell,
  Button,
} from '@mui/material';
import axios from 'axios';

const ReportsPage = () => {
  const [reports, setReports] = useState([]);
  const [selectedReport, setSelectedReport] = useState(null);

  useEffect(() => {
    const fetchReports = async () => {
      try {
        const res = await axios.get('http://localhost:5000/list-reports');
        setReports(res.data);
      } catch (err) {
        console.error(err);
      }
    };

    fetchReports();
  }, []);

  const handleViewReport = async (fileId) => {
    try {
      const res = await axios.get(`http://localhost:5000/get-report/${fileId}`);
      setSelectedReport(res.data);
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <Paper elevation={3} style={{ padding: '2rem' }}>
      <Typography variant="h5" gutterBottom>Past Prediction Reports</Typography>

      <Table>
        <TableHead>
          <TableRow>
            <TableCell>File ID</TableCell>
            <TableCell>Created At</TableCell>
            <TableCell>Actions</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {reports.map((r, idx) => (
            <TableRow key={idx}>
              <TableCell>{r.file_id}</TableCell>
              <TableCell>{r.created_at}</TableCell>
              <TableCell>
                <Button onClick={() => handleViewReport(r.file_id)} size="small" variant="outlined">
                  View
                </Button>
                <Button
                  href={`http://localhost:5000/download-report/${r.file_id}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  size="small"
                  style={{ marginLeft: 10 }}
                >
                  Download
                </Button>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>

      {selectedReport && (
        <>
          <Typography variant="h6" style={{ marginTop: '2rem' }}>
            Report for File ID: {selectedReport.file_id}
          </Typography>

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
              {selectedReport.predictions.map((p) => (
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
        </>
      )}
    </Paper>
  );
};

export default ReportsPage;
