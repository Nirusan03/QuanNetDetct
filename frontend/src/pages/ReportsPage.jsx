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
  Box
} from '@mui/material';
import axios from 'axios';
import PageWrapper from '../components/PageWrapper';

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
    <PageWrapper>
      <Box sx={{ maxWidth: '1900px', mx: 'auto' }}>
        <Paper elevation={3} sx={{ padding: '2.5rem', backgroundColor: '#1e1e1e' }}>
          <Typography variant="h5" gutterBottom sx={{ color: '#90caf9' }}>
            Past Prediction Reports
          </Typography>

          <Table size="small">
            <TableHead>
              <TableRow sx={{ backgroundColor: '#263238' }}>
                <TableCell sx={{ color: '#ffffff', fontWeight: 600 }}>File ID</TableCell>
                <TableCell sx={{ color: '#ffffff', fontWeight: 600 }}>Created At</TableCell>
                <TableCell sx={{ color: '#ffffff', fontWeight: 600 }}>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {reports.map((r, idx) => (
                <TableRow
                  key={idx}
                  hover
                  sx={{
                    backgroundColor: idx % 2 === 0 ? '#2c2c2c' : '#252525',
                    '&:hover': { backgroundColor: '#37474f' },
                  }}
                >
                  <TableCell>{r.file_id}</TableCell>
                  <TableCell>{r.created_at}</TableCell>
                  <TableCell>
                    <Button
                      onClick={() => handleViewReport(r.file_id)}
                      size="small"
                      variant="outlined"
                      sx={{ borderColor: '#42a5f5', color: '#42a5f5', fontWeight: 600 }}
                    >
                      View
                    </Button>
                    <Button
                      href={`http://localhost:5000/download-report/${r.file_id}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      size="small"
                      sx={{
                        ml: 2,
                        color: '#ffffff',
                        fontWeight: 600,
                        backgroundColor: '#1e88e5',
                        '&:hover': { backgroundColor: '#1565c0' },
                      }}
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
              <Typography variant="h6" sx={{ mt: 5, mb: 2, color: '#64b5f6' }}>
                Report for File ID: {selectedReport.file_id}
              </Typography>

              <Paper elevation={1} sx={{ borderRadius: 2, overflow: 'auto', backgroundColor: '#1e1e1e' }}>
                <Table size="medium">
                  <TableHead>
                    <TableRow sx={{ backgroundColor: '#263238' }}>
                      {['#', 'Predicted Class', 'BENIGN', 'DrDoS_DNS', 'DrDoS_LDAP', 'LDAP', 'Syn'].map((header, idx) => (
                        <TableCell key={idx} sx={{ color: '#ffffff', fontWeight: 600 }}>
                          {header}
                        </TableCell>
                      ))}
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {selectedReport.predictions.map((p, idx) => (
                      <TableRow
                        key={p.id}
                        hover
                        sx={{
                          backgroundColor: idx % 2 === 0 ? '#2c2c2c' : '#252525',
                          '&:hover': { backgroundColor: '#37474f' },
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
            </>
          )}
        </Paper>
      </Box>
    </PageWrapper>
  );
};

export default ReportsPage;
