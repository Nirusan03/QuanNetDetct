import React, { useState } from 'react';
import {
  Typography,
  Paper,
  TextField,
  Button,
  Table,
  TableHead,
  TableBody,
  TableRow,
  TableCell,
  Grid,
  Box
} from '@mui/material';
import axios from 'axios';
import { Pie, Bar } from 'react-chartjs-2';
import PageWrapper from '../components/PageWrapper';
import Footer from '../components/Footer';
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend,
  CategoryScale,
  LinearScale,
  BarElement
} from 'chart.js';

ChartJS.register(ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement);

const VisualizeUploaded = () => {
  const [fileId, setFileId] = useState('');
  const [data, setData] = useState(null);

  const fetchData = async () => {
    try {
      const res = await axios.get(`http://localhost:5000/visualize-upload/${fileId}`);
      setData(res.data);
    } catch (err) {
      console.error(err);
    }
  };

  const preparePieChartData = (distribution, label) => ({
    labels: Object.keys(distribution),
    datasets: [
      {
        label,
        data: Object.values(distribution),
        backgroundColor: ['#42a5f5', '#66bb6a', '#ffca28', '#ef5350', '#ab47bc'],
      },
    ],
  });

  const prepareBarChartData = (distribution, label, color = '#42a5f5') => ({
    labels: Object.keys(distribution),
    datasets: [
      {
        label,
        data: Object.values(distribution),
        backgroundColor: color,
      },
    ],
  });

  return (
    <PageWrapper>
      <Box sx={{ maxWidth: '1900px', mx: 'auto' }}>
        <Paper elevation={3} sx={{ padding: '2.5rem', backgroundColor: '#1e1e1e' }}>
          <Typography variant="h5" gutterBottom sx={{ color: '#90caf9' }}>
            Visualize Uploaded PCAP
          </Typography>

          <TextField
            label="Enter File ID"
            fullWidth
            margin="normal"
            value={fileId}
            onChange={(e) => setFileId(e.target.value)}
            sx={{ input: { color: '#e0e0e0' } }}
          />

          <Button variant="contained" sx={{ mt: 2, backgroundColor: '#42a5f5' }} onClick={fetchData}>
            Visualize
          </Button>

          {data && (
            <>
              <Typography sx={{ mt: 4, color: '#81c784' }}>
                Total Packets: {data.total_packets}
              </Typography>

              <Grid container spacing={3} sx={{ mt: 1 }}>
                <Grid item xs={12} md={4}>
                  <Typography variant="subtitle1" sx={{ mb: 1, color: '#64b5f6' }}>
                    Protocol Distribution
                  </Typography>
                  <Pie data={preparePieChartData(data.protocol_distribution, 'Protocols')} />
                </Grid>
                <Grid item xs={12} md={4}>
                  <Typography variant="subtitle1" sx={{ mb: 1, color: '#64b5f6' }}>
                    Source Port Usage
                  </Typography>
                  <Bar data={prepareBarChartData(data.source_port_distribution, 'Source Ports', '#66bb6a')} />
                </Grid>
                <Grid item xs={12} md={4}>
                  <Typography variant="subtitle1" sx={{ mb: 1, color: '#64b5f6' }}>
                    Destination Port Usage
                  </Typography>
                  <Bar data={prepareBarChartData(data.destination_port_distribution, 'Destination Ports', '#ba68c8')} />
                </Grid>
              </Grid>

              <Typography variant="h6" sx={{ mt: 5, mb: 2, color: '#64b5f6' }}>
                Flow Table
              </Typography>

              <Paper elevation={1} sx={{ borderRadius: 2, overflow: 'auto', backgroundColor: '#1e1e1e' }}>
                <Table size="medium">
                  <TableHead>
                    <TableRow sx={{ backgroundColor: '#263238' }}>
                      {['Source', 'Destination', 'Protocol', 'Info', 'Time', 'Length'].map((header, idx) => (
                        <TableCell key={idx} sx={{ color: '#ffffff', fontWeight: 600, py: 1.5 }}>
                          {header}
                        </TableCell>
                      ))}
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {data.flows.map((flow, idx) => (
                      <TableRow
                        key={idx}
                        hover
                        sx={{
                          backgroundColor: idx % 2 === 0 ? '#2c2c2c' : '#252525',
                          '&:hover': { backgroundColor: '#37474f' },
                        }}
                      >
                        <TableCell>{flow.Source}</TableCell>
                        <TableCell>{flow.Destination}</TableCell>
                        <TableCell>{flow.Protocol}</TableCell>
                        <TableCell>{flow.Info}</TableCell>
                        <TableCell>{flow.Time}</TableCell>
                        <TableCell>{flow.Length}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </Paper>

              <Typography variant="body2" sx={{ mt: 3, color: '#cccccc' }}>
                This file contains traffic using the {Object.keys(data.protocol_distribution).join(', ')} protocol(s),
                mostly through ports {Object.keys(data.destination_port_distribution).join(', ')}.
              </Typography>
            </>
          )}
        </Paper>
      </Box>
      <Footer />
    </PageWrapper>
  );
};

export default VisualizeUploaded;
