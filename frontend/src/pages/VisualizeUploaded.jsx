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
} from '@mui/material';
import axios from 'axios';
import { Pie, Bar } from 'react-chartjs-2';
import PageWrapper from '../components/PageWrapper';
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

  const prepareBarChartData = (distribution, label) => ({
    labels: Object.keys(distribution),
    datasets: [
      {
        label,
        data: Object.values(distribution),
        backgroundColor: '#42a5f5',
      },
    ],
  });

  return (
    <PageWrapper>
      <Paper elevation={3} style={{ padding: '2rem' }}>
        <Typography variant="h5" gutterBottom>
          Visualize Uploaded PCAP
        </Typography>

        <TextField
          label="Enter File ID"
          fullWidth
          margin="normal"
          value={fileId}
          onChange={(e) => setFileId(e.target.value)}
        />

        <Button variant="contained" onClick={fetchData}>
          Visualize
        </Button>

        {data && (
          <>
            <Typography variant="body1" style={{ marginTop: '1rem' }}>
              Total Packets: {data.total_packets}
            </Typography>

            <Grid container spacing={3} style={{ marginTop: '1rem' }}>
              <Grid item xs={12} md={4}>
                <Typography variant="subtitle1">Protocol Distribution</Typography>
                <Pie data={preparePieChartData(data.protocol_distribution, 'Protocols')} />
              </Grid>
              <Grid item xs={12} md={4}>
                <Typography variant="subtitle1">Source Port Usage</Typography>
                <Bar data={prepareBarChartData(data.source_port_distribution, 'Source Ports')} />
              </Grid>
              <Grid item xs={12} md={4}>
                <Typography variant="subtitle1">Destination Port Usage</Typography>
                <Bar data={prepareBarChartData(data.destination_port_distribution, 'Destination Ports')} />
              </Grid>
            </Grid>

            <Typography variant="h6" style={{ marginTop: '2rem' }}>
              Flow Table
            </Typography>

            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Source</TableCell>
                  <TableCell>Destination</TableCell>
                  <TableCell>Protocol</TableCell>
                  <TableCell>Info</TableCell>
                  <TableCell>Time</TableCell>
                  <TableCell>Length</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {data.flows.map((flow, idx) => (
                  <TableRow key={idx}>
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

            <Typography variant="body2" style={{ marginTop: '1rem' }}>
              This file contains traffic using the {Object.keys(data.protocol_distribution).join(', ')} protocol(s), 
              mostly through ports {Object.keys(data.destination_port_distribution).join(', ')}.
            </Typography>
          </>
        )}
      </Paper>
    </PageWrapper>
  );
};

export default VisualizeUploaded;
