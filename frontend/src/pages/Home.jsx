import React from 'react';
import { Typography, Paper } from '@mui/material';

const Home = () => {
  return (
    <Paper elevation={3} style={{ padding: '2rem' }}>
      <Typography variant="h4" gutterBottom>
        Welcome to QuanNetDetect
      </Typography>
      <Typography variant="body1">
        This dashboard lets you detect malicious TLS traffic using hybrid quantum-classical models. 
        Start by uploading a PCAP file, then simulate and run predictions.
      </Typography>
    </Paper>
  );
};

export default Home;
