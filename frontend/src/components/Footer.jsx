import React from 'react';
import { Box, Typography, Link } from '@mui/material';

const Footer = () => {
  return (
    <Box sx={{
      backgroundColor: '#1e1e1e',
      color: '#e0e0e0',
      padding: '1rem',
      textAlign: 'center',
      position: 'fixed',
      bottom: 0,
      left: 0,
      width: '100%',
      zIndex: 1000
    }}>
      <Typography variant="body2" sx={{ fontWeight: 600, color: '#90caf9' }}>
        Nirusan Hariharan | 20200094 | W1867405
      </Typography>
      <Typography variant="body2" sx={{ fontWeight: 400, color: '#90caf9' }}>
        © 2025 Quan Net Detect
      </Typography>
    </Box>
  );
};

export default Footer;
