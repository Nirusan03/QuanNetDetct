import React from 'react';
import { Drawer, List, ListItem, ListItemIcon, ListItemText, Toolbar, Typography, Box } from '@mui/material';
import { Home as HomeIcon, CloudUpload, Assessment, BarChart, Timeline } from '@mui/icons-material';
import { NavLink } from 'react-router-dom';

const drawerWidth = 240;

const Sidebar = () => {
  const navItems = [
    { text: 'Documentation', icon: <HomeIcon />, path: '/' },
    { text: 'Upload PCAP', icon: <CloudUpload />, path: '/upload' },
    { text: 'Simulate & Predict', icon: <Assessment />, path: '/simulate' },
    { text: 'Visualize Uploaded', icon: <Timeline />, path: '/visualize/upload/PLACEHOLDER' },
    { text: 'Visualize Simulated', icon: <BarChart />, path: '/visualize/simulated/PLACEHOLDER' },
    { text: 'Reports', icon: <Assessment />, path: '/reports' },
  ];

  return (
    <Drawer
      variant="permanent"
      anchor="left"
      sx={{
        width: drawerWidth,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: drawerWidth,
          boxSizing: 'border-box',
          backgroundColor: '#121212',
          color: '#fff',
        },
      }}
    >
      <Box sx={{ padding: '1rem' }}>
        <Typography variant="h6" sx={{ color: '#90caf9', fontWeight: 'bold', textAlign: 'center' }}>
          Quan Net Detect
        </Typography>
      </Box>
      <Toolbar />
      <List>
        {navItems.map(({ text, icon, path }) => (
          <ListItem
            key={text}
            component={NavLink}
            to={path}
            style={({ isActive }) => ({
              backgroundColor: isActive ? '#1e1e1e' : 'inherit',
              color: isActive ? '#2196f3' : 'inherit',
              textDecoration: 'none',
            })}
          >
            <ListItemIcon sx={{ color: 'inherit' }}>{icon}</ListItemIcon>
            <ListItemText primary={text} />
          </ListItem>
        ))}
      </List>
    </Drawer>
  );
};

export default Sidebar;
