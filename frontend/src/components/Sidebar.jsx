import React from 'react';
import { Drawer, List, ListItem, ListItemIcon, ListItemText, Toolbar } from '@mui/material';
import { Home, CloudUpload, Assessment, BarChart, Timeline } from '@mui/icons-material';
import { NavLink } from 'react-router-dom';

const drawerWidth = 240;

const Sidebar = () => {
  const navItems = [
    { text: 'Home', icon: <Home />, path: '/' },
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
        [`& .MuiDrawer-paper`]: { width: drawerWidth, boxSizing: 'border-box' },
      }}
    >
      <Toolbar />
      <List>
        {navItems.map(({ text, icon, path }) => (
          <ListItem
            button
            key={text}
            component={NavLink}
            to={path}
            sx={{
              '&.active': {
                backgroundColor: 'action.selected',
              },
            }}
          >
            <ListItemIcon>{icon}</ListItemIcon>
            <ListItemText primary={text} />
          </ListItem>
        ))}
      </List>
    </Drawer>
  );
};

export default Sidebar;
