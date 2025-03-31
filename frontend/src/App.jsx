import React, { useState, useMemo } from 'react';
import { ThemeProvider, CssBaseline, IconButton, Toolbar } from '@mui/material';
import { Brightness4, Brightness7 } from '@mui/icons-material';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

import { getTheme } from './theme/theme';
import Sidebar from './components/Sidebar';

import Home from './pages/Documentation';
import UploadPage from './pages/UploadPage';
import SimulatePage from './pages/SimulatePage';
import VisualizeUploaded from './pages/VisualizeUploaded';
import VisualizeSimulated from './pages/VisualizeSimulated';
import ReportsPage from './pages/ReportsPage';

const App = () => {
  const [mode, setMode] = useState('dark');
  const theme = useMemo(() => getTheme(mode), [mode]);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <div style={{ display: 'flex' }}>
          <Sidebar />
          <div style={{ flexGrow: 1, padding: '1rem' }}>
            {/* Optional top toolbar (for spacing + light/dark toggle) */}
            <Toolbar style={{ display: 'flex', justifyContent: 'flex-end' }}>
              <IconButton onClick={() => setMode(mode === 'dark' ? 'light' : 'dark')}>
                {mode === 'dark' ? <Brightness7 /> : <Brightness4 />}
              </IconButton>
            </Toolbar>

            {/* Routes */}
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/upload" element={<UploadPage />} />
              <Route path="/simulate" element={<SimulatePage />} />
              <Route path="/visualize/upload/:fileId" element={<VisualizeUploaded />} />
              <Route path="/visualize/simulated/:fileId" element={<VisualizeSimulated />} />
              <Route path="/reports" element={<ReportsPage />} />
            </Routes>
          </div>
        </div>
      </Router>
    </ThemeProvider>
  );
};

export default App;
