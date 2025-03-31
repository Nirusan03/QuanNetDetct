import { createTheme } from '@mui/material/styles';

export const getTheme = (mode) =>
  createTheme({
    palette: {
      mode,
      ...(mode === 'dark'
        ? {
            background: {
              default: '#121212',
              paper: '#1e1e1e',
            },
            text: {
              primary: '#ffffff',
              secondary: '#aaaaaa',
            },
          }
        : {
            background: {
              default: '#fafafa',
              paper: '#ffffff',
            },
            text: {
              primary: '#000000',
              secondary: '#444444',
            },
          }),
    },
    typography: {
      fontFamily: 'Roboto, sans-serif',
    },
  });
