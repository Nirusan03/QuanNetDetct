import React, { useState } from "react";
import axios from "axios";
import {
  Container, Card, CardContent, Button, Typography, CircularProgress, 
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper, 
  Dialog, DialogActions, DialogContent, DialogTitle, List, ListItem, ListItemIcon, ListItemText
} from "@mui/material";
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import WarningIcon from '@mui/icons-material/Warning';
import SecurityIcon from '@mui/icons-material/Security';
import { styled } from "@mui/material/styles";
import { Bar } from "react-chartjs-2";
import { Chart, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from "chart.js";

Chart.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const UploadButton = styled(Button)({
  backgroundColor: "#1976d2",
  color: "#fff",
  fontWeight: "bold",
  width: "100%",
  '&:hover': {
    backgroundColor: "#115293"
  }
});

const StyledContainer = styled(Container)({
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  minHeight: "100vh",
  backgroundColor: "#e3f2fd",
  paddingBottom: "50px"
});

const StyledCard = styled(Card)({
  width: "90%",
  maxWidth: "1200px",
  padding: "20px",
  backgroundColor: "#f4f6f8",
  borderRadius: "12px",
  textAlign: "center",
  border: "2px solid #d1d1d1"
});

const StyledTableContainer = styled(TableContainer)({
  borderRadius: "10px",
  border: "2px solid #d1d1d1",
  backgroundColor: "#fff"
});

const ChartContainer = styled("div")({
  width: "100%",
  height: "400px",
  display: "flex",
  justifyContent: "center",
  alignItems: "center"
});

const App = () => {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [open, setOpen] = useState(false);
  const [chartData, setChartData] = useState({
    labels: [],
    datasets: [
      {
        label: "TLS Classification Confidence",
        data: [],
        backgroundColor: ["#007bff"],
      }
    ]
  });

  const handleFileChange = (e) => {
    if (e.target.files.length > 0) {
      const selectedFile = e.target.files[0];
  
      // Check if the file is a CSV
      if (!selectedFile.name.endsWith(".csv")) {
        setError("Invalid file type. Please upload a CSV file.");
        setFile(null);
        setOpen(false);
        return;
      }
  
      setFile(selectedFile);
      setError(null);
      setOpen(false);
    }
  };
  
  const handleSubmit = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
  
    if (!file) {
      setError("No file selected. Please upload a CSV file before proceeding.");
      setLoading(false);
      return;
    }
  
    const formData = new FormData();
    formData.append("file", file);
  
    try {
      const response = await axios.post("http://127.0.0.1:5000/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" }
      });
  
      setResult(response.data);
      updateChartData(response.data);
    } catch (err) {
      setError(`Error: ${err.response?.data?.error || "Server not reachable."}`);
    } finally {
      setLoading(false);
    }
  };
  

  const updateChartData = (data) => {
    if (!data || !data.predictions) return;

    const newLabels = data.predictions.map((_, index) => `Sample ${index + 1}`);
    const newData = data.predictions.map((item) => parseFloat(item.confidence.replace("%", "")));

    const maxBars = 10;
    const limitedLabels = newLabels.slice(-maxBars);
    const limitedData = newData.slice(-maxBars);

    setChartData({
      labels: limitedLabels,
      datasets: [
        {
          label: "TLS Classification Confidence",
          data: limitedData,
          backgroundColor: ["#007bff"],
          borderColor: ["#0056b3"],
          borderWidth: 1,
          barThickness: 30,
        }
      ]
    });
  };

  return (
    <StyledContainer>
      <StyledCard elevation={5}>
        <CardContent>
          <Typography variant="h4" gutterBottom style={{ color: "#1976d2", fontWeight: "bold" }}>
            Quantum Neural Network TLS Traffic Classification
          </Typography>

          <Button
            variant="contained"
            startIcon={<CloudUploadIcon />}
            onClick={() => setOpen(true)}
            style={{ marginBottom: "20px", backgroundColor: "#0288d1", width: "100%" }}
          >
            Choose File
          </Button>

          {error && (
            <Typography variant="body2" style={{ color: "red", marginTop: "10px" }}>
              {error}
            </Typography>
          )}

          <Dialog open={open} onClose={() => setOpen(false)}>
            <DialogTitle>Select a File</DialogTitle>
            <DialogContent>
              <input type="file" accept=".csv" onChange={handleFileChange} />
            </DialogContent>
            <DialogActions>
              <Button onClick={() => setOpen(false)}>Close</Button>
            </DialogActions>
          </Dialog>

          {file && <Typography variant="body1" style={{ marginBottom: "15px", color: "#388e3c", fontWeight: "bold" }}>Selected File: {file.name}</Typography>}

          <UploadButton variant="contained" onClick={handleSubmit} disabled={loading}>
            {loading ? <CircularProgress size={24} style={{ color: "#fff" }} /> : "Upload & Predict"}
          </UploadButton>
        </CardContent>
      </StyledCard>

      {result && (
        <StyledCard elevation={5} style={{ marginTop: "20px", padding: "15px", textAlign: "left" }}>
          <Typography variant="h5" gutterBottom style={{ color: "#1976d2", fontWeight: "bold" }}>
            API Response:
          </Typography>
          <pre style={{ backgroundColor: "#eef", padding: "10px", borderRadius: "5px", whiteSpace: "pre-wrap" }}>
            {JSON.stringify(result, null, 2)}
          </pre>
        </StyledCard>
      )}

      {result && (
        <StyledCard elevation={5} style={{ marginTop: "20px" }}>
          <StyledTableContainer component={Paper}>
            <Table>
              <TableHead>
                <TableRow style={{ backgroundColor: "#1976d2", color: "white" }}>
                  <TableCell style={{ color: "white" }}>Prediction</TableCell>
                  <TableCell style={{ color: "white" }}>Description</TableCell>
                  <TableCell style={{ color: "white" }}>Confidence</TableCell>
                  <TableCell style={{ color: "white" }}>Potential Risks</TableCell>
                  <TableCell style={{ color: "white" }}>Recommended Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {result.predictions.map((item, index) => (
                  <TableRow key={index}>
                    <TableCell>
                      <ListItemIcon>
                        {item.prediction.includes("Benign") ? <CheckCircleIcon style={{ color: "green" }} /> : <ErrorIcon style={{ color: "red" }} />}
                      </ListItemIcon>
                      {item.prediction}
                    </TableCell>
                    <TableCell>{item.description}</TableCell>
                    <TableCell>{item.confidence}</TableCell>
                    <TableCell>
                      <List>
                        {item.potential_risks.map((risk, idx) => (
                          <ListItem key={idx}><WarningIcon style={{ color: "orange" }} /> {risk}</ListItem>
                        ))}
                      </List>
                    </TableCell>
                    <TableCell>
                      <List>
                        {item.recommended_actions.map((action, idx) => (
                          <ListItem key={idx}><SecurityIcon style={{ color: "blue" }} /> {action}</ListItem>
                        ))}
                      </List>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </StyledTableContainer>
        </StyledCard>
      )}

      {result && (
        <StyledCard elevation={5} style={{ marginTop: "20px" }}>
          <Typography variant="h5">TLS Traffic Trends</Typography>
          <ChartContainer>
            <Bar data={chartData} options={{ responsive: true, maintainAspectRatio: false }} />
          </ChartContainer>
        </StyledCard>
      )}
    </StyledContainer>
  );
};

export default App;
