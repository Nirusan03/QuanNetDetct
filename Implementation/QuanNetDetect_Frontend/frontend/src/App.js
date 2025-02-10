import React, { useState } from "react";
import axios from "axios";
import { Container, Card, CardContent, Button, Typography, Alert, CircularProgress, Dialog, DialogActions, DialogContent, DialogTitle, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper } from "@mui/material";
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import WarningIcon from '@mui/icons-material/Warning';
import BarChartIcon from '@mui/icons-material/BarChart';
import { styled } from "@mui/material/styles";

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
});

const AnalysisSection = styled("div")({
  width: "90%",
  maxWidth: "1200px",
  marginTop: "30px",
  padding: "20px",
  backgroundColor: "#ffffff",
  borderRadius: "12px",
  textAlign: "center",
  boxShadow: "0px 4px 10px rgba(0, 0, 0, 0.1)"
});

const App = () => {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [open, setOpen] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setOpen(false);
  };

  const handleSubmit = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    
    if (file) {
      const formData = new FormData();
      formData.append("file", file);
      
      try {
        const response = await axios.post("http://127.0.0.1:5000/predict", formData, {
          headers: {
            "Content-Type": "multipart/form-data"
          }
        });
        
        setResult(response.data);
      } catch (err) {
        setError(`Error: ${err.response?.data?.error || "Server not reachable."}`);
      } finally {
        setLoading(false);
      }
    } else {
      setError("Please upload a CSV file.");
      setLoading(false);
    }
  };

  const getIcon = (prediction) => {
    switch (prediction) {
      case "Benign TLS Traffic":
        return <CheckCircleIcon style={{ color: "green" }} />;
      case "Malicious TLS Traffic":
        return <ErrorIcon style={{ color: "red" }} />;
      case "Uncertain TLS Traffic":
        return <WarningIcon style={{ color: "orange" }} />;
      default:
        return null;
    }
  };

  return (
    <StyledContainer>
      <StyledCard elevation={5}>
        <CardContent>
          <Typography variant="h4" gutterBottom style={{ color: "#1976d2", fontWeight: "bold" }}>
            Quantum Neural Network TLS Traffic Classification
          </Typography>
          
          <Button variant="contained" startIcon={<CloudUploadIcon />} onClick={() => setOpen(true)} style={{ marginBottom: "20px", backgroundColor: "#0288d1", width: "100%" }}>
            Choose File
          </Button>
          
          {file && <Typography variant="body1" style={{ marginBottom: "15px", color: "#388e3c", fontWeight: "bold" }}>Selected File: {file.name}</Typography>}
          
          <UploadButton variant="contained" onClick={handleSubmit} disabled={loading}>
            {loading ? <CircularProgress size={24} style={{ color: "#fff" }} /> : "Upload & Predict"}
          </UploadButton>
          
          {result && (
            <>
              <TableContainer component={Paper} style={{ marginTop: "20px" }}>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Prediction</TableCell>
                      <TableCell>Description</TableCell>
                      <TableCell>Confidence</TableCell>
                      <TableCell>Potential Risks</TableCell>
                      <TableCell>Recommended Actions</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {result.predictions.map((item, index) => (
                      <TableRow key={index}>
                        <TableCell>{getIcon(item.prediction)} {item.prediction}</TableCell>
                        <TableCell>{item.description}</TableCell>
                        <TableCell>{item.confidence}</TableCell>
                        <TableCell>
                          <ul>
                            {item.potential_risks.map((risk, i) => (
                              <li key={i}>{risk}</li>
                            ))}
                          </ul>
                        </TableCell>
                        <TableCell>
                          <ul>
                            {item.recommended_actions.map((action, i) => (
                              <li key={i}>{action}</li>
                            ))}
                          </ul>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
              
              <AnalysisSection>
                <Typography variant="h5" style={{ fontWeight: "bold", color: "#1565c0" }}>
                  Traffic Analysis Summary <BarChartIcon />
                </Typography>
                <Typography variant="body1" style={{ marginTop: "10px" }}>
                  - Total Samples Processed: {result.predictions.length}
                </Typography>
                <Typography variant="body1">
                  - Malicious Samples Detected: {result.predictions.filter(p => p.prediction === "Malicious TLS Traffic").length}
                </Typography>
                <Typography variant="body1">
                  - Benign Samples: {result.predictions.filter(p => p.prediction === "Benign TLS Traffic").length}
                </Typography>
              </AnalysisSection>
            </>
          )}
        </CardContent>
      </StyledCard>
    </StyledContainer>
  );
};

export default App;