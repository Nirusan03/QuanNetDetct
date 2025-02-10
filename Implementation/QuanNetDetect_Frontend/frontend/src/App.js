import React, { useState } from "react";
import axios from "axios";
import { Container, Card, CardContent, Button, Typography, Alert, CircularProgress, Dialog, DialogActions, DialogContent, DialogTitle } from "@mui/material";
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import CloseIcon from '@mui/icons-material/Close';
import { styled } from "@mui/material/styles";

const UploadButton = styled(Button)({
  backgroundColor: "#1976d2",
  color: "#fff",
  fontWeight: "bold",
  '&:hover': {
    backgroundColor: "#115293"
  }
});

const StyledDialog = styled(Dialog)({
  "& .MuiDialog-paper": {
    borderRadius: "15px",
    padding: "20px",
    backgroundColor: "#f8f9fa"
  }
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

  return (
    <Container maxWidth="sm" style={{ marginTop: "50px", textAlign: "center" }}>
      <Card elevation={5} style={{ padding: "20px", backgroundColor: "#f4f6f8", borderRadius: "12px" }}>
        <CardContent>
          <Typography variant="h4" gutterBottom style={{ color: "#1976d2", fontWeight: "bold" }}>
            Quantum Neural Network TLS Traffic Classification
          </Typography>
          
          <Button variant="contained" startIcon={<CloudUploadIcon />} onClick={() => setOpen(true)} style={{ marginBottom: "20px", backgroundColor: "#0288d1" }}>
            Choose File
          </Button>
          
          <StyledDialog open={open} onClose={() => setOpen(false)}>
            <DialogTitle style={{ textAlign: "center", fontWeight: "bold" }}>Upload CSV File</DialogTitle>
            <DialogContent style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
              <input type="file" accept=".csv" onChange={handleFileChange} style={{ padding: "10px", border: "1px solid #ccc", borderRadius: "5px" }} />
            </DialogContent>
            <DialogActions style={{ justifyContent: "center" }}>
              <Button onClick={() => setOpen(false)} startIcon={<CloseIcon />} color="secondary" variant="contained">
                Cancel
              </Button>
            </DialogActions>
          </StyledDialog>
          
          {file && <Typography variant="body1" style={{ marginBottom: "15px", color: "#388e3c", fontWeight: "bold" }}>Selected File: {file.name}</Typography>}
          
          <UploadButton variant="contained" onClick={handleSubmit} fullWidth disabled={loading}>
            {loading ? <CircularProgress size={24} style={{ color: "#fff" }} /> : "Upload & Predict"}
          </UploadButton>
          
          {/* Display Prediction Result */}
          {result && (
            <Alert severity="success" style={{ marginTop: "20px", backgroundColor: "#e8f5e9", color: "#388e3c", fontWeight: "bold" }}>
              Predictions: <pre>{JSON.stringify(result, null, 2)}</pre>
            </Alert>
          )}

          {/* Display Error Message */}
          {error && (
            <Alert severity="error" style={{ marginTop: "20px", backgroundColor: "#ffebee", color: "#d32f2f", fontWeight: "bold" }}>
              {error}
            </Alert>
          )}
        </CardContent>
      </Card>
    </Container>
  );
};

export default App;
