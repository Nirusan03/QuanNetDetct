import React, { useState } from "react";
import axios from "axios";
import { Container, Card, CardContent, Button, Typography, Alert, CircularProgress } from "@mui/material";

const App = () => {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleSubmit = async () => {
    if (!file) {
      setError("Please upload a CSV file.");
      return;
    }
    
    setLoading(true);
    setError(null);
    setResult(null);
    
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
  };

  return (
    <Container maxWidth="sm" style={{ marginTop: "50px" }}>
      <Card elevation={3}>
        <CardContent>
          <Typography variant="h5" gutterBottom>
            Quantum Neural Network TLS Traffic Classification
          </Typography>
          
          <input type="file" accept=".csv" onChange={handleFileChange} style={{ marginBottom: "15px" }} />
          
          <Button variant="contained" color="primary" onClick={handleSubmit} fullWidth disabled={loading}>
            {loading ? <CircularProgress size={24} /> : "Upload & Predict"}
          </Button>
          
          {/* Display Prediction Result */}
          {result && (
            <Alert severity="success" style={{ marginTop: "20px" }}>
              Predictions: <pre>{JSON.stringify(result, null, 2)}</pre>
            </Alert>
          )}

          {/* Display Error Message */}
          {error && (
            <Alert severity="error" style={{ marginTop: "20px" }}>
              {error}
            </Alert>
          )}
        </CardContent>
      </Card>
    </Container>
  );
};

export default App;
