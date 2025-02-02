import React, { useState } from "react";
import axios from "axios";
import { Container, Card, CardContent, Button, TextField, Typography, Alert } from "@mui/material";

const App = () => {
  const [features, setFeatures] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    setFeatures(e.target.value);
  };

  const handleSubmit = async () => {
    try {
      setError(null);
      setResult(null);

      // Convert input into an array of numbers
      const featuresArray = features.split(",").map(val => {
        const num = Number(val.trim());
        return isNaN(num) ? null : num;
      });

      // Validate: Ensure no NaN values and correct feature length
      if (featuresArray.includes(null)) {
        setError("Invalid input! Ensure all values are numbers.");
        return;
      }

      console.log("Sending Features:", featuresArray); // Debugging

      // Send request to Flask backend
      const response = await axios.post("http://127.0.0.1:5000/predict", {
        features: featuresArray,
      });

      setResult(response.data);
    } catch (err) {
      setError(`Error: ${err.response?.data?.error || "Server not reachable."}`);
    }
  };

  return (
    <Container maxWidth="sm" style={{ marginTop: "50px" }}>
      <Card elevation={3}>
        <CardContent>
          <Typography variant="h5" gutterBottom>
            Quantum Neural Network TLS Traffic Classification
          </Typography>
          
          <TextField
            label="Enter Features (comma-separated)"
            variant="outlined"
            fullWidth
            value={features}
            onChange={handleChange}
            margin="normal"
            placeholder="Example: 1.2, 3.5, 0.7, 5.1"
          />
          
          <Button variant="contained" color="primary" onClick={handleSubmit} fullWidth>
            Predict
          </Button>

          {/* Display Prediction Result */}
          {result && (
            <Alert severity="success" style={{ marginTop: "20px" }}>
              Prediction: <strong>{result.prediction}</strong>
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
