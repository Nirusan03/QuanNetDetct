import React from 'react';
import { Typography, Paper, Box, Divider } from '@mui/material';
import PageWrapper from '../components/PageWrapper';

const Highlight = ({ children }) => (
  <span style={{ fontWeight: 600, color: '#1976d2' }}>{children}</span>
);

const ApiBox = ({ title, children }) => (
  <Box sx={{ mt: 4, mb: 4 }}>
    <Typography variant="h6" gutterBottom sx={{ color: '#90caf9' }}>
      {title}
    </Typography>
    <Paper variant="outlined" sx={{ padding: 2, backgroundColor: '#1e1e1e' }}>
      <Typography component="div" sx={{ whiteSpace: 'pre-wrap', fontSize: '0.95rem', color: '#e0e0e0' }}>
        {children}
      </Typography>
    </Paper>
  </Box>
);

const DocumentationPage = () => {
  return (
    <PageWrapper>
      <Box sx={{ maxWidth: '1800px', mx: 'auto', padding: '2rem' }}>
        <Typography variant="h4" gutterBottom>
          QuanNetDetect: <span style={{ color: '#64b5f6' }}>User Guide & API Flow</span>
        </Typography>

        <Typography sx={{ mb: 3 }}>
          This documentation helps you use the <strong>QuanNetDetect</strong> system — a hybrid quantum-classical malicious TLS traffic detection platform. Follow the instructions below to understand the system functionality and how to interact with its backend APIs.
        </Typography>

        <Divider sx={{ my: 2 }} />

        <Typography variant="h6" gutterBottom>Step-by-Step Dashboard Instructions:</Typography>

        <Box sx={{ mt: 3 }}>
          <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>1. Upload a PCAP File</Typography>
          <Typography sx={{ mt: 1 }}>
            Go to the <Highlight>"Upload PCAP"</Highlight> page and upload a <code>.pcap</code> file. You’ll be asked to select:
          </Typography>
          <ul style={{ marginTop: 8 }}>
            <li><strong>TLS Version</strong>: Choose TLS 1.2, TLS 1.3, or both</li>
            <li><strong>Mode</strong>: Auto uses all features; Custom allows selecting specific features</li>
            <li><strong>Record Limit</strong>: Max number of flows to extract</li>
          </ul>
          <Typography sx={{ mt: 1 }}>
            The file will be saved and simulated. You will receive a <strong>File ID</strong> for the next steps.
          </Typography>
        </Box>

        <Box sx={{ mt: 3 }}>
          <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>2. Simulate, Validate & Predict</Typography>
          <Typography sx={{ mt: 1 }}>Go to the <Highlight>"Simulate & Predict"</Highlight> page and enter your <code>File ID</code>.</Typography>
          <ul style={{ marginTop: 8 }}>
            <li><strong>Generate Simulated PCAP</strong>: Converts features into synthetic packet traffic</li>
            <li><strong>Preview Packets</strong>: Shows TCP/IP details like IPs, Ports, Flags, etc.</li>
            <li>
              <strong>Predict</strong>: Classifies flows using a QNN model. You’ll see:
              <ul style={{ marginTop: 4 }}>
                <li>Predicted class (e.g. <code>BENIGN</code>, <code>DrDoS_DNS</code>, <code>LDAP</code>)</li>
                <li>Probabilities for all classes</li>
                <li>Downloadable <code>CSV Report</code></li>
              </ul>
            </li>
          </ul>
        </Box>

        <Box sx={{ mt: 3 }}>
          <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>3. Visualize PCAP Traffic</Typography>
          <Typography sx={{ mt: 1 }}>Use the <Highlight>"Visualize Uploaded"</Highlight> and <Highlight>"Visualize Simulated"</Highlight> pages. Enter your <code>File ID</code> to see:</Typography>
          <ul style={{ marginTop: 8 }}>
            <li>Port distributions</li>
            <li>Protocol breakdown (TCP, UDP, etc.)</li>
            <li>Source/Destination IPs, packet lengths, and more</li>
          </ul>
        </Box>

        <Box sx={{ mt: 3 }}>
          <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>4. Review Past Reports</Typography>
          <Typography sx={{ mt: 1 }}>The <Highlight>"Reports"</Highlight> page lists all previous detections with:</Typography>
          <ul style={{ marginTop: 8 }}>
            <li><strong>File ID</strong></li>
            <li><strong>Created Timestamp</strong></li>
            <li><strong>Option to view or download results</strong></li>
          </ul>
        </Box>

        <Divider sx={{ my: 4 }} />

        <Typography variant="h6" gutterBottom>API Reference (Backend)</Typography>

        <ApiBox title="1. POST /upload-pcap">
          Upload a `.pcap` file and extract flow features. Returns a File ID.
          {"\n\nForm Data (Auto):"}
          {`\n  file: <yourfile.pcap>\n  metadata: {
  "tls_version": "1",
  "mode": "auto",
  "record_limit": 50
}`}
          {"\n\nForm Data (Custom):"}
          {`\n  file: <yourfile.pcap>\n  metadata: {
  "tls_version": "3",
  "mode": "custom",
  "record_limit": 100,
  "custom_features": {
    "Flow Duration": 0.1,
    "Source Port": 443
  }
}`}
        </ApiBox>

        <ApiBox title="2. POST /generate-pcap">
          Generates a simulated `.pcap` file from extracted features.
          {"\nPayload: { \"file_id\": \"<your_file_id>\" }"}
        </ApiBox>

        <ApiBox title="3. POST /validate-pcap (Optional)">
          Preview the first 100 packets of the generated `.pcap`.
          {"\nPayload: { \"file_id\": \"<your_file_id>\" }"}
        </ApiBox>

        <ApiBox title="4. POST /predict">
          Runs the QNN model to classify traffic and generate a report.
          {"\nPayload: { \"file_id\": \"<your_file_id>\" }"}
        </ApiBox>

        <ApiBox title="5. GET /download-report/<file_id>">
          Downloads the final CSV report from the model predictions.
        </ApiBox>

        <ApiBox title="6. GET /get-report/<file_id>">
          Fetches the full result with class probabilities from MongoDB.
        </ApiBox>

        <ApiBox title="7. GET /list-reports">
          Lists all scan reports with timestamps and File IDs.
        </ApiBox>

        <Divider sx={{ mt: 4, mb: 1 }} />

        <Typography variant="body2" sx={{ mt: 2 }}>
          <strong>Tip:</strong> Always keep the <code>File ID</code> returned from the upload step. It’s required for all other operations including simulation, prediction, and report download.
        </Typography>
      </Box>
    </PageWrapper>
  );
};

export default DocumentationPage;
