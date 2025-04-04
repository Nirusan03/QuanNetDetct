import React from 'react';

const PageWrapper = ({ children }) => (
  <div style={{
    padding: '2rem',
    minHeight: '100vh',
    backgroundColor: '#121212'
  }}>
    <div style={{ width: '100%' }}>
      {children}
    </div>
  </div>
);

export default PageWrapper;
