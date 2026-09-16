import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import ErrorBoundary from './ErrorBoundary.jsx'

// The boundary sits INSIDE StrictMode but OUTSIDE App, which is the only
// placement that catches a throw from App's own render. Put it inside App and
// the component that crashes is the one meant to catch it.
createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </StrictMode>,
)
