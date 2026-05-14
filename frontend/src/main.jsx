/**
 * main.jsx — Application Entry Point
 * =====================================
 * Mounts the React application into the DOM.
 *
 * Wraps the App component with:
 *   - StrictMode:     Enables additional development warnings and checks.
 *   - ThemeProvider:   Provides dark/light theme context to all child components.
 *
 * The root element (#root) is defined in index.html.
 */

import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import { ThemeProvider } from './context/ThemeContext.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ThemeProvider>
      <App />
    </ThemeProvider>
  </StrictMode>,
)
