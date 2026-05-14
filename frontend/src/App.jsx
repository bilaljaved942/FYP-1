/**
 * App.jsx — Root Application Component
 * ======================================
 * Manages the top-level navigation state and authentication flow.
 *
 * Navigation Flow:
 *   1. Landing Page  → User clicks "Get Started"
 *   2. Login Page    → User registers or signs in
 *   3. Dashboard     → Routed to TeacherDashboard or HODDashboard based on role
 *
 * Authentication:
 *   - JWT token and user name are stored in localStorage after login/registration.
 *   - On logout, both are cleared and the user is returned to the landing page.
 *   - The `userName` prop is passed to dashboards for display in the Navbar.
 *
 * State:
 *   - view: Controls which page is rendered ('landing' | 'login' | 'teacher' | 'hod')
 */

import { useState } from 'react'
import LandingPage from './pages/LandingPage'
import LoginPage from './pages/LoginPage'
import TeacherDashboard from './pages/TeacherDashboard'
import HODDashboard from './pages/HODDashboard'

export default function App() {
  // Current view/page state — determines which component is rendered
  const [view, setView] = useState('landing') // 'landing' | 'login' | 'teacher' | 'hod'

  // Navigation handlers
  const handleEnterApp = () => setView('login')
  const handleBack     = () => setView('landing')

  /**
   * Called after successful login or registration.
   * Persists auth data to localStorage and routes to the appropriate dashboard.
   *
   * @param {string} role  - 'teacher' or 'hod' (determines which dashboard to show)
   * @param {string} token - JWT authentication token (used for API requests)
   * @param {string} name  - User's full name (displayed in the navbar)
   */
  const handleLogin = (role, token, name) => {
    if (token) localStorage.setItem('token', token)
    if (name)  localStorage.setItem('userName', name)
    setView(role)
  }

  /**
   * Called when the user clicks "Logout" in the navbar.
   * Clears all stored auth data and returns to the landing page.
   */
  const handleLogout = () => {
    localStorage.removeItem('token')
    localStorage.removeItem('userName')
    setView('landing')
  }

  // Read the persisted user name for passing to dashboard navbars
  const userName = localStorage.getItem('userName') || ''

  // Render the active view
  if (view === 'teacher') return <TeacherDashboard onLogout={handleLogout} userName={userName} />
  if (view === 'hod')     return <HODDashboard     onLogout={handleLogout} userName={userName} />
  if (view === 'login')   return <LoginPage onLogin={handleLogin} onBack={handleBack} />
  return <LandingPage onEnterApp={handleEnterApp} />
}
