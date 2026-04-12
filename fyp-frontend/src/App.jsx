import { useState } from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import LoginPage from './pages/LoginPage'
import TeacherDashboard from './pages/TeacherDashboard'
import HODDashboard from './pages/HODDashboard'

export default function App() {
  const [role, setRole] = useState(null)

  const handleLogin = (selectedRole) => {
    setRole(selectedRole)
  }

  const handleLogout = () => {
    setRole(null)
  }

  return (
    <BrowserRouter>
      <Routes>
        <Route
          path="/"
          element={
            role === 'teacher' ? (
              <Navigate to="/teacher" replace />
            ) : role === 'hod' ? (
              <Navigate to="/hod" replace />
            ) : (
              <LoginPage onLogin={handleLogin} />
            )
          }
        />
        <Route
          path="/teacher"
          element={
            role === 'teacher' ? (
              <TeacherDashboard onLogout={handleLogout} />
            ) : (
              <Navigate to="/" replace />
            )
          }
        />
        <Route
          path="/hod"
          element={
            role === 'hod' ? (
              <HODDashboard onLogout={handleLogout} />
            ) : (
              <Navigate to="/" replace />
            )
          }
        />
      </Routes>
    </BrowserRouter>
  )
}
