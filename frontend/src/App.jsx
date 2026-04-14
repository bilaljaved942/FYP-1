import { useState } from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import LandingPage from './pages/LandingPage'
import LoginPage from './pages/LoginPage'
import TeacherDashboard from './pages/TeacherDashboard'
import HODDashboard from './pages/HODDashboard'

export default function App() {
  const [view, setView] = useState('landing') // 'landing' | 'login' | 'teacher' | 'hod'

  const handleEnterApp = () => setView('login')
  const handleBack     = () => setView('landing')
  const handleLogin    = (role) => setView(role)
  const handleLogout   = () => setView('landing')

  if (view === 'teacher') return <TeacherDashboard onLogout={handleLogout} />
  if (view === 'hod')     return <HODDashboard     onLogout={handleLogout} />
  if (view === 'login')   return <LoginPage onLogin={handleLogin} onBack={handleBack} />
  return <LandingPage onEnterApp={handleEnterApp} />
}
