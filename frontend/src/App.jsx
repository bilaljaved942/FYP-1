import { useState } from 'react'
import LandingPage from './pages/LandingPage'
import LoginPage from './pages/LoginPage'
import TeacherDashboard from './pages/TeacherDashboard'
import HODDashboard from './pages/HODDashboard'

export default function App() {
  const [view, setView] = useState('landing') // 'landing' | 'login' | 'teacher' | 'hod'

  const handleEnterApp = () => setView('login')
  const handleBack     = () => setView('landing')

  const handleLogin = (role, token, name) => {
    // Persist token and name so dashboards can read them
    if (token) localStorage.setItem('token', token)
    if (name)  localStorage.setItem('userName', name)
    setView(role)
  }

  const handleLogout = () => {
    localStorage.removeItem('token')
    localStorage.removeItem('userName')
    setView('landing')
  }

  const userName = localStorage.getItem('userName') || ''

  if (view === 'teacher') return <TeacherDashboard onLogout={handleLogout} userName={userName} />
  if (view === 'hod')     return <HODDashboard     onLogout={handleLogout} userName={userName} />
  if (view === 'login')   return <LoginPage onLogin={handleLogin} onBack={handleBack} />
  return <LandingPage onEnterApp={handleEnterApp} />
}
