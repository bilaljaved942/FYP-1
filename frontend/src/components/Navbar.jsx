/**
 * Navbar — Top Navigation Bar Component
 * ========================================
 * Persistent navigation bar displayed on Teacher and HOD dashboards.
 *
 * Features:
 *   - ClassroomEye logo with gradient branding
 *   - Role badge showing the authenticated user's name (or role fallback)
 *   - Dark/Light theme toggle switch
 *   - Logout button with hover animation
 *
 * Props:
 *   @param {string}   title    - Subtitle displayed below the logo (e.g., "Teacher Dashboard")
 *   @param {string}   role     - 'teacher' or 'hod' — controls badge color scheme
 *   @param {function} onLogout - Callback fired when the logout button is clicked
 *   @param {string}   userName - Authenticated user's full name for the badge display
 */

import { LogOut, Eye, Bell, Sun, Moon } from 'lucide-react'
import { useTheme } from '../context/ThemeContext'

/**
 * Role-based visual configuration for the user badge.
 * - Teacher: Green (brand) colors
 * - HOD: Purple (accent) colors
 */
const roleConfig = {
  teacher: {
    label: 'Teacher',
    style: {
      background: 'var(--brand-bg)',
      border: '1px solid var(--brand-border)',
      color: 'var(--brand-primary)',
    },
    dot: { background: 'var(--brand-primary)' },
  },
  hod: {
    label: 'HOD',
    style: {
      background: 'var(--accent-bg)',
      border: '1px solid var(--accent-border)',
      color: 'var(--accent-primary)',
    },
    dot: { background: 'var(--accent-primary)' },
  },
}

/**
 * ThemeToggle — Small toggle switch for dark/light mode.
 * Uses the global ThemeContext to read and switch themes.
 */
function ThemeToggle() {
  const { theme, toggle } = useTheme()
  const isDark = theme === 'dark'
  return (
    <button
      onClick={toggle}
      id="theme-toggle"
      title={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
      className="theme-toggle"
      style={{
        background: isDark ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.07)',
      }}
    >
      <div className="theme-toggle-knob" />
      {/* Moon icon (visible in dark mode) */}
      <Moon
        size={10}
        style={{
          position: 'absolute', left: 5, top: '50%', transform: 'translateY(-50%)',
          color: isDark ? '#94a3b8' : 'transparent', transition: 'color 0.3s',
        }}
      />
      {/* Sun icon (visible in light mode) */}
      <Sun
        size={10}
        style={{
          position: 'absolute', right: 5, top: '50%', transform: 'translateY(-50%)',
          color: !isDark ? 'var(--brand-primary)' : 'transparent', transition: 'color 0.3s',
        }}
      />
    </button>
  )
}

export default function Navbar({ title, role, onLogout, userName }) {
  const rc = roleConfig[role] || roleConfig.teacher

  return (
    <nav className="nav-surface sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">

          {/* Logo + App Title */}
          <div className="flex items-center gap-3">
            <div
              className="w-9 h-9 rounded-xl flex items-center justify-center shadow-md"
              style={{ background: 'linear-gradient(135deg, var(--brand-primary), var(--brand-dark))', boxShadow: 'var(--shadow-glow-sm)' }}
            >
              <Eye size={17} className="text-white" />
            </div>
            <div>
              <div className="flex items-baseline gap-0.5">
                <span className="font-bold text-base leading-none" style={{ color: 'var(--text-primary)', fontFamily: "'Plus Jakarta Sans', sans-serif" }}>Classroom</span>
                <span className="font-bold text-base leading-none gradient-text" style={{ fontFamily: "'Plus Jakarta Sans', sans-serif" }}>Eye</span>
              </div>
              {/* Dashboard title shown as subtitle */}
              <span className="text-[10px] font-medium" style={{ color: 'var(--text-muted)' }}>{title}</span>
            </div>
          </div>

          {/* Right side: Role Badge + Theme Toggle + Logout */}
          <div className="flex items-center gap-2">
            {/* Role badge — shows user name or falls back to role label */}
            <div
              className="hidden sm:flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-bold"
              style={rc.style}
            >
              {/* Animated status dot */}
              <div className="w-1.5 h-1.5 rounded-full animate-pulse" style={rc.dot} />
              {userName || rc.label}
            </div>

            {/* Dark/Light mode toggle */}
            <ThemeToggle />

            {/* Logout button with hover effect (red on hover) */}
            <button
              onClick={onLogout}
              id="navbar-logout"
              className="flex items-center gap-2 px-3.5 py-2 rounded-xl btn-outline text-sm font-semibold"
              style={{ color: 'var(--text-secondary)' }}
              onMouseEnter={e => { e.currentTarget.style.color = '#f43f5e'; e.currentTarget.style.borderColor = 'rgba(244,63,94,0.4)' }}
              onMouseLeave={e => { e.currentTarget.style.color = 'var(--text-secondary)'; e.currentTarget.style.borderColor = 'var(--border-color)' }}
            >
              <LogOut size={14} />
              <span className="hidden sm:inline">Logout</span>
            </button>
          </div>
        </div>
      </div>
    </nav>
  )
}
