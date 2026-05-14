/**
 * ThemeContext — Dark/Light Mode Manager
 * ========================================
 * Provides a global theme context to the entire application.
 *
 * Features:
 *   - Persists the user's theme preference in localStorage (key: 'ce-theme').
 *   - Applies the theme by setting a `data-theme` attribute on <html>.
 *   - CSS variables in index.css respond to `[data-theme="dark"]` / `[data-theme="light"]`.
 *   - Defaults to dark theme if no preference is stored.
 *
 * Usage:
 *   // In any component:
 *   const { theme, toggle } = useTheme()
 *   // theme = 'dark' | 'light'
 *   // toggle() switches between them
 */

import { createContext, useContext, useEffect, useState } from 'react'

// Create context with sensible defaults (prevents errors if used outside provider)
const ThemeContext = createContext({ theme: 'dark', toggle: () => {} })

/**
 * ThemeProvider — Wrap the app with this to enable theme switching.
 *
 * On mount, reads the saved theme from localStorage.
 * On change, updates both the DOM attribute and localStorage.
 */
export function ThemeProvider({ children }) {
  // Initialize from localStorage, defaulting to 'dark'
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem('ce-theme') || 'dark'
  })

  // Sync theme to the DOM and localStorage whenever it changes
  useEffect(() => {
    const root = document.documentElement
    root.setAttribute('data-theme', theme)
    localStorage.setItem('ce-theme', theme)
  }, [theme])

  // Toggle between dark and light
  const toggle = () => setTheme(t => t === 'dark' ? 'light' : 'dark')

  return (
    <ThemeContext.Provider value={{ theme, toggle }}>
      {children}
    </ThemeContext.Provider>
  )
}

/**
 * useTheme — Hook to access the current theme and toggle function.
 * @returns {{ theme: 'dark'|'light', toggle: () => void }}
 */
export const useTheme = () => useContext(ThemeContext)
