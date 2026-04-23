import { useState, useEffect } from 'react'
import { GraduationCap, ShieldCheck, Eye, EyeOff, ArrowLeft, Mail, Lock, User, Loader2, AlertCircle, Building2, Network } from 'lucide-react'
import { loginUser, registerUser } from '../services/api'

const ROLES = [
  { id: 'teacher', label: 'Teacher', icon: GraduationCap, type: 'brand' },
  { id: 'hod', label: 'Admin (HOD)', icon: ShieldCheck, type: 'accent' },
]

export default function LoginPage({ onLogin, onBack }) {
  const [isSignUp, setIsSignUp] = useState(false)
  const [selectedRole, setSelectedRole] = useState('teacher')
  const [formData, setFormData] = useState({ name: '', email: '', password: '', university: 'FAST NUCES', department: 'Computer Science' })
  const [isAuthenticating, setIsAuthenticating] = useState(false)
  const [errorMsg, setErrorMsg] = useState(null)
  const [showPassword, setShowPassword] = useState(false)

  // Clear errors when switching tabs
  useEffect(() => {
    setErrorMsg(null)
  }, [isSignUp])

  const handleSubmit = async (e) => {
    e.preventDefault()
    setErrorMsg(null)
    setIsAuthenticating(true)
    
    try {
      let response;
      if (isSignUp) {
        response = await registerUser(formData.name, formData.email, formData.password, selectedRole, formData.university, formData.department)
      } else {
        response = await loginUser(formData.email, formData.password)
      }
      
      // Artificial delay for smooth UX transition as requested
      await new Promise(r => setTimeout(r, 800))
      
      // Route the user based on the role verified by the backend database
      const verifiedRole = response?.role?.toLowerCase() || (isSignUp ? selectedRole : 'teacher')
      onLogin(verifiedRole, response?.access_token, response?.full_name || formData.name)
    } catch (err) {
      setErrorMsg(err.message)
    } finally {
      setIsAuthenticating(false)
    }
  }

  return (
    <div className="hero-section min-h-screen flex items-center justify-center p-4 relative">
      <div className="hero-grid" />
      <div className="hero-blob-1" />
      <div className="hero-blob-2" />

      <div className="relative w-full max-w-md z-10 animate-fade-up">
        {onBack && (
          <button onClick={onBack} className="mb-6 flex items-center gap-2 text-sm font-medium transition-colors cursor-pointer hover:text-[var(--text-primary)]" style={{ color: 'var(--text-secondary)' }}>
            <ArrowLeft size={16} /> Back to Home
          </button>
        )}

        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl mb-4 shadow-lg"
               style={{ background: 'linear-gradient(135deg, var(--brand-primary), var(--brand-dark))', boxShadow: 'var(--shadow-glow)' }}>
            <Eye size={24} className="text-white" />
          </div>
          <h1 className="text-2xl font-black font-display mb-1">
            <span style={{ color: 'var(--text-primary)' }}>Classroom</span><span className="gradient-text">Eye</span>
          </h1>
        </div>

        <div className="glass rounded-[2rem] p-8 shadow-2xl">
          {/* Toggle Sign In / Sign Up */}
          <div className="flex p-1 rounded-xl mb-6" style={{ background: 'var(--bg-input)' }}>
             <button onClick={() => setIsSignUp(false)} className={`flex-1 py-2 text-sm font-bold rounded-lg transition-all ${!isSignUp ? 'bg-[var(--bg-surface)] shadow-sm' : 'opacity-50'}`} style={{ color: 'var(--text-primary)' }}>Sign In</button>
             <button onClick={() => setIsSignUp(true)} className={`flex-1 py-2 text-sm font-bold rounded-lg transition-all ${isSignUp ? 'bg-[var(--bg-surface)] shadow-sm' : 'opacity-50'}`} style={{ color: 'var(--text-primary)' }}>Create Account</button>
          </div>

          <h2 className="text-xl font-black mb-6 text-center" style={{ color: 'var(--text-primary)', fontFamily: "'Plus Jakarta Sans', sans-serif" }}>
             {isSignUp ? 'Create your account' : 'Welcome back'}
          </h2>

          <form onSubmit={handleSubmit} className="space-y-4 mb-6">
            {isSignUp && (
              <>
                <div className="grid grid-cols-2 gap-3 mb-2">
                  {ROLES.map(role => {
                    const isSelected = selectedRole === role.id
                    const colorBase = role.type === 'brand' ? 'var(--brand-primary)' : 'var(--accent-primary)'
                    return (
                      <button key={role.id} type="button" onClick={() => setSelectedRole(role.id)}
                              className={`flex items-center justify-center gap-2 py-3 rounded-xl border text-sm font-bold transition-all ${isSelected ? 'shadow-sm' : ''}`}
                              style={{ 
                                background: isSelected ? (role.type === 'brand' ? 'var(--brand-bg)' : 'var(--accent-bg)') : 'var(--bg-input)', 
                                borderColor: isSelected ? colorBase : 'transparent',
                                color: isSelected ? colorBase : 'var(--text-muted)'
                              }}>
                        <role.icon size={16} /> {role.label}
                      </button>
                    )
                  })}
                </div>
                <div className="relative animate-fade-up">
                  <User size={18} className="absolute left-4 top-1/2 -translate-y-1/2" style={{ color: 'var(--text-muted)' }}/>
                  <input type="text" placeholder="Full Name" required value={formData.name} onChange={e => setFormData({...formData, name: e.target.value})} className="w-full pl-11 pr-4 py-3 rounded-xl border text-sm font-medium focus:outline-none focus:ring-2 focus:ring-[var(--brand-primary)] transition-all" style={{ background: 'var(--bg-input)', borderColor: 'var(--border-color)', color: 'var(--text-primary)' }} />
                </div>
                
                <div className="relative animate-fade-up" style={{ animationDelay: '100ms' }}>
                  <Building2 size={18} className="absolute left-4 top-1/2 -translate-y-1/2" style={{ color: 'var(--text-muted)' }}/>
                  <select value={formData.university} onChange={e => setFormData({...formData, university: e.target.value})} className="w-full pl-11 pr-10 py-3 rounded-xl border text-sm font-medium appearance-none focus:outline-none focus:ring-2 focus:ring-[var(--brand-primary)] transition-all" style={{ background: 'var(--bg-input)', borderColor: 'var(--border-color)', color: 'var(--text-primary)' }}>
                    <option value="FAST NUCES" style={{ background: 'var(--bg-dropdown)' }}>FAST NUCES</option>
                    <option value="NUST" style={{ background: 'var(--bg-dropdown)' }}>NUST</option>
                    <option value="LUMS" style={{ background: 'var(--bg-dropdown)' }}>LUMS</option>
                  </select>
                </div>
                
                <div className="relative animate-fade-up" style={{ animationDelay: '150ms' }}>
                  <Network size={18} className="absolute left-4 top-1/2 -translate-y-1/2" style={{ color: 'var(--text-muted)' }}/>
                  <select value={formData.department} onChange={e => setFormData({...formData, department: e.target.value})} className="w-full pl-11 pr-10 py-3 rounded-xl border text-sm font-medium appearance-none focus:outline-none focus:ring-2 focus:ring-[var(--brand-primary)] transition-all" style={{ background: 'var(--bg-input)', borderColor: 'var(--border-color)', color: 'var(--text-primary)' }}>
                    <option value="Computer Science" style={{ background: 'var(--bg-dropdown)' }}>Computer Science</option>
                    <option value="Electrical Engineering" style={{ background: 'var(--bg-dropdown)' }}>Electrical Engineering</option>
                    <option value="BBA" style={{ background: 'var(--bg-dropdown)' }}>BBA</option>
                  </select>
                </div>
              </>
            )}
            
            <div className="relative">
              <Mail size={18} className="absolute left-4 top-1/2 -translate-y-1/2" style={{ color: 'var(--text-muted)' }}/>
              <input type="email" placeholder="Email address" required value={formData.email} onChange={e => setFormData({...formData, email: e.target.value})} className="w-full pl-11 pr-4 py-3 rounded-xl border text-sm font-medium focus:outline-none focus:ring-2 focus:ring-[var(--brand-primary)] transition-all" style={{ background: 'var(--bg-input)', borderColor: 'var(--border-color)', color: 'var(--text-primary)' }} />
            </div>

            <div className="relative">
              <Lock size={18} className="absolute left-4 top-1/2 -translate-y-1/2" style={{ color: 'var(--text-muted)' }}/>
              <input 
                type={showPassword ? "text" : "password"} 
                placeholder="Password" required 
                value={formData.password} onChange={e => setFormData({...formData, password: e.target.value})} 
                className="w-full pl-11 pr-12 py-3 rounded-xl border text-sm font-medium focus:outline-none focus:ring-2 focus:ring-[var(--brand-primary)] transition-all" 
                style={{ background: 'var(--bg-input)', borderColor: 'var(--border-color)', color: 'var(--text-primary)' }} 
              />
              <button 
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-4 top-1/2 -translate-y-1/2 focus:outline-none transition-colors hover:text-[var(--text-primary)]"
                style={{ color: 'var(--text-muted)' }}
                title={showPassword ? 'Hide password' : 'Show password'}
              >
                {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
              </button>
            </div>

            {errorMsg && (
              <div className="flex items-center gap-2 text-red-500 text-xs font-bold bg-red-500/10 p-3 rounded-lg animate-fade-up">
                 <AlertCircle size={14} /> {errorMsg}
              </div>
            )}

            <button type="submit" disabled={isAuthenticating} className="w-full py-3.5 flex items-center justify-center gap-2 rounded-xl font-bold text-sm transition-all shadow-lg text-white mt-2 cursor-pointer disabled:opacity-70 disabled:cursor-not-allowed"
                    style={{ background: 'linear-gradient(90deg, var(--brand-dark), var(--brand-light))' }}>
              {isAuthenticating ? <Loader2 size={18} className="animate-spin" /> : (isSignUp ? 'Create Account' : 'Sign In')}
            </button>
          </form>



        </div>
      </div>
    </div>
  )
}
