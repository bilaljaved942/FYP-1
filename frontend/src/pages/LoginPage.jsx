import { useState } from 'react'
import { GraduationCap, ShieldCheck, Eye, ArrowLeft, Sparkles } from 'lucide-react'

const ROLES = [
  {
    id: 'teacher',
    label: 'Teacher',
    subtitle: 'Upload videos & analyze per-student engagement',
    icon: GraduationCap,
    type: 'brand',
    description: 'Access video upload tools, per-student emotion & action dashboards, and session reports.',
  },
  {
    id: 'hod',
    label: 'Head of Department',
    subtitle: 'Department-wide analytics & oversight',
    icon: ShieldCheck,
    type: 'accent',
    description: 'View department trends, teacher performance, and at-risk student alerts.',
  },
]

export default function LoginPage({ onLogin, onBack }) {
  const [selectedRole, setSelectedRole] = useState(null)
  const selected = ROLES.find(r => r.id === selectedRole)

  return (
    <div className="hero-section min-h-screen flex items-center justify-center p-4 relative">
      <div className="hero-grid" />
      <div className="hero-blob-1" />
      <div className="hero-blob-2" />

      <div className="relative w-full max-w-lg z-10 animate-fade-up">
        {onBack && (
          <button onClick={onBack} className="mb-6 flex items-center gap-2 text-sm font-medium transition-colors cursor-pointer" style={{ color: 'var(--text-secondary)' }}>
            <ArrowLeft size={16} /> Back to Home
          </button>
        )}

        <div className="text-center mb-10">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl mb-5 shadow-lg"
               style={{ background: 'linear-gradient(135deg, var(--brand-primary), var(--brand-dark))', boxShadow: 'var(--shadow-glow)' }}>
            <Eye size={28} className="text-white" />
          </div>
          <h1 className="text-3xl font-black font-display mb-1">
            <span style={{ color: 'var(--text-primary)' }}>Classroom</span><span className="gradient-text">Eye</span>
          </h1>
          <p className="text-sm font-medium" style={{ color: 'var(--text-muted)' }}>AI-Powered Classroom Engagement Analytics</p>
        </div>

        <div className="glass rounded-[2rem] p-8 md:p-10 shadow-2xl">
          <div className="flex items-center gap-2 mb-2">
            <Sparkles size={14} style={{ color: 'var(--brand-primary)' }} />
            <span className="text-xs font-bold uppercase tracking-wider" style={{ color: 'var(--brand-primary)' }}>Select Role</span>
          </div>
          <h2 className="text-2xl font-black mb-2" style={{ color: 'var(--text-primary)', fontFamily: "'Plus Jakarta Sans', sans-serif" }}>Welcome Back</h2>
          <p className="text-sm mb-8" style={{ color: 'var(--text-muted)' }}>Choose your role to access your personalized dashboard</p>

          <div className="space-y-4 mb-8">
            {ROLES.map(role => {
              const isSelected = selectedRole === role.id
              const isBrand = role.type === 'brand'
              const colorBase = isBrand ? 'var(--brand-primary)' : 'var(--accent-primary)'
              const bgBase = isBrand ? 'var(--brand-bg)' : 'var(--accent-bg)'
              const borderBase = isBrand ? 'var(--brand-border)' : 'var(--accent-border)'

              return (
                <button
                  key={role.id}
                  onClick={() => setSelectedRole(role.id)}
                  className="w-full text-left flex items-start gap-4 p-5 rounded-2xl transition-all duration-200 group cursor-pointer border"
                  style={{
                    background: isSelected ? bgBase : 'var(--bg-input)',
                    borderColor: isSelected ? borderBase : 'var(--border-color)',
                  }}
                >
                  <div className="w-12 h-12 rounded-xl flex items-center justify-center flex-shrink-0 transition-transform duration-200 group-hover:scale-110"
                       style={{ background: bgBase, color: colorBase }}>
                    <role.icon size={22} />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="font-bold text-base" style={{ color: 'var(--text-primary)' }}>{role.label}</span>
                      {isSelected && <span className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ background: colorBase }} />}
                    </div>
                    <p className="text-xs leading-relaxed" style={{ color: 'var(--text-muted)' }}>{role.description}</p>
                  </div>
                  <div className="w-5 h-5 rounded-full border-2 flex items-center justify-center flex-shrink-0 mt-1 transition-all"
                       style={{ borderColor: isSelected ? colorBase : 'var(--border-color)' }}>
                    {isSelected && <div className="w-2.5 h-2.5 rounded-full" style={{ background: colorBase }} />}
                  </div>
                </button>
              )
            })}
          </div>

          <button
            onClick={() => selectedRole && onLogin(selectedRole)}
            disabled={!selectedRole}
            className="w-full py-4 rounded-xl font-bold text-base transition-all duration-200 shadow-lg text-white"
            style={{
              background: !selectedRole ? 'var(--bg-input)' : (selected?.type === 'brand' ? 'linear-gradient(90deg, var(--brand-dark), var(--brand-light))' : 'linear-gradient(90deg, var(--accent-primary), var(--accent-light))'),
              opacity: !selectedRole ? 0.5 : 1,
              cursor: !selectedRole ? 'not-allowed' : 'pointer'
            }}
          >
            {selectedRole ? `Continue as ${selected?.label}` : 'Continue'}
          </button>
        </div>
      </div>
    </div>
  )
}
