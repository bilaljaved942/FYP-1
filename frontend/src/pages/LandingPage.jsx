/**
 * LandingPage — Marketing & Introduction Page
 * ===============================================
 * The first page users see when visiting ClassroomEye.
 * Showcases the platform's features, role-based dashboards, and technology.
 *
 * Sections:
 *   1. Hero — Main headline, tagline, and CTA buttons
 *   2. Features — 6 feature cards (Video Analysis, Emotion Recognition, etc.)
 *   3. Role Portals — Teacher vs HOD dashboard previews
 *   4. Footer — Branding and project attribution
 *
 * Props:
 *   @param {function} onEnterApp - Navigates to the login page
 */
import { useState, useEffect, useRef } from 'react'
import {
  Eye, Brain, TrendingUp, Users, Upload, BarChart3,
  Play, Shield, Zap, CheckCircle, GraduationCap,
  Activity, ArrowRight, Sparkles, Camera, LineChart,
  Moon, Sun, Menu, X
} from 'lucide-react'
import { useTheme } from '../context/ThemeContext'

/** Theme toggle button for the landing page header (dark/light mode). */
function HeaderThemeToggle() {
  const { theme, toggle } = useTheme()
  const isDark = theme === 'dark'
  return (
    <button onClick={toggle} className="theme-toggle" title={isDark ? 'Switch to light mode' : 'Switch to dark mode'}>
      <div className="theme-toggle-knob shadow-sm" />
      <Moon size={11} style={{ position: 'absolute', left: 5, top: 4, color: isDark ? '#94a3b8' : 'transparent', transition: 'color 0.3s' }} />
      <Sun  size={11} style={{ position: 'absolute', right: 5, top: 4, color: !isDark ? 'var(--brand-primary)' : 'transparent', transition: 'color 0.3s' }} />
    </button>
  )
}

/** Reusable feature card component with animated icon and hover effects. */
function FeatureCard({ icon: Icon, title, description, badgeColor, delay = 0 }) {
  const isAccent = badgeColor === 'indigo'
  return (
    <div className="glass p-8 group animate-fade-up" style={{ animationDelay: `${delay}ms` }}>
      <div className="w-14 h-14 rounded-2xl flex items-center justify-center mb-6 transition-transform duration-300 group-hover:scale-110"
           style={{ background: isAccent ? 'var(--accent-bg)' : 'var(--brand-bg)' }}>
        <Icon size={26} style={{ color: isAccent ? 'var(--accent-primary)' : 'var(--brand-primary)' }} />
      </div>
      <h3 className="font-bold text-xl mb-3 text-primary">{title}</h3>
      <p className="text-sm leading-relaxed text-secondary">{description}</p>
    </div>
  )
}

export default function LandingPage({ onEnterApp }) {
  const [scrolled, setScrolled] = useState(false)

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 20)
    window.addEventListener('scroll', handleScroll)
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  return (
    <div className="relative min-h-screen" style={{ background: 'var(--bg-base)', overflowX: 'hidden' }}>
      <div className="hero-grid" />
      <div className="hero-blob-1" />
      <div className="hero-blob-2" />

      {/* Header */}
      <header className={`fixed top-0 w-full z-50 transition-all duration-300 ${scrolled ? 'nav-surface shadow-sm' : 'bg-transparent'}`}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-20 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl flex items-center justify-center shadow-lg bg-gradient-to-br from-emerald-500 to-emerald-700">
              <Eye size={20} className="text-white" />
            </div>
            <span className="font-black text-xl tracking-tight font-display text-primary">
              Classroom<span className="gradient-text">Eye</span>
            </span>
          </div>
          
          <nav className="hidden md:flex items-center gap-8 font-semibold text-sm text-secondary">
            <a href="#features" className="hover:text-primary transition-colors">Features</a>
            <a href="#roles" className="hover:text-primary transition-colors">Role Dashboards</a>
            <a href="#tech" className="hover:text-primary transition-colors">Technology</a>
          </nav>

          <div className="flex items-center gap-5">
            <HeaderThemeToggle />
            <div className="hidden sm:block w-px h-6 bg-[var(--border-color)]"></div>
            <button onClick={onEnterApp} className="hidden sm:block text-sm font-bold text-secondary hover:text-primary transition-colors">
              Sign In
            </button>
            <button onClick={onEnterApp} className="btn-brand px-6 py-2.5 rounded-xl text-sm font-bold tracking-wide shadow-lg">
              Get Started
            </button>
          </div>
        </div>
      </header>

      {/* Hero */}
      <section className="relative z-10 pt-40 pb-24 px-4 text-center max-w-5xl mx-auto">
        <div className="animate-fade-up">
          <span className="badge-brand mb-8 shadow-sm px-4 py-1.5 text-sm"><Sparkles size={16}/> AI-Powered Education Analytics</span>
        </div>
        <h1 className="text-5xl sm:text-6xl md:text-7xl font-black mb-8 leading-tight animate-fade-up delay-100 font-display text-primary">
          Understand Every <br className="hidden sm:block" />
          <span className="gradient-text">Student's</span> Engagement
        </h1>
        <p className="text-lg md:text-xl max-w-3xl mx-auto mb-12 animate-fade-up delay-200 text-secondary leading-relaxed">
          ClassroomEye uses computer vision and deep learning to analyze student behavior in real-time — tracking emotions, attention, and actions automatically.
        </p>
        <div className="flex flex-col sm:flex-row justify-center items-center gap-5 animate-fade-up delay-300">
          <button onClick={onEnterApp} className="btn-brand w-full sm:w-auto px-8 py-4 rounded-2xl text-lg font-bold shadow-xl flex items-center justify-center gap-2">
            Start Analyzing <ArrowRight size={20} />
          </button>
          <a href="#features" className="btn-outline w-full sm:w-auto px-8 py-4 rounded-2xl text-lg font-bold shadow-sm flex items-center justify-center gap-2">
            <Play size={20} style={{ color: 'var(--brand-primary)' }} /> See How It Works
          </a>
        </div>
      </section>



      {/* Features */}
      <section id="features" className="relative z-10 py-28 px-4 max-w-7xl mx-auto">
        <div className="text-center mb-20">
          <span className="badge-accent mb-6 px-4 py-1.5 text-sm"><Zap size={16}/> Powerful Features</span>
          <h2 className="text-4xl md:text-5xl font-black font-display text-primary mb-6 leading-tight">
            Everything You Need to <br /><span className="gradient-text-accent">Transform Education</span>
          </h2>
          <p className="text-lg text-secondary max-w-2xl mx-auto">From real-time video analysis to comprehensive department reports, we provide the insights that matter.</p>
        </div>
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          <FeatureCard icon={Camera} title="Video Analysis" description="Upload recordings and get instant per-student engagement breakdown with exact timelines." badgeColor="green" delay={0} />
          <FeatureCard icon={Brain} title="Emotion Recognition" description="Advanced deep learning model detects focus, confusion, boredom, frustration, and distraction." badgeColor="indigo" delay={100} />
          <FeatureCard icon={Activity} title="Action Tracking" description="Identifies 12+ physical behaviors including note-taking, phone usage, and sleeping in class." badgeColor="green" delay={200} />
          <FeatureCard icon={LineChart} title="Engagement Timeline" description="Second-by-second graphical scores reveal exactly when students tune in or check out." badgeColor="indigo" delay={300} />
          <FeatureCard icon={Users} title="At-Risk Alerts" description="Automated alerts sent to HODs for students with consistently low engagement across weeks." badgeColor="green" delay={400} />
          <FeatureCard icon={BarChart3} title="Department Analytics" description="Compare teacher effectiveness, track trends across courses, and make data-driven decisions." badgeColor="indigo" delay={500} />
        </div>
      </section>

      {/* Roles */}
      <section id="roles" className="relative z-10 py-24 px-4 border-t" style={{ borderColor: 'var(--border-color)', background: 'var(--bg-surface)' }}>
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-black font-display text-primary mb-4">Dedicated Role Portals</h2>
            <p className="text-lg text-secondary">Specific tools designed explicitly for the needs of Faculty and Administration.</p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-10">
            <div className="glass rounded-[2rem] p-10 animate-fade-up shadow-xl" style={{ border: '2px solid var(--brand-border)' }}>
              <div className="w-16 h-16 rounded-2xl flex items-center justify-center mb-8 shadow-sm" style={{ background: 'var(--brand-bg)' }}>
                <GraduationCap size={32} style={{ color: 'var(--brand-primary)' }} />
              </div>
              <span className="badge-brand mb-4 px-3 py-1">Teacher Access</span>
              <h3 className="text-3xl font-bold mb-6 font-display text-primary">Classroom Insights</h3>
              <ul className="space-y-4 mb-10">
                 {['Upload MP4 lecture videos for scanning', 'See individual student emotion breakdown', 'Download comprehensive session reports'].map(f => (
                   <li key={f} className="flex items-center gap-3 text-base text-secondary font-medium">
                     <CheckCircle size={20} style={{ color: 'var(--brand-primary)' }} /> {f}
                   </li>
                 ))}
              </ul>
              <button onClick={onEnterApp} className="w-full py-4 rounded-xl font-bold text-base shadow-sm transition-all hover:-translate-y-1" style={{ background: 'var(--brand-bg)', color: 'var(--brand-primary)'}}>
                Explore Teacher Dashboard &rarr;
              </button>
            </div>

            <div className="glass rounded-[2rem] p-10 animate-fade-up delay-100 shadow-xl" style={{ border: '2px solid var(--accent-border)' }}>
              <div className="w-16 h-16 rounded-2xl flex items-center justify-center mb-8 shadow-sm" style={{ background: 'var(--accent-bg)' }}>
                <Shield size={32} style={{ color: 'var(--accent-primary)' }} />
              </div>
              <span className="badge-accent mb-4 px-3 py-1">HOD Access</span>
              <h3 className="text-3xl font-bold mb-6 font-display text-primary">Department Overview</h3>
              <ul className="space-y-4 mb-10">
                 {['View department-wide engagement trends', 'Compare average metrics between teachers', 'Receive alerts for at-risk struggling students'].map(f => (
                   <li key={f} className="flex items-center gap-3 text-base text-secondary font-medium">
                     <CheckCircle size={20} style={{ color: 'var(--accent-primary)' }} /> {f}
                   </li>
                 ))}
              </ul>
              <button onClick={onEnterApp} className="w-full py-4 rounded-xl font-bold text-base shadow-sm transition-all hover:-translate-y-1" style={{ background: 'var(--accent-bg)', color: 'var(--accent-primary)'}}>
                Explore HOD Dashboard &rarr;
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer id="tech" className="relative z-10 border-t py-12 px-4" style={{ borderColor: 'var(--border-color)', background: 'var(--bg-base)' }}>
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="flex items-center gap-3">
            <Eye size={24} style={{ color: 'var(--brand-primary)' }} />
            <span className="font-black text-xl font-display text-primary">ClassroomEye</span>
          </div>
          <div className="text-secondary text-sm font-medium text-center md:text-left">
            AI-Powered Engagement System &middot; Final Year Project
          </div>
        </div>
      </footer>
    </div>
  )
}
