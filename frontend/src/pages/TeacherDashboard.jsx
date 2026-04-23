import { useState, useEffect, useCallback, useRef } from 'react'
import Navbar from '../components/Navbar'
import KPICard from '../components/KPICard'
import { uploadVideo, getJobStatus } from '../services/api'
import {
  Upload, Brain, Activity, TrendingUp, Users, FileVideo, ArrowLeft,
  Sparkles, BarChart3, Clock, XCircle, ChevronDown
} from 'lucide-react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, BarChart, Bar, Rectangle
} from 'recharts'
import { useTheme } from '../context/ThemeContext'

// Colors for charts that work in both modes
const CHART_COLORS = ['#10b981', '#6366f1', '#06b6d4', '#f59e0b', '#f43f5e', '#8b5cf6', '#ec4899']

function getTopEntry(obj) {
  if (!obj || Object.keys(obj).length === 0) return 'N/A'
  return Object.entries(obj).sort((a, b) => b[1] - a[1])[0][0]
}

function avgEngagement(timeline) {
  if (!timeline || timeline.length === 0) return 0
  return Math.round(timeline.reduce((a, b) => a + b.score, 0) / timeline.length)
}

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className="glass rounded-xl px-4 py-3 text-sm shadow-xl border">
      <p style={{ color: 'var(--text-muted)', marginBottom: 2 }}>{label !== undefined ? `Second ${label}` : payload[0]?.name}</p>
      <p style={{ color: 'var(--text-primary)', fontWeight: 'bold' }}>{payload[0]?.value}{typeof payload[0]?.value === 'number' && payload[0]?.name === 'score' ? '%' : ''}</p>
    </div>
  )
}

function UploadSection({ onUploadComplete }) {
  const [file, setFile] = useState(null)
  const [dragActive, setDragActive] = useState(false)
  const [status, setStatus] = useState('idle')
  const [jobId, setJobId] = useState(null)
  const [error, setError] = useState(null)
  const [progress, setProgress] = useState(0)
  const [classSection, setClassSection] = useState('')
  const [courseName, setCourseName] = useState('')
  const intervalRef = useRef(null)
  const inputRef = useRef(null)

  const handleDrag = (e) => { e.preventDefault(); e.stopPropagation(); setDragActive(e.type === 'dragenter' || e.type === 'dragover') }
  const handleDrop = (e) => { e.preventDefault(); e.stopPropagation(); setDragActive(false); if (e.dataTransfer.files?.[0]) setFile(e.dataTransfer.files[0]) }

  const clearPolling = useCallback(() => { if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null } }, [])
  useEffect(() => () => clearPolling(), [clearPolling])

  useEffect(() => {
    if (status === 'processing') {
      const t = setInterval(() => setProgress(p => p >= 90 ? p : p + Math.random() * 8), 1200)
      return () => clearInterval(t)
    }
    if (status === 'completed') setProgress(100)
  }, [status])

  const handleAnalyze = async () => {
    if (!file) return
    setStatus('uploading'); setError(null); setProgress(5)
    try {
      const token = localStorage.getItem('token')
      const data = await uploadVideo(file, classSection, courseName, token)
      setJobId(data.job_id); setStatus('processing'); setProgress(20)
      intervalRef.current = setInterval(async () => {
        try {
          const job = await getJobStatus(data.job_id)
          if (job.status === 'COMPLETED') { clearPolling(); setStatus('completed'); setProgress(100); onUploadComplete(job.ai_results) }
          else if (job.status === 'FAILED') { clearPolling(); setStatus('failed'); setError('AI analysis failed.') }
        } catch (err) { clearPolling(); setStatus('failed'); setError(err.message) }
      }, 3000)
    } catch (err) { setStatus('failed'); setError(err.message) }
  }

  if (status === 'completed') return null

  return (
    <div className="w-full max-w-xl mx-auto animate-fade-up">
      {status === 'idle' || status === 'failed' ? (
        <div className="text-center">
          <div className="badge-brand mb-6 shadow-sm mx-auto inline-flex"><Sparkles size={12} /> AI Video Analysis</div>
          <h2 className="text-3xl font-black font-display mb-3" style={{ color: 'var(--text-primary)' }}>Analyze Classroom</h2>
          <p className="text-sm mb-8" style={{ color: 'var(--text-muted)' }}>Upload MP4 recording for instant behavioral breakdown</p>

          <div className="grid grid-cols-2 gap-4 mb-6">
            <input 
              type="text" placeholder="Class Section (e.g., CS-6A)" required
              value={classSection} onChange={(e) => setClassSection(e.target.value)}
              className="w-full px-4 py-3 rounded-xl border text-sm font-medium focus:outline-none focus:ring-2 focus:ring-[var(--brand-primary)] transition-all"
              style={{ background: 'var(--bg-input)', borderColor: 'var(--border-color)', color: 'var(--text-primary)' }}
            />
            <input 
              type="text" placeholder="Course Name (e.g., English)" required
              value={courseName} onChange={(e) => setCourseName(e.target.value)}
              className="w-full px-4 py-3 rounded-xl border text-sm font-medium focus:outline-none focus:ring-2 focus:ring-[var(--brand-primary)] transition-all"
              style={{ background: 'var(--bg-input)', borderColor: 'var(--border-color)', color: 'var(--text-primary)' }}
            />
          </div>

          <div
            className={`transition-all duration-300 rounded-3xl p-8 cursor-pointer border-2 hover:border-[var(--brand-primary)] ${dragActive ? 'border-[var(--brand-primary)] bg-[var(--brand-primary)]/5 scale-105' : 'border-dashed border-[var(--border-color)] bg-transparent'}`}
            onDragEnter={handleDrag} onDragLeave={handleDrag} onDragOver={handleDrag} onDrop={handleDrop}
            onClick={() => inputRef.current?.click()}
          >
            <input ref={inputRef} type="file" accept="video/mp4" className="hidden" onChange={(e) => setFile(e.target.files?.[0] || null)} />
            <div className="w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4 transition-transform group-hover:scale-110" style={{ background: file ? 'var(--brand-bg)' : 'var(--bg-input)' }}>
              {file ? <FileVideo size={24} style={{ color: 'var(--brand-primary)' }} /> : <Upload size={24} style={{ color: 'var(--text-secondary)' }} />}
            </div>
            
            {file ? (
              <div className="animate-fade-up">
                <p className="font-bold text-lg" style={{ color: 'var(--text-primary)' }}>{file.name}</p>
                <p className="text-sm mt-1" style={{ color: 'var(--text-muted)' }}>{(file.size / 1024 / 1024).toFixed(1)} MB</p>
              </div>
            ) : (
              <div>
                <p className="font-semibold text-base mb-1" style={{ color: 'var(--text-primary)' }}>Click to upload or drag and drop</p>
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Maximum file size 500MB</p>
              </div>
            )}
          </div>
          
          {error && <div className="mt-6 flex items-center justify-center gap-2 text-red-500 text-sm font-semibold animate-fade-up"><XCircle size={16} /> {error}</div>}
          
          {file && classSection && courseName && (
            <button onClick={handleAnalyze} className="mt-8 px-10 py-4 rounded-full font-bold text-white shadow-xl hover:scale-105 transition-all text-sm uppercase tracking-wide animate-fade-up"
                    style={{ background: 'linear-gradient(135deg, var(--brand-primary), var(--brand-dark))', boxShadow: 'var(--shadow-glow)' }}>
              Start Analysis
            </button>
          )}
        </div>
      ) : (
        <div className="text-center py-10">
          <div className="relative inline-flex items-center justify-center mb-8">
            <svg className="w-32 h-32 -rotate-90 drop-shadow-xl" viewBox="0 0 100 100">
              <circle cx="50" cy="50" r="46" fill="none" stroke="var(--bg-input)" strokeWidth="6" />
              <circle cx="50" cy="50" r="46" fill="none" stroke="url(#progressGrad)" strokeWidth="6" strokeLinecap="round"
                      strokeDasharray={`${2 * Math.PI * 46}`} strokeDashoffset={`${2 * Math.PI * 46 * (1 - progress / 100)}`}
                      style={{ transition: 'stroke-dashoffset 0.8s cubic-bezier(0.4, 0, 0.2, 1)' }} />
              <defs>
                <linearGradient id="progressGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="var(--brand-primary)" />
                  <stop offset="100%" stopColor="var(--brand-dark)" />
                </linearGradient>
              </defs>
            </svg>
            <div className="absolute inset-0 flex items-center justify-center flex-col">
              <span className="text-2xl font-black font-display" style={{ color: 'var(--text-primary)' }}>{Math.round(progress)}<span className="text-sm font-medium" style={{ color: 'var(--text-muted)' }}>%</span></span>
            </div>
          </div>
          <div className="font-bold text-2xl mb-2 font-display" style={{ color: 'var(--text-primary)' }}>
            {status === 'uploading' ? 'Uploading...' : 'Processing Video'}
          </div>
          <p className="text-sm font-medium animate-pulse" style={{ color: 'var(--brand-primary)' }}>
            {status === 'processing' ? 'AI analyzing behaviors frame by frame...' : 'Encrypting and transferring file...'}
          </p>
        </div>
      )}
    </div>
  )
}

function ResultsSection({ data }) {
  const { theme } = useTheme()
  const isDark = theme === 'dark'
  const students = data?.students || []
  const [selectedIdx, setSelectedIdx] = useState(0)

  if (students.length === 0) {
    return (
      <div className="text-center py-24">
        <div className="w-20 h-20 rounded-3xl flex items-center justify-center mx-auto mb-4" style={{ background: 'var(--bg-card)', border: '1px solid var(--border-color)' }}>
          <Users size={36} style={{ color: 'var(--text-muted)' }} />
        </div>
        <p className="text-lg font-medium" style={{ color: 'var(--text-secondary)' }}>No students detected.</p>
      </div>
    )
  }

  const student = students[selectedIdx]
  const emotions = student.emotions || {}
  const actions  = student.actions  || {}
  const timeline = student.engagement_over_time || []
  const emotionData = Object.entries(emotions).map(([name, value]) => ({ name, value }))
  const actionData  = Object.entries(actions).map(([name, value]) => ({ name, value }))
  const avg = avgEngagement(timeline)

  const chartGridColor = isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)'
  const chartTickColor = isDark ? '#64748b' : '#94a3b8'

  return (
    <div className="space-y-6 animate-fade-up">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="badge-brand mb-2"><BarChart3 size={12}/> Analysis Complete</div>
          <h2 className="text-2xl font-black font-display" style={{ color: 'var(--text-primary)' }}>Engagement Results</h2>
          <p className="text-sm mt-0.5" style={{ color: 'var(--text-muted)' }}>{students.length} student(s) detected</p>
        </div>
        <div className="relative">
          <select value={selectedIdx} onChange={(e) => setSelectedIdx(Number(e.target.value))}
                  className="appearance-none glass border rounded-xl pl-4 pr-10 py-2.5 text-sm font-semibold cursor-pointer"
                  style={{ color: 'var(--text-primary)' }}>
            {students.map((s, i) => <option key={s.student_id} value={i} style={{ background: 'var(--bg-dropdown)' }}>Student {s.student_id}</option>)}
          </select>
          <ChevronDown size={15} className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none" style={{ color: 'var(--text-muted)' }} />
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <KPICard icon={Brain}      label="Dominant Emotion"  value={getTopEntry(emotions)} sub={`${Object.keys(emotions).length} detected`} color="indigo" />
        <KPICard icon={Activity}   label="Primary Action"    value={getTopEntry(actions)}  sub={`${Object.keys(actions).length} tracked`}   color="amber"  />
        <KPICard icon={TrendingUp} label="Avg Engagement"    value={`${avg}%`}             sub={`over ${timeline.length}s`}                 color={avg >= 70 ? 'emerald' : avg >= 50 ? 'amber' : 'rose'} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <div className="glass rounded-2xl border p-6 lg:col-span-2">
          <div className="flex items-center gap-2 mb-5">
             <Clock size={15} style={{ color: 'var(--brand-primary)' }} />
             <h3 className="text-sm font-bold uppercase tracking-wide" style={{ color: 'var(--text-primary)' }}>Engagement Timeline</h3>
          </div>
          {timeline.length > 0 ? (
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={timeline}>
                <defs>
                  <linearGradient id="engGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="var(--brand-primary)" stopOpacity="0.2" />
                    <stop offset="100%" stopColor="var(--brand-primary)" stopOpacity="0" />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke={chartGridColor} />
                <XAxis dataKey="second" tick={{ fill: chartTickColor, fontSize: 11 }} />
                <YAxis domain={[0, 100]} tick={{ fill: chartTickColor, fontSize: 11 }} />
                <Tooltip content={<CustomTooltip />} />
                <Line type="monotone" dataKey="score" stroke="var(--brand-primary)" strokeWidth={2.5} dot={false} activeDot={{ r: 5, fill: 'var(--brand-primary)', stroke: '#fff', strokeWidth: 2 }} />
              </LineChart>
            </ResponsiveContainer>
          ) : <p className="text-center py-12" style={{ color: 'var(--text-muted)' }}>No data</p>}
        </div>

        <div className="glass rounded-2xl border p-6">
          <div className="flex items-center gap-2 mb-5">
             <Brain size={15} style={{ color: 'var(--accent-primary)' }} />
             <h3 className="text-sm font-bold uppercase tracking-wide" style={{ color: 'var(--text-primary)' }}>Emotion Distribution</h3>
          </div>
          {emotionData.length > 0 ? (
            <ResponsiveContainer width="100%" height={260}>
              <PieChart>
                <Pie data={emotionData} cx="50%" cy="50%" innerRadius={55} outerRadius={90} paddingAngle={3} dataKey="value"
                     label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`} labelLine={{ stroke: chartGridColor }}>
                  {emotionData.map((_, i) => <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />)}
                </Pie>
                <Tooltip content={<CustomTooltip />} />
              </PieChart>
            </ResponsiveContainer>
          ) : <p className="text-center py-12" style={{ color: 'var(--text-muted)' }}>No data</p>}
        </div>

        <div className="glass rounded-2xl border p-6">
          <div className="flex items-center gap-2 mb-5">
             <Activity size={15} color="#f59e0b" />
             <h3 className="text-sm font-bold uppercase tracking-wide" style={{ color: 'var(--text-primary)' }}>Action Breakdown</h3>
          </div>
          {actionData.length > 0 ? (
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={actionData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke={chartGridColor} />
                <XAxis type="number" tick={{ fill: chartTickColor, fontSize: 11 }} />
                <YAxis type="category" dataKey="name" tick={{ fill: chartTickColor, fontSize: 11 }} width={110} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="value" fill="#f59e0b" radius={[0, 8, 8, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : <p className="text-center py-12" style={{ color: 'var(--text-muted)' }}>No data</p>}
        </div>
      </div>
    </div>
  )
}

export default function TeacherDashboard({ onLogout, userName }) {
  const [results, setResults] = useState(null)

  return (
    <>
      <Navbar title="Teacher Dashboard" role="teacher" onLogout={onLogout} userName={userName} />
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 relative">
        <div className="hero-grid" style={{ opacity: 0.5 }} />
        <div className="relative z-10">
          {results ? (
            <>
              <button onClick={() => setResults(null)} className="mb-6 flex items-center gap-2 text-sm font-medium hover:text-[var(--text-primary)] transition-colors cursor-pointer" style={{ color: 'var(--text-secondary)' }}>
                <ArrowLeft size={16} /> Upload another video
              </button>
              <ResultsSection data={results} />
            </>
          ) : (
            <div className="min-h-[70vh] flex items-center justify-center">
              <UploadSection onUploadComplete={setResults} />
            </div>
          )}
        </div>
      </main>
    </>
  )
}
