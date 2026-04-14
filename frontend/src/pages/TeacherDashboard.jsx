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
      const data = await uploadVideo(file)
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
    <div className="max-w-2xl mx-auto animate-fade-up">
      <div className="mb-8 text-center">
        <div className="badge-brand mb-4 shadow-sm"><Sparkles size={12} /> AI Video Analysis</div>
        <h2 className="text-3xl font-black font-display" style={{ color: 'var(--text-primary)' }}>Upload Classroom Video</h2>
        <p className="mt-2 text-sm" style={{ color: 'var(--text-muted)' }}>Our AI analyzes student engagement second by second</p>
      </div>

      <div className="glass rounded-[2rem] p-8 md:p-10 shadow-2xl">
        {status === 'idle' || status === 'failed' ? (
          <>
            <div
              className={`drop-zone rounded-2xl p-10 text-center cursor-pointer ${dragActive ? 'active' : ''}`}
              onDragEnter={handleDrag} onDragLeave={handleDrag} onDragOver={handleDrag} onDrop={handleDrop}
              onClick={() => inputRef.current?.click()}
            >
              <input ref={inputRef} type="file" accept="video/mp4" className="hidden" onChange={(e) => setFile(e.target.files?.[0] || null)} />
              <div className="w-16 h-16 rounded-2xl flex items-center justify-center mx-auto mb-4" style={{ background: 'var(--brand-bg)' }}>
                {file ? <FileVideo size={28} style={{ color: 'var(--brand-primary)' }} /> : <Upload size={28} style={{ color: 'var(--brand-primary)' }} />}
              </div>
              {file ? (
                <div>
                  <p className="font-semibold" style={{ color: 'var(--text-primary)' }}>{file.name}</p>
                  <p className="text-sm mt-1" style={{ color: 'var(--text-muted)' }}>{(file.size / 1024 / 1024).toFixed(1)} MB &middot; MP4</p>
                  <span className="badge-brand mt-3">✓ Ready to analyze</span>
                </div>
              ) : (
                <>
                  <p className="font-semibold mb-1" style={{ color: 'var(--text-primary)' }}>Drag & drop your video</p>
                  <p className="text-sm" style={{ color: 'var(--text-muted)' }}>or <span style={{ color: 'var(--brand-primary)' }}>browse</span> &middot; MP4 only</p>
                </>
              )}
            </div>
            {error && <div className="mt-4 flex items-center gap-2 text-red-500 text-sm bg-red-100 dark:bg-red-900/20 rounded-xl p-3"><XCircle size={16} /> {error}</div>}
            <button onClick={handleAnalyze} disabled={!file} className="mt-6 w-full py-4 rounded-xl font-bold btn-brand flex justify-center gap-2">
               Analyze Video
            </button>
          </>
        ) : (
          <div className="text-center py-8">
            <div className="relative inline-flex items-center justify-center mb-6">
              <svg className="w-24 h-24 -rotate-90" viewBox="0 0 100 100">
                <circle cx="50" cy="50" r="42" fill="none" stroke="var(--border-color)" strokeWidth="8" />
                <circle cx="50" cy="50" r="42" fill="none" stroke="var(--brand-primary)" strokeWidth="8" strokeLinecap="round"
                        strokeDasharray={`${2 * Math.PI * 42}`} strokeDashoffset={`${2 * Math.PI * 42 * (1 - progress / 100)}`}
                        style={{ transition: 'stroke-dashoffset 0.5s ease' }} />
              </svg>
              <div className="absolute inset-0 flex items-center justify-center">
                {status === 'uploading' ? <Upload size={22} style={{ color: 'var(--brand-primary)' }} /> : <Brain size={22} className="animate-pulse" style={{ color: 'var(--brand-primary)' }} />}
              </div>
            </div>
            <div className="font-bold text-xl mb-2 font-display" style={{ color: 'var(--text-primary)' }}>
              {status === 'uploading' ? 'Uploading Video...' : 'AI Analyzing Engagement...'}
            </div>
            <p className="text-sm mb-4" style={{ color: 'var(--text-muted)' }}>
              {status === 'processing' ? 'Detecting students, emotions & behaviors frame by frame' : 'Sending file to server'}
            </p>
            <div className="w-full rounded-full h-1.5 overflow-hidden mt-4 max-w-xs mx-auto" style={{ background: 'var(--bg-input)' }}>
              <div className="h-full rounded-full transition-all duration-500" style={{ width: `${progress}%`, background: 'var(--brand-primary)' }} />
            </div>
            <p className="text-sm font-semibold mt-2" style={{ color: 'var(--brand-primary)' }}>{Math.round(progress)}%</p>
          </div>
        )}
      </div>
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

export default function TeacherDashboard({ onLogout }) {
  const [results, setResults] = useState(null)

  return (
    <>
      <Navbar title="Teacher Dashboard" role="teacher" onLogout={onLogout} />
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
