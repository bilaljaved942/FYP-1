/**
 * HODDashboard — Department-Wide Analytics Dashboard
 * ====================================================
 * Aggregated view for Head of Department (HOD) users.
 * Displays analytics from all teachers in the same university + department.
 *
 * Visualizations (6 charts):
 *   1. Teacher-wise Engagement (bar chart)
 *   2. Overall Emotion Distribution (pie/donut chart)
 *   3. Overall Action Breakdown (horizontal bar chart)
 *   4. Emotions by Course (grouped bar chart)
 *   5. Actions by Course (grouped bar chart)
 *   6. Course Engagement (progress bars)
 *
 * Data is fetched from GET /analytics/hod (requires HOD role).
 * Shows an empty state message when no data is available yet.
 *
 * Props:
 *   @param {function} onLogout  - Callback to log out
 *   @param {string}   userName  - HOD's name for navbar display
 */
import { useState, useEffect } from 'react'
import Navbar from '../components/Navbar'
import KPICard from '../components/KPICard'
import { TrendingUp, BookOpen, Award, Users, Activity, BarChart3, Brain, Filter, Loader2 } from 'lucide-react'
import {
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, Cell, PieChart, Pie
} from 'recharts'
import { useTheme } from '../context/ThemeContext'
import { getHodAnalytics } from '../services/api'

// Color palettes for charts
const TEACHER_COLORS = ['#10b981', '#6366f1', '#f59e0b', '#06b6d4', '#f43f5e']
const CHART_COLORS = ['#10b981', '#6366f1', '#06b6d4', '#f59e0b', '#f43f5e', '#8b5cf6', '#ec4899', '#14b8a6']

/** Reusable section header with icon, title, and optional subtitle. */
function SectionHeader({ icon: Icon, title, sub, color = 'var(--brand-primary)' }) {
  return (
    <div className="flex items-center gap-2 mb-5">
      <Icon size={15} style={{ color }} />
      <h3 className="text-sm font-bold uppercase tracking-wide" style={{ color: 'var(--text-primary)' }}>{title}</h3>
      {sub && <span className="ml-auto text-xs" style={{ color: 'var(--text-muted)' }}>{sub}</span>}
    </div>
  )
}

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className="glass rounded-xl px-4 py-3 text-sm shadow-xl border">
      <p style={{ color: 'var(--text-muted)', marginBottom: 2 }}>{label}</p>
      {payload.map((p, i) => (
        <p key={i} className="font-bold" style={{ color: p.color || 'var(--text-primary)' }}>
          {p.name}: {p.value}{typeof p.value === 'number' ? '%' : ''}
        </p>
      ))}
    </div>
  )
}

export default function HODDashboard({ onLogout, userName }) {
  const { theme } = useTheme()
  const isDark = theme === 'dark'
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    async function loadData() {
      try {
        const token = localStorage.getItem('token')
        const result = await getHodAnalytics(token)
        setData(result)
      } catch (err) {
        setError(err.message)
      } finally {
        setLoading(false)
      }
    }
    loadData()
  }, [])

  const chartGridColor = isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)'
  const chartTickColor = isDark ? '#64748b' : '#94a3b8'

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Loader2 size={32} className="animate-spin text-[var(--brand-primary)]" />
      </div>
    )
  }

  const isEmpty = error || !data || !data.kpis

  const defaultData = {
    kpis: { deptAvg: 0, lecturesAnalyzed: 0, mostActiveCourse: 'N/A' },
    teacherComparison: [],
    classEngagement: [],
    emotionData: [],
    actionData: [],
    emotionsByCourse: [],
    actionsByCourse: [],
    recentActivity: []
  }

  const { kpis, teacherComparison, classEngagement, emotionData, actionData, emotionsByCourse, actionsByCourse, recentActivity } = isEmpty ? defaultData : data

  // Extract unique emotion/action keys for grouped bar charts
  const emotionKeys = [...new Set(emotionsByCourse.flatMap(e => Object.keys(e).filter(k => k !== 'course')))]
  const actionKeys  = [...new Set(actionsByCourse.flatMap(a => Object.keys(a).filter(k => k !== 'course')))]

  return (
    <>
      <Navbar title="HOD Dashboard" role="hod" onLogout={onLogout} userName={userName} />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6 relative">
        <div className="hero-grid" style={{ opacity: 0.5 }} />

        <div className="relative z-10 space-y-6">
          {isEmpty && (
            <div className="glass rounded-2xl border p-6 flex items-center gap-4" style={{ borderColor: 'var(--brand-border)', background: 'var(--brand-bg)' }}>
              <BookOpen size={24} style={{ color: 'var(--brand-primary)', flexShrink: 0 }} />
              <div>
                <p className="font-bold text-sm" style={{ color: 'var(--brand-primary)' }}>No Data Available Yet</p>
                <p className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>Analytics will appear here once teachers in your department upload and analyze classroom videos.</p>
              </div>
            </div>
          )}

          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-black font-display" style={{ color: 'var(--text-primary)' }}>Department Overview</h1>
              <p className="text-sm mt-0.5" style={{ color: 'var(--text-secondary)' }}>Spring 2025 &middot; Computer Science Department</p>
            </div>
            <button className="flex items-center gap-2 btn-outline px-4 py-2 rounded-xl text-sm font-medium">
              <Filter size={14} /> Filters
            </button>
          </div>

          {/* KPI Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <KPICard icon={TrendingUp} label="Dept. Avg Engagement" value={`${kpis.deptAvg}%`} sub="Based on all videos" color="emerald" trend={0} />
            <KPICard icon={BookOpen}   label="Lectures Analyzed" value={kpis.lecturesAnalyzed} sub="This semester" color="indigo" />
            <KPICard icon={Award}      label="Top Course" value={kpis.mostActiveCourse} sub="Highest avg engagement" color="amber" />
          </div>

          {/* Teacher-wise Engagement (full width) */}
          <div className="glass rounded-2xl border p-6 animate-fade-up">
            <SectionHeader icon={Users} title="Teacher-wise Engagement" sub="All teachers" color="var(--accent-primary)" />
            {teacherComparison.length > 0 ? (
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={teacherComparison}>
                  <CartesianGrid strokeDasharray="3 3" stroke={chartGridColor} />
                  <XAxis dataKey="name" tick={{ fill: chartTickColor, fontSize: 11 }} />
                  <YAxis domain={[0, 100]} tick={{ fill: chartTickColor, fontSize: 12 }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="engagement" radius={[6, 6, 0, 0]}>
                    {teacherComparison.map((_, i) => <Cell key={i} fill={TEACHER_COLORS[i % TEACHER_COLORS.length]} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : <p className="text-center py-12" style={{ color: 'var(--text-muted)' }}>No teacher data yet</p>}
          </div>

          {/* Emotion Distribution (Pie) + Action Breakdown (Bar) */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
            <div className="glass rounded-2xl border p-6 animate-fade-up">
              <SectionHeader icon={Brain} title="Overall Emotion Distribution" color="var(--accent-primary)" />
              {emotionData.length > 0 ? (
                <ResponsiveContainer width="100%" height={280}>
                  <PieChart>
                    <Pie
                      data={emotionData} cx="50%" cy="50%" innerRadius={60} outerRadius={100}
                      paddingAngle={3} dataKey="value"
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                      labelLine={{ stroke: chartGridColor }}
                    >
                      {emotionData.map((_, i) => <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />)}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              ) : <p className="text-center py-12" style={{ color: 'var(--text-muted)' }}>No emotion data</p>}
            </div>

            <div className="glass rounded-2xl border p-6 animate-fade-up delay-100">
              <SectionHeader icon={Activity} title="Overall Action Breakdown" color="#f59e0b" />
              {actionData.length > 0 ? (
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={actionData} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke={chartGridColor} />
                    <XAxis type="number" tick={{ fill: chartTickColor, fontSize: 11 }} />
                    <YAxis type="category" dataKey="name" tick={{ fill: chartTickColor, fontSize: 10 }} width={120} />
                    <Tooltip />
                    <Bar dataKey="value" fill="#f59e0b" radius={[0, 8, 8, 0]}>
                      {actionData.map((_, i) => <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              ) : <p className="text-center py-12" style={{ color: 'var(--text-muted)' }}>No action data</p>}
            </div>
          </div>

          {/* Emotions by Course (Grouped Bar) */}
          {emotionsByCourse.length > 0 && (
            <div className="glass rounded-2xl border p-6 animate-fade-up">
              <SectionHeader icon={Brain} title="Emotions by Course" sub="Grouped by course" color="#8b5cf6" />
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={emotionsByCourse}>
                  <CartesianGrid strokeDasharray="3 3" stroke={chartGridColor} />
                  <XAxis dataKey="course" tick={{ fill: chartTickColor, fontSize: 11 }} />
                  <YAxis tick={{ fill: chartTickColor, fontSize: 12 }} />
                  <Tooltip content={<CustomTooltip />} />
                  {emotionKeys.map((key, i) => (
                    <Bar key={key} dataKey={key} fill={CHART_COLORS[i % CHART_COLORS.length]} radius={[4, 4, 0, 0]} />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Actions by Course (Grouped Bar) */}
          {actionsByCourse.length > 0 && (
            <div className="glass rounded-2xl border p-6 animate-fade-up">
              <SectionHeader icon={Activity} title="Actions by Course" sub="Grouped by course" color="#06b6d4" />
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={actionsByCourse}>
                  <CartesianGrid strokeDasharray="3 3" stroke={chartGridColor} />
                  <XAxis dataKey="course" tick={{ fill: chartTickColor, fontSize: 11 }} />
                  <YAxis tick={{ fill: chartTickColor, fontSize: 12 }} />
                  <Tooltip content={<CustomTooltip />} />
                  {actionKeys.map((key, i) => (
                    <Bar key={key} dataKey={key} fill={CHART_COLORS[i % CHART_COLORS.length]} radius={[4, 4, 0, 0]} />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Course Engagement + Recent Activity */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
            <div className="glass rounded-2xl border p-6 animate-fade-up">
              <SectionHeader icon={BarChart3} title="Course Engagement" color="#f59e0b" />
              <div className="space-y-3 mt-2">
                {classEngagement.length > 0 ? classEngagement.map(c => (
                  <div key={c.name}>
                    <div className="flex justify-between text-sm mb-1.5">
                      <span className="font-medium" style={{ color: 'var(--text-primary)' }}>{c.name}</span>
                      <span className="font-bold" style={{ color: c.fill }}>{c.avg}%</span>
                    </div>
                    <div className="w-full h-2 rounded-full overflow-hidden" style={{ background: 'var(--bg-input)' }}>
                      <div className="h-full rounded-full transition-all duration-1000" style={{ width: `${c.avg}%`, background: c.fill }} />
                    </div>
                  </div>
                )) : <p className="text-center py-4 text-sm" style={{ color: 'var(--text-muted)' }}>No course data</p>}
              </div>
            </div>

            <div className="glass rounded-2xl border p-6 animate-fade-up delay-100 lg:col-span-2">
              <SectionHeader icon={Activity} title="Recent Activity" sub="Last 7 days" color="#06b6d4" />
              <div className="space-y-3">
                {recentActivity.length > 0 ? recentActivity.map((a, i) => {
                  const dot = a.status === 'high' ? 'var(--brand-primary)' : a.status === 'medium' ? '#f59e0b' : '#f43f5e'
                  return (
                    <div key={i} className="flex items-center gap-3 py-2 px-3 rounded-xl transition-colors hover:bg-[var(--bg-card-hover)]">
                      <div className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: dot }} />
                      <div className="flex-1 min-w-0">
                        <span className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>{a.teacher}</span>
                        <span className="text-sm" style={{ color: 'var(--text-secondary)' }}> &mdash; {a.class}</span>
                      </div>
                      <span className="font-bold text-sm" style={{ color: dot }}>{a.score}%</span>
                      <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{a.time}</span>
                    </div>
                  )
                }) : <p className="text-sm text-center py-4" style={{ color: 'var(--text-muted)' }}>No recent activity</p>}
              </div>
            </div>
          </div>

        </div>
      </main>
    </>
  )
}
