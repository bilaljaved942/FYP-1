import Navbar from '../components/Navbar'
import KPICard from '../components/KPICard'
import { TrendingUp, BookOpen, Award, AlertTriangle, Users, Activity, BarChart3, Filter } from 'lucide-react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, Rectangle, Cell
} from 'recharts'
import { useTheme } from '../context/ThemeContext'

// ── Mock Data ──────────────────────────────────────────────────
const weeklyTrend = [
  { day: 'Mon', engagement: 72 },
  { day: 'Tue', engagement: 78 },
  { day: 'Wed', engagement: 65 },
  { day: 'Thu', engagement: 82 },
  { day: 'Fri', engagement: 76 },
]

const teacherComparison = [
  { name: 'Dr. Ahmed',  engagement: 85, sessions: 8  },
  { name: 'Ms. Fatima', engagement: 78, sessions: 6  },
  { name: 'Mr. Bilal',  engagement: 72, sessions: 10 },
  { name: 'Dr. Sara',   engagement: 90, sessions: 5  },
  { name: 'Mr. Usman',  engagement: 65, sessions: 7  },
]

const atRiskStudents = [
  { id: 1, name: 'Ali Hassan',     class: 'CS-6A', avgEngagement: 32, topAction: 'sleeping',      sessions: 8 },
  { id: 2, name: 'Zainab Malik',   class: 'CS-6B', avgEngagement: 38, topAction: 'using_mobile',  sessions: 6 },
  { id: 3, name: 'Hamza Qureshi',  class: 'CS-6A', avgEngagement: 41, topAction: 'looking_away',  sessions: 7 },
  { id: 4, name: 'Ayesha Siddiq',  class: 'CS-6C', avgEngagement: 35, topAction: 'sleeping',      sessions: 5 },
  { id: 5, name: 'Omar Farooq',    class: 'CS-6B', avgEngagement: 29, topAction: 'using_mobile',  sessions: 9 },
]

const classEngagement = [
  { name: 'CS-6A', avg: 85, fill: 'var(--brand-primary)' },
  { name: 'CS-6B', avg: 71, fill: 'var(--accent-primary)' },
  { name: 'CS-6C', avg: 78, fill: '#f59e0b' },
  { name: 'CS-5A', avg: 63, fill: '#06b6d4' },
]

const TEACHER_COLORS = ['#10b981', '#6366f1', '#f59e0b', '#06b6d4', '#f43f5e']

function engagementBadge(val) {
  if (val >= 70) return { style: { background: 'rgba(16,185,129,0.15)', color: '#10b981', border: '1px solid rgba(16,185,129,0.3)' }, label: 'Good' }
  if (val >= 50) return { style: { background: 'rgba(245,158,11,0.15)', color: '#f59e0b', border: '1px solid rgba(245,158,11,0.3)' }, label: 'Fair' }
  return           { style: { background: 'rgba(244,63,94,0.15)', color: '#f43f5e', border: '1px solid rgba(244,63,94,0.3)' }, label: 'At Risk' }
}

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
        <p key={i} className="font-bold" style={{ color: 'var(--text-primary)' }}>
          {p.value}{typeof p.value === 'number' ? '%' : ''}
        </p>
      ))}
    </div>
  )
}

export default function HODDashboard({ onLogout }) {
  const { theme } = useTheme()
  const isDark = theme === 'dark'

  const chartGridColor = isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)'
  const chartTickColor = isDark ? '#64748b' : '#94a3b8'

  return (
    <>
      <Navbar title="HOD Dashboard" role="hod" onLogout={onLogout} />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6 relative">
        <div className="hero-grid" style={{ opacity: 0.5 }} />

        <div className="relative z-10 space-y-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-black font-display" style={{ color: 'var(--text-primary)' }}>Department Overview</h1>
              <p className="text-sm mt-0.5" style={{ color: 'var(--text-secondary)' }}>Spring 2025 &middot; Computer Science Department</p>
            </div>
            <button className="flex items-center gap-2 btn-outline px-4 py-2 rounded-xl text-sm font-medium">
              <Filter size={14} /> Filters
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <KPICard icon={TrendingUp} label="Dept. Avg Engagement" value="78%" sub="Up 3% from last week" color="emerald" trend={3} />
            <KPICard icon={BookOpen}   label="Lectures Analyzed" value="24" sub="This semester" color="indigo" />
            <KPICard icon={Award}      label="Most Active Class" value="CS-6A" sub="Avg 85% engagement" color="amber" />
            <KPICard icon={AlertTriangle} label="At-Risk Students" value="5" sub="Needs immediate attention" color="rose" />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
            <div className="glass rounded-2xl border p-6 animate-fade-up">
              <SectionHeader icon={TrendingUp} title="Weekly Engagement Trend" sub="This week" />
              <ResponsiveContainer width="100%" height={240}>
                <LineChart data={weeklyTrend}>
                  <defs>
                    <linearGradient id="trendGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="var(--brand-primary)" stopOpacity="0.2" />
                      <stop offset="100%" stopColor="var(--brand-primary)" stopOpacity="0" />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke={chartGridColor} />
                  <XAxis dataKey="day" tick={{ fill: chartTickColor, fontSize: 12 }} />
                  <YAxis domain={[50, 100]} tick={{ fill: chartTickColor, fontSize: 12 }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Line type="monotone" dataKey="engagement" stroke="var(--brand-primary)" strokeWidth={2.5}
                        dot={{ fill: 'var(--brand-primary)', r: 4, strokeWidth: 2 }} activeDot={{ r: 6, stroke: '#fff', strokeWidth: 2 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="glass rounded-2xl border p-6 animate-fade-up delay-100">
              <SectionHeader icon={Users} title="Teacher-wise Engagement" sub="All teachers" color="var(--accent-primary)" />
              <ResponsiveContainer width="100%" height={240}>
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
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
            <div className="glass rounded-2xl border p-6 animate-fade-up">
              <SectionHeader icon={BarChart3} title="Class Engagement" color="#f59e0b" />
              <div className="space-y-3 mt-2">
                {classEngagement.map(c => (
                  <div key={c.name}>
                    <div className="flex justify-between text-sm mb-1.5">
                      <span className="font-medium" style={{ color: 'var(--text-primary)' }}>{c.name}</span>
                      <span className="font-bold" style={{ color: c.fill }}>{c.avg}%</span>
                    </div>
                    <div className="w-full h-2 rounded-full overflow-hidden" style={{ background: 'var(--bg-input)' }}>
                      <div className="h-full rounded-full transition-all duration-1000" style={{ width: `${c.avg}%`, background: c.fill }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="glass rounded-2xl border p-6 animate-fade-up delay-100 lg:col-span-2">
              <SectionHeader icon={Activity} title="Recent Activity" sub="Last 7 days" color="#06b6d4" />
              <div className="space-y-3">
                {[
                  { teacher: 'Dr. Sara',    class: 'CS-6A', score: 90, time: '2h ago',  status: 'high'   },
                  { teacher: 'Dr. Ahmed',   class: 'CS-6B', score: 85, time: '5h ago',  status: 'high'   },
                  { teacher: 'Ms. Fatima',  class: 'CS-6C', score: 78, time: '1d ago',  status: 'medium' },
                  { teacher: 'Mr. Bilal',   class: 'CS-6A', score: 72, time: '1d ago',  status: 'medium' },
                  { teacher: 'Mr. Usman',   class: 'CS-6B', score: 48, time: '2d ago',  status: 'low'    },
                ].map((a, i) => {
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
                })}
              </div>
            </div>
          </div>

          <div className="glass rounded-2xl border p-6 animate-fade-up">
            <SectionHeader icon={AlertTriangle} title="At-Risk Students" sub="Consistently low engagement" color="#f43f5e" />
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left border-b" style={{ borderColor: 'var(--border-color)' }}>
                    {['Student', 'Class', 'Avg Engagement', 'Top Distraction', 'Sessions', 'Status'].map(h => (
                      <th key={h} className="pb-3 pr-4 text-xs uppercase tracking-wider font-semibold" style={{ color: 'var(--text-muted)' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y" style={{ borderColor: 'var(--border-color)' }}>
                  {atRiskStudents.map(s => {
                    const badge = engagementBadge(s.avgEngagement)
                    return (
                      <tr key={s.id} className="transition-colors hover:bg-[var(--bg-card-hover)]">
                        <td className="py-3.5 pr-4 font-semibold" style={{ color: 'var(--text-primary)' }}>{s.name}</td>
                        <td className="py-3.5 pr-4">
                          <span className="px-2 py-0.5 rounded-lg text-xs font-medium border" style={{ background: 'var(--bg-input)', borderColor: 'var(--border-color)', color: 'var(--text-secondary)' }}>{s.class}</span>
                        </td>
                        <td className="py-3.5 pr-4">
                          <div className="flex items-center gap-2">
                            <div className="w-20 h-1.5 rounded-full overflow-hidden" style={{ background: 'var(--bg-input)' }}>
                              <div className="h-full rounded-full bg-rose-500" style={{ width: `${s.avgEngagement}%` }} />
                            </div>
                            <span className="font-bold text-xs" style={{ color: '#f43f5e' }}>{s.avgEngagement}%</span>
                          </div>
                        </td>
                        <td className="py-3.5 pr-4 capitalize" style={{ color: 'var(--text-secondary)' }}>{s.topAction.replace(/_/g, ' ')}</td>
                        <td className="py-3.5 pr-4" style={{ color: 'var(--text-secondary)' }}>{s.sessions}</td>
                        <td className="py-3.5">
                          <span className="inline-flex px-2.5 py-0.5 rounded-full text-xs font-semibold" style={badge.style}>{badge.label}</span>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </main>
    </>
  )
}
