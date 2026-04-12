import Navbar from '../components/Navbar'
import KPICard from '../components/KPICard'
import { TrendingUp, BookOpen, Award, AlertTriangle } from 'lucide-react'
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    BarChart, Bar, Rectangle, Cell
} from 'recharts'

// ─── Mock Data ──────────────────────────────────────────────

const weeklyTrend = [
    { day: 'Monday', engagement: 72 },
    { day: 'Tuesday', engagement: 78 },
    { day: 'Wednesday', engagement: 65 },
    { day: 'Thursday', engagement: 82 },
    { day: 'Friday', engagement: 76 },
]

const teacherComparison = [
    { name: 'Dr. Ahmed', engagement: 85 },
    { name: 'Ms. Fatima', engagement: 78 },
    { name: 'Mr. Bilal', engagement: 72 },
    { name: 'Dr. Sara', engagement: 90 },
    { name: 'Mr. Usman', engagement: 65 },
]

const atRiskStudents = [
    { id: 1, name: 'Ali Hassan', class: 'CS-6A', avgEngagement: 32, topAction: 'sleeping', sessions: 8 },
    { id: 2, name: 'Zainab Malik', class: 'CS-6B', avgEngagement: 38, topAction: 'using_mobile', sessions: 6 },
    { id: 3, name: 'Hamza Qureshi', class: 'CS-6A', avgEngagement: 41, topAction: 'looking_away', sessions: 7 },
    { id: 4, name: 'Ayesha Siddiq', class: 'CS-6C', avgEngagement: 35, topAction: 'sleeping', sessions: 5 },
    { id: 5, name: 'Omar Farooq', class: 'CS-6B', avgEngagement: 29, topAction: 'using_mobile', sessions: 9 },
]

const BAR_COLORS = ['#6366f1', '#8b5cf6', '#06b6d4', '#10b981', '#f59e0b']

function getEngagementColor(val) {
    if (val >= 70) return 'text-emerald-600 bg-emerald-50'
    if (val >= 50) return 'text-amber-600 bg-amber-50'
    return 'text-red-600 bg-red-50'
}

// ─── Main Page ──────────────────────────────────────────────

export default function HODDashboard({ onLogout }) {
    return (
        <div className="min-h-screen bg-surface">
            <Navbar title="HOD Dashboard" role="hod" onLogout={onLogout} />

            <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
                {/* KPI Cards */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <KPICard
                        icon={TrendingUp}
                        label="Dept. Avg Engagement"
                        value="78%"
                        sub="Up 3% from last week"
                        color="primary"
                    />
                    <KPICard
                        icon={BookOpen}
                        label="Lectures Analyzed"
                        value="24"
                        sub="This semester"
                        color="emerald"
                    />
                    <KPICard
                        icon={Award}
                        label="Most Active Class"
                        value="CS-6A"
                        sub="Avg 85% engagement"
                        color="violet"
                    />
                </div>

                {/* Charts Row */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* Weekly Trend */}
                    <div className="bg-white rounded-xl border border-slate-200 p-5 animate-fade-in">
                        <h3 className="text-sm font-semibold text-slate-700 mb-4">Department Engagement Trend</h3>
                        <ResponsiveContainer width="100%" height={280}>
                            <LineChart data={weeklyTrend}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                                <XAxis dataKey="day" tick={{ fontSize: 12 }} />
                                <YAxis domain={[50, 100]} tick={{ fontSize: 12 }} />
                                <Tooltip contentStyle={{ borderRadius: '8px', border: '1px solid #e2e8f0', fontSize: '13px' }} formatter={(v) => [`${v}%`, 'Engagement']} />
                                <Line type="monotone" dataKey="engagement" stroke="#6366f1" strokeWidth={2.5} dot={{ fill: '#6366f1', r: 4 }} activeDot={{ r: 6 }} />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>

                    {/* Teacher Comparison */}
                    <div className="bg-white rounded-xl border border-slate-200 p-5 animate-fade-in">
                        <h3 className="text-sm font-semibold text-slate-700 mb-4">Teacher-wise Engagement</h3>
                        <ResponsiveContainer width="100%" height={280}>
                            <BarChart data={teacherComparison}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                                <YAxis domain={[0, 100]} tick={{ fontSize: 12 }} />
                                <Tooltip contentStyle={{ borderRadius: '8px', border: '1px solid #e2e8f0', fontSize: '13px' }} formatter={(v) => [`${v}%`, 'Engagement']} />
                                <Bar dataKey="engagement" radius={[6, 6, 0, 0]} activeBar={<Rectangle fill="#4f46e5" />}>
                                    {teacherComparison.map((_, i) => (
                                        <Cell key={i} fill={BAR_COLORS[i % BAR_COLORS.length]} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* At-Risk Students Table */}
                <div className="bg-white rounded-xl border border-slate-200 p-5 animate-fade-in">
                    <div className="flex items-center gap-2 mb-4">
                        <AlertTriangle size={18} className="text-amber-500" />
                        <h3 className="text-sm font-semibold text-slate-700">At-Risk Students</h3>
                        <span className="ml-auto text-xs text-slate-400">Students with consistently low engagement</span>
                    </div>
                    <div className="overflow-x-auto">
                        <table className="w-full text-sm">
                            <thead>
                                <tr className="text-left text-xs text-slate-500 uppercase tracking-wider border-b border-slate-100">
                                    <th className="pb-3 pr-4 font-medium">Student</th>
                                    <th className="pb-3 pr-4 font-medium">Class</th>
                                    <th className="pb-3 pr-4 font-medium">Avg Engagement</th>
                                    <th className="pb-3 pr-4 font-medium">Top Distraction</th>
                                    <th className="pb-3 font-medium">Sessions</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-slate-50">
                                {atRiskStudents.map((s) => (
                                    <tr key={s.id} className="hover:bg-slate-50 transition-colors">
                                        <td className="py-3 pr-4 font-medium text-slate-800">{s.name}</td>
                                        <td className="py-3 pr-4 text-slate-600">{s.class}</td>
                                        <td className="py-3 pr-4">
                                            <span className={`inline-flex px-2.5 py-0.5 rounded-full text-xs font-semibold ${getEngagementColor(s.avgEngagement)}`}>
                                                {s.avgEngagement}%
                                            </span>
                                        </td>
                                        <td className="py-3 pr-4 text-slate-600 capitalize">{s.topAction.replace('_', ' ')}</td>
                                        <td className="py-3 text-slate-600">{s.sessions}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            </main>
        </div>
    )
}
