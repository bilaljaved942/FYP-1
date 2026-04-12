import { useState, useEffect, useCallback, useRef } from 'react'
import Navbar from '../components/Navbar'
import KPICard from '../components/KPICard'
import { uploadVideo, getJobStatus } from '../services/api'
import {
    Upload, Loader2, CheckCircle2, XCircle, ChevronDown,
    Brain, Activity, TrendingUp, Users
} from 'lucide-react'
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    PieChart, Pie, Cell, Legend,
    BarChart, Bar, Rectangle
} from 'recharts'

const PIE_COLORS = ['#6366f1', '#8b5cf6', '#06b6d4', '#10b981', '#f59e0b', '#f43f5e', '#ec4899']

function getTopEntry(obj) {
    if (!obj || Object.keys(obj).length === 0) return 'N/A'
    return Object.entries(obj).sort((a, b) => b[1] - a[1])[0][0]
}

function avgEngagement(timeline) {
    if (!timeline || timeline.length === 0) return 0
    const total = timeline.reduce((a, b) => a + b.score, 0)
    return Math.round(total / timeline.length)
}

// ─── Upload Section ─────────────────────────────────────────

function UploadSection({ onUploadComplete }) {
    const [file, setFile] = useState(null)
    const [dragActive, setDragActive] = useState(false)
    const [status, setStatus] = useState('idle') // idle | uploading | processing | completed | failed
    const [jobId, setJobId] = useState(null)
    const [error, setError] = useState(null)
    const intervalRef = useRef(null)
    const inputRef = useRef(null)

    const handleDrag = (e) => {
        e.preventDefault()
        e.stopPropagation()
        setDragActive(e.type === 'dragenter' || e.type === 'dragover')
    }

    const handleDrop = (e) => {
        e.preventDefault()
        e.stopPropagation()
        setDragActive(false)
        if (e.dataTransfer.files?.[0]) setFile(e.dataTransfer.files[0])
    }

    const clearPolling = useCallback(() => {
        if (intervalRef.current) {
            clearInterval(intervalRef.current)
            intervalRef.current = null
        }
    }, [])

    useEffect(() => () => clearPolling(), [clearPolling])

    const handleAnalyze = async () => {
        if (!file) return
        setStatus('uploading')
        setError(null)

        try {
            const data = await uploadVideo(file)
            setJobId(data.job_id)
            setStatus('processing')

            intervalRef.current = setInterval(async () => {
                try {
                    const job = await getJobStatus(data.job_id)
                    if (job.status === 'COMPLETED') {
                        clearPolling()
                        setStatus('completed')
                        onUploadComplete(job.ai_results)
                    } else if (job.status === 'FAILED') {
                        clearPolling()
                        setStatus('failed')
                        setError('AI analysis failed. Check backend logs.')
                    }
                } catch (err) {
                    clearPolling()
                    setStatus('failed')
                    setError(err.message)
                }
            }, 3000)
        } catch (err) {
            setStatus('failed')
            setError(err.message)
        }
    }

    if (status === 'completed') return null

    return (
        <div className="max-w-2xl mx-auto animate-fade-in">
            <div className="bg-white rounded-2xl border border-slate-200 p-8 shadow-sm">
                <h2 className="text-xl font-semibold text-slate-900 mb-1">Upload Classroom Video</h2>
                <p className="text-sm text-slate-500 mb-6">Upload an .mp4 video to analyze student engagement</p>

                {status === 'idle' || status === 'failed' ? (
                    <>
                        <div
                            className={`drop-zone rounded-xl p-10 text-center cursor-pointer ${dragActive ? 'active' : ''}`}
                            onDragEnter={handleDrag}
                            onDragLeave={handleDrag}
                            onDragOver={handleDrag}
                            onDrop={handleDrop}
                            onClick={() => inputRef.current?.click()}
                        >
                            <input
                                ref={inputRef}
                                type="file"
                                accept="video/mp4"
                                className="hidden"
                                onChange={(e) => setFile(e.target.files?.[0] || null)}
                            />
                            <Upload size={40} className="mx-auto text-slate-400 mb-3" />
                            {file ? (
                                <p className="text-sm font-medium text-slate-700">{file.name} <span className="text-slate-400">({(file.size / 1024 / 1024).toFixed(1)} MB)</span></p>
                            ) : (
                                <>
                                    <p className="text-sm font-medium text-slate-600">Drag & drop your video here</p>
                                    <p className="text-xs text-slate-400 mt-1">or click to browse (.mp4 only)</p>
                                </>
                            )}
                        </div>

                        {error && (
                            <div className="mt-4 flex items-center gap-2 text-red-600 text-sm bg-red-50 rounded-lg p-3">
                                <XCircle size={16} /> {error}
                            </div>
                        )}

                        <button
                            onClick={handleAnalyze}
                            disabled={!file}
                            className="mt-6 w-full py-3 rounded-xl bg-gradient-to-r from-primary-500 to-primary-600 text-white font-semibold hover:from-primary-600 hover:to-primary-700 disabled:opacity-40 disabled:cursor-not-allowed transition-all cursor-pointer shadow-lg shadow-primary-500/20"
                        >
                            Analyze Video
                        </button>
                    </>
                ) : (
                    <div className="text-center py-12">
                        <div className="relative inline-flex items-center justify-center mb-4">
                            <div className="w-16 h-16 rounded-full bg-primary-100 flex items-center justify-center">
                                <Loader2 size={28} className="text-primary-600 animate-spin" />
                            </div>
                            <div className="absolute w-16 h-16 rounded-full border-2 border-primary-400 animate-pulse-ring" />
                        </div>
                        <p className="text-lg font-semibold text-slate-800">
                            {status === 'uploading' ? 'Uploading video...' : 'AI is analyzing your video...'}
                        </p>
                        <p className="text-sm text-slate-500 mt-1">
                            {status === 'processing' && 'This may take a few minutes depending on video length'}
                        </p>
                        {jobId && (
                            <p className="text-xs text-slate-400 mt-3 font-mono">Job ID: {jobId}</p>
                        )}
                    </div>
                )}
            </div>
        </div>
    )
}

// ─── Results Section ────────────────────────────────────────

function ResultsSection({ data }) {
    const students = data?.students || []
    const [selectedIdx, setSelectedIdx] = useState(0)

    if (students.length === 0) {
        return (
            <div className="text-center py-20 text-slate-500">
                <Users size={48} className="mx-auto mb-3 text-slate-300" />
                <p className="text-lg font-medium">No students detected in this video.</p>
            </div>
        )
    }

    const student = students[selectedIdx]
    const emotions = student.emotions || {}
    const actions = student.actions || {}
    const timeline = student.engagement_over_time || []

    const emotionData = Object.entries(emotions).map(([name, value]) => ({ name, value }))
    const actionData = Object.entries(actions).map(([name, value]) => ({ name, value }))

    return (
        <div className="space-y-6 animate-fade-in">
            {/* Header with selector */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <div>
                    <h2 className="text-xl font-semibold text-slate-900">Analysis Results</h2>
                    <p className="text-sm text-slate-500">{students.length} student(s) detected</p>
                </div>
                <div className="relative">
                    <select
                        value={selectedIdx}
                        onChange={(e) => setSelectedIdx(Number(e.target.value))}
                        className="appearance-none bg-white border border-slate-200 rounded-lg pl-4 pr-10 py-2.5 text-sm font-medium text-slate-700 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500 cursor-pointer"
                    >
                        {students.map((s, i) => (
                            <option key={s.student_id} value={i}>Student {s.student_id}</option>
                        ))}
                    </select>
                    <ChevronDown size={16} className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none" />
                </div>
            </div>

            {/* KPI Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <KPICard
                    icon={Brain}
                    label="Dominant Emotion"
                    value={getTopEntry(emotions)}
                    sub={`out of ${Object.keys(emotions).length} detected`}
                    color="violet"
                />
                <KPICard
                    icon={Activity}
                    label="Primary Action"
                    value={getTopEntry(actions)}
                    sub={`out of ${Object.keys(actions).length} detected`}
                    color="emerald"
                />
                <KPICard
                    icon={TrendingUp}
                    label="Avg Engagement"
                    value={`${avgEngagement(timeline)}%`}
                    sub={`over ${timeline.length} seconds`}
                    color="primary"
                />
            </div>

            {/* Charts Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Engagement Timeline */}
                <div className="bg-white rounded-xl border border-slate-200 p-5 lg:col-span-2">
                    <h3 className="text-sm font-semibold text-slate-700 mb-4">Engagement Over Time</h3>
                    {timeline.length > 0 ? (
                        <ResponsiveContainer width="100%" height={280}>
                            <LineChart data={timeline}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                                <XAxis dataKey="second" tick={{ fontSize: 12 }} label={{ value: 'Second', position: 'insideBottom', offset: -5, fontSize: 12 }} />
                                <YAxis domain={[0, 100]} tick={{ fontSize: 12 }} label={{ value: 'Score', angle: -90, position: 'insideLeft', fontSize: 12 }} />
                                <Tooltip
                                    contentStyle={{ borderRadius: '8px', border: '1px solid #e2e8f0', fontSize: '13px' }}
                                    formatter={(val) => [`${val}%`, 'Engagement']}
                                />
                                <Line type="monotone" dataKey="score" stroke="#6366f1" strokeWidth={2} dot={false} activeDot={{ r: 5, fill: '#6366f1' }} />
                            </LineChart>
                        </ResponsiveContainer>
                    ) : (
                        <p className="text-sm text-slate-400 text-center py-10">No timeline data available</p>
                    )}
                </div>

                {/* Emotion Pie Chart */}
                <div className="bg-white rounded-xl border border-slate-200 p-5">
                    <h3 className="text-sm font-semibold text-slate-700 mb-4">Emotion Distribution</h3>
                    {emotionData.length > 0 ? (
                        <ResponsiveContainer width="100%" height={280}>
                            <PieChart>
                                <Pie
                                    data={emotionData}
                                    cx="50%"
                                    cy="50%"
                                    innerRadius={60}
                                    outerRadius={100}
                                    paddingAngle={3}
                                    dataKey="value"
                                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                                >
                                    {emotionData.map((_, i) => (
                                        <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />
                                    ))}
                                </Pie>
                                <Tooltip formatter={(val, name) => [val, name]} />
                            </PieChart>
                        </ResponsiveContainer>
                    ) : (
                        <p className="text-sm text-slate-400 text-center py-10">No emotion data</p>
                    )}
                </div>

                {/* Action Bar Chart */}
                <div className="bg-white rounded-xl border border-slate-200 p-5">
                    <h3 className="text-sm font-semibold text-slate-700 mb-4">Action Breakdown</h3>
                    {actionData.length > 0 ? (
                        <ResponsiveContainer width="100%" height={280}>
                            <BarChart data={actionData} layout="vertical">
                                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                                <XAxis type="number" tick={{ fontSize: 12 }} />
                                <YAxis type="category" dataKey="name" tick={{ fontSize: 12 }} width={100} />
                                <Tooltip contentStyle={{ borderRadius: '8px', border: '1px solid #e2e8f0', fontSize: '13px' }} />
                                <Bar
                                    dataKey="value"
                                    fill="#6366f1"
                                    radius={[0, 6, 6, 0]}
                                    activeBar={<Rectangle fill="#4f46e5" />}
                                />
                            </BarChart>
                        </ResponsiveContainer>
                    ) : (
                        <p className="text-sm text-slate-400 text-center py-10">No action data</p>
                    )}
                </div>
            </div>
        </div>
    )
}

// ─── Main Page ──────────────────────────────────────────────

export default function TeacherDashboard({ onLogout }) {
    const [results, setResults] = useState(null)

    return (
        <div className="min-h-screen bg-surface">
            <Navbar title="Teacher Dashboard" role="teacher" onLogout={onLogout} />
            <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                {results ? (
                    <>
                        <button
                            onClick={() => setResults(null)}
                            className="mb-6 text-sm text-primary-600 hover:text-primary-700 font-medium cursor-pointer"
                        >
                            &larr; Upload another video
                        </button>
                        <ResultsSection data={results} />
                    </>
                ) : (
                    <UploadSection onUploadComplete={setResults} />
                )}
            </main>
        </div>
    )
}
