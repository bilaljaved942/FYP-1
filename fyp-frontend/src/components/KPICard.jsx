export default function KPICard({ icon: Icon, label, value, sub, color = 'primary' }) {
    const colorMap = {
        primary: 'from-primary-500 to-primary-600',
        emerald: 'from-emerald-500 to-emerald-600',
        amber: 'from-amber-500 to-amber-600',
        rose: 'from-rose-500 to-rose-600',
        violet: 'from-violet-500 to-violet-600',
        sky: 'from-sky-500 to-sky-600',
    }

    const bgMap = {
        primary: 'bg-primary-50',
        emerald: 'bg-emerald-50',
        amber: 'bg-amber-50',
        rose: 'bg-rose-50',
        violet: 'bg-violet-50',
        sky: 'bg-sky-50',
    }

    return (
        <div className="bg-white rounded-xl border border-slate-200 p-5 hover:shadow-md transition-shadow animate-fade-in">
            <div className="flex items-start justify-between">
                <div className="flex-1">
                    <p className="text-sm font-medium text-slate-500">{label}</p>
                    <p className="text-2xl font-bold text-slate-900 mt-1">{value}</p>
                    {sub && <p className="text-xs text-slate-400 mt-1">{sub}</p>}
                </div>
                {Icon && (
                    <div className={`${bgMap[color] || bgMap.primary} p-2.5 rounded-lg`}>
                        <Icon size={20} className={`bg-gradient-to-r ${colorMap[color] || colorMap.primary} bg-clip-text`} style={{ color: color === 'emerald' ? '#10b981' : color === 'amber' ? '#f59e0b' : color === 'rose' ? '#f43f5e' : color === 'violet' ? '#8b5cf6' : color === 'sky' ? '#0ea5e9' : '#6366f1' }} />
                    </div>
                )}
            </div>
        </div>
    )
}
