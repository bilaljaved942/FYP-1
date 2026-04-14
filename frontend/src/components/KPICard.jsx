export default function KPICard({ icon: Icon, label, value, sub, color = 'emerald', trend }) {
  const colorMap = {
    emerald: {
      bg:     'bg-emerald-500/10',
      icon:   'text-emerald-400',
      value:  'text-emerald-400',
      border: 'border-emerald-500/20 hover:border-emerald-500/40',
      glow:   'shadow-emerald-900/20',
    },
    indigo: {
      bg:     'bg-indigo-500/10',
      icon:   'text-indigo-400',
      value:  'text-indigo-400',
      border: 'border-indigo-500/20 hover:border-indigo-500/40',
      glow:   'shadow-indigo-900/20',
    },
    amber: {
      bg:     'bg-amber-500/10',
      icon:   'text-amber-400',
      value:  'text-amber-400',
      border: 'border-amber-500/20 hover:border-amber-500/40',
      glow:   'shadow-amber-900/20',
    },
    rose: {
      bg:     'bg-rose-500/10',
      icon:   'text-rose-400',
      value:  'text-rose-400',
      border: 'border-rose-500/20 hover:border-rose-500/40',
      glow:   'shadow-rose-900/20',
    },
    violet: {
      bg:     'bg-violet-500/10',
      icon:   'text-violet-400',
      value:  'text-violet-400',
      border: 'border-violet-500/20 hover:border-violet-500/40',
      glow:   'shadow-violet-900/20',
    },
    cyan: {
      bg:     'bg-cyan-500/10',
      icon:   'text-cyan-400',
      value:  'text-cyan-400',
      border: 'border-cyan-500/20 hover:border-cyan-500/40',
      glow:   'shadow-cyan-900/20',
    },
    primary: {
      bg:     'bg-emerald-500/10',
      icon:   'text-emerald-400',
      value:  'text-emerald-400',
      border: 'border-emerald-500/20 hover:border-emerald-500/40',
      glow:   'shadow-emerald-900/20',
    },
  }

  const c = colorMap[color] || colorMap.emerald

  return (
    <div className={`glass rounded-2xl border p-5 transition-all duration-300 animate-fade-up shadow-lg ${c.border} ${c.glow}`}>
      <div className="flex items-start justify-between mb-4">
        <div className={`${c.bg} w-11 h-11 rounded-xl flex items-center justify-center`}>
          {Icon && <Icon size={20} className={c.icon} />}
        </div>
        {trend !== undefined && (
          <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${
            trend >= 0 ? 'bg-emerald-500/15 text-emerald-400' : 'bg-rose-500/15 text-rose-400'
          }`}>
            {trend >= 0 ? '↑' : '↓'} {Math.abs(trend)}%
          </span>
        )}
      </div>
      <p className="text-slate-400 text-xs font-medium uppercase tracking-wider mb-1">{label}</p>
      <p className={`text-2xl font-black font-display ${c.value}`}>{value}</p>
      {sub && <p className="text-slate-500 text-xs mt-1.5">{sub}</p>}
    </div>
  )
}
