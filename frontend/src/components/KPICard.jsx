/**
 * KPICard — Key Performance Indicator Card Component
 * ====================================================
 * Reusable card component for displaying a single metric/statistic.
 * Used in both the Teacher dashboard (Dominant Emotion, Primary Action, Avg Engagement)
 * and the HOD dashboard (Dept Avg, Lectures Analyzed, Top Course).
 *
 * Features:
 *   - Color-coded icon, value, and border based on the `color` prop.
 *   - Optional trend indicator (↑ or ↓ with percentage).
 *   - Glassmorphism styling with subtle glow effects.
 *   - Smooth fade-up entrance animation.
 *
 * Props:
 *   @param {Component} icon  - Lucide icon component (e.g., TrendingUp, Brain).
 *   @param {string}    label - Short label above the value (e.g., "Dept. Avg Engagement").
 *   @param {string}    value - The main metric value (e.g., "72%", "N/A").
 *   @param {string}    sub   - Small subtitle below the value (e.g., "Based on all videos").
 *   @param {string}    color - Color scheme key: 'emerald'|'indigo'|'amber'|'rose'|'violet'|'cyan'.
 *   @param {number}    trend - Optional trend percentage (positive = green ↑, negative = red ↓).
 */

export default function KPICard({ icon: Icon, label, value, sub, color = 'emerald', trend }) {
  /**
   * Color map — maps semantic color names to Tailwind utility classes.
   * Each color scheme provides consistent styling for:
   *   bg:     Background of the icon container
   *   icon:   Icon color
   *   value:  Main value text color
   *   border: Card border (with hover effect)
   *   glow:   Subtle shadow glow effect
   */
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

  // Fall back to emerald if an unknown color is passed
  const c = colorMap[color] || colorMap.emerald

  return (
    <div className={`glass rounded-2xl border p-5 transition-all duration-300 animate-fade-up shadow-lg ${c.border} ${c.glow}`}>
      {/* Top row: Icon + optional trend badge */}
      <div className="flex items-start justify-between mb-4">
        <div className={`${c.bg} w-11 h-11 rounded-xl flex items-center justify-center`}>
          {Icon && <Icon size={20} className={c.icon} />}
        </div>
        {/* Trend indicator — only shown when `trend` prop is provided */}
        {trend !== undefined && (
          <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${
            trend >= 0 ? 'bg-emerald-500/15 text-emerald-400' : 'bg-rose-500/15 text-rose-400'
          }`}>
            {trend >= 0 ? '↑' : '↓'} {Math.abs(trend)}%
          </span>
        )}
      </div>

      {/* Label (e.g., "Dept. Avg Engagement") */}
      <p className="text-slate-400 text-xs font-medium uppercase tracking-wider mb-1">{label}</p>

      {/* Main value (e.g., "72%") */}
      <p className={`text-2xl font-black font-display ${c.value}`}>{value}</p>

      {/* Subtitle (e.g., "Based on all videos") */}
      {sub && <p className="text-slate-500 text-xs mt-1.5">{sub}</p>}
    </div>
  )
}
