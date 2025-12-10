import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  Legend,
} from 'recharts'

const COLORS = {
  emotions: ['#ef4444', '#f97316', '#22c55e', '#6b7280', '#3b82f6', '#8b5cf6'],
  actions: ['#0ea5e9', '#14b8a6', '#f59e0b', '#ec4899'],
  engagement: ['#22c55e', '#ef4444'],
}

function ChartCard({ title, children }) {
  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">{title}</h3>
      <div className="h-64">{children}</div>
    </div>
  )
}

export function EmotionChart({ data }) {
  const chartData = Object.entries(data).map(([name, value]) => ({
    name: name.charAt(0).toUpperCase() + name.slice(1),
    count: value,
  }))

  return (
    <ChartCard title="Emotion Analysis">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis dataKey="name" tick={{ fontSize: 12 }} />
          <YAxis tick={{ fontSize: 12 }} />
          <Tooltip
            contentStyle={{
              backgroundColor: '#fff',
              border: '1px solid #e5e7eb',
              borderRadius: '8px',
            }}
          />
          <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]}>
            {chartData.map((_, index) => (
              <Cell key={`cell-${index}`} fill={COLORS.emotions[index % COLORS.emotions.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </ChartCard>
  )
}

export function ActionsChart({ data }) {
  const chartData = Object.entries(data).map(([name, value]) => ({
    name: name
      .split('_')
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' '),
    count: value,
  }))

  return (
    <ChartCard title="Student Actions">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis dataKey="name" tick={{ fontSize: 12 }} />
          <YAxis tick={{ fontSize: 12 }} />
          <Tooltip
            contentStyle={{
              backgroundColor: '#fff',
              border: '1px solid #e5e7eb',
              borderRadius: '8px',
            }}
          />
          <Bar dataKey="count" fill="#0ea5e9" radius={[4, 4, 0, 0]}>
            {chartData.map((_, index) => (
              <Cell key={`cell-${index}`} fill={COLORS.actions[index % COLORS.actions.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </ChartCard>
  )
}

export function EngagementSummaryChart({ data }) {
  // Calculate engagement based on actions
  // Engaged: raising_hand, writing_notes, looking_at_board
  // Disengaged: sleeping, looking_away, talking
  const engagedActions = ['raising_hand', 'writing_notes', 'looking_at_board']
  const disengagedActions = ['sleeping', 'looking_away', 'talking']

  let engaged = 0
  let disengaged = 0

  Object.entries(data).forEach(([action, count]) => {
    if (engagedActions.includes(action)) {
      engaged += count
    } else if (disengagedActions.includes(action)) {
      disengaged += count
    } else {
      // Default: sleeping is disengaged, others are engaged
      if (action === 'sleeping') {
        disengaged += count
      } else {
        engaged += count
      }
    }
  })

  const chartData = [
    { name: 'Engaged', value: engaged },
    { name: 'Disengaged', value: disengaged },
  ]

  return (
    <ChartCard title="Engagement Summary">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={chartData}
            cx="50%"
            cy="50%"
            innerRadius={50}
            outerRadius={80}
            paddingAngle={5}
            dataKey="value"
            label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
          >
            {chartData.map((_, index) => (
              <Cell key={`cell-${index}`} fill={COLORS.engagement[index]} />
            ))}
          </Pie>
          <Tooltip
            contentStyle={{
              backgroundColor: '#fff',
              border: '1px solid #e5e7eb',
              borderRadius: '8px',
            }}
          />
          <Legend />
        </PieChart>
      </ResponsiveContainer>
    </ChartCard>
  )
}

export function EngagementTimelineChart({ data }) {
  const chartData = data.map((item) => ({
    second: `${item.second}s`,
    score: item.score,
  }))

  return (
    <ChartCard title="Engagement Over Time">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis dataKey="second" tick={{ fontSize: 12 }} />
          <YAxis tick={{ fontSize: 12 }} domain={[0, 100]} />
          <Tooltip
            contentStyle={{
              backgroundColor: '#fff',
              border: '1px solid #e5e7eb',
              borderRadius: '8px',
            }}
          />
          <Line
            type="monotone"
            dataKey="score"
            stroke="#8b5cf6"
            strokeWidth={3}
            dot={{ fill: '#8b5cf6', strokeWidth: 2, r: 4 }}
            activeDot={{ r: 6 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </ChartCard>
  )
}
