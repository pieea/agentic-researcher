import React from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { ClusterInfo } from '../types'

interface TrendTimelineProps {
  clusters: ClusterInfo[]
}

const COLORS = [
  '#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6',
  '#ec4899', '#14b8a6', '#f97316', '#06b6d4', '#6366f1'
]

export function TrendTimeline({ clusters }: TrendTimelineProps) {
  if (!clusters || clusters.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-muted-foreground">
        <p>클러스터 데이터가 없습니다</p>
      </div>
    )
  }

  // Mock timeline data (in real implementation, extract from published_date)
  const data = Array.from({ length: 7 }, (_, i) => {
    const entry: any = { date: `Day ${i + 1}` }
    clusters.forEach((cluster) => {
      entry[cluster.name] = Math.floor(Math.random() * cluster.size) + 1
    })
    return entry
  })

  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart
        data={data}
        margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
      >
        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
        <XAxis
          dataKey="date"
          stroke="#9ca3af"
          fontSize={12}
        />
        <YAxis
          stroke="#9ca3af"
          fontSize={12}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: 'hsl(var(--card))',
            border: '1px solid hsl(var(--border))',
            borderRadius: '0.5rem',
            fontSize: '0.875rem'
          }}
        />
        <Legend
          wrapperStyle={{ fontSize: '0.875rem' }}
        />
        {clusters.slice(0, 5).map((cluster, index) => (
          <Line
            key={cluster.id}
            type="monotone"
            dataKey={cluster.name}
            stroke={COLORS[index % COLORS.length]}
            strokeWidth={2}
            dot={{ r: 3 }}
            activeDot={{ r: 5 }}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  )
}
