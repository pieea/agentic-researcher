import React from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts'
import { ClusterInfo } from '../types'

interface TrendTimelineProps {
  clusters: ClusterInfo[]
}

export function TrendTimeline({ clusters }: TrendTimelineProps) {
  // Mock timeline data (in real implementation, extract from published_date)
  const data = Array.from({ length: 7 }, (_, i) => {
    const entry: any = { date: `Day ${i + 1}` }
    clusters.forEach((cluster) => {
      entry[cluster.name] = Math.floor(Math.random() * cluster.size)
    })
    return entry
  })

  const colors = [
    '#8884d8', '#82ca9d', '#ffc658', '#ff8042', '#8dd1e1'
  ]

  return (
    <div className="trend-timeline">
      <h3>Trend Timeline</h3>
      <LineChart
        width={800}
        height={300}
        data={data}
        margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
      >
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="date" />
        <YAxis />
        <Tooltip />
        <Legend />
        {clusters.map((cluster, index) => (
          <Line
            key={cluster.id}
            type="monotone"
            dataKey={cluster.name}
            stroke={colors[index % colors.length]}
            strokeWidth={2}
          />
        ))}
      </LineChart>
    </div>
  )
}
