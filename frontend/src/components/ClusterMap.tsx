import React from 'react'
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Cell, ResponsiveContainer, LabelList } from 'recharts'
import { ClusterInfo } from '../types'

interface ClusterMapProps {
  clusters: ClusterInfo[]
}

const COLORS = [
  '#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6',
  '#ec4899', '#14b8a6', '#f97316', '#06b6d4', '#6366f1'
]

export function ClusterMap({ clusters }: ClusterMapProps) {
  if (!clusters || clusters.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-muted-foreground">
        <p>클러스터 데이터가 없습니다</p>
      </div>
    )
  }

  // Transform clusters to scatter plot data with better positioning
  const data = clusters.map((cluster, index) => {
    const angle = (index * 2 * Math.PI) / clusters.length
    const radius = 30 + Math.random() * 20
    return {
      x: 50 + Math.cos(angle) * radius,
      y: 50 + Math.sin(angle) * radius,
      size: Math.max(cluster.size * 50, 100),
      name: cluster.name,
      count: cluster.size,
      label: `${cluster.name} (${cluster.size})`,
      cluster: cluster
    }
  })

  return (
    <ResponsiveContainer width="100%" height={300}>
      <ScatterChart
        margin={{ top: 10, right: 10, bottom: 10, left: 10 }}
      >
        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
        <XAxis
          type="number"
          dataKey="x"
          domain={[0, 100]}
          hide
        />
        <YAxis
          type="number"
          dataKey="y"
          domain={[0, 100]}
          hide
        />
        <Tooltip
          cursor={{ strokeDasharray: '3 3' }}
          content={({ payload }) => {
            if (!payload || payload.length === 0) return null
            const data = payload[0].payload
            return (
              <div className="bg-card border border-border rounded-lg p-3 shadow-lg">
                <p className="font-semibold text-sm">{data.name}</p>
                <p className="text-xs text-muted-foreground mt-1">{data.count}개 문서</p>
              </div>
            )
          }}
        />
        <Scatter data={data} fill="#8884d8">
          {data.map((entry, index) => (
            <Cell
              key={`cell-${index}`}
              fill={COLORS[index % COLORS.length]}
              opacity={0.8}
            />
          ))}
          <LabelList
            dataKey="label"
            position="top"
            style={{
              fontSize: '12px',
              fontWeight: 600,
              fill: '#374151'
            }}
          />
        </Scatter>
      </ScatterChart>
    </ResponsiveContainer>
  )
}
