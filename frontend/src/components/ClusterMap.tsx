import React from 'react'
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Cell } from 'recharts'
import { ClusterInfo } from '../types'

interface ClusterMapProps {
  clusters: ClusterInfo[]
}

const COLORS = [
  '#8884d8', '#82ca9d', '#ffc658', '#ff8042', '#8dd1e1',
  '#d084d0', '#82ca82', '#ffc042', '#ff5042', '#8db1e1'
]

export function ClusterMap({ clusters }: ClusterMapProps) {
  // Transform clusters to scatter plot data
  const data = clusters.map((cluster, index) => ({
    x: index * 100 + Math.random() * 50,
    y: Math.random() * 100,
    size: cluster.size * 20,
    name: cluster.name,
    cluster: cluster
  }))

  return (
    <div className="cluster-map">
      <h3>Cluster Visualization</h3>
      <ScatterChart
        width={800}
        height={400}
        margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
      >
        <CartesianGrid />
        <XAxis type="number" dataKey="x" name="x" hide />
        <YAxis type="number" dataKey="y" name="y" hide />
        <Tooltip
          cursor={{ strokeDasharray: '3 3' }}
          content={({ payload }) => {
            if (!payload || payload.length === 0) return null
            const data = payload[0].payload
            return (
              <div className="custom-tooltip">
                <p><strong>{data.name}</strong></p>
                <p>{data.cluster.size} documents</p>
              </div>
            )
          }}
        />
        <Scatter data={data} fill="#8884d8">
          {data.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
          ))}
        </Scatter>
      </ScatterChart>
    </div>
  )
}
