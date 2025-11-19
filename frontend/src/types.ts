export interface ResearchRequest {
  query: string
}

export interface ResearchResponse {
  request_id: string
  status: string
}

export interface ClusterInfo {
  id: number
  name: string
  size: number
  keywords: string[]
  documents: any[]
}

export interface InsightInfo {
  insights: string[]
  summary: string
  cluster_count: number
  total_documents: number
}

export interface ResearchResult {
  request_id: string
  query: string
  status: string
  clusters: ClusterInfo[]
  insights?: InsightInfo
  created_at: string
  completed_at?: string
}

export type ResearchStatus =
  | 'idle'
  | 'searching'
  | 'analyzing'
  | 'clustering'
  | 'generating_insights'
  | 'completed'
  | 'failed'
