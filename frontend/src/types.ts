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
  success_cases: string[]
  failure_cases: string[]
  market_outlook: string[]
  summary: string
  cluster_count: number
  total_documents: number
}

export interface SearchResult {
  title: string
  url: string
  content: string
  score: number
  published_date?: string
  source: string
}

export interface ResearchResult {
  request_id: string
  query: string
  status: string
  raw_results: SearchResult[]
  clusters: ClusterInfo[]
  insights?: InsightInfo
  created_at: string
  completed_at?: string
}

export type ResearchStatus =
  | 'idle'
  | 'searching'
  | 'search_completed'
  | 'analyzing'
  | 'clustering_completed'
  | 'clustering_skipped'
  | 'generating_insights'
  | 'completed'
  | 'failed'

export interface ProgressUpdate {
  status: ResearchStatus
  query?: string
  message?: string
  node?: 'search' | 'analysis' | 'insight'
  results_count?: number
  clusters_count?: number
  insights_count?: number
  error?: string
}
