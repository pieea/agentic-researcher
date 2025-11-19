import React, { useState } from 'react'
import { createResearchRequest, streamResearchProgress, getResearchResult } from './api/research'
import { ResearchResult, ResearchStatus } from './types'
import { ClusterMap } from './components/ClusterMap'
import { TrendTimeline } from './components/TrendTimeline'

function App() {
  const [query, setQuery] = useState('')
  const [status, setStatus] = useState<ResearchStatus>('idle')
  const [result, setResult] = useState<ResearchResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleSearch = async () => {
    if (!query.trim()) return

    setStatus('searching')
    setError(null)
    setResult(null)

    try {
      // Create research request
      const response = await createResearchRequest(query)

      // Stream progress updates
      streamResearchProgress(
        response.request_id,
        (newStatus) => {
          setStatus(newStatus as ResearchStatus)
        },
        async () => {
          // Fetch final results
          const finalResult = await getResearchResult(response.request_id)
          setResult(finalResult)
          setStatus('completed')
        },
        (err) => {
          setError(err.message)
          setStatus('failed')
        }
      )
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
      setStatus('failed')
    }
  }

  return (
    <div className="app">
      <header>
        <h1>Market Research Platform</h1>
      </header>

      <main>
        <div className="search-box">
          <input
            type="text"
            placeholder="Enter topic to research..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={status !== 'idle' && status !== 'completed' && status !== 'failed'}
          />
          <button
            onClick={handleSearch}
            disabled={status !== 'idle' && status !== 'completed' && status !== 'failed'}
          >
            Research
          </button>
        </div>

        {status !== 'idle' && status !== 'completed' && (
          <div className="status">
            <p>Status: {status}</p>
          </div>
        )}

        {error && (
          <div className="error">
            <p>Error: {error}</p>
          </div>
        )}

        {result && (
          <div className="results">
            <h2>Results for: {result.query}</h2>

            {result.insights && (
              <div className="insights">
                <h3>Key Insights</h3>
                <ul>
                  {result.insights.insights.map((insight, i) => (
                    <li key={i}>{insight}</li>
                  ))}
                </ul>
              </div>
            )}

            <div className="visualizations">
              <ClusterMap clusters={result.clusters} />
              <TrendTimeline clusters={result.clusters} />
            </div>

            <div className="clusters">
              {result.clusters.map((cluster) => (
                <div key={cluster.id} className="cluster-card">
                  <h3>{cluster.name}</h3>
                  <p>{cluster.size} documents</p>
                  <div className="keywords">
                    {cluster.keywords.map((kw, i) => (
                      <span key={i} className="keyword">{kw}</span>
                    ))}
                  </div>
                  <div className="documents">
                    {cluster.documents.slice(0, 3).map((doc, i) => (
                      <div key={i} className="document">
                        <a href={doc.url} target="_blank" rel="noopener noreferrer">
                          {doc.title}
                        </a>
                        <span className="source">{doc.source}</span>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  )
}

export default App
