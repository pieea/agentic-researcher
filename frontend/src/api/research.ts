import { ResearchRequest, ResearchResponse, ResearchResult } from '../types'

const API_BASE = '/api'

export async function createResearchRequest(
  query: string
): Promise<ResearchResponse> {
  const response = await fetch(`${API_BASE}/research`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ query }),
  })

  if (!response.ok) {
    throw new Error('Failed to create research request')
  }

  return response.json()
}

export async function getResearchResult(
  requestId: string
): Promise<ResearchResult> {
  const response = await fetch(`${API_BASE}/research/${requestId}`)

  if (!response.ok) {
    throw new Error('Failed to fetch research result')
  }

  return response.json()
}

export function streamResearchProgress(
  requestId: string,
  onProgress: (status: string) => void,
  onComplete: () => void,
  onError: (error: Error) => void
): EventSource {
  const eventSource = new EventSource(
    `${API_BASE}/research/${requestId}/stream`
  )

  eventSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data)
      onProgress(data.status)

      if (data.status === 'completed' || data.status === 'failed') {
        eventSource.close()
        onComplete()
      }
    } catch (error) {
      onError(error as Error)
      eventSource.close()
    }
  }

  eventSource.onerror = (error) => {
    onError(new Error('SSE connection error'))
    eventSource.close()
  }

  return eventSource
}
