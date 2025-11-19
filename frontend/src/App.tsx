import { useState } from 'react'
import { Search, Loader2, TrendingUp, AlertCircle } from 'lucide-react'
import { createResearchRequest, streamResearchProgress, getResearchResult } from './api/research'
import { ResearchResult, ResearchStatus, ProgressUpdate } from './types'
import { ClusterMap } from './components/ClusterMap'
import { TrendTimeline } from './components/TrendTimeline'
import { Button } from './components/ui/button'
import { Input } from './components/ui/input'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './components/ui/card'
import { Badge } from './components/ui/badge'

function App() {
  const [query, setQuery] = useState('')
  const [status, setStatus] = useState<ResearchStatus>('idle')
  const [result, setResult] = useState<ResearchResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [progress, setProgress] = useState<ProgressUpdate | null>(null)

  const handleSearch = async () => {
    if (!query.trim()) return

    setStatus('searching')
    setError(null)
    setResult(null)
    setProgress(null)

    try {
      const response = await createResearchRequest(query)

      streamResearchProgress(
        response.request_id,
        (progressUpdate) => {
          setStatus(progressUpdate.status)
          setProgress(progressUpdate)
        },
        async () => {
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

  const isLoading = status !== 'idle' && status !== 'completed' && status !== 'failed'

  // Determine which node to display based on status
  const getCurrentNode = () => {
    if (!progress) return null
    if (progress.node) return progress.node

    // Fallback to status-based detection
    if (['initialized', 'searching', 'search_completed'].includes(status)) return 'search'
    if (['analyzing', 'clustering_completed', 'clustering_skipped'].includes(status)) return 'analysis'
    if (['generating_insights'].includes(status)) return 'insight'
    return null
  }

  const currentNode = getCurrentNode()

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        {/* Header */}
        <header className="text-center mb-12">
          <div className="inline-flex items-center gap-2 mb-4">
            <TrendingUp className="h-10 w-10 text-primary" />
            <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-violet-600 bg-clip-text text-transparent">
              Market Research Platform
            </h1>
          </div>
          <p className="text-muted-foreground text-lg">
            AI-powered trend analysis with automatic clustering and insights
          </p>
        </header>

        {/* Search Box */}
        <Card className="mb-8 shadow-lg">
          <CardContent className="pt-6">
            <div className="flex gap-3">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  type="text"
                  placeholder="Enter topic to research (e.g., 'AI agents', 'sustainable tech')..."
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && !isLoading && handleSearch()}
                  disabled={isLoading}
                  className="pl-10 h-12 text-base"
                />
              </div>
              <Button
                onClick={handleSearch}
                disabled={isLoading || !query.trim()}
                size="lg"
                className="px-8"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    분석 중
                  </>
                ) : (
                  '분석 시작'
                )}
              </Button>
            </div>

            {/* Compact Progress Bar */}
            {isLoading && progress && currentNode && (
              <div className="mt-4 pt-4 border-t">
                <div className="flex items-center gap-3">
                  <div className="flex items-center gap-2 min-w-0 flex-1">
                    {currentNode === 'search' && (
                      <>
                        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center animate-pulse">
                          <span className="text-lg">🔍</span>
                        </div>
                        <div className="min-w-0">
                          <div className="font-semibold text-sm text-blue-700">검색 단계</div>
                          <div className="text-xs text-muted-foreground truncate">
                            {progress.message || '검색 중...'}
                            {progress.results_count && ` (${progress.results_count}개 발견)`}
                          </div>
                        </div>
                      </>
                    )}
                    {currentNode === 'analysis' && (
                      <>
                        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-green-100 flex items-center justify-center animate-pulse">
                          <span className="text-lg">📊</span>
                        </div>
                        <div className="min-w-0">
                          <div className="font-semibold text-sm text-green-700">분석 단계</div>
                          <div className="text-xs text-muted-foreground truncate">
                            {progress.message || '데이터 분석 중...'}
                            {progress.clusters_count && ` (${progress.clusters_count}개 주제)`}
                          </div>
                        </div>
                      </>
                    )}
                    {currentNode === 'insight' && (
                      <>
                        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-violet-100 flex items-center justify-center animate-pulse">
                          <span className="text-lg">💡</span>
                        </div>
                        <div className="min-w-0">
                          <div className="font-semibold text-sm text-violet-700">인사이트 생성 단계</div>
                          <div className="text-xs text-muted-foreground truncate">
                            {progress.message || '인사이트 생성 중...'}
                          </div>
                        </div>
                      </>
                    )}
                  </div>
                  <Loader2 className="flex-shrink-0 h-4 w-4 animate-spin text-primary" />
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Progress Indicator */}
        {isLoading && progress && (
          <Card className="mb-8 border-primary/20 shadow-lg">
            <CardHeader>
              <CardTitle className="text-lg">진행 상황</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-between gap-8 mb-6">
                {/* Search Step */}
                <div className={`flex-1 flex flex-col items-center gap-3 transition-all ${
                  progress.node === 'search' || status === 'search_completed' ? 'opacity-100 scale-105' : 'opacity-40'
                }`}>
                  <div className={`w-16 h-16 rounded-full flex items-center justify-center text-3xl transition-all ${
                    status === 'search_completed'
                      ? 'bg-green-100 ring-4 ring-green-200'
                      : progress.node === 'search'
                      ? 'bg-blue-100 ring-4 ring-blue-200 animate-pulse'
                      : 'bg-slate-100'
                  }`}>
                    🔍
                  </div>
                  <div className="text-center">
                    <div className="font-semibold text-sm">검색</div>
                    {progress.node === 'search' && progress.results_count && (
                      <div className="text-xs text-primary font-medium mt-1">
                        {progress.results_count}개 결과
                      </div>
                    )}
                  </div>
                </div>

                <div className="h-1 flex-1 bg-gradient-to-r from-slate-200 to-slate-300 rounded-full" />

                {/* Analysis Step */}
                <div className={`flex-1 flex flex-col items-center gap-3 transition-all ${
                  progress.node === 'analysis' || status === 'clustering_completed' || status === 'clustering_skipped'
                    ? 'opacity-100 scale-105' : 'opacity-40'
                }`}>
                  <div className={`w-16 h-16 rounded-full flex items-center justify-center text-3xl transition-all ${
                    status === 'clustering_completed' || status === 'clustering_skipped'
                      ? 'bg-green-100 ring-4 ring-green-200'
                      : progress.node === 'analysis'
                      ? 'bg-blue-100 ring-4 ring-blue-200 animate-pulse'
                      : 'bg-slate-100'
                  }`}>
                    📊
                  </div>
                  <div className="text-center">
                    <div className="font-semibold text-sm">분석</div>
                    {(status === 'clustering_completed' || status === 'clustering_skipped') && progress.clusters_count && (
                      <div className="text-xs text-primary font-medium mt-1">
                        {progress.clusters_count}개 주제
                      </div>
                    )}
                  </div>
                </div>

                <div className="h-1 flex-1 bg-gradient-to-r from-slate-200 to-slate-300 rounded-full" />

                {/* Insight Step */}
                <div className={`flex-1 flex flex-col items-center gap-3 transition-all ${
                  progress.node === 'insight' || status === 'completed' ? 'opacity-100 scale-105' : 'opacity-40'
                }`}>
                  <div className={`w-16 h-16 rounded-full flex items-center justify-center text-3xl transition-all ${
                    status === 'completed'
                      ? 'bg-green-100 ring-4 ring-green-200'
                      : progress.node === 'insight'
                      ? 'bg-blue-100 ring-4 ring-blue-200 animate-pulse'
                      : 'bg-slate-100'
                  }`}>
                    💡
                  </div>
                  <div className="text-center">
                    <div className="font-semibold text-sm">인사이트</div>
                    {status === 'completed' && progress.insights_count && (
                      <div className="text-xs text-primary font-medium mt-1">
                        {progress.insights_count}개 도출
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {progress.message && (
                <div className="text-center py-3 px-4 bg-primary/5 rounded-lg">
                  <p className="text-sm font-medium text-primary">{progress.message}</p>
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {/* Error */}
        {error && (
          <Card className="mb-8 border-destructive bg-destructive/5">
            <CardContent className="pt-6">
              <div className="flex items-start gap-3">
                <AlertCircle className="h-5 w-5 text-destructive mt-0.5" />
                <div>
                  <p className="font-semibold text-destructive">오류가 발생했습니다</p>
                  <p className="text-sm text-destructive/80 mt-1">{error}</p>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Results */}
        {result && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-2xl font-bold">
                분석 결과: <span className="text-primary">{result.query}</span>
              </h2>
              <Badge variant="secondary" className="text-sm px-3 py-1">
                {result.clusters.length}개 클러스터
              </Badge>
            </div>

            {/* Key Insights */}
            {result.insights && (
              <div className="space-y-6">
                {/* 핵심 인사이트 */}
                <Card className="shadow-lg">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <span className="text-2xl">💡</span>
                      핵심 인사이트
                    </CardTitle>
                    <CardDescription>
                      AI가 분석한 주요 트렌드와 인사이트
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <ul className="space-y-3">
                      {result.insights.insights.map((insight, i) => (
                        <li key={i} className="flex items-start gap-3 p-3 rounded-lg bg-primary/5 border border-primary/10">
                          <Badge className="mt-0.5">{i + 1}</Badge>
                          <span className="text-sm leading-relaxed">{insight}</span>
                        </li>
                      ))}
                    </ul>
                  </CardContent>
                </Card>

                {/* 성공사례 & 실패사례 */}
                <div className="grid md:grid-cols-2 gap-6">
                  {result.insights.success_cases && result.insights.success_cases.length > 0 && (
                    <Card className="shadow-lg border-green-200 bg-green-50/50">
                      <CardHeader>
                        <CardTitle className="flex items-center gap-2 text-green-700">
                          <span className="text-2xl">✅</span>
                          성공 사례
                        </CardTitle>
                        <CardDescription>
                          시장에서 검증된 성공 전략
                        </CardDescription>
                      </CardHeader>
                      <CardContent>
                        <ul className="space-y-3">
                          {result.insights.success_cases.map((case_item, i) => (
                            <li key={i} className="flex items-start gap-3 p-3 rounded-lg bg-white border border-green-200">
                              <Badge variant="outline" className="mt-0.5 border-green-600 text-green-700">{i + 1}</Badge>
                              <span className="text-sm leading-relaxed text-gray-700">{case_item}</span>
                            </li>
                          ))}
                        </ul>
                      </CardContent>
                    </Card>
                  )}

                  {result.insights.failure_cases && result.insights.failure_cases.length > 0 && (
                    <Card className="shadow-lg border-red-200 bg-red-50/50">
                      <CardHeader>
                        <CardTitle className="flex items-center gap-2 text-red-700">
                          <span className="text-2xl">⚠️</span>
                          실패 사례
                        </CardTitle>
                        <CardDescription>
                          피해야 할 함정과 교훈
                        </CardDescription>
                      </CardHeader>
                      <CardContent>
                        <ul className="space-y-3">
                          {result.insights.failure_cases.map((case_item, i) => (
                            <li key={i} className="flex items-start gap-3 p-3 rounded-lg bg-white border border-red-200">
                              <Badge variant="outline" className="mt-0.5 border-red-600 text-red-700">{i + 1}</Badge>
                              <span className="text-sm leading-relaxed text-gray-700">{case_item}</span>
                            </li>
                          ))}
                        </ul>
                      </CardContent>
                    </Card>
                  )}
                </div>

                {/* 향후 시장 전망 */}
                {result.insights.market_outlook && result.insights.market_outlook.length > 0 && (
                  <Card className="shadow-lg border-violet-200 bg-gradient-to-br from-violet-50 to-blue-50">
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2 text-violet-700">
                        <span className="text-2xl">🔮</span>
                        향후 시장 전망
                      </CardTitle>
                      <CardDescription>
                        미래 트렌드와 예측
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <ul className="space-y-3">
                        {result.insights.market_outlook.map((outlook, i) => (
                          <li key={i} className="flex items-start gap-3 p-3 rounded-lg bg-white border border-violet-200">
                            <Badge variant="outline" className="mt-0.5 border-violet-600 text-violet-700">{i + 1}</Badge>
                            <span className="text-sm leading-relaxed text-gray-700">{outlook}</span>
                          </li>
                        ))}
                      </ul>
                    </CardContent>
                  </Card>
                )}
              </div>
            )}

            {/* Visualizations */}
            <div className="grid md:grid-cols-2 gap-6">
              <Card className="shadow-lg">
                <CardHeader>
                  <CardTitle>클러스터 맵</CardTitle>
                  <CardDescription>주제별 분포 시각화</CardDescription>
                </CardHeader>
                <CardContent>
                  <ClusterMap clusters={result.clusters} />
                </CardContent>
              </Card>

              <Card className="shadow-lg">
                <CardHeader>
                  <CardTitle>트렌드 타임라인</CardTitle>
                  <CardDescription>시간별 트렌드 변화</CardDescription>
                </CardHeader>
                <CardContent>
                  <TrendTimeline clusters={result.clusters} />
                </CardContent>
              </Card>
            </div>

            {/* Cluster Details */}
            <div>
              <h3 className="text-xl font-bold mb-4">상세 클러스터</h3>
              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
                {result.clusters.map((cluster) => (
                  <Card key={cluster.id} className="shadow-md hover:shadow-lg transition-shadow">
                    <CardHeader>
                      <CardTitle className="text-lg">{cluster.name}</CardTitle>
                      <CardDescription>{cluster.size}개 문서</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      {/* Keywords */}
                      <div>
                        <p className="text-sm font-semibold text-muted-foreground mb-2">키워드</p>
                        <div className="flex flex-wrap gap-2">
                          {cluster.keywords.map((kw, i) => (
                            <Badge key={i} variant="secondary" className="text-xs">
                              {kw}
                            </Badge>
                          ))}
                        </div>
                      </div>

                      {/* Documents */}
                      <div>
                        <p className="text-sm font-semibold text-muted-foreground mb-2">주요 문서</p>
                        <div className="space-y-2">
                          {cluster.documents.slice(0, 3).map((doc, i) => (
                            <div key={i} className="text-sm">
                              <a
                                href={doc.url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-primary hover:underline line-clamp-2"
                              >
                                {doc.title}
                              </a>
                              <p className="text-xs text-muted-foreground mt-0.5">
                                {doc.source}
                              </p>
                            </div>
                          ))}
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default App
