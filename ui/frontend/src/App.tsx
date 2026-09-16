import { useEffect, useRef, useState } from 'react'
import './App.css'
import { getPersonalization, listUsers } from './api'
import { streamQuery } from './api'
import { IndexFolder } from './components/IndexFolder'
import { PersonalizationPanel } from './components/PersonalizationPanel'
import { ResultsList } from './components/ResultsList'
import { TracePanel } from './components/TracePanel'
import type { PersonalizationInsights, ResultItem, RoutingTraceData, StageTrace, UserSummary } from './types'

const EXAMPLE_QUERIES = [
  'Find my latest presentation',
  'Show spreadsheets I used recently',
  'Find the notes about my project from last week',
]

function App() {
  const [users, setUsers] = useState<UserSummary[]>([])
  const [selectedUserId, setSelectedUserId] = useState<number | null>(null)
  const [query, setQuery] = useState('')

  const [tier, setTier] = useState<string | null>(null)
  const [rationale, setRationale] = useState('')
  const [stagesByName, setStagesByName] = useState<Record<string, StageTrace>>({})
  const [results, setResults] = useState<ResultItem[]>([])
  const [finalTrace, setFinalTrace] = useState<RoutingTraceData | null>(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const [lastQuery, setLastQuery] = useState('')

  const [insights, setInsights] = useState<PersonalizationInsights | null>(null)
  const [insightsLoading, setInsightsLoading] = useState(false)
  const [apiError, setApiError] = useState<string | null>(null)

  const cancelRef = useRef<(() => void) | null>(null)

  useEffect(() => {
    listUsers()
      .then((u) => {
        setUsers(u)
        if (u.length > 0) setSelectedUserId(u[0].id)
      })
      .catch(() => setApiError('Sift cannot connect right now. Please make sure the Sift service is running, then refresh this page.'))
  }, [])

  useEffect(() => {
    if (selectedUserId === null) return
    setInsightsLoading(true)
    getPersonalization(selectedUserId)
      .then(setInsights)
      .catch(() => setInsights(null))
      .finally(() => setInsightsLoading(false))
  }, [selectedUserId, results])

  function runQuery() {
    if (selectedUserId === null || !query.trim()) return
    cancelRef.current?.()

    setTier(null)
    setRationale('')
    setStagesByName({})
    setResults([])
    setFinalTrace(null)
    setIsStreaming(true)
    setLastQuery(query)
    setApiError(null)

    cancelRef.current = streamQuery(
      query,
      selectedUserId,
      (event) => {
        if (event.type === 'route') {
          setTier(event.tier)
          setRationale(event.rationale)
        } else if (event.type === 'stage') {
          setStagesByName((prev) => ({
            ...prev,
            [event.name]: { name: event.name, duration_ms: event.duration_ms, skipped: event.skipped, detail: event.detail },
          }))
        } else if (event.type === 'done') {
          setResults(event.results)
          setFinalTrace(event.routing_trace)
          setIsStreaming(false)
        }
      },
      () => {
        setIsStreaming(false)
        setApiError('Query stream failed — check that the API is still running.')
      },
    )
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="brand-mark" aria-hidden="true">S</div>
        <div>
          <h1>Sift</h1>
          <p className="app-subtitle">Find the right file, even when you only remember part of it.</p>
        </div>
      </header>

      {apiError && <div className="error-banner">{apiError}</div>}

      <section className="search-hero">
        <div className="search-copy">
          <h2>What are you looking for?</h2>
          <p>Use everyday words. Try a name, topic, file type, or when you last used it.</p>
        </div>
        <div className="query-bar">
          <span className="search-icon" aria-hidden="true">⌕</span>
          <input
            className="query-input"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && runQuery()}
            placeholder="e.g. the budget spreadsheet I opened last month"
            aria-label="Describe the file you want to find"
          />
          <button className="run-button" onClick={runQuery} disabled={isStreaming || !query.trim() || selectedUserId === null}>
            {isStreaming ? 'Finding files…' : 'Find files'}
          </button>
        </div>
        <div className="example-area">
          <span>Try:</span>
          {EXAMPLE_QUERIES.map((q) => (
            <button key={q} className="example-chip" onClick={() => setQuery(q)}>
              {q}
            </button>
          ))}
        </div>
        <div className="search-options">
          <label className="profile-picker">
            <span>Personalize for</span>
            <select
              value={selectedUserId ?? ''}
              onChange={(e) => setSelectedUserId(Number(e.target.value))}
              className="user-select"
              aria-label="Choose the profile whose file history should personalize results"
            >
              {users.map((u) => (
                <option key={u.id} value={u.id}>{u.name}</option>
              ))}
            </select>
          </label>
          <span className="privacy-note">Your files stay on this device.</span>
        </div>
      </section>

      <IndexFolder />

      <div className="main-grid">
        <main className="main-column">
          <section className="results-panel panel" aria-live="polite">
            <div className="results-heading">
              <div>
                <h2>{isStreaming ? 'Looking through your files…' : results.length ? 'Recommended files' : 'Your recommendations will appear here'}</h2>
                <p>{lastQuery && !isStreaming ? `Results for “${lastQuery}”` : 'Search by what you remember — Sift handles the rest.'}</p>
              </div>
              {results.length > 0 && <span className="results-count">{results.length} found</span>}
            </div>
            <ResultsList results={results} userId={selectedUserId ?? 0} query={lastQuery} />
          </section>

          {(tier || finalTrace) && (
            <details className="advanced-details">
              <summary>How Sift found these files</summary>
              <p>This is the technical search path used for this request.</p>
              <TracePanel
                tier={tier}
                rationale={rationale}
                stagesByName={stagesByName}
                isStreaming={isStreaming}
                finalTrace={finalTrace}
              />
            </details>
          )}
        </main>

        <aside className="side-column">
          <PersonalizationPanel insights={insights} loading={insightsLoading} />
        </aside>
      </div>
    </div>
  )
}

export default App
