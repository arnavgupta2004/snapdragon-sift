import type { PersonalizationInsights } from '../types'

interface Props {
  insights: PersonalizationInsights | null
  loading: boolean
}

export function PersonalizationPanel({ insights, loading }: Props) {
  if (loading) return <div className="panel">Loading personalization…</div>
  if (!insights) return null

  return (
    <div className="panel">
      <h3>Made for you</h3>

      <div className="insight-block">
        <span className="insight-label">Files you use most</span>
        <div className="chip-row">
          {insights.preferred_file_types.length === 0 && <span className="muted">none yet</span>}
          {insights.preferred_file_types.map((t) => (
            <span className="chip" key={t}>
              {t}
            </span>
          ))}
        </div>
      </div>

      <div className="insight-block">
        <span className="insight-label">Frequently opened</span>
        <ul className="insight-list">
          {insights.top_files_by_frequency.map((f) => (
            <li key={f.file_id}>
              {f.filename} <span className="muted">({(f.frequency * 100).toFixed(0)}%)</span>
            </li>
          ))}
        </ul>
      </div>

      <div className="insight-block">
        <span className="insight-label">Your usual times</span>
        {insights.recurring_patterns.length === 0 && <span className="muted">none detected</span>}
        <ul className="insight-list">
          {insights.recurring_patterns.map((p, i) => (
            <li key={i}>
              {p.weekday} {p.hour}:00 → {p.filenames.join(', ')}{' '}
              <span className="muted">(confidence {p.confidence.toFixed(2)})</span>
            </li>
          ))}
        </ul>
      </div>

      {insights.active_context_boost_now && (
        <div className="context-boost-banner">
          We noticed a familiar time pattern — {insights.active_context_files.length} usual file
          {insights.active_context_files.length === 1 ? '' : 's'} boosted: {insights.active_context_files.join(', ')}
        </div>
      )}
    </div>
  )
}
