import { useState } from 'react'
import type { ResultItem } from '../types'
import { sendFeedback } from '../api'

interface Props {
  results: ResultItem[]
  userId: number
  query: string
}

export function ResultsList({ results, userId, query }: Props) {
  const [feedbackGiven, setFeedbackGiven] = useState<Record<number, 'thumbs_up' | 'thumbs_down'>>({})

  async function handleFeedback(fileId: number, signal: 'thumbs_up' | 'thumbs_down') {
    setFeedbackGiven((prev) => ({ ...prev, [fileId]: signal }))
    await sendFeedback(userId, fileId, query, signal)
  }

  if (results.length === 0) {
    return <p className="empty-state">Start with a search above. Your best matches will show up here.</p>
  }

  return (
    <div className="results-list">
      {results.map((r) => (
        <div className="result-card" key={r.file_id}>
          <div className="result-main">
            <div className="result-title-row">
              <span className="result-filename">{r.filename}</span>
              <span className="result-filetype">{r.file_type}</span>
              <span className="result-topic">{r.topic_cluster}</span>
            </div>
            <div className="result-score-bar-track">
              <div className="result-score-bar-fill" style={{ width: `${Math.min(100, r.score * 100)}%` }} />
            </div>
            <details className="result-why">
              <summary>Why this may be the one</summary>
              <p className="result-why-detail">{r.explanation}</p>
            </details>
          </div>
          <div className="result-side">
            <span className="result-score-value">{r.score.toFixed(3)}</span>
            <div className="feedback-buttons">
              <button
                className={feedbackGiven[r.file_id] === 'thumbs_up' ? 'feedback-btn active' : 'feedback-btn'}
                onClick={() => handleFeedback(r.file_id, 'thumbs_up')}
                aria-label="helpful"
                title="Helpful"
              >
                <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M7 9v9H4a1 1 0 0 1-1-1v-7a1 1 0 0 1 1-1h3Zm0 0 4.5-7a1.8 1.8 0 0 1 3.2 1.5L13.5 8H16a2 2 0 0 1 2 2.3l-1.1 6.5A2 2 0 0 1 14.9 18H9a2 2 0 0 1-2-2V9Z" />
                </svg>
              </button>
              <button
                className={feedbackGiven[r.file_id] === 'thumbs_down' ? 'feedback-btn active' : 'feedback-btn'}
                onClick={() => handleFeedback(r.file_id, 'thumbs_down')}
                aria-label="not helpful"
                title="Not helpful"
              >
                <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M13 11V2h3a1 1 0 0 1 1 1v7a1 1 0 0 1-1 1h-3Zm0 0-4.5 7a1.8 1.8 0 0 1-3.2-1.5L6.5 12H4a2 2 0 0 1-2-2.3l1.1-6.5A2 2 0 0 1 5.1 2H11a2 2 0 0 1 2 2v7Z" />
                </svg>
              </button>
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}
