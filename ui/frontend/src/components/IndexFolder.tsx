import { useState } from 'react'
import { ingestFolder } from '../api'
import type { IngestResult } from '../types'

const SUGGESTED_ROOTS = ['~/Downloads', '~/Documents', '~/Desktop']

interface Props {
  onIndexed?: () => void
}

export function IndexFolder({ onIndexed }: Props) {
  const [open, setOpen] = useState(false)
  const [root, setRoot] = useState('')
  const [maxFiles, setMaxFiles] = useState(300)
  const [clearExisting, setClearExisting] = useState(false)
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<IngestResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  async function handleIndex() {
    if (!root.trim()) return
    setBusy(true)
    setError(null)
    setResult(null)
    try {
      const res = await ingestFolder(root.trim(), maxFiles, clearExisting)
      setResult(res)
      onIndexed?.()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to index folder')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="panel index-folder-panel">
      <button className="index-folder-toggle" onClick={() => setOpen((v) => !v)}>
        <h3>
          <span className={`index-folder-chevron${open ? ' open' : ''}`} />
          Index a folder
        </h3>
        <span className="index-folder-hint">
          {open ? '' : 'Point Sift at your Downloads, Documents, or any local directory'}
        </span>
      </button>

      {open && (
        <div className="index-folder-body">
          <p className="index-folder-description">
            Choose a folder on this computer. Sift reads its file names and contents so you can search them
            using normal language.
          </p>

          <div className="index-folder-suggestions">
            {SUGGESTED_ROOTS.map((r) => (
              <button key={r} className="example-chip" onClick={() => setRoot(r)}>
                {r}
              </button>
            ))}
          </div>

          <div className="index-folder-row">
            <input
              className="query-input"
              value={root}
              onChange={(e) => setRoot(e.target.value)}
              placeholder="/Users/you/Downloads or C:\\Users\\you\\Documents"
            />
            <input
              className="index-folder-maxfiles"
              type="number"
              min={1}
              max={5000}
              value={maxFiles}
              onChange={(e) => setMaxFiles(Number(e.target.value))}
              aria-label="Maximum number of files to add"
            />
          </div>

          <label className="index-folder-checkbox">
            <input
              type="checkbox"
              checked={clearExisting}
              onChange={(e) => setClearExisting(e.target.checked)}
            />
            Start fresh and replace the current files
          </label>

          <button className="run-button" onClick={handleIndex} disabled={busy || !root.trim()}>
            {busy ? 'Indexing…' : 'Index this folder'}
          </button>

          {error && <div className="error-banner index-folder-error">{error}</div>}

          {result && (
            <div className="index-folder-result">
              Ready to search: <strong>{result.n_files_crawled}</strong> files added from <code>{result.root}</code> —{' '}
              {Object.entries(result.by_type)
                .map(([type, count]) => `${count} .${type}`)
                .join(', ')}
              . <strong>{result.n_indexed_total}</strong> files are now searchable.
            </div>
          )}
        </div>
      )}
    </div>
  )
}
