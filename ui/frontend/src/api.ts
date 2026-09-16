import type {
  BaselineComparison,
  IngestResult,
  PersonalizationInsights,
  SSEEvent,
  UserSummary,
} from './types'

export async function listUsers(): Promise<UserSummary[]> {
  const res = await fetch('/api/users')
  return res.json()
}

export async function ingestFolder(
  root: string,
  maxFiles: number,
  clearExisting: boolean,
): Promise<IngestResult> {
  const res = await fetch('/api/ingest', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ root, max_files: maxFiles, clear_existing: clearExisting }),
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(body.detail ?? 'Failed to index folder')
  }
  return res.json()
}

export async function getPersonalization(userId: number): Promise<PersonalizationInsights> {
  const res = await fetch(`/api/personalization/${userId}`)
  return res.json()
}

export async function getBaselineComparison(): Promise<BaselineComparison> {
  const res = await fetch('/api/eval/baseline-comparison')
  if (!res.ok) throw new Error('baseline comparison not available')
  return res.json()
}

export async function sendFeedback(
  userId: number,
  fileId: number,
  query: string,
  signal: 'thumbs_up' | 'thumbs_down',
): Promise<void> {
  await fetch('/api/feedback', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: userId, file_id: fileId, query, signal }),
  })
}

/** Streams the live routing trace via SSE. Returns an abort function. */
export function streamQuery(
  query: string,
  userId: number,
  onEvent: (event: SSEEvent) => void,
  onError: (error: unknown) => void,
): () => void {
  const params = new URLSearchParams({ query, user_id: String(userId) })
  const source = new EventSource(`/api/query/stream?${params.toString()}`)

  source.onmessage = (evt) => {
    try {
      const data = JSON.parse(evt.data) as SSEEvent
      onEvent(data)
      if (data.type === 'done') {
        source.close()
      }
    } catch (err) {
      onError(err)
    }
  }
  source.onerror = (err) => {
    onError(err)
    source.close()
  }

  return () => source.close()
}
