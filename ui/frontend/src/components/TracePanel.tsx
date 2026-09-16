import type { RoutingTraceData, StageTrace } from '../types'

// Fixed canonical order so the panel layout doesn't jump around between tiers —
// stages not reached yet stay in "pending" state until an SSE event resolves them.
const CANONICAL_STAGES = [
  'entity_extraction',
  'routing',
  'query_enrichment',
  'filename_search',
  'metadata_search',
  'keyword_search',
  'semantic_search',
  'hybrid_fusion',
  'reranker',
  'personalization',
  'llm_explanation',
  'finalize',
]

const STAGE_LABELS: Record<string, string> = {
  entity_extraction: 'Entity extraction',
  routing: 'Routing',
  query_enrichment: 'Query enrichment',
  filename_search: 'Filename search',
  metadata_search: 'Metadata search',
  keyword_search: 'Keyword (BM25)',
  semantic_search: 'Semantic search',
  hybrid_fusion: 'Hybrid fusion (RRF)',
  reranker: 'Cross-encoder rerank',
  personalization: 'Personalization',
  llm_explanation: 'LLM explanation',
  finalize: 'Finalize',
}

interface Props {
  tier: string | null
  rationale: string
  stagesByName: Record<string, StageTrace>
  isStreaming: boolean
  finalTrace: RoutingTraceData | null
}

export function TracePanel({ tier, rationale, stagesByName, isStreaming, finalTrace }: Props) {
  return (
    <div className="panel trace-panel">
      <div className="trace-header">
        <h3>Routing trace</h3>
        {tier && <span className={`tier-badge tier-${tier}`}>{tier.toUpperCase()}</span>}
      </div>

      {rationale && <p className="trace-rationale">{rationale}</p>}

      <div className="trace-stages">
        {CANONICAL_STAGES.map((name) => {
          const stage = stagesByName[name]
          const state = !stage ? 'pending' : stage.skipped ? 'skipped' : 'done'
          return (
            <div key={name} className={`trace-stage trace-stage-${state}`}>
              <span className="trace-stage-icon" />
              <span className="trace-stage-name">{STAGE_LABELS[name] ?? name}</span>
              {state === 'done' && <span className="trace-stage-timing">{stage.duration_ms.toFixed(1)}ms</span>}
              {state === 'skipped' && stage.detail && (
                <span className="trace-stage-detail" title={stage.detail}>
                  skipped
                </span>
              )}
            </div>
          )
        })}
      </div>

      {isStreaming && <div className="trace-streaming-indicator">running…</div>}

      {finalTrace && (
        <div className="trace-footer">
          <span>
            <strong>{finalTrace.total_duration_ms.toFixed(1)}ms</strong> total
          </span>
          <span>
            <strong>{finalTrace.llm_call_count}</strong> LLM call{finalTrace.llm_call_count === 1 ? '' : 's'}
          </span>
          {finalTrace.used_learned_router && <span className="badge-small">learned router</span>}
          {finalTrace.used_llm_fallback_classification && <span className="badge-small">LLM fallback</span>}
        </div>
      )}
    </div>
  )
}
