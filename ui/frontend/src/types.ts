export interface UserSummary {
  id: number
  name: string
  persona_key: string
}

export interface StageTrace {
  name: string
  duration_ms: number
  skipped: boolean
  detail: string
}

export interface RoutingTraceData {
  tier: string
  rationale: string
  used_llm_fallback_classification: boolean
  used_learned_router: boolean
  llm_call_count: number
  total_duration_ms: number
  stages: StageTrace[]
}

export interface ResultItem {
  file_id: number
  filename: string
  path: string
  file_type: string
  topic_cluster: string
  modified_at: string
  score: number
  explanation: string
}

export interface QueryResponse {
  query: string
  results: ResultItem[]
  routing_trace: RoutingTraceData
}

export interface RecurringPattern {
  weekday: string
  hour: number
  file_ids: number[]
  filenames: string[]
  confidence: number
}

export interface TopFile {
  file_id: number
  filename: string
  frequency: number
}

export interface PersonalizationInsights {
  preferred_file_types: string[]
  top_files_by_frequency: TopFile[]
  recurring_patterns: RecurringPattern[]
  active_context_boost_now: boolean
  active_context_files: string[]
}

export interface BaselineStats {
  precision_at_5: number
  recall_at_5: number
  ndcg_at_10: number
  mrr: number
  mean_latency_ms: number
}

export type BaselineComparison = Record<string, BaselineStats>

export interface IngestResult {
  root: string
  n_files_crawled: number
  by_type: Record<string, number>
  n_indexed_total: number
  cleared_existing: boolean
}

export type SSEEvent =
  | { type: 'route'; tier: string; rationale: string }
  | { type: 'stage'; name: string; duration_ms: number; skipped: boolean; detail: string }
  | { type: 'done'; results: ResultItem[]; routing_trace: RoutingTraceData }
