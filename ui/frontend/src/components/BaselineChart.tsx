import { useEffect, useState } from 'react'
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { BaselineComparison } from '../types'
import { getBaselineComparison } from '../api'

const SYSTEM_LABELS: Record<string, string> = {
  naive_keyword: 'Naive keyword',
  naive_semantic: 'Naive semantic',
  always_full_pipeline: 'Always full pipeline',
  full_system: 'Full system (this project)',
}

const SYSTEM_ORDER = ['naive_keyword', 'naive_semantic', 'always_full_pipeline', 'full_system']

export function BaselineChart() {
  const [data, setData] = useState<BaselineComparison | null>(null)
  const [error, setError] = useState(false)

  useEffect(() => {
    getBaselineComparison()
      .then(setData)
      .catch(() => setError(true))
  }, [])

  if (error) {
    return (
      <div className="panel">
        <h3>Baseline comparison</h3>
        <p className="muted">
          Not available yet — run <code>python eval/baseline_comparison.py</code>.
        </p>
      </div>
    )
  }
  if (!data) return <div className="panel">Loading baseline comparison…</div>

  const qualityRows = SYSTEM_ORDER.filter((k) => data[k]).map((key) => ({
    system: SYSTEM_LABELS[key] ?? key,
    'Precision@5': Number(data[key].precision_at_5.toFixed(3)),
    'NDCG@10': Number(data[key].ndcg_at_10.toFixed(3)),
    MRR: Number(data[key].mrr.toFixed(3)),
  }))

  const latencyRows = SYSTEM_ORDER.filter((k) => data[k]).map((key) => ({
    system: SYSTEM_LABELS[key] ?? key,
    'Latency (ms)': Number(data[key].mean_latency_ms.toFixed(1)),
  }))

  return (
    <div className="panel">
      <h3>Baseline comparison</h3>
      <p className="muted">Precision@5 / NDCG@10 / MRR / latency across four systems, same eval set.</p>

      <div className="chart-block">
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={qualityRows} margin={{ top: 8, right: 8, left: -12, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis dataKey="system" tick={{ fontSize: 11 }} interval={0} angle={-15} textAnchor="end" height={50} />
            <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
            <Tooltip />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Bar dataKey="Precision@5" fill="var(--accent)" radius={[4, 4, 0, 0]} />
            <Bar dataKey="NDCG@10" fill="var(--success)" radius={[4, 4, 0, 0]} />
            <Bar dataKey="MRR" fill="var(--warning)" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="chart-block">
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={latencyRows} margin={{ top: 8, right: 8, left: -12, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis dataKey="system" tick={{ fontSize: 11 }} interval={0} angle={-15} textAnchor="end" height={50} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip />
            <Bar dataKey="Latency (ms)" fill="var(--accent)" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
