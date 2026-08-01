import React, { useState, useCallback } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ScatterChart, Scatter, ZAxis,
  PieChart, Pie, Cell, Legend,
  LineChart, Line
} from 'recharts';

// ── Types ─────────────────────────────────────────────────────────────────────
interface AlgoResult {
  algorithm: string; status: string; error?: string;
  metrics: Record<string, any>;
  visualization: Record<string, any>;
  benchmark: {
    execution_time_s: number; simulator: string; iterations: number;
    objective_value: number; convergence_status: string;
    qpu_available: boolean; classical_comparison: string;
  };
  resource_usage: {
    qubits: number | null; circuit_depth: number | null;
    optimizer_iterations: number; num_shots: number;
    backend: string; annealing_sweeps: number | null;
  };
  business_summary: string; technical_summary: string;
}

interface QuantumResults {
  status: string; session_id: string; n_customers: number;
  feature_names: string[]; total_time_s: number;
  preprocessing_time_s: number;
  algorithms: Record<string, AlgoResult>;
}

// ── Palette ───────────────────────────────────────────────────────────────────
const COLORS = ['#8B5CF6','#3B82F6','#10B981','#F59E0B','#EF4444','#06B6D4','#EC4899','#84CC16'];
const TT = {
  contentStyle: { backgroundColor: '#1F2937', border: '1px solid #374151', borderRadius: 8, fontSize: 11 },
  labelStyle: { color: '#E5E7EB' },
};

// ── Shared sub-components ─────────────────────────────────────────────────────
function AlgoCard({ title, icon, children, status, time }: {
  title: string; icon: string; children: React.ReactNode;
  status: string; time?: number;
}) {
  const ok = status === 'success';
  return (
    <div className={`rounded-2xl border p-5 flex flex-col gap-4 bg-gray-800/60
      ${ok ? 'border-gray-700' : 'border-red-800/50'}`}>
      <div className="flex items-center justify-between">
        <h3 className="font-bold text-white flex items-center gap-2">
          <span>{icon}</span>{title}
        </h3>
        <div className="flex items-center gap-2">
          {time && <span className="text-gray-500 text-xs font-mono">{time.toFixed(2)}s</span>}
          <span className={`text-xs px-2 py-0.5 rounded-full font-mono
            ${ok ? 'bg-green-900/40 text-green-400 border border-green-800'
                 : 'bg-red-900/40 text-red-400 border border-red-800'}`}>
            {ok ? '✓ success' : '⚠ error'}
          </span>
        </div>
      </div>
      {children}
    </div>
  );
}

function SummaryBox({ business, technical }: { business: string; technical: string }) {
  const [show, setShow] = useState<'business' | 'technical'>('business');
  return (
    <div className="mt-1">
      <div className="flex gap-2 mb-2">
        {(['business', 'technical'] as const).map(t => (
          <button key={t} onClick={() => setShow(t)}
            className={`text-xs px-3 py-1 rounded-full border transition-colors
              ${show === t
                ? 'bg-purple-900/60 text-purple-300 border-purple-600'
                : 'bg-gray-900 text-gray-500 border-gray-700 hover:border-gray-500'}`}>
            {t === 'business' ? '💼 Business' : '🔬 Technical'}
          </button>
        ))}
      </div>
      <p className="text-xs text-gray-400 leading-relaxed">{show === 'business' ? business : technical}</p>
    </div>
  );
}

function BenchmarkBar({ bench, res }: { bench: AlgoResult['benchmark']; res: AlgoResult['resource_usage'] }) {
  const items = [
    { label: 'Simulator', val: bench.simulator.split('(')[0].trim() },
    { label: 'Time',      val: `${bench.execution_time_s}s` },
    { label: 'Iters',     val: bench.iterations?.toString() ?? '—' },
    { label: 'Qubits',    val: res.qubits?.toString() ?? '—' },
    { label: 'Depth',     val: res.circuit_depth?.toString() ?? '—' },
    { label: 'Shots',     val: res.num_shots?.toString() ?? '—' },
    { label: 'SA Sweeps', val: res.annealing_sweeps?.toString() ?? '—' },
    { label: 'QPU',       val: bench.qpu_available ? '✓ Real' : '⌁ Sim' },
  ];
  return (
    <div className="flex flex-wrap gap-2 mt-2">
      {items.map(({ label, val }) => (
        <div key={label} className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-1.5 text-center">
          <p className="text-purple-400 font-bold text-sm">{val}</p>
          <p className="text-gray-500 text-[9px] uppercase">{label}</p>
        </div>
      ))}
    </div>
  );
}

// ── Algorithm Panels ──────────────────────────────────────────────────────────
function SegmentationPanel({ r }: { r: AlgoResult }) {
  const sil = r.visualization?.silhouette_comparison ?? [];
  const scatter = (r.visualization?.scatter_quantum ?? []).slice(0, 100);
  const clusters = [...new Set(scatter.map((p: any) => p.cluster))];

  return (
    <AlgoCard title="QUBO Max-Cut Customer Segmentation" icon="🔗"
      status={r.status} time={r.benchmark?.execution_time_s}>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <p className="text-gray-400 text-xs mb-2">Silhouette Score Comparison</p>
          <div className="h-40">
            <ResponsiveContainer><BarChart data={sil} margin={{ left: -20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="method" stroke="#6B7280" tick={{ fontSize: 10 }} />
              <YAxis stroke="#6B7280" tick={{ fontSize: 10 }} domain={[-1, 1]} />
              <Tooltip {...TT} />
              <Bar dataKey="score" radius={[4, 4, 0, 0]}>
                {sil.map((_: any, i: number) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Bar>
            </BarChart></ResponsiveContainer>
          </div>
        </div>
        <div>
          <p className="text-gray-400 text-xs mb-2">Quantum Cluster Scatter (PCA 2D)</p>
          <div className="h-40">
            <ResponsiveContainer><ScatterChart margin={{ left: -20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="x" name="PC1" stroke="#6B7280" tick={{ fontSize: 9 }} />
              <YAxis dataKey="y" name="PC2" stroke="#6B7280" tick={{ fontSize: 9 }} />
              <ZAxis range={[18, 18]} />
              <Tooltip {...TT} />
              {clusters.map((c: any) => (
                <Scatter key={c} name={`Cluster ${c}`} fillOpacity={0.7}
                  data={scatter.filter((p: any) => p.cluster === c)}
                  fill={COLORS[c % COLORS.length]} />
              ))}
            </ScatterChart></ResponsiveContainer>
          </div>
        </div>
      </div>
      <BenchmarkBar bench={r.benchmark} res={r.resource_usage} />
      <SummaryBox business={r.business_summary} technical={r.technical_summary} />
    </AlgoCard>
  );
}

function ValueTieringPanel({ r }: { r: AlgoResult }) {
  const donut = r.visualization?.tier_donut ?? [];
  const bar   = r.visualization?.tier_bar   ?? [];
  return (
    <AlgoCard title="QUBO Multi-Criteria Value Tiering" icon="💎"
      status={r.status} time={r.benchmark?.execution_time_s}>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="h-52">
          <ResponsiveContainer><PieChart>
            <Pie data={donut} dataKey="value" nameKey="name"
              cx="50%" cy="50%" innerRadius={40} outerRadius={80} paddingAngle={4}
              label={({ name, percent }: any) => `${name} ${(percent * 100).toFixed(0)}%`}
              labelLine={false}>
              {donut.map((d: any, i: number) => <Cell key={i} fill={d.color ?? COLORS[i]} />)}
            </Pie>
            <Tooltip {...TT} /><Legend />
          </PieChart></ResponsiveContainer>
        </div>
        <div className="h-52">
          <ResponsiveContainer><BarChart data={bar} margin={{ left: -20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="tier" stroke="#6B7280" tick={{ fontSize: 11 }} />
            <YAxis stroke="#6B7280" tick={{ fontSize: 10 }} />
            <Tooltip {...TT} />
            <Bar dataKey="count" radius={[4, 4, 0, 0]} name="Customers">
              {bar.map((_: any, i: number) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
            </Bar>
          </BarChart></ResponsiveContainer>
        </div>
      </div>
      <BenchmarkBar bench={r.benchmark} res={r.resource_usage} />
      <SummaryBox business={r.business_summary} technical={r.technical_summary} />
    </AlgoCard>
  );
}

function FeatureInteractPanel({ r }: { r: AlgoResult }) {
  const chart = (r.visualization?.interaction_chart ?? []).slice(0, 15);
  const selected = r.visualization?.selected_interactions ?? [];
  return (
    <AlgoCard title="Quantum Feature Interaction Discovery" icon="🔬"
      status={r.status} time={r.benchmark?.execution_time_s}>
      <div className="h-52">
        <ResponsiveContainer><BarChart data={chart} layout="vertical" margin={{ left: 60 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis type="number" stroke="#6B7280" tick={{ fontSize: 10 }} />
          <YAxis dataKey="name" type="category" stroke="#6B7280" tick={{ fontSize: 9 }} width={120} />
          <Tooltip {...TT} />
          <Bar dataKey="mi" radius={[0, 4, 4, 0]} name="Mutual Info">
            {chart.map((d: any, i: number) => (
              <Cell key={i} fill={d.selected ? '#8B5CF6' : '#374151'} />
            ))}
          </Bar>
        </BarChart></ResponsiveContainer>
      </div>
      <div className="flex flex-wrap gap-2">
        {selected.map((s: any) => (
          <span key={s.name} className="px-2 py-1 bg-purple-900/40 border border-purple-600 rounded-lg text-purple-300 text-xs">
            {s.name} · MI: {s.mi}
          </span>
        ))}
      </div>
      <BenchmarkBar bench={r.benchmark} res={r.resource_usage} />
      <SummaryBox business={r.business_summary} technical={r.technical_summary} />
    </AlgoCard>
  );
}

function LookalikePanelComp({ initialData, sessionId }: {
  initialData: AlgoResult | null;
  sessionId: string | null;
}) {
  const [data, setData]       = useState<AlgoResult | null>(initialData);
  const [localIdx, setLocalIdx] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState<string | null>(null);

  const refresh = useCallback(async (idx: number) => {
    if (!sessionId) return;
    setLoading(true); setError(null);
    try {
      const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const res = await fetch(
        `${API_BASE}/api/v1/quantum/lookalike?session_id=${sessionId}&query_customer_idx=${idx}&top_k=10`
      );
      if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'Lookalike failed'); }
      const json = await res.json();
      setData(json.lookalike);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Error');
    } finally { setLoading(false); }
  }, [sessionId]);

  const r = data;
  const bars  = r?.visualization?.similarity_bars ?? [];
  const q     = r?.visualization?.grover_circuit_poc ?? {};
  const query = r?.visualization?.query_customer    ?? {};

  return (
    <AlgoCard title="Grover-Inspired Similarity Search" icon="🔍"
      status={r?.status ?? 'pending'} time={r?.benchmark?.execution_time_s}>

      {/* Inline query control — fast, no full re-run */}
      <div className="flex items-center gap-3 p-3 bg-gray-900/60 rounded-xl border border-gray-700">
        <span className="text-gray-400 text-xs whitespace-nowrap">Query Customer #</span>
        <input
          type="number" value={localIdx} min={0}
          onChange={e => setLocalIdx(parseInt(e.target.value) || 0)}
          className="w-24 bg-gray-800 border border-gray-600 rounded-lg px-2 py-1.5
                     text-white text-sm text-center focus:outline-none focus:border-purple-500" />
        <button
          onClick={() => refresh(localIdx)}
          disabled={loading || !sessionId}
          className={`flex items-center gap-2 px-4 py-1.5 rounded-lg text-sm font-semibold transition-all
            ${ loading
               ? 'bg-blue-900/40 text-blue-300 cursor-wait'
               : !sessionId
                 ? 'bg-gray-700 text-gray-500 cursor-not-allowed'
                 : 'bg-blue-600 hover:bg-blue-500 text-white'}`}>
          {loading
            ? <><span className="animate-spin inline-block">↻</span> Searching…</>
            : <>🔍 Find Look-Alikes</>}
        </button>
        <span className="text-green-500 text-xs font-mono ml-auto">⚡ instant · no re-run</span>
      </div>

      {error && (
        <p className="text-red-400 text-xs px-2">{error}</p>
      )}

      {r && (
        <>
          <div className="p-3 bg-gray-900 rounded-xl border border-gray-700 text-xs text-gray-400">
            <p className="text-purple-300 font-semibold mb-1">Query Customer Profile</p>
            <div className="flex flex-wrap gap-3">
              {Object.entries(query).map(([k, v]) => (
                <span key={k}><span className="text-gray-500">{k}:</span> {String(v)}</span>
              ))}
            </div>
          </div>
          <div className="h-44">
            <ResponsiveContainer><BarChart data={bars} margin={{ left: -10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="customer" stroke="#6B7280" tick={{ fontSize: 9 }} />
              <YAxis stroke="#6B7280" tick={{ fontSize: 10 }} domain={[0, 1]} />
              <Tooltip {...TT} />
              <Bar dataKey="similarity" fill="#8B5CF6" radius={[3, 3, 0, 0]} fillOpacity={0.8} name="Cosine Sim" />
            </BarChart></ResponsiveContainer>
          </div>
          <div className="grid grid-cols-4 gap-2">
            {[['n_qubits','Qubits (PoC)'],['search_space','Search Space'],['grover_iterations','G-Iters'],['circuit_depth','Circuit Depth']].map(([k,l]) => (
              <div key={k} className="bg-gray-900 border border-gray-700 rounded-lg text-center p-2">
                <p className="text-blue-400 font-bold">{q[k] ?? '—'}</p>
                <p className="text-gray-500 text-[9px]">{l}</p>
              </div>
            ))}
          </div>
          <BenchmarkBar bench={r.benchmark} res={r.resource_usage} />
          <SummaryBox business={r.business_summary} technical={r.technical_summary} />
        </>
      )}

      {!r && !loading && (
        <p className="text-gray-500 text-sm text-center py-8">
          Click <span className="text-blue-400">Find Look-Alikes</span> to search for similar customers.
        </p>
      )}
    </AlgoCard>
  );
}

function QSVCPanel({ r }: { r: AlgoResult }) {
  const bars  = r.visualization?.comparison_bars ?? [];
  const rocQ  = r.visualization?.roc_qsvc ?? [];
  const rocX  = r.visualization?.roc_xgb  ?? [];
  const cmQ   = r.visualization?.confusion_qsvc ?? [[0,0],[0,0]];

  const rocData = rocQ.slice(0, 50).map((d: any, i: number) => ({
    fpr: d.fpr, qsvc: d.tpr, xgb: rocX[i]?.tpr ?? null
  }));

  return (
    <AlgoCard title="Quantum Kernel SVM vs XGBoost" icon="⚛"
      status={r.status} time={r.benchmark?.execution_time_s}>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="md:col-span-1 h-48">
          <p className="text-gray-400 text-xs mb-1">Metrics Comparison</p>
          <ResponsiveContainer><BarChart data={bars} margin={{ left: -20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="metric" stroke="#6B7280" tick={{ fontSize: 10 }} />
            <YAxis stroke="#6B7280" tick={{ fontSize: 10 }} domain={[0, 100]} unit="%" />
            <Tooltip {...TT} formatter={(v: any) => `${v}%`} />
            <Bar dataKey="QSVC"    fill="#8B5CF6" radius={[3,3,0,0]} />
            <Bar dataKey="XGBoost" fill="#10B981" radius={[3,3,0,0]} />
            <Legend />
          </BarChart></ResponsiveContainer>
        </div>
        <div className="md:col-span-1 h-48">
          <p className="text-gray-400 text-xs mb-1">ROC Curve</p>
          <ResponsiveContainer><LineChart data={rocData} margin={{ left: -20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="fpr" stroke="#6B7280" tick={{ fontSize: 9 }} label={{ value: 'FPR', position: 'insideBottom', offset: -3, fill: '#6B7280', fontSize: 9 }} />
            <YAxis stroke="#6B7280" tick={{ fontSize: 9 }} label={{ value: 'TPR', angle: -90, position: 'insideLeft', fill: '#6B7280', fontSize: 9 }} />
            <Tooltip {...TT} />
            <Line type="monotone" dataKey="qsvc" stroke="#8B5CF6" dot={false} name="QSVC" />
            <Line type="monotone" dataKey="xgb"  stroke="#10B981" dot={false} name="XGBoost" />
          </LineChart></ResponsiveContainer>
        </div>
        <div className="md:col-span-1">
          <p className="text-gray-400 text-xs mb-1">Confusion Matrix (QSVC)</p>
          <div className="grid grid-cols-2 gap-1 mt-2">
            {['TN', 'FP', 'FN', 'TP'].map((label, idx) => {
              const row = Math.floor(idx / 2), col = idx % 2;
              const val = cmQ[row]?.[col] ?? 0;
              const isCorrect = (row === 0 && col === 0) || (row === 1 && col === 1);
              return (
                <div key={label} className={`rounded-lg p-3 text-center border
                  ${isCorrect ? 'border-green-700 bg-green-900/20' : 'border-red-800 bg-red-900/10'}`}>
                  <p className={`font-bold text-2xl ${isCorrect ? 'text-green-400' : 'text-red-400'}`}>{val}</p>
                  <p className="text-gray-500 text-[10px]">{label}</p>
                </div>
              );
            })}
          </div>
        </div>
      </div>
      <BenchmarkBar bench={r.benchmark} res={r.resource_usage} />
      <SummaryBox business={r.business_summary} technical={r.technical_summary} />
    </AlgoCard>
  );
}

function RetentionPanel({ r }: { r: AlgoResult }) {
  const tiers   = r.visualization?.tier_distribution ?? [];
  const targets = r.visualization?.retention_targets  ?? [];
  return (
    <AlgoCard title="Quantum Retention Target Optimization" icon="🎯"
      status={r.status} time={r.benchmark?.execution_time_s}>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <p className="text-gray-400 text-xs mb-2">Customer Loyalty Tier Distribution</p>
          <div className="h-40">
            <ResponsiveContainer><BarChart data={tiers} margin={{ left: -20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="tier" stroke="#6B7280" tick={{ fontSize: 11 }} />
              <YAxis stroke="#6B7280" tick={{ fontSize: 10 }} />
              <Tooltip {...TT} />
              <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                {tiers.map((_: any, i: number) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Bar>
            </BarChart></ResponsiveContainer>
          </div>
        </div>
        <div>
          <p className="text-gray-400 text-xs mb-2">Top QUBO Retention Targets</p>
          <div className="space-y-1.5 overflow-y-auto max-h-40">
            {targets.map((t: any, i: number) => (
              <div key={i} className="flex items-center justify-between bg-gray-900
                border border-gray-700 rounded-lg px-3 py-1.5">
                <span className="text-gray-300 text-xs">Customer #{t.customer_index}</span>
                <div className="flex items-center gap-2">
                  <div className="w-24 h-1.5 bg-gray-700 rounded-full overflow-hidden">
                    <div className="h-full bg-gradient-to-r from-purple-500 to-blue-500 rounded-full"
                      style={{ width: `${(t.ev_score ?? 0) * 100}%` }} />
                  </div>
                  <span className="text-purple-400 text-xs font-mono">{(t.ev_score ?? 0).toFixed(3)}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
      <BenchmarkBar bench={r.benchmark} res={r.resource_usage} />
      <SummaryBox business={r.business_summary} technical={r.technical_summary} />
    </AlgoCard>
  );
}

function CategoryPanel({ r }: { r: AlgoResult }) {
  const chart = r.visualization?.affinity_chart ?? [];
  return (
    <AlgoCard title="QAOA Category Affinity Matching" icon="🛍️"
      status={r.status} time={r.benchmark?.execution_time_s}>
      <div className="h-52">
        <ResponsiveContainer><BarChart data={chart} layout="vertical" margin={{ left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis type="number" stroke="#6B7280" tick={{ fontSize: 10 }} domain={[0, 1]} />
          <YAxis dataKey="category" type="category" stroke="#6B7280" tick={{ fontSize: 10 }} width={90} />
          <Tooltip {...TT} />
          <Bar dataKey="avg_affinity" radius={[0, 4, 4, 0]} name="Avg Affinity">
            {chart.map((_: any, i: number) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
          </Bar>
        </BarChart></ResponsiveContainer>
      </div>
      <BenchmarkBar bench={r.benchmark} res={r.resource_usage} />
      <SummaryBox business={r.business_summary} technical={r.technical_summary} />
    </AlgoCard>
  );
}

function FeatureImportancePanel({ r }: { r: AlgoResult }) {
  const full  = r.visualization?.full_shap  ?? [];
  const quant = r.visualization?.quantum_assisted_shap ?? [];
  const cmp   = r.visualization?.shap_comparison ?? [];

  return (
    <AlgoCard title="Quantum-Assisted Feature Importance" icon="📊"
      status={r.status} time={r.benchmark?.execution_time_s}>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <p className="text-gray-400 text-xs mb-2">Full SHAP Values (All Features)</p>
          <div className="h-44">
            <ResponsiveContainer><BarChart data={full} layout="vertical" margin={{ left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis type="number" stroke="#6B7280" tick={{ fontSize: 9 }} />
              <YAxis dataKey="feature" type="category" stroke="#6B7280" tick={{ fontSize: 9 }} width={100} />
              <Tooltip {...TT} />
              <Bar dataKey="shap_value" radius={[0, 3, 3, 0]} name="SHAP">
                {full.map((d: any, i: number) => (
                  <Cell key={i} fill={d.selected_by_qubo ? '#8B5CF6' : '#4B5563'} />
                ))}
              </Bar>
            </BarChart></ResponsiveContainer>
          </div>
        </div>
        <div>
          <p className="text-gray-400 text-xs mb-2">Quantum-Assisted SHAP (QUBO Coalition)</p>
          <div className="h-44">
            <ResponsiveContainer><BarChart data={quant} layout="vertical" margin={{ left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis type="number" stroke="#6B7280" tick={{ fontSize: 9 }} />
              <YAxis dataKey="feature" type="category" stroke="#6B7280" tick={{ fontSize: 9 }} width={100} />
              <Tooltip {...TT} />
              <Bar dataKey="shap_value" fill="#8B5CF6" radius={[0, 3, 3, 0]} fillOpacity={0.9} name="QA-SHAP" />
            </BarChart></ResponsiveContainer>
          </div>
        </div>
      </div>
      <div className="flex gap-4 text-xs text-gray-400">
        {cmp.map((c: any) => (
          <div key={c.name} className="flex-1 bg-gray-900 border border-gray-700 rounded-xl p-3 text-center">
            <p className="text-purple-400 font-bold text-base">{c.coalitions.toLocaleString()}</p>
            <p className="text-gray-500">{c.name}</p>
            <p className="text-gray-600">{c.time_s?.toFixed(3)}s</p>
          </div>
        ))}
      </div>
      <BenchmarkBar bench={r.benchmark} res={r.resource_usage} />
      <SummaryBox business={r.business_summary} technical={r.technical_summary} />
    </AlgoCard>
  );
}

// ── Main QuantumInsights Page ─────────────────────────────────────────────────
interface Props { sessionId: string | null; }

export default function QuantumInsights({ sessionId }: Props) {
  const [results, setResults] = useState<QuantumResults | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState<string | null>(null);

  // Lookalike is NOT part of the main run — it has its own panel state
  // (see LookalikePanelComp above)

  const runAnalysis = async () => {
    if (!sessionId) { setError('Upload a CSV first.'); return; }
    setLoading(true); setError(null);
    try {
      // Note: no query_customer_idx here — lookalike is fetched independently
      const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const res = await fetch(`${API_BASE}/api/v1/quantum/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, n_clusters: 4 }),
      });
      if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'Quantum analysis failed'); }
      setResults(await res.json());
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Unknown error');
    } finally { setLoading(false); }
  };

  const a = results?.algorithms ?? {};

  return (
    <div className="space-y-6">
      {/* Header + Run Button */}
      <div className="flex flex-col sm:flex-row sm:items-center gap-4 justify-between">
        <div>
          <h2 className="text-2xl font-extrabold text-white flex items-center gap-2">
            <span>⚛</span>
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-blue-400">
              Quantum Analytics Suite
            </span>
          </h2>
          <p className="text-gray-400 text-sm mt-1">
            8 quantum algorithms · D-Wave Ocean SDK · Qiskit AerSimulator
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button onClick={runAnalysis} disabled={loading || !sessionId}
            className={`px-5 py-2.5 rounded-xl font-bold text-sm transition-all duration-200 flex items-center gap-2
              ${!sessionId
                ? 'bg-gray-700 text-gray-500 cursor-not-allowed'
                : loading
                  ? 'bg-purple-800 text-purple-300 cursor-wait'
                  : 'bg-gradient-to-r from-purple-600 to-blue-600 text-white hover:from-purple-500 hover:to-blue-500 shadow-lg hover:shadow-purple-900/40'}`}>
            {loading ? (
              <><span className="animate-spin">⚙</span> Running 7 algorithms…</>
            ) : (
              <><span>⚛</span> Run Quantum Analysis</>
            )}
          </button>
        </div>
      </div>

      {!sessionId && (
        <div className="text-center py-16 border border-dashed border-gray-700 rounded-2xl">
          <p className="text-5xl mb-3">⚛</p>
          <p className="text-xl font-semibold text-gray-500">Upload a CSV on the Dashboard tab first</p>
          <p className="text-sm text-gray-600 mt-1">Then return here and click Run Quantum Analysis</p>
        </div>
      )}

      {error && (
        <div className="px-4 py-3 rounded-xl bg-red-900/30 border border-red-700 text-red-400 text-sm">
          ⚠ {error}
        </div>
      )}

      {loading && (
        <div className="py-20 text-center space-y-4">
          <div className="text-5xl animate-pulse">⚛</div>
          <p className="text-lg font-semibold text-purple-300">Running 8 quantum algorithms in parallel…</p>
          <div className="flex flex-wrap gap-2 justify-center text-xs text-gray-500">
            {['QUBO Max-Cut', 'Value Tiering', 'Feature Interactions', 'Grover Search',
              'QSVC', 'Retention Opt.', 'Category Matching', 'Feature Importance'].map(n => (
              <span key={n} className="px-2 py-1 bg-gray-800 border border-gray-700 rounded-full animate-pulse">{n}</span>
            ))}
          </div>
          <p className="text-gray-600 text-sm">This may take 5–15 seconds for 200 customers</p>
        </div>
      )}

      {results && !loading && (
        <>
          {/* Summary bar */}
          <div className="flex flex-wrap gap-3">
            {[
              { label: 'Total Time', val: `${results.total_time_s}s` },
              { label: 'Preprocessing', val: `${results.preprocessing_time_s}s` },
              { label: 'Customers Processed', val: results.n_customers.toLocaleString() },
              { label: 'Features Used', val: results.feature_names.length.toString() },
              { label: 'Algorithms Run', val: Object.keys(a).length.toString() },
              { label: 'Succeeded', val: Object.values(a).filter((r: any) => r.status === 'success').length.toString() },
            ].map(({ label, val }) => (
              <div key={label} className="bg-gray-800 border border-gray-700 rounded-xl px-4 py-2 text-center">
                <p className="text-purple-400 font-bold text-lg">{val}</p>
                <p className="text-gray-500 text-[10px]">{label}</p>
              </div>
            ))}
          </div>

          {/* Algorithm cards */}
          <div className="space-y-5">
            {a.segmentation       && <SegmentationPanel       r={a.segmentation} />}
            {a.value_tiering      && <ValueTieringPanel       r={a.value_tiering} />}
            {a.feature_interact   && <FeatureInteractPanel    r={a.feature_interact} />}
            {/* Lookalike has its own state — changing # never re-runs heavy algorithms */}
            <LookalikePanelComp initialData={a.lookalike ?? null} sessionId={sessionId} />
            {a.qsvc               && <QSVCPanel               r={a.qsvc} />}
            {a.loyalty_retention  && <RetentionPanel          r={a.loyalty_retention} />}
            {a.category_match     && <CategoryPanel           r={a.category_match} />}
            {a.feature_importance && <FeatureImportancePanel  r={a.feature_importance} />}
          </div>
        </>
      )}
    </div>
  );
}
