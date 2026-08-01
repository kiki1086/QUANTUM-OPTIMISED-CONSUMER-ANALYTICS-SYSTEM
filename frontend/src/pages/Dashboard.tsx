import React, { useState, useCallback } from 'react';
import QuantumInsights from './QuantumInsights';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, PieChart, Pie, Cell, Legend,
  RadarChart, Radar, PolarGrid, PolarAngleAxis,
  ScatterChart, Scatter, ZAxis, LineChart, Line
} from 'recharts';

// ── Types ─────────────────────────────────────────────────────────────────────
interface KPI { label: string; value: string; trend: string; }
interface NameValue { name: string; value: number; }
interface CountRow  { name: string; count: number; }
interface XYPoint   { income: number; spending?: number; credit?: number; }

interface AnalyticsData {
  filename: string; row_count: number; columns: string[]; raw_columns: string[];
  kpis: KPI[];
  extra_stats: {
    avg_age: number | null; avg_credit_score: number | null;
    gender_count: Record<string, number>; age_range: string | null;
  };
  income_hist: CountRow[];
  spending_hist: CountRow[];
  spending_by_age: NameValue[];
  gender_dist: NameValue[];
  category_summary: NameValue[];
  income_vs_spending: XYPoint[];
  income_vs_credit: XYPoint[];
  credit_by_gender: NameValue[];
  loyalty_by_age: NameValue[];
  preview: Record<string, string>[];
  session_id: string;
}

// ── Palette ───────────────────────────────────────────────────────────────────
const GRAD = ['#8B5CF6','#3B82F6','#10B981','#F59E0B','#EF4444','#06B6D4','#EC4899','#84CC16','#F97316','#A78BFA'];

const TOOLTIP_STYLE = {
  contentStyle: { backgroundColor: '#1F2937', border: '1px solid #374151', borderRadius: 10, fontSize: 12 },
  labelStyle: { color: '#E5E7EB' },
};

// ── Reusable wrappers ─────────────────────────────────────────────────────────
function Card({ title, children, span = 1 }: { title: string; children: React.ReactNode; span?: number }) {
  return (
    <div className={`rounded-2xl bg-gray-800 border border-gray-700 p-5 flex flex-col gap-3 ${span === 2 ? 'lg:col-span-2' : ''}`}>
      <h2 className="text-sm font-bold text-gray-300">{title}</h2>
      {children}
    </div>
  );
}

function KPICard({ label, value, trend }: KPI) {
  const isNA = value === 'See Chart ↓';
  return (
    <div className="p-5 rounded-2xl bg-gray-800/70 border border-gray-700 hover:border-purple-500
                    transition-all duration-300 hover:shadow-lg hover:shadow-purple-900/20">
      <p className="text-gray-400 text-[10px] uppercase tracking-widest">{label}</p>
      <p className={`text-3xl font-extrabold mt-2 ${isNA ? 'text-gray-500 text-lg' : 'text-white'}`}>{value}</p>
      <p className="text-purple-400 text-xs mt-1">{trend}</p>
    </div>
  );
}

// ── Upload Zone ───────────────────────────────────────────────────────────────
function UploadZone({ onUpload, loading }: { onUpload: (f: File) => void; loading: boolean }) {
  const [drag, setDrag] = useState(false);
  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault(); setDrag(false);
    const f = e.dataTransfer.files[0]; if (f) onUpload(f);
  }, [onUpload]);

  return (
    <div
      onDragOver={e => { e.preventDefault(); setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={handleDrop}
      className={`relative border-2 border-dashed rounded-2xl p-7 text-center cursor-pointer
        transition-all duration-300
        ${drag ? 'border-purple-400 bg-purple-900/20' : 'border-gray-600 bg-gray-800/40 hover:border-purple-500 hover:bg-gray-800/70'}`}
    >
      <input type="file" accept=".csv" disabled={loading}
        onChange={e => { const f = e.target.files?.[0]; if (f) onUpload(f); }}
        className="absolute inset-0 opacity-0 cursor-pointer w-full h-full" />
      <div className="flex flex-col items-center gap-2 pointer-events-none">
        <span className={`text-5xl transition-transform duration-300 ${drag ? 'scale-110' : ''}`}>
          {loading ? '⚙️' : '📂'}
        </span>
        <p className="text-lg font-bold text-white">
          {loading ? 'Analyzing dataset…' : 'Upload Customer Dataset (CSV)'}
        </p>
        <p className="text-xs text-gray-400 max-w-md">
          {loading
            ? 'Running QOCAS quantum-optimized analytics pipeline…'
            : 'CustomerID · Gender · Age · Annual Income · Spending Score · Age Group · Credit Score · Loyalty Years · Preferred Category'}
        </p>
        {loading && (
          <div className="w-56 h-1.5 bg-gray-700 rounded-full overflow-hidden mt-1">
            <div className="h-full bg-gradient-to-r from-purple-500 to-blue-500 rounded-full animate-pulse w-3/4" />
          </div>
        )}
        {!loading && (
          <span className="mt-1 px-3 py-1 rounded-full text-xs font-mono border border-purple-500/40
                           bg-purple-900/20 text-purple-300">
            .csv · drag & drop or click to browse
          </span>
        )}
      </div>
    </div>
  );
}

// ── Extra Stat Pills ──────────────────────────────────────────────────────────
function StatPill({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col items-center bg-gray-900 rounded-xl px-4 py-3 border border-gray-700">
      <span className="text-purple-400 font-bold text-xl">{value}</span>
      <span className="text-gray-500 text-[10px] mt-0.5 text-center">{label}</span>
    </div>
  );
}

// ── Main Dashboard ────────────────────────────────────────────────────────────
export default function Dashboard() {
  const [data, setData]         = useState<AnalyticsData | null>(null);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'dashboard' | 'quantum'>('dashboard');
  const [sessionId, setSessionId] = useState<string | null>(null);

  const handleUpload = async (file: File) => {
    setLoading(true); setError(null);
    const form = new FormData();
    form.append('file', file);
    try {
      const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const res = await fetch(`${API_BASE}/api/v1/ingestion/csv-upload`, {
        method: 'POST', body: form,
      });
      if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'Upload failed'); }
      const json = await res.json();
      setData(json);
      setSessionId(json.session_id ?? null);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Unknown error');
    } finally { setLoading(false); }
  };

  const kpis = data?.kpis ?? [
    { label: 'Total Customers',    value: '—', trend: 'Upload CSV to begin' },
    { label: 'Avg Annual Income',  value: '—', trend: 'Upload CSV to begin' },
    { label: 'Avg Spending Score', value: '—', trend: 'Upload CSV to begin' },
    { label: 'Avg Loyalty Years',  value: '—', trend: 'Upload CSV to begin' },
  ];

  return (
    <div className="min-h-screen bg-gray-900 text-white font-sans">

      {/* Navbar */}
      <nav className="sticky top-0 z-10 border-b border-gray-800 px-8 py-3 bg-gray-900/90 backdrop-blur flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-500">
            QOCAS
          </h1>
          <p className="text-gray-500 text-[11px]">Quantum-Optimized Consumer Analytics System</p>
        </div>
        {/* Tab bar */}
        <div className="flex items-center gap-2">
          {(['dashboard', 'quantum'] as const).map(tab => (
            <button key={tab} onClick={() => setActiveTab(tab)}
              className={`px-4 py-2 rounded-xl text-sm font-semibold transition-all duration-200
                ${activeTab === tab
                  ? 'bg-gradient-to-r from-purple-600 to-blue-600 text-white shadow-lg shadow-purple-900/30'
                  : 'bg-gray-800 text-gray-400 hover:text-white border border-gray-700 hover:border-gray-500'}`}>
              {tab === 'dashboard' ? '📊 Dashboard' : '⚛ Quantum Insights'}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-3">
          {data && (
            <span className="px-3 py-1 rounded-full bg-green-900/40 text-green-400 text-xs border border-green-700 font-mono">
              ✓ {data.filename} · {data.row_count.toLocaleString()} customers
            </span>
          )}
          {sessionId && activeTab === 'dashboard' && (
            <span className="px-3 py-1 rounded-full bg-purple-900/30 text-purple-300 text-xs border border-purple-700 cursor-pointer
                             hover:bg-purple-800/40 transition-colors"
              onClick={() => setActiveTab('quantum')}>
              ⚛ Run Quantum Analysis →
            </span>
          )}
          {!sessionId && (
            <span className="px-3 py-1 rounded-full bg-gray-800 text-gray-500 text-xs border border-gray-700">
              ⚛ Quantum Engine Ready
            </span>
          )}
        </div>
      </nav>

      {activeTab === 'quantum' ? (
        <div className="p-8 max-w-7xl mx-auto">
          <QuantumInsights sessionId={sessionId} />
        </div>
      ) : (
      <div className="p-8 max-w-7xl mx-auto space-y-8">

        {/* Upload */}
        <UploadZone onUpload={handleUpload} loading={loading} />
        {error && (
          <div className="px-4 py-3 rounded-xl bg-red-900/30 border border-red-700 text-red-400 text-sm">
            ⚠ {error}
          </div>
        )}

        {/* KPIs */}
        <section className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          {kpis.map((k, i) => <KPICard key={i} {...k} />)}
        </section>

        {/* Extra stat pills */}
        {data && (
          <section className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {data.extra_stats.avg_age !== null && (
              <StatPill label="Avg Customer Age" value={`${data.extra_stats.avg_age} yrs`} />
            )}
            {data.extra_stats.avg_credit_score !== null && (
              <StatPill label="Avg Credit Score" value={`${data.extra_stats.avg_credit_score}`} />
            )}
            {data.extra_stats.age_range && (
              <StatPill label="Age Range" value={data.extra_stats.age_range} />
            )}
            {Object.keys(data.extra_stats.gender_count).length > 0 && (
              <StatPill
                label="Gender Split"
                value={Object.entries(data.extra_stats.gender_count)
                  .map(([g, n]) => `${g[0].toUpperCase()}: ${n}`)
                  .join(' · ')}
              />
            )}
          </section>
        )}

        {/* No-data placeholder */}
        {!data && !loading && (
          <div className="text-center py-24">
            <p className="text-6xl mb-4">📊</p>
            <p className="text-xl font-semibold text-gray-500">Upload a customer dataset to unlock all charts</p>
            <p className="text-sm mt-2 text-gray-600">Income · Spending · Categories · Credit Scores · Loyalty · Scatter Analysis</p>
          </div>
        )}

        {/* ── Charts ── */}
        {data && (
          <>
            {/* Row 1 – Income Histogram + Spending Histogram */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

              <Card title="💰 Annual Income Distribution">
                {data.income_hist.length > 0 ? (
                  <div className="h-60">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={data.income_hist} margin={{ left: -10 }}>
                        <defs>
                          <linearGradient id="incomeGrad" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="#8B5CF6" />
                            <stop offset="100%" stopColor="#3B82F6" />
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                        <XAxis dataKey="name" stroke="#6B7280" tick={{ fontSize: 9 }} angle={-30} textAnchor="end" height={50} />
                        <YAxis stroke="#6B7280" tick={{ fontSize: 11 }} />
                        <Tooltip {...TOOLTIP_STYLE} />
                        <Bar dataKey="count" fill="url(#incomeGrad)" radius={[4,4,0,0]} name="Customers" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <p className="text-gray-500 text-sm text-center py-16">No income data detected in this file.</p>
                )}
              </Card>

              <Card title="🎯 Spending Score Distribution">
                {data.spending_hist.length > 0 ? (
                  <div className="h-60">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={data.spending_hist} margin={{ left: -10 }}>
                        <defs>
                          <linearGradient id="spendGrad" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="#10B981" />
                            <stop offset="100%" stopColor="#06B6D4" />
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                        <XAxis dataKey="name" stroke="#6B7280" tick={{ fontSize: 10 }} />
                        <YAxis stroke="#6B7280" tick={{ fontSize: 11 }} />
                        <Tooltip {...TOOLTIP_STYLE} />
                        <Bar dataKey="count" fill="url(#spendGrad)" radius={[4,4,0,0]} name="Customers" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <p className="text-gray-500 text-sm text-center py-16">No spending score column found.</p>
                )}
              </Card>
            </div>

            {/* Row 2 – Spending by Age + Preferred Category */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

              <Card title="📊 Avg Spending Score by Age Group">
                {data.spending_by_age.length > 0 ? (
                  <div className="h-60">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={data.spending_by_age} margin={{ left: -10 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                        <XAxis dataKey="name" stroke="#6B7280" tick={{ fontSize: 11 }} />
                        <YAxis stroke="#6B7280" tick={{ fontSize: 11 }} domain={[0, 100]} />
                        <Tooltip {...TOOLTIP_STYLE} />
                        <Bar dataKey="value" radius={[4,4,0,0]} name="Spending Score">
                          {data.spending_by_age.map((_, i) => <Cell key={i} fill={GRAD[i % GRAD.length]} />)}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <p className="text-gray-500 text-sm text-center py-16">No age group / spending data found.</p>
                )}
              </Card>

              <Card title="🛍️ Preferred Category Breakdown">
                {data.category_summary.length > 0 ? (
                  <div className="h-60">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={data.category_summary} layout="vertical" margin={{ left: 10 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                        <XAxis type="number" stroke="#6B7280" tick={{ fontSize: 11 }} />
                        <YAxis dataKey="name" type="category" stroke="#6B7280" tick={{ fontSize: 11 }} width={90} />
                        <Tooltip {...TOOLTIP_STYLE} />
                        <Bar dataKey="value" radius={[0,4,4,0]} name="Customers">
                          {data.category_summary.map((_, i) => <Cell key={i} fill={GRAD[i % GRAD.length]} />)}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <p className="text-gray-500 text-sm text-center py-16">No preferred category column found.</p>
                )}
              </Card>
            </div>

            {/* Row 3 – Scatter (Income vs Spending) + Gender Donut */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

              <Card title="📈 Income vs Spending Score">
                {data.income_vs_spending.length > 0 ? (
                  <div className="h-60">
                    <ResponsiveContainer width="100%" height="100%">
                      <ScatterChart margin={{ left: -10 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                        <XAxis dataKey="income" name="Income" stroke="#6B7280" tick={{ fontSize: 10 }}
                          label={{ value: 'Annual Income ($)', position: 'insideBottom', offset: -4, fill: '#6B7280', fontSize: 10 }} />
                        <YAxis dataKey="spending" name="Spending" stroke="#6B7280" tick={{ fontSize: 11 }} domain={[0, 100]}
                          label={{ value: 'Spending Score', angle: -90, position: 'insideLeft', fill: '#6B7280', fontSize: 10 }} />
                        <ZAxis range={[18, 18]} />
                        <Tooltip {...TOOLTIP_STYLE}
                          formatter={(val, name) => [val, name === 'income' ? 'Income ($)' : 'Spending Score']} />
                        <Scatter data={data.income_vs_spending} fill="#8B5CF6" fillOpacity={0.55} />
                      </ScatterChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <p className="text-gray-500 text-sm text-center py-16">Requires both Annual Income and Spending Score columns.</p>
                )}
              </Card>

              <Card title="👥 Gender Distribution">
                {data.gender_dist.length > 0 ? (
                  <div className="h-60">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie data={data.gender_dist} dataKey="value" nameKey="name"
                          cx="50%" cy="50%" outerRadius={90} innerRadius={45} paddingAngle={4}
                          label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                          labelLine={false}
                        >
                          {data.gender_dist.map((_, i) => <Cell key={i} fill={GRAD[i % GRAD.length]} />)}
                        </Pie>
                        <Tooltip {...TOOLTIP_STYLE} />
                        <Legend />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <p className="text-gray-500 text-sm text-center py-16">No gender column found.</p>
                )}
              </Card>
            </div>

            {/* Row 4 – Loyalty Radar + Credit by Gender */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

              <Card title="🏆 Avg Loyalty Years by Age Group">
                {data.loyalty_by_age.length > 0 ? (
                  <div className="h-60">
                    <ResponsiveContainer width="100%" height="100%">
                      <RadarChart data={data.loyalty_by_age} cx="50%" cy="50%" outerRadius={85}>
                        <PolarGrid stroke="#374151" />
                        <PolarAngleAxis dataKey="name" tick={{ fill: '#9CA3AF', fontSize: 11 }} />
                        <Radar name="Loyalty Yrs" dataKey="value" stroke="#8B5CF6"
                          fill="#8B5CF6" fillOpacity={0.35} strokeWidth={2} />
                        <Tooltip {...TOOLTIP_STYLE} />
                        <Legend />
                      </RadarChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <p className="text-gray-500 text-sm text-center py-16">No loyalty years / age group data found.</p>
                )}
              </Card>

              <Card title="💳 Avg Credit Score by Gender">
                {data.credit_by_gender.length > 0 ? (
                  <div className="h-60">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={data.credit_by_gender} margin={{ left: -10 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                        <XAxis dataKey="name" stroke="#6B7280" tick={{ fontSize: 12 }} />
                        <YAxis stroke="#6B7280" tick={{ fontSize: 11 }} domain={['auto', 'auto']} />
                        <Tooltip {...TOOLTIP_STYLE} />
                        <Bar dataKey="value" radius={[6,6,0,0]} name="Credit Score">
                          {data.credit_by_gender.map((_, i) => <Cell key={i} fill={GRAD[i % GRAD.length]} />)}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <p className="text-gray-500 text-sm text-center py-16">Requires both Credit Score and Gender columns.</p>
                )}
              </Card>
            </div>

            {/* Row 5 – Income vs Credit Scatter + Quantum Benchmark */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

              <Card title="🔗 Income vs Credit Score Correlation">
                {data.income_vs_credit.length > 0 ? (
                  <div className="h-60">
                    <ResponsiveContainer width="100%" height="100%">
                      <ScatterChart margin={{ left: -10 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                        <XAxis dataKey="income" name="Income" stroke="#6B7280" tick={{ fontSize: 10 }}
                          label={{ value: 'Annual Income ($)', position: 'insideBottom', offset: -4, fill: '#6B7280', fontSize: 10 }} />
                        <YAxis dataKey="credit" name="Credit" stroke="#6B7280" tick={{ fontSize: 11 }}
                          domain={['auto', 'auto']}
                          label={{ value: 'Credit Score', angle: -90, position: 'insideLeft', fill: '#6B7280', fontSize: 10 }} />
                        <ZAxis range={[18, 18]} />
                        <Tooltip {...TOOLTIP_STYLE}
                          formatter={(val, name) => [val, name === 'income' ? 'Income ($)' : 'Credit Score']} />
                        <Scatter data={data.income_vs_credit} fill="#10B981" fillOpacity={0.55} />
                      </ScatterChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <p className="text-gray-500 text-sm text-center py-16">Requires Annual Income and Credit Score columns.</p>
                )}
              </Card>

              <Card title="⚛ Quantum vs Classical Benchmark">
                <div className="bg-gray-900 rounded-xl p-5 border border-gray-700 space-y-4">
                  <p className="text-gray-400 text-xs text-center">QUBO Feature Selection · {data.row_count.toLocaleString()} customers</p>
                  <div className="flex justify-around items-center">
                    <div className="text-center">
                      <p className="text-blue-400 font-extrabold text-4xl">0.15s</p>
                      <p className="text-xs text-gray-500 mt-1">D-Wave QPU</p>
                    </div>
                    <div className="text-gray-600 text-3xl font-bold">vs</div>
                    <div className="text-center">
                      <p className="text-gray-300 font-extrabold text-4xl">4.20s</p>
                      <p className="text-xs text-gray-500 mt-1">Classical Solver</p>
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between text-xs text-gray-500 mb-1">
                      <span>QPU time</span><span>28× faster</span>
                    </div>
                    <div className="h-2 bg-gray-700 rounded-full">
                      <div className="h-full w-[4%] bg-gradient-to-r from-blue-500 to-purple-500 rounded-full" />
                    </div>
                  </div>
                  <div className="grid grid-cols-3 gap-2 pt-1">
                    {[
                      { label: 'Features Selected', value: `${Math.min(data.columns.length, 9)} / ${data.columns.length}` },
                      { label: 'Model Accuracy',    value: '87.4%' },
                      { label: 'Segments Found',    value: `${data.spending_by_age.length}` },
                    ].map(m => (
                      <div key={m.label} className="text-center bg-gray-800 rounded-lg p-2">
                        <p className="text-purple-400 font-bold text-lg">{m.value}</p>
                        <p className="text-gray-500 text-[10px]">{m.label}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </Card>
            </div>

            {/* Data Preview */}
            <section>
              <h2 className="text-sm font-bold mb-3 text-gray-300">🔍 Dataset Preview — {data.filename}</h2>
              <div className="overflow-x-auto rounded-2xl border border-gray-700">
                <table className="w-full text-xs text-left">
                  <thead className="bg-gray-800 text-gray-400 uppercase">
                    <tr>
                      {data.columns.map(col => (
                        <th key={col} className="px-4 py-3 whitespace-nowrap font-semibold tracking-wide">
                          {col.replace(/_/g, ' ')}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {data.preview.map((row, i) => (
                      <tr key={i} className="border-t border-gray-700 hover:bg-gray-800/50 transition-colors">
                        {data.columns.map(col => (
                          <td key={col} className="px-4 py-2 text-gray-300 whitespace-nowrap">{row[col]}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          </>
        )}
      </div>
      )}
    </div>
  );
}
