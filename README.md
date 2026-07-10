# QOCAS — Quantum-Optimised Consumer Analytics System

QOCAS is a **full-stack quantum-classical analytics platform** that ingests a customer dataset and runs **8 genuinely formulated quantum algorithms** on it, each paired with a classical benchmark for comparison.

## 🗂 General Features

**Classical side (instant):** KPI cards, income/spending histograms, scatter plots, radar chart, gender donut, loyalty trends, category breakdown, dataset preview.

**Quantum side (on-demand):**

| # | Algorithm | Quantum Method |
|---|---|---|
| 1 | Customer Segmentation | QUBO Max-Cut clustering via D-Wave Simulated Annealing |
| 2 | Value Tiering | QUBO multi-criteria optimisation (5 features simultaneously) |
| 3 | Feature Interaction Discovery | Annealing-based MI-scored coalition selection |
| 4 | Look-Alike Search | Grover-Inspired circuit PoC + cosine similarity |
| 5 | Churn Classifier | QSVC (ZZFeatureMap + FidelityQuantumKernel) vs XGBoost |
| 6 | Retention Targeting | Budget-constrained QUBO campaign optimisation |
| 7 | Category Affinity Matching | Max-weight bipartite QUBO (customers ↔ categories) |
| 8 | Feature Importance | QUBO coalition pruning → Quantum-Assisted SHAP |

---

## 🛠 Tech Stack

- **Backend:** FastAPI + Uvicorn, D-Wave Ocean SDK (`dimod`, `dwave-samplers`, `dwave-neal`), Qiskit 2.5 + Qiskit-Aer, Qiskit Machine Learning, scikit-learn, XGBoost, SHAP, pandas/numpy/networkx
- **Frontend:** React 18 + TypeScript, Vite, Recharts, Tailwind CSS
- **Concurrency:** `ThreadPoolExecutor` — 6 QUBO algorithms run in parallel

---

## 🏛 Architecture

```
Browser → Dashboard.tsx / QuantumInsights.tsx
          ↓ POST /ingestion/csv-upload → session_id stored in SESSION_STORE
          ↓ POST /quantum/analyze → CTX_CACHE → 8 algorithms (4 parallel threads)
          ↓ GET  /quantum/lookalike → fast re-query from cached QuantumContext
          ↓
     ml/quantum/shared/preprocessing.py
          builds QuantumContext once: normalised features, cosine matrix,
          distance matrix, networkx similarity graph
          → shared by all 8 algorithm modules
```

---

## 🔄 Workflow

1. **Upload CSV** → fuzzy column mapping, instant classical dashboard + `session_id`
2. **Switch to ⚛ Quantum Insights tab**
3. **Click "Run Quantum Analysis"** → 30–90s for all 8 algorithms
4. **Explore panels** — each has charts, benchmark metrics (qubits/depth/shots/sweeps), and switchable Business / Technical summary tabs
5. **Look-Alike re-queries** → < 200ms each, fully independent of the main run

---

## 🏆 Evaluation Perspective

- **Real quantum SDKs** — D-Wave Ocean + Qiskit actually invoked, nothing mocked
- **No false speedup claims** — Grover is labelled "Proof of Concept", SHAP uses "Quantum-Assisted" framing
- **Every quantum result paired with classical benchmark** — silhouette scores, ROC-AUC, F1, coalition counts
- **Per-algorithm resource panel** — qubits, circuit depth, shots, annealing sweeps visible to judges
- **Dual-audience explanations** — plain-English business insight + precise technical formulation per algorithm
- **End-to-end pipeline** — no step is mocked or hardcoded; every result depends on the uploaded dataset

## 🚀 Running Locally

### Backend
```bash
cd backend
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r ../requirements.txt

# Start FastAPI server
set PYTHONPATH=%PYTHONPATH%;..\ml
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```