# Quantum Allocation & Staffing Co-Pilot
**Vanguard x WISER Quantum+AI Challenge 2026**

Hybrid quantum-classical optimization for two Vanguard challenge tracks:
multi-asset portfolio construction and call-center staffing optimization.

## TL;DR result
At the problem scale tested (12-20 qubits), **classical exact solving beats
QAOA on every axis measured** -- speed, optimality, and feasibility. We
report this as the headline finding, not a caveat: see
[`portfolio/results/scalability_sweep_results.csv`](portfolio/results/scalability_sweep_results.csv)
and [`staffing/results/staffing_scalability_sweep_results.csv`](staffing/results/staffing_scalability_sweep_results.csv)
for the measured breakdown points, reproducible by rerunning the sweep
scripts below.

## Repo structure
```
portfolio/
  01-03_*.md          formulation, research positioning, asset universe docs
  data/                synthetic asset data + generator script
  05_baseline.py        QUBO baseline: brute force -> exact -> QAOA
  10-11_*.py            scalability sweep (pushes qubit count to breakdown)
  12_constrained.py     sector guardrails + turnover cost
  16_hardware_test.py   real IBM hardware robustness test (run on qBraid)
  results/               output CSVs from the above

staffing/
  06_qubo_formulation.md
  data/                 synthetic call-center data + generator script
  08_baseline.py         QUBO baseline (same 3-method validation pattern)
  09_disruption_replanning_demo.py   volume-spike re-solve stretch goal
  13_scalability_sweep.py
  14_constrained.py      skill-specific coverage constraints
  17_hardware_test.py    real IBM hardware robustness test (run on qBraid)
  results/

dashboard/
  streamlit_app/            LIVE interactive dashboard -- real solver, not mocked
    .streamlit/
      config.toml             theme config
    portfolio_data/           copy of portfolio CSVs (self-contained)
    staffing_data/            copy of staffing CSVs (self-contained)
    app.py
    requirements.txt
    runtime.txt

docs/
  ASSUMPTIONS.md          consolidated data/methodology assumptions
```

## Live dashboard
**[wiser-vanguard-challenge.streamlit.app](https://wiser-vanguard-challenge.streamlit.app/)**

Every slider triggers a real QUBO solve on click -- not precomputed. Tested
across the full slider range (16 Portfolio combinations, 18 Staffing
combinations including the demand-spike checkbox) before deployment.

## Running the QUBO solver scripts locally
```bash
pip install -r requirements.txt
cd portfolio && python3 05_baseline.py      # ~10s
cd ../staffing && python3 08_baseline.py    # ~10s
```
Each script is self-contained: reads from its local `data/`, writes to its
local `results/`, and prints a brute-force vs. exact-eigensolver vs. QAOA
comparison with feasibility checks.

## Running the real-hardware tests (qBraid)
See the "qBraid setup" section below. Requires an IBM Quantum account linked
to your qBraid session; these scripts submit ONE real-hardware job each
(not a full re-optimization loop) to stay within CPU-hour budgets.

## Key findings (see docs/ASSUMPTIONS.md for full detail)
- Two independent QUBO-encoding bugs were caught by cross-checking against
  brute force (off-diagonal covariance doubling in Portfolio; penalty
  magnitude vs. labor cost in Staffing) -- both documented in-line where
  they were fixed, not hidden.
- Parameter-shift gradients were tested and rejected for QAOA at this
  circuit size: 540 circuit evaluations per gradient step (measured), vs.
  COBYLA's full 300-iteration run in under 6 seconds.
- QAOA's practical wall sits at 16-20 qubits regardless of problem type
  (timeout on Portfolio, OOM crash on Staffing) -- confirmed independently
  twice per track.
- Warm-starting QAOA from a pre-disruption solution did not improve
  re-solve speed or quality (reported honestly as a negative result).

## qBraid setup
```bash
# 1. Clone this repo into your qBraid Lab environment
git clone <your-repo-url>
cd <repo-name>

# 2. Install dependencies (qBraid's Qiskit environment usually has most of
#    this already; this ensures nothing is missing)
pip install -r requirements.txt

# 3. Set IBM Quantum credentials (one-time, persists across sessions)
python3 -c "
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(
    channel='ibm_quantum_platform',
    token='YOUR_IBM_TOKEN',   # from your IBM Quantum Platform account
    overwrite=True
)
"

# 4. Run the hardware robustness tests (each submits ONE job)
cd portfolio && python3 16_hardware_test.py
cd ../staffing && python3 17_hardware_test.py
```

Backend selection is automatic (`least_busy()`, excluding the exploratory
`ibm_miami` Nighthawk chip -- see comments in `16_hardware_test.py` for why).

## Data & methodology transparency
All financial data is synthetic. Expected returns are anchored to
Vanguard's own published March 2026 VCMM 10-year return forecasts;
volatility and correlation structure are documented assumptions, not
fitted or scraped from proprietary data. Full disclosure in
`docs/ASSUMPTIONS.md`. No real Vanguard operational data was used, per
challenge data-privacy rules.
