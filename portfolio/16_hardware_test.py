"""
Portfolio QUBO -- Real Hardware Robustness Test (run in qBraid Lab)
Vanguard x WISER Quantum Challenge 2026

CANNOT BE RUN HERE -- this sandbox has no network access to IBM Quantum or
qBraid endpoints. Copy this into a qBraid Lab notebook/session, where
qiskit-ibm-runtime is pre-configured, and run it there.

Design choice: does NOT re-run COBYLA optimization on real hardware. Each
classical-optimizer iteration would be a separate queued hardware job --
expensive in both wall-clock time and your qBraid CPU-hour budget (visible
in your qBraid Lab screenshot: 10:04 / 100:00 used). Instead:
  1. Train QAOA angles on the free noiseless simulator (as already validated
     in 05_portfolio_qubo_baseline.py)
  2. Submit ONE hardware job with those fixed, pre-trained angles
  3. Compare the real-hardware output distribution against the simulator's
This isolates "how much does real noise degrade an already-good circuit"
-- the actual robustness question -- for the cost of a single job.

SETUP (in qBraid Lab):
  pip install qiskit-ibm-runtime  # usually pre-installed in qBraid's Qiskit env
  Set credentials once via:
    from qiskit_ibm_runtime import QiskitRuntimeService
    QiskitRuntimeService.save_account(channel="ibm_quantum_platform",
                                       token="YOUR_IBM_TOKEN", overwrite=True)
  Or use qBraid's own wrapper (equivalent, qBraid-native):
    from qbraid.runtime import QiskitRuntimeProvider
    provider = QiskitRuntimeProvider()  # picks up saved/env credentials
"""
import numpy as np
import pandas as pd

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.circuit.library import QAOAAnsatz
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# --- Real hardware imports -- only resolve inside qBraid Lab ---
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as RuntimeSampler

# Pick one of your available IBM backends (from your qBraid device list):
# ibm_marrakesh, ibm_pittsburgh, ibm_kingston, ibm_miami, ibm_boston, ibm_fez
BACKEND_NAME = "ibm_kingston"   # swap freely -- pick whichever has shortest queue
SHOTS = 4096                    # single job, modest shot count

# ---------------------------------------------------------------------------
# 1. Rebuild the validated 12-qubit Portfolio QUBO (same as 05_portfolio_qubo_baseline.py)
# ---------------------------------------------------------------------------
mu_df = pd.read_csv("data/expected_returns.csv")
cov_df = pd.read_csv("data/covariance_matrix.csv", index_col=0)
asset_ids = mu_df["asset_id"].tolist()
mu = mu_df["expected_return_annual"].values
sigma = cov_df.values
n = len(asset_ids)

RISK_AVERSION_Q = 0.5
BUDGET_B = 4
PENALTY_P = 2.0

qp = QuadraticProgram(name="portfolio_selection")
for aid in asset_ids:
    qp.binary_var(name=aid)
linear = {aid: -mu[i] for i, aid in enumerate(asset_ids)}
quadratic = {}
for i in range(n):
    for j in range(n):
        if i == j:
            quadratic[(asset_ids[i], asset_ids[j])] = RISK_AVERSION_Q * sigma[i, j]
        elif i < j:
            quadratic[(asset_ids[i], asset_ids[j])] = 2 * RISK_AVERSION_Q * sigma[i, j]
qp.minimize(linear=linear, quadratic=quadratic)
qp.linear_constraint(linear={aid: 1 for aid in asset_ids}, sense="==", rhs=BUDGET_B, name="budget")

qubo_converter = QuadraticProgramToQubo(penalty=PENALTY_P)
qubo = qubo_converter.convert(qp)
op, offset = qubo.to_ising()
print(f"Problem: {op.num_qubits} qubits, {len(op)} Pauli terms")

# ---------------------------------------------------------------------------
# 2. Train QAOA angles on the FREE noiseless simulator (no hardware cost)
# ---------------------------------------------------------------------------
sim_sampler = StatevectorSampler()
sim_pm = generate_preset_pass_manager(optimization_level=1, basis_gates=["rz", "sx", "x", "cx"])
qaoa_mes = QAOA(sampler=sim_sampler, optimizer=COBYLA(maxiter=300), reps=3, transpiler=sim_pm)
qaoa_solver = MinimumEigenOptimizer(qaoa_mes)
sim_result = qaoa_solver.solve(qp)
optimal_angles = sim_result.min_eigen_solver_result.optimal_point
sim_assets = [asset_ids[i] for i, v in enumerate(sim_result.x) if v > 0.5]
print(f"\n[Simulator-trained] cost={sim_result.fval:.4f}  assets={sim_assets}")
print(f"Optimal angles (reused on hardware, NOT retrained): {optimal_angles}")

# ---------------------------------------------------------------------------
# 3. Submit ONE job to real IBM hardware with these fixed angles
# ---------------------------------------------------------------------------
service = QiskitRuntimeService()  # reads saved/env credentials
backend = service.backend(BACKEND_NAME)
print(f"\nBackend: {backend.name}, {backend.num_qubits} qubits, "
      f"queue depth: {backend.status().pending_jobs}")

# Real hardware needs backend-aware transpilation (respects actual qubit
# connectivity/coupling map) -- NOT the generic basis-gate transpile used
# for the simulator. This is the key difference from the simulator pipeline.
hw_pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
ansatz = QAOAAnsatz(cost_operator=op, reps=3)
ansatz_transpiled = hw_pm.run(ansatz)
bound_circuit = ansatz_transpiled.assign_parameters(optimal_angles)
bound_circuit.measure_all()

sampler = RuntimeSampler(mode=backend)
job = sampler.run([bound_circuit], shots=SHOTS)
print(f"Job submitted: {job.job_id()}  (check status in qBraid Lab's job monitor)")
result = job.result()
counts = result[0].data.meas.get_counts()

# ---------------------------------------------------------------------------
# 4. Decode and compare against the simulator result
# ---------------------------------------------------------------------------
def decode_bitstring(bitstring, n_asset_vars):
    # Qiskit bit ordering is little-endian relative to qubit index; the
    # asset-selection variables are the first n_asset_vars qubits in our
    # ansatz construction.
    bits = bitstring[::-1]
    return [int(b) for b in bits[:n_asset_vars]]


sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
print(f"\nTop 5 measured bitstrings (of {SHOTS} shots on {BACKEND_NAME}):")
for bitstring, count in sorted_counts[:5]:
    x = decode_bitstring(bitstring, n)
    selected = [asset_ids[i] for i, v in enumerate(x) if v == 1]
    feasible = sum(x) == BUDGET_B
    cost = qp.objective.evaluate(np.array(x)) if feasible else None
    print(f"  {bitstring}  count={count} ({count/SHOTS*100:.1f}%)  "
          f"assets={selected}  feasible={feasible}"
          + (f"  cost={cost:.4f}" if feasible else ""))

top_x = decode_bitstring(sorted_counts[0][0], n)
top_feasible = sum(top_x) == BUDGET_B
print(f"\n=== COMPARISON ===")
print(f"Simulator (noiseless):  cost={sim_result.fval:.4f}  feasible=True   assets={sim_assets}")
print(f"Real hardware (top result): "
      + (f"cost={qp.objective.evaluate(np.array(top_x)):.4f}  feasible=True" if top_feasible else "feasible=False")
      + f"  count share={sorted_counts[0][1]/SHOTS*100:.1f}%")
print("\nThis comparison IS the robustness finding -- report the degradation "
      "(or lack thereof) honestly, whichever direction it goes.")
