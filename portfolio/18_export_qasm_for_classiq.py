"""
Portfolio QUBO -- QASM Export for Classiq IDE
Vanguard x WISER Quantum Challenge 2026

No qBraid, no IBM Cloud auth, no external quantum SDK dependency beyond
Qiskit itself. Runs anywhere Python + Qiskit is installed. Trains QAOA on
the free local simulator, then exports the final circuit as OpenQASM 3 for
manual import into the Classiq IDE (platform.classiq.io), which can execute
it against real IBM hardware using Classiq's own managed credentials.
"""
import numpy as np
import pandas as pd
from qiskit import qasm3

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.circuit.library import QAOAAnsatz
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# ---------------------------------------------------------------------------
# 1. Rebuild the validated 12-qubit Portfolio QUBO
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

qubo = QuadraticProgramToQubo(penalty=PENALTY_P).convert(qp)
op, offset = qubo.to_ising()
print(f"Problem: {op.num_qubits} qubits, {len(op)} Pauli terms")

# ---------------------------------------------------------------------------
# 2. Train QAOA angles on the free local simulator
# ---------------------------------------------------------------------------
sim_sampler = StatevectorSampler()
sim_pm = generate_preset_pass_manager(optimization_level=1, basis_gates=["rz", "sx", "x", "cx"])
qaoa_mes = QAOA(sampler=sim_sampler, optimizer=COBYLA(maxiter=300), reps=3, transpiler=sim_pm)
qaoa_solver = MinimumEigenOptimizer(qaoa_mes)
sim_result = qaoa_solver.solve(qp)
optimal_angles = sim_result.min_eigen_solver_result.optimal_point
sim_assets = [asset_ids[i] for i, v in enumerate(sim_result.x) if v > 0.5]
print(f"\n[Simulator-trained] cost={sim_result.fval:.4f}  assets={sim_assets}")
print(f"Optimal angles: {optimal_angles}")

# ---------------------------------------------------------------------------
# 3. Build the final circuit and export as OpenQASM 3
# ---------------------------------------------------------------------------
ansatz = QAOAAnsatz(cost_operator=op, reps=3)
bound_circuit = ansatz.assign_parameters(optimal_angles)
bound_circuit.measure_all()

qasm_str = qasm3.dumps(bound_circuit)
with open("portfolio_qaoa_circuit.qasm", "w") as f:
    f.write(qasm_str)

print(f"\nSaved: portfolio_qaoa_circuit.qasm ({bound_circuit.num_qubits} qubits, "
      f"{bound_circuit.size()} gates)")
print("\n--- QASM (paste this into the Classiq IDE) ---\n")
print(qasm_str)
