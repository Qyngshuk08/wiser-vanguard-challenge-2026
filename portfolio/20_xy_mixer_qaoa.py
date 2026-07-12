"""
Portfolio QUBO -- Constraint-Preserving Ring XY-Mixer QAOA
Vanguard x WISER Quantum Challenge 2026

Replaces the penalty-method budget constraint with a structural one: a ring
XY-mixer (Hadfield et al., "quantum alternating operator ansatz") preserves
Hamming weight, so combined with a fixed-weight initial state, EVERY
measured outcome satisfies the budget constraint by construction -- no
penalty tuning, no infeasible results to filter out.

Measured result (noiseless simulator, 1000 shots): 0 infeasible outcomes,
vs. 89.5-95.2% infeasible rate measured across four penalty-method runs
(simulator + real Fez hardware, both tracks) earlier in this project.
Quality: -0.1478 vs. true optimum -0.1803 (~18% gap) -- a real trade-off
against the penalty method's best-case 3.6% gap, but every shot here is
usable, vs. needing to filter for the rare feasible ones with penalties.
"""
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

mu_df = pd.read_csv("data/expected_returns.csv")
cov_df = pd.read_csv("data/covariance_matrix.csv", index_col=0)
asset_ids = mu_df["asset_id"].tolist()
mu = mu_df["expected_return_annual"].values
sigma = cov_df.values
n = len(asset_ids)
RISK_AVERSION_Q = 0.5
BUDGET_B = 4

# Cost operator WITHOUT the budget penalty term -- the mixer enforces it
# structurally instead, so no penalty coefficient is needed at all.
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
qubo_noconstraint = QuadraticProgramToQubo().convert(qp)
op, offset = qubo_noconstraint.to_ising()


def ring_xy_mixer(n_qubits):
    """Sum over neighboring qubit pairs of (XX + YY)/2 -- preserves total
    Hamming weight for any parameter value, so the mixer alone can never
    move probability mass outside the fixed-weight subspace."""
    pauli_list = []
    for i in range(n_qubits):
        j = (i + 1) % n_qubits
        xx = ["I"] * n_qubits
        yy = ["I"] * n_qubits
        xx[i], xx[j] = "X", "X"
        yy[i], yy[j] = "Y", "Y"
        pauli_list.append(("".join(xx)[::-1], 0.5))
        pauli_list.append(("".join(yy)[::-1], 0.5))
    return SparsePauliOp.from_list(pauli_list)


mixer_op = ring_xy_mixer(n)

# Fixed-weight initial state: exactly B qubits set to |1>. Any computational
# basis state with Hamming weight B works as the starting point -- the
# mixer explores the rest of that weight-B subspace from there.
init_circuit = QuantumCircuit(n)
for i in range(BUDGET_B):
    init_circuit.x(i)

sampler = StatevectorSampler()
pm = generate_preset_pass_manager(optimization_level=1, basis_gates=["rz", "sx", "x", "cx", "rxx", "ryy"])

qaoa_mes = QAOA(sampler=sampler, optimizer=COBYLA(maxiter=300), reps=3,
                 mixer=mixer_op, initial_state=init_circuit, transpiler=pm)
qaoa_solver = MinimumEigenOptimizer(qaoa_mes)
result = qaoa_solver.solve(qp)
x = [int(round(v)) for v in result.x]
selected = [asset_ids[i] for i, v in enumerate(x) if v == 1]
print(f"Top result: selected={selected} (n={sum(x)}, target B={BUDGET_B})")
print(f"Cost: {result.fval:.4f}  (true optimum: -0.1803)")

# Verify feasibility-by-construction empirically, not just trust the theory
optimal_angles = result.min_eigen_solver_result.optimal_point
final_ansatz = QAOAAnsatz(cost_operator=op, reps=3, mixer_operator=mixer_op, initial_state=init_circuit)
final_transpiled = pm.run(final_ansatz)
bound = final_transpiled.assign_parameters(optimal_angles)
bound.measure_all()
job = sampler.run([bound], shots=1000)
counts = job.result()[0].data.meas.get_counts()

infeasible = sum(c for bs, c in counts.items() if sum(int(b) for b in bs) != BUDGET_B)
print(f"\nFeasibility check (1000 shots): {1000-infeasible}/1000 feasible ({(1000-infeasible)/10:.1f}%)")

# ---------------------------------------------------------------------------
# Export as OpenQASM 3 for the Classiq IDE (same pattern as
# 18_export_qasm_for_classiq.py -- no qBraid/IBM auth needed for this step)
# ---------------------------------------------------------------------------
from qiskit import qasm3
qasm_str = qasm3.dumps(bound)
with open("portfolio_xy_mixer_circuit.qasm", "w") as f:
    f.write(qasm_str)
print(f"\nSaved: portfolio_xy_mixer_circuit.qasm ({bound.num_qubits} qubits, {bound.size()} gates)")
print("\n--- QASM (paste this into the Classiq IDE) ---\n")
print(qasm_str)
