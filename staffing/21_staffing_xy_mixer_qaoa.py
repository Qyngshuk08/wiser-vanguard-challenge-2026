"""
Call Center Staffing -- Constraint-Preserving Mixer (per-agent one-hot)
Vanguard x WISER Quantum Challenge 2026

Portfolio's budget constraint ("exactly B of N") maps directly onto a
fixed-Hamming-weight ring XY-mixer. Staffing's constraint ("at most 1 shift
per agent") does NOT -- weight-preserving mixers can't move between weight-0
and weight-1, so a plain ring-XY can't represent "0 or 1" per agent.

Fix: encode each agent as ONE-HOT over 4 categories (Shift1, Shift2, Shift3,
Off) instead of 3 independent bits. "At most 1 real shift" becomes "exactly
1 category" -- which a ring-XY mixer handles natively. Applied as 4
SEPARATE per-agent rings (block-diagonal), not one mixer across all 16
qubits, since agents' constraints are independent of each other.

4 agents x 4 categories = 16 qubits (vs. 12 in the original encoding --
the one-hot trick costs one extra qubit per agent for the "Off" category).
"""
import time
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

agents_df = pd.read_csv("data/agents.csv")
shifts_df = pd.read_csv("data/shifts.csv")
demand_df = pd.read_csv("data/demand_forecast.csv")

agent_ids = agents_df["agent_id"].tolist()
shift_ids = shifts_df["shift_id"].tolist()  # ["S1", "S2", "S3"]
categories = shift_ids + ["OFF"]  # one-hot categories per agent
n_agents = len(agent_ids)
n_cat = len(categories)  # 4
n = n_agents * n_cat  # 16 qubits

cost_per_hour = dict(zip(agents_df["agent_id"], agents_df["hourly_cost"]))
shift_hours = dict(zip(shifts_df["shift_id"], shifts_df["hours"]))

AGENT_THROUGHPUT = 6
required_agents = {}
for _, s in shifts_df.iterrows():
    mask = (demand_df["interval_start_hour"] >= s["start_hour"]) & (demand_df["interval_start_hour"] < s["end_hour"])
    total_calls = demand_df.loc[mask, "forecast_calls"].sum()
    intervals = mask.sum()
    calls_per_hour = total_calls / (intervals * 0.5)
    required_agents[s["shift_id"]] = max(1, int(np.ceil(calls_per_hour / AGENT_THROUGHPUT)))

COVERAGE_PENALTY = 500.0


def vname(a, c):
    return f"{a}_{c}"


var_names = [vname(a, c) for a in agent_ids for c in categories]
qubit_index = {v: i for i, v in enumerate(var_names)}

# ---------------------------------------------------------------------------
# Cost operator WITHOUT the one-shift-per-agent penalty -- the mixer enforces
# it structurally. Coverage-gap penalty still needed (that's a cross-agent
# constraint, not per-agent, so it can't be folded into a per-agent mixer).
# ---------------------------------------------------------------------------
qp = QuadraticProgram(name="staffing_onehot")
for v in var_names:
    qp.binary_var(name=v)

linear = {}
for a in agent_ids:
    for c in categories:
        linear[vname(a, c)] = cost_per_hour[a] * shift_hours[c] if c != "OFF" else 0.0

quadratic = {}
for s in shift_ids:
    req = required_agents[s]
    vars_s = [vname(a, s) for a in agent_ids]
    for v in vars_s:
        linear[v] = linear.get(v, 0) + COVERAGE_PENALTY * (1 - 2 * req)
    for i in range(len(vars_s)):
        for j in range(i + 1, len(vars_s)):
            key = (vars_s[i], vars_s[j])
            quadratic[key] = quadratic.get(key, 0) + 2 * COVERAGE_PENALTY

qp.minimize(linear=linear, quadratic=quadratic)
qubo_noconstraint = QuadraticProgramToQubo().convert(qp)
op, offset = qubo_noconstraint.to_ising()
print(f"Cost operator: {op.num_qubits} qubits, {len(op)} terms (no per-agent penalty needed)")


def block_ring_xy_mixer(n_agents, n_cat):
    """One ring-XY mixer PER AGENT, connecting only that agent's own
    category-qubits. Preserves weight=1 independently within each agent's
    block -- an agent can never end up with 0 or 2+ categories active,
    structurally, regardless of parameters."""
    pauli_list = []
    total_qubits = n_agents * n_cat
    for a in range(n_agents):
        base = a * n_cat
        for i in range(n_cat):
            j = (i + 1) % n_cat
            qi, qj = base + i, base + j
            xx = ["I"] * total_qubits
            yy = ["I"] * total_qubits
            xx[qi], xx[qj] = "X", "X"
            yy[qi], yy[qj] = "Y", "Y"
            pauli_list.append(("".join(xx)[::-1], 0.5))
            pauli_list.append(("".join(yy)[::-1], 0.5))
    return SparsePauliOp.from_list(pauli_list)


mixer_op = block_ring_xy_mixer(n_agents, n_cat)

# Fixed one-hot initial state: every agent starts in "OFF" (last category
# in each 4-qubit block). Any valid one-hot state works as the seed --
# the mixer explores the rest of each agent's weight-1 subspace from there.
init_circuit = QuantumCircuit(n)
for a_idx in range(n_agents):
    off_qubit = a_idx * n_cat + (n_cat - 1)  # "OFF" is the last category
    init_circuit.x(off_qubit)

sampler = StatevectorSampler()
pm = generate_preset_pass_manager(optimization_level=1, basis_gates=["rz", "sx", "x", "cx", "rxx", "ryy"])

t0 = time.time()
qaoa_mes = QAOA(sampler=sampler, optimizer=COBYLA(maxiter=300), reps=3,
                 mixer=mixer_op, initial_state=init_circuit, transpiler=pm)
qaoa_solver = MinimumEigenOptimizer(qaoa_mes)
result = qaoa_solver.solve(qp)
elapsed = time.time() - t0

x = [int(round(v)) for v in result.x]
assignment = {}
for a in agent_ids:
    for c in categories:
        if x[qubit_index[vname(a, c)]] == 1 and c != "OFF":
            assignment[a] = c
print(f"\nTop result: assignment={assignment}  cost={result.fval:.0f}  time={elapsed:.2f}s")

# ---------------------------------------------------------------------------
# Verify feasibility-by-construction: every agent's 4-qubit block should sum
# to EXACTLY 1, always, regardless of measurement outcome.
# ---------------------------------------------------------------------------
optimal_angles = result.min_eigen_solver_result.optimal_point
final_ansatz = QAOAAnsatz(cost_operator=op, reps=3, mixer_operator=mixer_op, initial_state=init_circuit)
final_transpiled = pm.run(final_ansatz)
bound = final_transpiled.assign_parameters(optimal_angles)
bound.measure_all()
job = sampler.run([bound], shots=1000)
counts = job.result()[0].data.meas.get_counts()

infeasible = 0
for bitstring, count in counts.items():
    bits = bitstring[::-1]
    xv = [int(b) for b in bits]
    for a_idx in range(n_agents):
        block = xv[a_idx * n_cat:(a_idx + 1) * n_cat]
        if sum(block) != 1:
            infeasible += count
            break

print(f"\nFeasibility check (1000 shots, per-agent one-hot): "
      f"{1000-infeasible}/1000 feasible ({(1000-infeasible)/10:.1f}%)")

# ---------------------------------------------------------------------------
# Export as OpenQASM 3 for the Classiq IDE (same pattern as the portfolio
# mixer export)
# ---------------------------------------------------------------------------
from qiskit import qasm3
qasm_str = qasm3.dumps(bound)
with open("staffing_xy_mixer_circuit.qasm", "w") as f:
    f.write(qasm_str)
print(f"\nSaved: staffing_xy_mixer_circuit.qasm ({bound.num_qubits} qubits, {bound.size()} gates)")
