"""
Call Center Staffing QUBO -- Real Hardware Robustness Test (run in qBraid Lab)
Vanguard x WISER Quantum Challenge 2026

CANNOT BE RUN HERE -- no network access to IBM Quantum or qBraid endpoints
in this sandbox. Copy into a qBraid Lab notebook and run there.

Same design as 16_portfolio_hardware_test.py: train QAOA angles on the free
noiseless simulator, submit ONE hardware job with fixed angles, compare
distributions. See that file's header for the full rationale (budget-
conscious, isolates the noise-robustness question specifically).
"""
import numpy as np
import pandas as pd

from qiskit_optimization import QuadraticProgram
from qiskit.circuit.library import QAOAAnsatz
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_optimization.converters import QuadraticProgramToQubo

from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as RuntimeSampler

BACKEND_NAME = "ibm_kingston"   # swap for whichever of your 6 IBM devices has shortest queue
SHOTS = 4096

# ---------------------------------------------------------------------------
# 1. Rebuild the validated 12-qubit Staffing QUBO (same as 08_staffing_qubo_baseline.py)
# ---------------------------------------------------------------------------
agents_df = pd.read_csv("data/agents.csv")
shifts_df = pd.read_csv("data/shifts.csv")
demand_df = pd.read_csv("data/demand_forecast.csv")

agent_ids = agents_df["agent_id"].tolist()
shift_ids = shifts_df["shift_id"].tolist()
n_agents = len(agent_ids)
cost_per_hour = dict(zip(agents_df["agent_id"], agents_df["hourly_cost"]))
shift_hours = dict(zip(shifts_df["shift_id"], shifts_df["hours"]))

AGENT_THROUGHPUT_CALLS_PER_HOUR = 6
required_agents = {}
for _, s in shifts_df.iterrows():
    mask = (demand_df["interval_start_hour"] >= s["start_hour"]) & (demand_df["interval_start_hour"] < s["end_hour"])
    total_calls = demand_df.loc[mask, "forecast_calls"].sum()
    intervals = mask.sum()
    calls_per_hour = total_calls / (intervals * 0.5)
    required_agents[s["shift_id"]] = max(1, int(np.ceil(calls_per_hour / AGENT_THROUGHPUT_CALLS_PER_HOUR)))

COVERAGE_PENALTY = 500.0


def vname(a, s):
    return f"{a}_{s}"


var_names = [vname(a, s) for a in agent_ids for s in shift_ids]
n = len(var_names)

qp = QuadraticProgram(name="staffing")
for v in var_names:
    qp.binary_var(name=v)
linear = {vname(a, s): cost_per_hour[a] * shift_hours[s] for a in agent_ids for s in shift_ids}
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
for a in agent_ids:
    qp.linear_constraint(linear={vname(a, s): 1 for s in shift_ids}, sense="<=", rhs=1, name=f"one_shift_{a}")

qubo = QuadraticProgramToQubo().convert(qp)
op, offset = qubo.to_ising()
print(f"Problem: {op.num_qubits} qubits, {len(op)} Pauli terms")

# ---------------------------------------------------------------------------
# 2. Train on free simulator
# ---------------------------------------------------------------------------
sim_sampler = StatevectorSampler()
sim_pm = generate_preset_pass_manager(optimization_level=1, basis_gates=["rz", "sx", "x", "cx"])
qaoa_mes = QAOA(sampler=sim_sampler, optimizer=COBYLA(maxiter=300), reps=3, transpiler=sim_pm)
qaoa_solver = MinimumEigenOptimizer(qaoa_mes)
sim_result = qaoa_solver.solve(qp)
optimal_angles = sim_result.min_eigen_solver_result.optimal_point
sim_assignment = {v.split("_")[0]: v.split("_")[1] for v, val in zip(var_names, sim_result.x) if val > 0.5}
print(f"\n[Simulator-trained] cost={sim_result.fval:.0f}  assignment={sim_assignment}")

# ---------------------------------------------------------------------------
# 3. Submit ONE job to real hardware
# ---------------------------------------------------------------------------
service = QiskitRuntimeService()
backend = service.backend(BACKEND_NAME)
print(f"\nBackend: {backend.name}, {backend.num_qubits} qubits, "
      f"queue depth: {backend.status().pending_jobs}")

hw_pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
ansatz = QAOAAnsatz(cost_operator=op, reps=3)
ansatz_transpiled = hw_pm.run(ansatz)
bound_circuit = ansatz_transpiled.assign_parameters(optimal_angles)
bound_circuit.measure_all()

sampler = RuntimeSampler(mode=backend)
job = sampler.run([bound_circuit], shots=SHOTS)
print(f"Job submitted: {job.job_id()}")
result = job.result()
counts = result[0].data.meas.get_counts()

# ---------------------------------------------------------------------------
# 4. Decode and compare
# ---------------------------------------------------------------------------
def decode_bitstring(bitstring, n_vars):
    bits = bitstring[::-1]
    return [int(b) for b in bits[:n_vars]]


sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
print(f"\nTop 5 measured bitstrings (of {SHOTS} shots on {BACKEND_NAME}):")
for bitstring, count in sorted_counts[:5]:
    x = decode_bitstring(bitstring, n)
    assignment = {var_names[i].split("_")[0]: var_names[i].split("_")[1] for i, v in enumerate(x) if v == 1}
    cost = qp.objective.evaluate(np.array(x))
    print(f"  {bitstring}  count={count} ({count/SHOTS*100:.1f}%)  assignment={assignment}  cost={cost:.0f}")

print(f"\n=== COMPARISON ===")
print(f"Simulator (noiseless): cost={sim_result.fval:.0f}  assignment={sim_assignment}")
print(f"Real hardware top result: see above -- report the degradation (or lack thereof) honestly.")
