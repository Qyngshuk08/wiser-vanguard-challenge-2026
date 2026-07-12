"""
Call Center Staffing QUBO -- QASM Export for Classiq IDE
Vanguard x WISER Quantum Challenge 2026

Same approach as portfolio/18_export_qasm_for_classiq.py -- no qBraid, no
IBM Cloud auth. Runs anywhere Python + Qiskit is installed.
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

sim_sampler = StatevectorSampler()
sim_pm = generate_preset_pass_manager(optimization_level=1, basis_gates=["rz", "sx", "x", "cx"])
qaoa_mes = QAOA(sampler=sim_sampler, optimizer=COBYLA(maxiter=300), reps=3, transpiler=sim_pm)
qaoa_solver = MinimumEigenOptimizer(qaoa_mes)
sim_result = qaoa_solver.solve(qp)
optimal_angles = sim_result.min_eigen_solver_result.optimal_point
sim_assignment = {v.split("_")[0]: v.split("_")[1] for v, val in zip(var_names, sim_result.x) if val > 0.5}
print(f"\n[Simulator-trained] cost={sim_result.fval:.0f}  assignment={sim_assignment}")

ansatz = QAOAAnsatz(cost_operator=op, reps=3)
bound_circuit = ansatz.assign_parameters(optimal_angles)
bound_circuit.measure_all()

qasm_str = qasm3.dumps(bound_circuit)
with open("staffing_qaoa_circuit.qasm", "w") as f:
    f.write(qasm_str)

print(f"\nSaved: staffing_qaoa_circuit.qasm ({bound_circuit.num_qubits} qubits, "
      f"{bound_circuit.size()} gates)")
print("\n--- QASM (paste this into the Classiq IDE) ---\n")
print(qasm_str)
