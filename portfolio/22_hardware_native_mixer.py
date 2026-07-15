"""
Portfolio QUBO -- Hardware-Native Mixer Topology
Vanguard x WISER Quantum Challenge 2026

Tests the hypothesis stated on the deck's closing slide: does building the
ring-XY mixer from the REAL hardware coupling map (instead of an abstract
0-1-2-...-11-0 ring) reduce SWAP overhead and narrow the 100% (simulator)
-> 16-19% (real Fez, abstract ring) feasibility gap found earlier?

Runs entirely locally -- no qBraid or live credentials needed. Uses
qiskit-ibm-runtime's FakeFez, a snapshot of the REAL Fez device's actual
calibration data and coupling map (confirmed: 156 qubits, 352 edges,
matches the real device's public specs). This gives us genuine hardware
topology for building the mixer; only the final submission step (5) needs
manual execution via the Classiq IDE, same as the other two circuits.

Approach: find a genuinely connected 12-qubit PATH in the real coupling
graph (a path, not a strict ring -- easier to find in a sparse degree-3
heavy-hex lattice, and still Hamming-weight-preserving), build the mixer
from THOSE real edges, and pin our logical qubits to that exact physical
layout via initial_layout so the transpiler doesn't need to insert SWAPs.
"""
import random
import numpy as np
import pandas as pd
import networkx as nx

from qiskit import QuantumCircuit, transpile, qasm3
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeFez

# ---------------------------------------------------------------------------
# 1. Get the REAL coupling map -- no live access needed, this is a local
#    snapshot of the actual Fez device.
# ---------------------------------------------------------------------------
backend = FakeFez()
print(f"Backend: {backend.name}, {backend.num_qubits} qubits")
coupling_edges = list(backend.coupling_map.get_edges())
print(f"Coupling map has {len(coupling_edges)} edges")

# ---------------------------------------------------------------------------
# 2. Find a genuinely connected 12-qubit path in the REAL graph
# ---------------------------------------------------------------------------
G = nx.Graph()
G.add_edges_from(coupling_edges)


def find_qubit_path(graph, length, tries=200, seed=1):
    random.seed(seed)
    nodes = list(graph.nodes())
    random.shuffle(nodes)
    for start in nodes[:tries]:
        path = [start]
        visited = {start}
        current = start
        while len(path) < length:
            neighbors = [n for n in graph.neighbors(current) if n not in visited]
            if not neighbors:
                break
            current = random.choice(neighbors)
            path.append(current)
            visited.add(current)
        if len(path) == length:
            return path
    return None


physical_path = find_qubit_path(G, 12)
if physical_path is None:
    raise RuntimeError("Could not find a connected 12-qubit path -- increase `tries`.")
print(f"Found physical qubit path on real Fez: {physical_path}")
print(f"All consecutive pairs connected: "
      f"{all(G.has_edge(physical_path[i], physical_path[i+1]) for i in range(11))}")

# ---------------------------------------------------------------------------
# 3. Rebuild the QUBO and a CHAIN (not ring) XY-mixer using these real edges.
#    A chain still preserves Hamming weight (still built from XX+YY terms);
#    it just has 11 connections instead of 12 (no wraparound), which is
#    exactly what a physical path -- rather than a closed ring -- gives you.
# ---------------------------------------------------------------------------
mu_df = pd.read_csv("data/expected_returns.csv")
cov_df = pd.read_csv("data/covariance_matrix.csv", index_col=0)
asset_ids = mu_df["asset_id"].tolist()
mu = mu_df["expected_return_annual"].values
sigma = cov_df.values
n = len(asset_ids)
RISK_AVERSION_Q = 0.5
BUDGET_B = 4

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


def chain_xy_mixer(n_qubits):
    """Logical qubits 0..n-1, edges only between LOGICALLY consecutive
    qubits (0-1, 1-2, ..., 10-11) -- these get mapped 1:1 onto the REAL
    physical path found above via initial_layout, so every mixer edge is
    a genuine hardware connection, not a SWAP-requiring abstract one."""
    pauli_list = []
    for i in range(n_qubits - 1):  # chain, not ring -- no wraparound edge
        j = i + 1
        xx = ["I"] * n_qubits
        yy = ["I"] * n_qubits
        xx[i], xx[j] = "X", "X"
        yy[i], yy[j] = "Y", "Y"
        pauli_list.append(("".join(xx)[::-1], 0.5))
        pauli_list.append(("".join(yy)[::-1], 0.5))
    return SparsePauliOp.from_list(pauli_list)


mixer_op = chain_xy_mixer(n)
init_circuit = QuantumCircuit(n)
for i in range(BUDGET_B):
    init_circuit.x(i)

# ---------------------------------------------------------------------------
# 4. Train on the free simulator (same mixer structure, so trained angles
#    transfer to the hardware-mapped version)
# ---------------------------------------------------------------------------
sim_sampler = StatevectorSampler()
sim_pm = generate_preset_pass_manager(optimization_level=1, basis_gates=["rz", "sx", "x", "cx", "rxx", "ryy"])
qaoa_mes = QAOA(sampler=sim_sampler, optimizer=COBYLA(maxiter=300), reps=3,
                 mixer=mixer_op, initial_state=init_circuit, transpiler=sim_pm)
qaoa_solver = MinimumEigenOptimizer(qaoa_mes)
sim_result = qaoa_solver.solve(qp)
optimal_angles = sim_result.min_eigen_solver_result.optimal_point
sim_assets = [asset_ids[i] for i, v in enumerate(sim_result.x) if v > 0.5]
print(f"\n[Simulator-trained, chain mixer] cost={sim_result.fval:.4f}  assets={sim_assets}")

# ---------------------------------------------------------------------------
# 5. Transpile with the REAL physical path pinned via initial_layout, then
#    export as QASM for the Classiq IDE -- same manual-paste workflow as
#    the other two circuits, since we don't have live IBM/qBraid access.
# ---------------------------------------------------------------------------
ansatz = QAOAAnsatz(cost_operator=op, reps=3, mixer_operator=mixer_op, initial_state=init_circuit)
bound_circuit = ansatz.assign_parameters(optimal_angles)
bound_circuit.measure_all()

# Pin logical qubits 0..11 onto the exact physical path found in step 2,
# using the real FakeFez backend for transpilation -- this locks in the
# hardware-native layout before export, so Classiq should not need to
# insert its own SWAP routing for the mixer's connectivity.
transpiled = transpile(bound_circuit, backend=backend, initial_layout=physical_path, optimization_level=1)
print(f"\nTranspiled circuit: {transpiled.size()} gates, depth {transpiled.depth()}")

qasm_str = qasm3.dumps(transpiled)
with open("portfolio_hw_native_mixer_circuit.qasm", "w") as f:
    f.write(qasm_str)
print(f"Saved: portfolio_hw_native_mixer_circuit.qasm")
print("\nPaste this into the Classiq IDE, run on Fez, and compare feasibility")
print("against the abstract-ring mixer's 16.4-18.9% result.")

