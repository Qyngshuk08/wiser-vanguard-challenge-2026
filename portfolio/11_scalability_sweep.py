"""
Scalability Sweep — Portfolio QUBO
Pushes n (number of assets = qubits, selection-QUBO encoding) up until
something breaks: brute force becomes infeasible, exact eigensolver runs out
of memory/time, or QAOA's runtime/quality degrades unacceptably.

This produces the scalability curve required by Vanguard's judging criteria
(speed, optimality, scalability) -- a single data point (n=12) is not
sufficient evidence of scalability; this sweep is.
"""
import time
import signal
import math
import numpy as np
import pandas as pd
from itertools import combinations

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from scaling_universe_generator import generate_universe

RISK_AVERSION_Q = 0.5
TIME_BUDGET_S = 45  # per-method cutoff -- if a method exceeds this at some n, we stop scaling it


class TimeoutException(Exception):
    pass


def with_timeout(seconds):
    def decorator(func):
        def wrapper(*args, **kwargs):
            def handler(signum, frame):
                raise TimeoutException()
            old_handler = signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)
            try:
                return func(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        return wrapper
    return decorator


def build_qp(asset_ids, mu, sigma, B):
    n = len(asset_ids)
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
    qp.linear_constraint(linear={aid: 1 for aid in asset_ids}, sense="==", rhs=B, name="budget")
    return qp


@with_timeout(TIME_BUDGET_S)
def run_brute_force(asset_ids, mu, sigma, B):
    n = len(asset_ids)
    best_cost = np.inf
    for combo in combinations(range(n), B):
        x = np.zeros(n)
        x[list(combo)] = 1
        cost = RISK_AVERSION_Q * x @ sigma @ x - mu @ x
        if cost < best_cost:
            best_cost = cost
    return best_cost


def compute_penalty(mu, sigma, B):
    # Must dominate the largest possible objective swing from violating the
    # budget constraint by one asset -- same principle as the staffing-track
    # bug (COVERAGE_PENALTY too small vs. labor cost). Scale with data, not a
    # fixed constant, since larger universes have larger raw objective ranges.
    return 10 * (np.max(np.abs(mu)) + RISK_AVERSION_Q * np.max(np.abs(sigma)) * B)


def check_feasible(x, B):
    return abs(np.sum(x) - B) < 1e-6


@with_timeout(TIME_BUDGET_S)
def run_exact(qp, penalty):
    solver = MinimumEigenOptimizer(NumPyMinimumEigensolver(), penalty=penalty)
    return solver.solve(qp)


@with_timeout(TIME_BUDGET_S)
def run_qaoa(qp, penalty):
    sampler = StatevectorSampler()
    pm = generate_preset_pass_manager(optimization_level=1, basis_gates=["rz", "sx", "x", "cx"])
    qaoa_mes = QAOA(sampler=sampler, optimizer=COBYLA(maxiter=200), reps=3, transpiler=pm)
    solver = MinimumEigenOptimizer(qaoa_mes, penalty=penalty)
    return solver.solve(qp)


results = []
for n in [8, 12, 16, 20, 24, 28, 32]:
    print(f"\n{'='*50}\nn = {n} assets / qubits\n{'='*50}")
    asset_ids, mu, sigma = generate_universe(n)
    B = max(2, n // 3)

    row = {"n_qubits": n, "budget_B": B}

    # Brute force
    try:
        t0 = time.time()
        bf_cost = run_brute_force(asset_ids, mu, sigma, B)
        row["brute_force_time_s"] = time.time() - t0
        row["brute_force_cost"] = bf_cost
        print(f"  Brute force: {bf_cost:.4f} in {row['brute_force_time_s']:.2f}s "
              f"(searched C({n},{B}) = {math.comb(n, B):,} combos)")
    except TimeoutException:
        row["brute_force_time_s"] = None
        row["brute_force_cost"] = None
        print(f"  Brute force: TIMED OUT (>{TIME_BUDGET_S}s) -- C({n},{B}) too large to enumerate")

    # Exact eigensolver -- skip above ~20 qubits: dense diagonalization of a
    # 2^n x 2^n matrix needs ~(2^n)^2 * 16 bytes; at n=24 that's ~4.5 TB,
    # which previously killed the whole process via OOM rather than raising
    # a catchable exception. Cap it explicitly instead of relying on a timeout.
    EXACT_QUBIT_LIMIT = 20
    qp = build_qp(asset_ids, mu, sigma, B)
    penalty = compute_penalty(mu, sigma, B)
    row["penalty_used"] = penalty
    if n > EXACT_QUBIT_LIMIT:
        row["exact_time_s"] = None
        row["exact_cost"] = None
        row["exact_feasible"] = None
        print(f"  Exact eigensolver: SKIPPED -- n={n} exceeds dense-diagonalization "
              f"memory limit (2^{n} x 2^{n} matrix, ~{(2**n)**2*16/1e9:.0f} GB)")
    else:
        try:
            t0 = time.time()
            exact_result = run_exact(qp, penalty)
            row["exact_time_s"] = time.time() - t0
            exact_feasible = check_feasible(exact_result.x, B)
            row["exact_feasible"] = exact_feasible
            row["exact_cost"] = exact_result.fval if exact_feasible else None
            print(f"  Exact eigensolver: {exact_result.fval:.4f} in {row['exact_time_s']:.2f}s "
                  f"(feasible={exact_feasible}, selected={int(exact_result.x.sum())}/{B})")
        except TimeoutException:
            row["exact_time_s"] = None
            row["exact_cost"] = None
            row["exact_feasible"] = None
            print(f"  Exact eigensolver: TIMED OUT (>{TIME_BUDGET_S}s)")

    # QAOA
    try:
        t0 = time.time()
        qaoa_result = run_qaoa(qp, penalty)
        row["qaoa_time_s"] = time.time() - t0
        qaoa_feasible = check_feasible(qaoa_result.x, B)
        row["qaoa_feasible"] = qaoa_feasible
        row["qaoa_cost"] = qaoa_result.fval if qaoa_feasible else None
        gap = None
        ref = row.get("exact_cost") or row.get("brute_force_cost")
        if ref is not None and qaoa_feasible:
            gap = qaoa_result.fval - ref
        row["qaoa_gap"] = gap
        print(f"  QAOA: {qaoa_result.fval:.4f} in {row['qaoa_time_s']:.2f}s "
              f"(feasible={qaoa_feasible}, selected={int(qaoa_result.x.sum())}/{B})"
              + (f"  gap={gap:.4f}" if gap is not None else "  (infeasible or no reference -- not comparable)"))
    except TimeoutException:
        row["qaoa_time_s"] = None
        row["qaoa_cost"] = None
        row["qaoa_gap"] = None
        row["qaoa_feasible"] = None
        print(f"  QAOA: TIMED OUT (>{TIME_BUDGET_S}s)")

    results.append(row)

    # Stop scaling once QAOA itself breaks down -- that's the real
    # quantum-relevant scalability limit; the exact solver is only a
    # reference and is expected to be skipped/unavailable well before that.
    if row.get("qaoa_time_s") is None:
        print(f"\nQAOA broke down at n={n}. Stopping sweep.")
        break

df = pd.DataFrame(results)
df.to_csv("results/scalability_sweep_results.csv", index=False)
print("\nSaved: scalability_sweep_results.csv")
print(df.to_string(index=False))
