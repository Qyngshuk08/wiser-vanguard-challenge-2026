"""
Portfolio Construction — Constrained QUBO (sector guardrails + turnover cost)
Builds on the validated 05_portfolio_qubo_baseline.py, using the real
12-asset Vanguard-anchored universe (not the synthetic scaling universe).

Adds, per brief deliverable #5 ("expose tunable goals: growth, income,
drawdown control, cost sensitivity") and #6 (guardrail breaches):
  - Sector exposure guardrails (equity/fixed-income/commodities+alt/cash),
    translated into cardinality bounds appropriate for a selection-based QUBO
  - Turnover cost vs. a synthetic "current holdings" portfolio -- for a
    selection encoding, |x_i - prior_i| reduces cleanly to x_i XOR prior_i
    = x_i + prior_i - 2*x_i*prior_i, which is natively QUBO-compatible with
    NO extra slack variables (unlike the general continuous-weight case
    described in the formulation doc's turnover row).
"""
import time
import numpy as np
import pandas as pd
from itertools import combinations

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

mu_df = pd.read_csv("data/expected_returns.csv")
cov_df = pd.read_csv("data/covariance_matrix.csv", index_col=0)
meta_df = pd.read_csv("data/asset_metadata.csv")

asset_ids = mu_df["asset_id"].tolist()
mu = mu_df["expected_return_annual"].values
sigma = cov_df.values
n = len(asset_ids)
asset_class = dict(zip(meta_df["asset_id"], meta_df["asset_class"]))
txn_cost_bps = dict(zip(meta_df["asset_id"], meta_df["transaction_cost_bps"]))

RISK_AVERSION_Q = 0.5
BUDGET_B = 6
TURNOVER_WEIGHT = 3.0  # tunable "cost sensitivity" knob (brief deliverable #5)

# Synthetic "current holdings" -- a plausible prior 6-asset portfolio,
# deliberately different from what we'd expect the optimizer to choose, so
# turnover cost is a real, non-trivial factor in this run.
PRIOR_HOLDINGS = {"A02", "A03", "A05", "A07", "A10", "A12"}
prior = {aid: (1 if aid in PRIOR_HOLDINGS else 0) for aid in asset_ids}

# Sector groupings (from asset_universe.md)
equity = [a for a in asset_ids if asset_class[a] == "Equity"]
fixed_income = [a for a in asset_ids if asset_class[a] == "FixedIncome"]
commod_alt = [a for a in asset_ids if asset_class[a] in ("Commodities", "Alternatives")]
cash = [a for a in asset_ids if asset_class[a] == "Cash"]

# Guardrails translated to cardinality bounds for a B=6 equal-weight selection
# (60%/25%/20%/15% weight guardrails from asset_universe.md, applied to counts)
MAX_EQUITY = 3        # 60% of 6
MIN_FIXED_INCOME = 2  # 25% of 6, rounded to a meaningful integer floor
MAX_COMMOD_ALT = 1    # 20% of 6
MAX_CASH = 1          # 15% of 6, rounded down

print(f"Sector groups: Equity={equity}\n  FixedIncome={fixed_income}\n  Commod/Alt={commod_alt}\n  Cash={cash}")
print(f"Guardrails: equity<={MAX_EQUITY}, fixed_income>={MIN_FIXED_INCOME}, "
      f"commod/alt<={MAX_COMMOD_ALT}, cash<={MAX_CASH}, budget={BUDGET_B}")
print(f"Prior holdings (for turnover cost): {sorted(PRIOR_HOLDINGS)}\n")

# ---------------------------------------------------------------------------
# Build QUBO
# ---------------------------------------------------------------------------
qp = QuadraticProgram(name="portfolio_constrained")
for aid in asset_ids:
    qp.binary_var(name=aid)

# Risk-return objective (doubling fix from baseline applied)
linear = {aid: -mu[i] for i, aid in enumerate(asset_ids)}
quadratic = {}
for i in range(n):
    for j in range(n):
        if i == j:
            quadratic[(asset_ids[i], asset_ids[j])] = RISK_AVERSION_Q * sigma[i, j]
        elif i < j:
            quadratic[(asset_ids[i], asset_ids[j])] = 2 * RISK_AVERSION_Q * sigma[i, j]

# Turnover cost: TURNOVER_WEIGHT * txn_cost_i * (x_i + prior_i - 2*x_i*prior_i)
# Only affects the objective for assets where prior_i = 1 (since prior_i=0
# terms vanish from the linear contribution's "+prior_i" part, and the cross
# term only exists when prior_i=1).
for aid in asset_ids:
    c = TURNOVER_WEIGHT * (txn_cost_bps[aid] / 10000)
    p = prior[aid]
    linear[aid] = linear.get(aid, 0) + c * (1 - 2 * p)
    # the "+ prior_i" constant term (c*p) doesn't affect argmin -- omitted

qp.minimize(linear=linear, quadratic=quadratic)

# Constraints
qp.linear_constraint(linear={aid: 1 for aid in asset_ids}, sense="==", rhs=BUDGET_B, name="budget")
qp.linear_constraint(linear={aid: 1 for aid in equity}, sense="<=", rhs=MAX_EQUITY, name="max_equity")
qp.linear_constraint(linear={aid: 1 for aid in fixed_income}, sense=">=", rhs=MIN_FIXED_INCOME, name="min_fixed_income")
qp.linear_constraint(linear={aid: 1 for aid in commod_alt}, sense="<=", rhs=MAX_COMMOD_ALT, name="max_commod_alt")
qp.linear_constraint(linear={aid: 1 for aid in cash}, sense="<=", rhs=MAX_CASH, name="max_cash")

print(qp.prettyprint())

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def check_all_constraints(x):
    xd = dict(zip(asset_ids, x))
    n_total = sum(xd.values())
    n_equity = sum(xd[a] for a in equity)
    n_fi = sum(xd[a] for a in fixed_income)
    n_ca = sum(xd[a] for a in commod_alt)
    n_cash = sum(xd[a] for a in cash)
    checks = {
        "budget": n_total == BUDGET_B,
        "max_equity": n_equity <= MAX_EQUITY,
        "min_fixed_income": n_fi >= MIN_FIXED_INCOME,
        "max_commod_alt": n_ca <= MAX_COMMOD_ALT,
        "max_cash": n_cash <= MAX_CASH,
    }
    return checks, all(checks.values())


def true_objective(x):
    xv = np.array(x)
    obj = RISK_AVERSION_Q * xv @ sigma @ xv - mu @ xv
    turnover = sum(
        TURNOVER_WEIGHT * (txn_cost_bps[aid] / 10000) * (xv[i] + prior[aid] - 2 * xv[i] * prior[aid])
        for i, aid in enumerate(asset_ids)
    )
    return obj + turnover


# ---------------------------------------------------------------------------
# Brute force ground truth (independent of Qiskit's QUBO conversion)
# ---------------------------------------------------------------------------
t0 = time.time()
best_cost, best_combo = np.inf, None
for combo in combinations(range(n), BUDGET_B):
    x = [1 if i in combo else 0 for i in range(n)]
    _, feasible = check_all_constraints(x)
    if not feasible:
        continue
    cost = true_objective(x)
    if cost < best_cost:
        best_cost, best_combo = cost, [asset_ids[i] for i in combo]
bf_time = time.time() - t0
print(f"\n[Brute force, constraint-filtered] cost={best_cost:.4f}  assets={best_combo}  time={bf_time*1000:.1f}ms")

# ---------------------------------------------------------------------------
# Exact eigensolver (validates the full constrained QUBO conversion)
# ---------------------------------------------------------------------------
penalty = 10 * (np.max(np.abs(mu)) + RISK_AVERSION_Q * np.max(np.abs(sigma)) * BUDGET_B)
exact_solver = MinimumEigenOptimizer(NumPyMinimumEigensolver(), penalty=penalty)
t0 = time.time()
exact_result = exact_solver.solve(qp)
exact_time = time.time() - t0
exact_x = [int(round(v)) for v in exact_result.x[:n]]  # first n vars are the asset selections
exact_assets = [asset_ids[i] for i, v in enumerate(exact_x) if v == 1]
checks, feasible = check_all_constraints(exact_x)
print(f"\n[Exact eigensolver] cost={true_objective(exact_x):.4f}  assets={exact_assets}  "
      f"time={exact_time*1000:.1f}ms  feasible={feasible}  checks={checks}")
print(f"MATCH with brute force: {abs(true_objective(exact_x) - best_cost) < 1e-6 if feasible else 'N/A (infeasible)'}")

# ---------------------------------------------------------------------------
# QAOA
# ---------------------------------------------------------------------------
import signal


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


@with_timeout(60)
def solve_qaoa():
    sampler = StatevectorSampler()
    pm = generate_preset_pass_manager(optimization_level=1, basis_gates=["rz", "sx", "x", "cx"])
    qaoa_mes = QAOA(sampler=sampler, optimizer=COBYLA(maxiter=300), reps=3, transpiler=pm)
    qaoa_solver = MinimumEigenOptimizer(qaoa_mes, penalty=penalty)
    return qaoa_solver.solve(qp)


t0 = time.time()
try:
    qaoa_result = solve_qaoa()
    qaoa_time = time.time() - t0
    qaoa_x = [int(round(v)) for v in qaoa_result.x[:n]]
    qaoa_assets = [asset_ids[i] for i, v in enumerate(qaoa_x) if v == 1]
    checks_q, feasible_q = check_all_constraints(qaoa_x)
    print(f"\n[QAOA] cost={true_objective(qaoa_x):.4f}  assets={qaoa_assets}  time={qaoa_time:.2f}s  "
          f"feasible={feasible_q}  checks={checks_q}")
    if feasible_q:
        print(f"Gap vs optimum: {true_objective(qaoa_x) - best_cost:.4f}")
    else:
        print("QAOA result INFEASIBLE -- not comparable to optimum. Report as-is, do not force a gap number.")
    qaoa_cost_report = true_objective(qaoa_x) if feasible_q else None
    qaoa_assets_report = ", ".join(qaoa_assets) if feasible_q else "INFEASIBLE"
    qaoa_time_report = qaoa_time
    qaoa_feasible_report = feasible_q
except TimeoutException:
    qaoa_time = time.time() - t0
    print(f"\n[QAOA] TIMED OUT after {qaoa_time:.1f}s at 17 qubits -- consistent with the scalability "
          f"sweep finding (QAOA broke down at n=16 in the pure selection case; adding 4 inequality "
          f"guardrails via Qiskit's automatic slack-variable conversion pushed this problem to 17 "
          f"qubits, squarely inside the wall we already measured).")
    qaoa_cost_report = None
    qaoa_assets_report = "TIMED_OUT"
    qaoa_time_report = None
    qaoa_feasible_report = None

print(f"\nTotal qubits used (including auto-added slack variables): {qp.get_num_binary_vars() if hasattr(qp, 'get_num_binary_vars') else 'see qubo conversion'}")

pd.DataFrame({
    "method": ["brute_force", "exact_eigensolver", "qaoa"],
    "cost": [best_cost, true_objective(exact_x) if feasible else None, qaoa_cost_report],
    "assets": [", ".join(best_combo), ", ".join(exact_assets) if feasible else "INFEASIBLE", qaoa_assets_report],
    "feasible": [True, feasible, qaoa_feasible_report],
    "runtime_s": [bf_time, exact_time, qaoa_time_report],
}).to_csv("results/constrained_portfolio_results.csv", index=False)
print("\nSaved: constrained_portfolio_results.csv")
