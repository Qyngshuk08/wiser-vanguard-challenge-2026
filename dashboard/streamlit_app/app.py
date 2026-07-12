"""
Quantum Allocation & Staffing Co-Pilot -- Streamlit App
Vanguard x WISER Quantum Challenge 2026

Real, live-solving interactive dashboard -- NOT a static mockup. Every
number on screen comes from actually re-running the validated QUBO solver
against whatever sliders the user sets, using the same formulation code
validated throughout this project (05_baseline.py / 12_constrained.py logic
for Portfolio, 08_baseline.py / 14_constrained.py logic for Staffing).

Run with: streamlit run app.py
"""
import time
import numpy as np
import pandas as pd
import streamlit as st
from itertools import combinations, product

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import NumPyMinimumEigensolver, QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

st.set_page_config(page_title="Quantum Co-Pilot", layout="wide", page_icon="\u269b\ufe0f")

# ---------------------------------------------------------------------------
# Styling -- same visual language as the earlier dashboard, adapted for
# Streamlit's theming system rather than raw HTML/CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .stApp { background-color: #0B0F17; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; color: #E7ECEF; }
    p, div, span, label { color: #E7ECEF; }
    .stMetric { background-color: #131A26; border: 1px solid #26324A; border-radius: 10px; padding: 10px; }
    div[data-testid="stMetricValue"] { color: #4FD1C5; }
    .stButton button { background-color: #2C6B66; color: #4FD1C5; border: 1px solid #4FD1C5; }
    .stButton button:hover { background-color: #4FD1C5; color: #0B0F17; }
</style>
""", unsafe_allow_html=True)

st.markdown("<p style='color:#4FD1C5; font-size:12px; letter-spacing:.08em; text-transform:uppercase;'>Vanguard x WISER Quantum+AI Challenge 2026</p>", unsafe_allow_html=True)
st.title("Quantum Allocation & Staffing Co-Pilot")
st.markdown("<p style='color:#8B98A9;'>Live QUBO solving behind every slider below -- not precomputed. Move a slider, click Solve, watch the actual optimizer re-run.</p>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Portfolio Co-Pilot", "Workforce Planner"])

# ===========================================================================
# PORTFOLIO TAB
# ===========================================================================
with tab1:
    mu_df = pd.read_csv("portfolio_data/expected_returns.csv")
    cov_df = pd.read_csv("portfolio_data/covariance_matrix.csv", index_col=0)
    meta_df = pd.read_csv("portfolio_data/asset_metadata.csv")
    asset_ids = mu_df["asset_id"].tolist()
    mu = mu_df["expected_return_annual"].values
    sigma = cov_df.values
    n = len(asset_ids)
    asset_class = dict(zip(meta_df["asset_id"], meta_df["asset_class"]))

    col_controls, col_results = st.columns([1, 2])

    with col_controls:
        st.subheader("Tunable goals")
        risk_aversion = st.slider("Risk aversion (q)", 0.1, 2.0, 0.5, 0.1,
                                    help="Higher = more drawdown control, lower = more growth-oriented")
        budget = st.slider("Number of holdings (B)", 3, 8, 4, 1)
        solver_choice = st.radio("Solver", ["Exact (classical, instant)", "QAOA (quantum simulator, ~5s)"])
        solve_btn = st.button("\u25b6 Solve Portfolio", type="primary")

    with col_results:
        if solve_btn:
            qp = QuadraticProgram(name="portfolio_selection")
            for aid in asset_ids:
                qp.binary_var(name=aid)
            linear = {aid: -mu[i] for i, aid in enumerate(asset_ids)}
            quadratic = {}
            for i in range(n):
                for j in range(n):
                    if i == j:
                        quadratic[(asset_ids[i], asset_ids[j])] = risk_aversion * sigma[i, j]
                    elif i < j:
                        quadratic[(asset_ids[i], asset_ids[j])] = 2 * risk_aversion * sigma[i, j]
            qp.minimize(linear=linear, quadratic=quadratic)
            qp.linear_constraint(linear={aid: 1 for aid in asset_ids}, sense="==", rhs=budget, name="budget")

            t0 = time.time()
            with st.spinner(f"Solving live via {solver_choice}..."):
                if solver_choice.startswith("Exact"):
                    solver = MinimumEigenOptimizer(NumPyMinimumEigensolver())
                    result = solver.solve(qp)
                else:
                    sampler = StatevectorSampler()
                    pm = generate_preset_pass_manager(optimization_level=1, basis_gates=["rz", "sx", "x", "cx"])
                    qaoa = QAOA(sampler=sampler, optimizer=COBYLA(maxiter=200), reps=3, transpiler=pm)
                    solver = MinimumEigenOptimizer(qaoa)
                    result = solver.solve(qp)
            elapsed = time.time() - t0

            selected = [asset_ids[i] for i, v in enumerate(result.x) if v > 0.5]
            feasible = len(selected) == budget

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Holdings", len(selected))
            m2.metric("Objective cost", f"{result.fval:.4f}")
            m3.metric("Solve time", f"{elapsed:.2f}s")
            m4.metric("Feasible", "Yes" if feasible else "No")

            st.subheader("Recommended allocation")
            if selected:
                weight = 100 / len(selected)
                alloc_df = pd.DataFrame({
                    "Asset": selected,
                    "Class": [asset_class[a] for a in selected],
                    "Weight %": [weight] * len(selected),
                })
                st.bar_chart(alloc_df.set_index("Asset")["Weight %"])
                st.dataframe(alloc_df, use_container_width=True, hide_index=True)
            else:
                st.warning("No assets selected -- solver did not converge to a valid allocation.")

            st.caption(f"Solved live via {solver_choice} at q={risk_aversion}, B={budget}. "
                       f"This is a real optimizer run, not a cached result.")
        else:
            st.info("Set your goals on the left and click **Solve Portfolio** to run the live optimizer.")

# ===========================================================================
# STAFFING TAB
# ===========================================================================
with tab2:
    agents_df = pd.read_csv("staffing_data/agents.csv")
    shifts_df = pd.read_csv("staffing_data/shifts.csv")
    demand_df = pd.read_csv("staffing_data/demand_forecast.csv")
    agent_ids = agents_df["agent_id"].tolist()
    shift_ids = shifts_df["shift_id"].tolist()
    n_agents = len(agent_ids)
    cost_per_hour = dict(zip(agents_df["agent_id"], agents_df["hourly_cost"]))
    shift_hours = dict(zip(shifts_df["shift_id"], shifts_df["hours"]))

    col_controls2, col_results2 = st.columns([1, 2])

    with col_controls2:
        st.subheader("Tunable goals")
        throughput = st.slider("Agent throughput (calls/hr)", 3, 10, 6, 1)
        coverage_penalty = st.slider("Coverage penalty weight", 50, 1000, 500, 50,
                                       help="Must dominate labor cost or the solver will under-staff -- try lowering this to see the 'hire nobody' failure mode we found during development")
        volume_spike = st.checkbox("Simulate demand spike (+40% on all shifts)")
        solve_btn2 = st.button("\u25b6 Solve Staffing", type="primary")

    with col_results2:
        if solve_btn2:
            spike_mult = 1.4 if volume_spike else 1.0
            required_agents = {}
            for _, s in shifts_df.iterrows():
                mask = (demand_df["interval_start_hour"] >= s["start_hour"]) & (demand_df["interval_start_hour"] < s["end_hour"])
                total_calls = demand_df.loc[mask, "forecast_calls"].sum() * spike_mult
                intervals = mask.sum()
                calls_per_hour = total_calls / (intervals * 0.5)
                required_agents[s["shift_id"]] = max(1, int(np.ceil(calls_per_hour / throughput)))

            def vname(a, s):
                return f"{a}_{s}"

            var_names = [vname(a, s) for a in agent_ids for s in shift_ids]
            qp = QuadraticProgram(name="staffing")
            for v in var_names:
                qp.binary_var(name=v)
            linear = {vname(a, s): cost_per_hour[a] * shift_hours[s] for a in agent_ids for s in shift_ids}
            quadratic = {}
            for s in shift_ids:
                req = required_agents[s]
                vars_s = [vname(a, s) for a in agent_ids]
                for v in vars_s:
                    linear[v] = linear.get(v, 0) + coverage_penalty * (1 - 2 * req)
                for i in range(len(vars_s)):
                    for j in range(i + 1, len(vars_s)):
                        key = (vars_s[i], vars_s[j])
                        quadratic[key] = quadratic.get(key, 0) + 2 * coverage_penalty
            qp.minimize(linear=linear, quadratic=quadratic)
            for a in agent_ids:
                qp.linear_constraint(linear={vname(a, s): 1 for s in shift_ids}, sense="<=", rhs=1, name=f"one_shift_{a}")

            t0 = time.time()
            with st.spinner("Solving live (exact eigensolver)..."):
                solver = MinimumEigenOptimizer(NumPyMinimumEigensolver())
                result = solver.solve(qp)
            elapsed = time.time() - t0

            assignment = {v.split("_")[0]: v.split("_")[1] for v, val in zip(var_names, result.x) if val > 0.5}

            m1, m2, m3 = st.columns(3)
            m1.metric("Total cost", f"{result.fval:.0f}")
            m2.metric("Agents deployed", f"{len(assignment)}/{n_agents}")
            m3.metric("Solve time", f"{elapsed*1000:.0f}ms")

            if volume_spike:
                st.warning("Demand spike active: required coverage increased 40% on all shifts.")

            st.subheader("Recommended schedule")
            for s in shift_ids:
                assigned = [a for a, sh in assignment.items() if sh == s]
                req = required_agents[s]
                status = "\u2705 covered" if len(assigned) >= req else "\u26a0\ufe0f short"
                st.markdown(f"**{s}** -- required: {req}, assigned: {', '.join(assigned) if assigned else 'none'} ({status})")

            st.caption(f"Solved live at throughput={throughput}/hr, penalty={coverage_penalty}, "
                       f"spike={'ON' if volume_spike else 'OFF'}. Real optimizer run, not cached.")
        else:
            st.info("Set your goals on the left and click **Solve Staffing** to run the live optimizer.")

st.divider()
st.caption("All solves above run the actual validated QUBO formulation from this project's repo -- "
           "same math as every number in the presentation deck. Nothing on this page is precomputed.")
