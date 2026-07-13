"""
Quantum Allocation & Staffing Co-Pilot -- Streamlit App
Vanguard x WISER Quantum Challenge 2026

Real, live-solving interactive dashboard -- every number on screen comes
from actually re-running the validated QUBO solver against whatever
controls the user sets. Nothing here is precomputed or mocked.

Run with: streamlit run app.py
"""
import os
import time
import datetime
import numpy as np
import pandas as pd
import streamlit as st

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import NumPyMinimumEigensolver, QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PORTFOLIO_DATA = os.path.join(APP_DIR, "portfolio_data")
STAFFING_DATA = os.path.join(APP_DIR, "staffing_data")

st.set_page_config(page_title="Quantum Co-Pilot | Vanguard WISER 2026", layout="wide", page_icon="\u25c6")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #0A0E14; }
    section[data-testid="stSidebar"] { background-color: #0D1219; border-right: 1px solid #1E2530; }
    h1, h2, h3 { font-family: 'Inter', sans-serif; color: #E4E7EB; font-weight: 700; }
    p, div, span, label { color: #C4CAD4; }
    .brand-wordmark { font-size: 15px; font-weight: 700; letter-spacing: 0.02em; color: #E4E7EB; margin-bottom: 0px; }
    .brand-sub { font-size: 11px; color: #6B7280; letter-spacing: 0.06em; text-transform: uppercase; margin-top: -4px; }
    .eyebrow { font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase; color: #6E8FA3; font-weight: 600; margin-bottom: 2px; }
    .page-title { font-size: 26px; font-weight: 700; color: #E4E7EB; margin-top: 0px; }
    .page-desc { color: #8A93A3; font-size: 14px; margin-top: -6px; }
    div[data-testid="stMetric"] { background-color: #12161F; border: 1px solid #1E2530; border-radius: 8px; padding: 14px 16px; }
    div[data-testid="stMetricLabel"] { color: #6B7280 !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 0.05em; }
    div[data-testid="stMetricValue"] { color: #E4E7EB !important; font-family: 'IBM Plex Mono', monospace !important; }
    .stButton button { background-color: #1B3A4B; color: #6E8FA3; border: 1px solid #2D5266; border-radius: 6px; font-weight: 600; font-size: 13px; }
    .stButton button:hover { background-color: #2D5266; color: #E4E7EB; border-color: #6E8FA3; }
    .stButton button[kind="primary"] { background-color: #2D5266; color: #E4E7EB; border: 1px solid #6E8FA3; }
    .stButton button[kind="primary"]:hover { background-color: #3A6B85; }
    .status-panel { background-color: #12161F; border: 1px solid #1E2530; border-radius: 8px; padding: 12px 14px; margin-top: 16px; font-size: 12px; }
    .status-row { display: flex; justify-content: space-between; margin-bottom: 4px; }
    .status-ok { color: #5FA88A; }
    .status-label { color: #6B7280; }
    div[data-testid="stExpander"] { background-color: #12161F; border: 1px solid #1E2530; border-radius: 8px; }
    hr { border-color: #1E2530; }
    .footer-bar { color: #4B5563; font-size: 11px; padding-top: 12px; border-top: 1px solid #1E2530; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<div class='brand-wordmark'>\u25c6 QUANTUM CO-PILOT</div>", unsafe_allow_html=True)
    st.markdown("<div class='brand-sub'>Vanguard &times; WISER 2026</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    page = st.radio("Navigate", ["Portfolio Co-Pilot", "Workforce Planner"], label_visibility="collapsed")

    st.markdown("""
    <div class='status-panel'>
        <div class='status-row'><span class='status-label'>Solver engine</span><span class='status-ok'>Qiskit / Ready</span></div>
        <div class='status-row'><span class='status-label'>Data source</span><span>Synthetic (Vanguard-anchored)</span></div>
        <div class='status-row'><span class='status-label'>Mode</span><span class='status-ok'>Live solving</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        f"<p style='font-size:11px; color:#4B5563; margin-top:16px;'>Session started {datetime.datetime.now().strftime('%H:%M')}</p>",
        unsafe_allow_html=True,
    )

# ===========================================================================
# PORTFOLIO CO-PILOT
# ===========================================================================
if page == "Portfolio Co-Pilot":
    st.markdown("<div class='eyebrow'>Multi-Asset Allocation</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-title'>Portfolio Co-Pilot</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-desc'>Quantum-assisted allocation recommendation with live constraint validation. Every result below is computed on demand.</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    mu_df = pd.read_csv(os.path.join(PORTFOLIO_DATA, "expected_returns.csv"))
    cov_df = pd.read_csv(os.path.join(PORTFOLIO_DATA, "covariance_matrix.csv"), index_col=0)
    meta_df = pd.read_csv(os.path.join(PORTFOLIO_DATA, "asset_metadata.csv"))
    asset_ids = mu_df["asset_id"].tolist()
    asset_names = dict(zip(mu_df["asset_id"], mu_df["name"]))
    mu = mu_df["expected_return_annual"].values
    sigma = cov_df.values
    n = len(asset_ids)
    asset_class = dict(zip(meta_df["asset_id"], meta_df["asset_class"]))

    col_controls, col_results = st.columns([1, 2.2], gap="large")

    with col_controls:
        with st.container(border=True):
            st.markdown("**Investment goals**")
            risk_aversion = st.slider("Risk aversion", 0.1, 2.0, 0.5, 0.1,
                                        help="Higher = prioritize drawdown control; lower = prioritize growth")
            budget = st.slider("Target number of holdings", 3, 8, 4, 1)
            solver_choice = st.selectbox("Solver engine", ["Exact (classical)", "QAOA (quantum simulator)"])
            st.markdown("")
            solve_btn = st.button("Generate Recommendation", type="primary", use_container_width=True)

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
            with st.spinner("Running optimizer..."):
                if solver_choice.startswith("Exact"):
                    solver = MinimumEigenOptimizer(NumPyMinimumEigensolver())
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
            m2.metric("Objective Cost", f"{result.fval:.4f}")
            m3.metric("Solve Time", f"{elapsed:.2f}s")
            m4.metric("Constraint Status", "Satisfied" if feasible else "Violated")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Recommended Allocation**")

            if selected:
                weight = 100 / len(selected)
                alloc_df = pd.DataFrame({
                    "Asset": selected,
                    "Name": [asset_names[a] for a in selected],
                    "Class": [asset_class[a] for a in selected],
                    "Weight": [weight / 100] * len(selected),
                })
                st.dataframe(
                    alloc_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Weight": st.column_config.ProgressColumn(
                            "Allocation", min_value=0, max_value=1, format="%.1f%%"
                        ),
                    },
                )
            else:
                st.warning("Solver did not converge to a valid allocation at this configuration.")

            with st.expander("Run details"):
                st.markdown(f"""
                - **Solver:** {solver_choice}
                - **Risk aversion:** {risk_aversion}
                - **Target holdings:** {budget}
                - **Feasible:** {feasible}
                - **Executed:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                """)
        else:
            st.markdown("<div style='padding:60px 0; text-align:center; color:#4B5563;'>Set investment goals and click <b>Generate Recommendation</b> to run the live optimizer.</div>", unsafe_allow_html=True)

# ===========================================================================
# WORKFORCE PLANNER
# ===========================================================================
else:
    st.markdown("<div class='eyebrow'>Call Center Operations</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-title'>Workforce Planner</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-desc'>Quantum-assisted shift scheduling with live demand-driven re-optimization.</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    agents_df = pd.read_csv(os.path.join(STAFFING_DATA, "agents.csv"))
    shifts_df = pd.read_csv(os.path.join(STAFFING_DATA, "shifts.csv"))
    demand_df = pd.read_csv(os.path.join(STAFFING_DATA, "demand_forecast.csv"))
    agent_ids = agents_df["agent_id"].tolist()
    shift_ids = shifts_df["shift_id"].tolist()
    n_agents = len(agent_ids)
    cost_per_hour = dict(zip(agents_df["agent_id"], agents_df["hourly_cost"]))
    shift_hours = dict(zip(shifts_df["shift_id"], shifts_df["hours"]))

    col_controls2, col_results2 = st.columns([1, 2.2], gap="large")

    with col_controls2:
        with st.container(border=True):
            st.markdown("**Operating parameters**")
            throughput = st.slider("Agent throughput (calls/hr)", 3, 10, 6, 1)
            coverage_penalty = st.slider("Coverage priority weight", 50, 1000, 500, 50,
                                           help="Must exceed labor cost or the solver will under-staff")
            volume_spike = st.checkbox("Simulate demand spike (+40%)")
            st.markdown("")
            solve_btn2 = st.button("Generate Schedule", type="primary", use_container_width=True)

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
            with st.spinner("Running optimizer..."):
                solver = MinimumEigenOptimizer(NumPyMinimumEigensolver())
                result = solver.solve(qp)
            elapsed = time.time() - t0

            assignment = {v.split("_")[0]: v.split("_")[1] for v, val in zip(var_names, result.x) if val > 0.5}

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Labor Cost", f"${result.fval:.0f}" if result.fval >= 0 else f"-${abs(result.fval):.0f}")
            m2.metric("Agents Deployed", f"{len(assignment)}/{n_agents}")
            m3.metric("Solve Time", f"{elapsed*1000:.0f}ms")

            if volume_spike:
                st.warning("Demand spike active: required coverage increased 40% across all shifts.")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Recommended Schedule**")

            schedule_rows = []
            for s in shift_ids:
                assigned = [a for a, sh in assignment.items() if sh == s]
                req = required_agents[s]
                schedule_rows.append({
                    "Shift": s,
                    "Required": req,
                    "Assigned": ", ".join(assigned) if assigned else "\u2014",
                    "Status": "Covered" if len(assigned) >= req else "Short",
                })
            schedule_df = pd.DataFrame(schedule_rows)
            st.dataframe(
                schedule_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Status": st.column_config.TextColumn("Status"),
                },
            )

            with st.expander("Run details"):
                st.markdown(f"""
                - **Throughput:** {throughput} calls/hr per agent
                - **Coverage penalty weight:** {coverage_penalty}
                - **Demand spike:** {'Active (+40%)' if volume_spike else 'Inactive'}
                - **Executed:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                """)
        else:
            st.markdown("<div style='padding:60px 0; text-align:center; color:#4B5563;'>Set operating parameters and click <b>Generate Schedule</b> to run the live optimizer.</div>", unsafe_allow_html=True)

st.markdown(
    "<div class='footer-bar'>Vanguard &times; WISER Quantum+AI Challenge 2026 &middot; "
    "All computations run live against validated QUBO formulations &middot; No results are precomputed or cached</div>",
    unsafe_allow_html=True,
)
