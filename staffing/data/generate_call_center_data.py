"""
Synthetic Call Center Data Generator — Vanguard x WISER Quantum Challenge 2026
Call Center Staffing Optimization track

Generates:
  - agents.csv           : agent roster with skills and cost
  - shifts.csv            : shift patterns (start/end/hours/cost)
  - call_arrivals_history.csv : 90 days of interval-level synthetic call arrivals
  - demand_forecast.csv   : next-day forecasted demand per interval/queue (derived
                             from history, used as the optimizer's input)

Scope note: sized for the SMALL validation slice (4 agents x 3 shifts = 12
qubits) matching what we already validated end-to-end on the Portfolio track,
per the scaling discussion in 06_staffing_qubo_formulation.md. Scale up only
after this validates.
"""

import numpy as np
import pandas as pd

RNG_SEED = 7
rng = np.random.default_rng(RNG_SEED)

# ---------------------------------------------------------------------------
# 1. Shift patterns (3 shifts, matches the 12-qubit baseline: 4 agents x 3 shifts)
# ---------------------------------------------------------------------------
shifts = pd.DataFrame([
    {"shift_id": "S1", "start_hour": 8,  "end_hour": 16, "hours": 8, "cost_per_hour": 28},
    {"shift_id": "S2", "start_hour": 10, "end_hour": 18, "hours": 8, "cost_per_hour": 28},
    {"shift_id": "S3", "start_hour": 12, "end_hour": 20, "hours": 8, "cost_per_hour": 31},  # evening premium
])
shifts.to_csv("shifts.csv", index=False)

# ---------------------------------------------------------------------------
# 2. Agent roster (4 agents, mixed skills -- matches 12-qubit baseline)
# ---------------------------------------------------------------------------
agents = pd.DataFrame([
    {"agent_id": "AG1", "skills": "Sales,Support", "hourly_cost": 26, "max_hours_week": 40},
    {"agent_id": "AG2", "skills": "Support",       "hourly_cost": 24, "max_hours_week": 40},
    {"agent_id": "AG3", "skills": "Sales",         "hourly_cost": 25, "max_hours_week": 40},
    {"agent_id": "AG4", "skills": "Sales,Support", "hourly_cost": 27, "max_hours_week": 32},
])
agents.to_csv("agents.csv", index=False)

# ---------------------------------------------------------------------------
# 3. Synthetic historical call arrivals: 90 days, 30-min intervals, 8am-8pm,
#    2 queues (Sales, Support), 2 channels (Phone, Chat)
# ---------------------------------------------------------------------------
INTERVAL_MIN = 30
DAY_START_HOUR = 8
DAY_END_HOUR = 20
N_INTERVALS_PER_DAY = int((DAY_END_HOUR - DAY_START_HOUR) * 60 / INTERVAL_MIN)  # 24
N_DAYS = 90

queues = ["Sales", "Support"]
channels = ["Phone", "Chat"]

# Time-of-day base arrival-rate shape (calls per interval), roughly bimodal:
# a late-morning peak and an early-afternoon peak, tapering at open/close.
interval_hours = np.array([DAY_START_HOUR + i * INTERVAL_MIN / 60 for i in range(N_INTERVALS_PER_DAY)])
base_shape = (
    3.0
    + 6.0 * np.exp(-((interval_hours - 10.5) ** 2) / (2 * 1.5 ** 2))   # morning peak ~10:30
    + 5.0 * np.exp(-((interval_hours - 14.0) ** 2) / (2 * 1.8 ** 2))   # afternoon peak ~14:00
)

# Queue/channel mix and relative volume weight
mix = {
    ("Sales", "Phone"): 0.35,
    ("Sales", "Chat"): 0.15,
    ("Support", "Phone"): 0.30,
    ("Support", "Chat"): 0.20,
}

records = []
dates = pd.bdate_range(end="2026-07-03", periods=N_DAYS)  # business days only
for date in dates:
    dow_factor = 1.15 if date.dayofweek == 0 else (0.9 if date.dayofweek == 4 else 1.0)  # Monday busier, Friday quieter
    for i, hour in enumerate(interval_hours):
        for (q, c), weight in mix.items():
            lam = max(0.2, base_shape[i] * weight * dow_factor)
            calls = rng.poisson(lam)
            avg_handle_time_min = rng.normal(6.5 if q == "Sales" else 8.0, 1.0)
            records.append({
                "date": date.date().isoformat(),
                "interval_start_hour": hour,
                "queue": q,
                "channel": c,
                "calls_arrived": calls,
                "avg_handle_time_min": max(2.0, avg_handle_time_min),
            })

history_df = pd.DataFrame(records)
history_df.to_csv("call_arrivals_history.csv", index=False)

# ---------------------------------------------------------------------------
# 4. Next-day demand forecast: mean + a safety buffer (simple, transparent
#    baseline forecast -- swap for a proper time-series model later; the
#    QUBO/staffing logic doesn't depend on which forecast method feeds it)
# ---------------------------------------------------------------------------
forecast = (
    history_df.groupby(["interval_start_hour", "queue", "channel"])["calls_arrived"]
    .agg(["mean", "std"])
    .reset_index()
)
forecast["forecast_calls"] = np.ceil(forecast["mean"] + 0.5 * forecast["std"].fillna(0)).astype(int)
forecast = forecast[["interval_start_hour", "queue", "channel", "forecast_calls"]]
forecast.to_csv("demand_forecast.csv", index=False)

print("Generated files:")
print(" - shifts.csv")
print(" - agents.csv")
print(" - call_arrivals_history.csv")
print(" - demand_forecast.csv")
print()
print(f"History: {N_DAYS} business days x {N_INTERVALS_PER_DAY} intervals x {len(mix)} queue/channel combos")
print(f"Total historical rows: {len(history_df)}")
print()
print("Sanity check -- peak interval forecast (should reflect the ~10:30/14:00 peaks):")
print(forecast.sort_values("forecast_calls", ascending=False).head(6).to_string(index=False))
