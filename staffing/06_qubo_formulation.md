# Call Center Staffing — QUBO Formulation

*Reuses the same penalty-method framework validated in the Portfolio track.*

## 1. Decision variables

`x_{a,s} ∈ {0,1}` — agent `a` is assigned to shift `s` (binary, per brief's
requirement for binary decision variables).

Extend with a skill/channel index if needed: `x_{a,s,k}` = agent `a` works
shift `s` handling skill/channel `k`. Start with the simpler `x_{a,s}` for the
baseline; add the `k` index only after the baseline validates (same
incremental-complexity discipline as the Portfolio track).

## 2. Objective

Minimize total staffing cost while penalizing under/over-coverage relative to
forecasted demand per interval:

```
H = Σ_s cost_s * (Σ_a x_{a,s})                          [labor cost]
  + P_under * Σ_i max(0, demand_i - staffed_i)²          [SLA risk]
  + P_over  * Σ_i max(0, staffed_i - demand_i)²          [idle cost]
```

Where `staffed_i` = number of agents covering interval `i`, derived from which
shifts `s` cover interval `i`. The `max(0, ...)` terms need linearization for
QUBO (standard slack-variable trick, same as the sector-limit inequality in
the Portfolio doc).

## 3. Constraints

| Constraint | Form | QUBO treatment |
|---|---|---|
| One shift per agent per day | `Σ_s x_{a,s} = 1` | Equality penalty (same as portfolio budget constraint) |
| Max overtime | `Σ_s hours_s * x_{a,s} ≤ max_hours` | Slack + penalty |
| Skill coverage | each interval needs ≥1 agent with required skill | Coverage penalty per skill-interval |
| Break rules | shift blocks must include mandated breaks | Encode as fixed structure in shift definitions, not a variable — keeps qubit count down |

## 4. Key difference from Portfolio track: this is a bigger combinatorial space
`agents × shifts` grows fast — 20 agents × 10 shift patterns = 200 binary
variables, already past comfortable statevector simulation (2^200). **Plan to
validate on a small slice** (e.g., 6 agents × 4 shifts = 24 qubits — still
large; realistically start at 4 agents × 3 shifts = 12 qubits, matching what
we just validated works end-to-end in the Portfolio track) and be explicit
about this scale limit in the scalability write-up. This is not a weakness to
hide — Vanguard's own judging criteria include scalability, so showing you
know exactly where the wall is is worth more than pretending it isn't there.

## 5. Next steps
1. Generate synthetic call arrival data (by interval, queue, channel)
2. Build the small-scale QUBO (4 agents × 3 shifts) and validate vs. brute force
3. Reuse the exact validation pattern from Portfolio: brute force → exact
   eigensolver → QAOA (with the transpiler fix already in hand — don't repeat
   that debugging cycle)
4. Layer in the disruption-replanning stretch goal once the static baseline
   is solid
