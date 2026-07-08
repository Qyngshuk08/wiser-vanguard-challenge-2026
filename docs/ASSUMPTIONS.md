# Assumptions, Tools, and Methodology -- Consolidated

*Per the challenge brief's requirement: "clear notes on the tools used,
assumptions made, and any post-processing methods applied." This document
consolidates notes that also appear inline in individual script docstrings.*

## Tools used
- **Qiskit 2.5.0** + `qiskit-optimization` 0.7.0 + `qiskit-algorithms` 0.4.0
  + `qiskit-aer` 0.17.2 -- QUBO formulation, QAOA, exact eigensolving
- **qiskit-ibm-runtime** -- real IBM hardware execution (via qBraid Lab)
- Chosen over PennyLane/D-Wave Ocean because `qiskit-optimization`'s
  `QuadraticProgram` maps directly onto our QUBO formulations and provides
  both QAOA and an exact classical solver in one framework, avoiding
  stitching together multiple libraries for formulation, solving, and
  validation.

## Portfolio track -- data assumptions
- **Expected returns**: anchored to Vanguard's own published March 31, 2026
  VCMM 10-year nominal return ranges (real, publicly reported figures --
  see `portfolio/03_asset_universe.md` for the exact source and per-asset
  citation). We use the midpoint of each published range.
- **Volatility and correlation structure**: NOT individually confirmed
  VCMM outputs (not fully public at sub-asset-class level). These are
  documented assumptions, set to be broadly consistent with well-known
  historical asset-class risk characteristics. Explicitly flagged as
  assumptions, not fitted or scraped data, in `portfolio/03_asset_universe.md`.
- **Synthetic historical returns** (750 days): generated via multivariate
  normal sampling from the above mu/covariance, for potential bootstrap
  risk estimation. Not real market data.

## Portfolio track -- methodology assumptions
- **Selection-based QUBO encoding** (binary "is asset i held" rather than
  continuous weight encoding): chosen as the Week-1 baseline per the
  brief's own suggested progression ("build baseline... then add
  constraints"). Trades continuous-weight fidelity for a much smaller
  qubit footprint -- see `portfolio/01_qubo_formulation.md` Section 2 for
  the explicit Option A vs. Option B tradeoff discussion.
- **Sector guardrails translated to cardinality bounds**: the brief's
  percentage-based guardrails (e.g. "max 60% equity") are approximated as
  integer holding-count limits under an equal-weight assumption (e.g. max
  3 of 6 holdings), since a selection-based encoding can't directly
  enforce a weight percentage. This is a known approximation -- exact
  percentage enforcement would require the continuous-weight (Option B)
  encoding described but not built.
- **Turnover/transaction cost**: computed via the XOR identity
  `|x_i - prior_i| = x_i + prior_i - 2*x_i*prior_i`, which is exact and
  QUBO-native for a selection encoding with a known prior portfolio --
  no auxiliary slack variables needed (simpler than the general
  continuous-weight case).
- **Penalty coefficients**: computed via
  `P = 10 * (max|mu| + risk_aversion * max|sigma| * budget)`, scaled to
  the actual data rather than a fixed constant. This value was verified
  to prevent the QUBO's true minimum from being an infeasible or trivial
  solution (see next section for why this check matters).

## Staffing track -- data assumptions
- **Call arrival data**: fully synthetic, generated via a Poisson process
  with a bimodal time-of-day intensity shape (morning ~10:30, afternoon
  ~14:00 peaks) and day-of-week effects. Not real call-center data, per
  challenge data-privacy rules.
- **Agent throughput**: assumed 6 calls/agent/hour, a simplifying constant
  used to convert forecasted call volume into a required-agent-count per
  shift. Real throughput varies by agent, skill, and call complexity.
- **Break rule**: a fixed 30-minute unpaid break at each shift's midpoint,
  applied as a structural/policy rule, NOT modeled as a QUBO decision
  variable -- it isn't a decision the optimizer needs to make, and adding
  it as a variable would cost qubits for zero decision value.

## Staffing track -- methodology assumptions
- **Skill-specific coverage** (Sales vs. Support, not aggregate headcount):
  chosen in the constrained version to more faithfully match the brief's
  "agents across shifts, skills, and communication channels" language.
  The unconstrained baseline uses simpler aggregate per-shift coverage.
- **Capacity-constrained scenario is intentional, not a bug**: in the
  4-agent baseline, required coverage (5 agent-slots across 3 shifts)
  exceeds available capacity (4 agents, one shift each). The true optimum
  itself leaves a shift understaffed -- we kept this rather than adding a
  5th agent to make the problem trivially solvable, because it's a more
  realistic and demonstrable manager trade-off.

## Cross-track finding: penalty coefficients must be checked, not assumed
Both tracks independently hit the same class of bug during development:
an under-scaled penalty coefficient let the QUBO's true minimum become a
degenerate or infeasible solution (Staffing: "hire nobody" beat any real
staffing plan when the coverage penalty was smaller than labor cost;
Portfolio: QAOA returned an infeasible cardinality violation reported as
if it were a better-than-optimal result). Both were caught by cross-
checking against independent brute-force enumeration, not by inspection.
**We recommend this as a standard validation step for any QUBO-based
submission** -- a plausible-looking result is not evidence of a correctly
encoded objective.

## Known limitations, stated explicitly
- Single-period only (no multi-period rebalancing modeled)
- Selection-based (not continuous-weight) portfolio encoding
- Synthetic data throughout; no real Vanguard operational data used
- QAOA tested only on noiseless simulation and, where hardware access was
  available, a single fixed-parameter real-hardware job (not re-optimized
  on hardware, for CPU-hour budget reasons -- see `*_hardware_test.py`)
- Scalability tested up to ~20 qubits; classical exact methods were not
  pushed past ~20 qubits due to dense-diagonalization memory limits
  (documented explicitly in the sweep scripts rather than silently
  truncated)
