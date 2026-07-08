# Open Problems in the Field — and How to Position Your Submission

*Research summary, July 2026 — informs both Portfolio Construction and Call Center
Staffing tracks*

## Why this matters for judging
Vanguard's rubric rewards **speed, optimality, and scalability** — not "does quantum
work at all." The literature below tells you exactly where the field's real ceiling
is right now. Your strongest move is not to claim you've beaten that ceiling
(nobody has), but to **clearly characterize where your solution sits relative to it**,
and to pick one specific unsolved sub-problem to push on rather than attempting
everything shallowly.

---

## Track 1: Multi-Asset Portfolio Construction

### Open problem A — Scale gap
Published QAOA portfolio work: ~4-8 assets. Quantum annealing: ~40-60 assets.
Real institutional portfolios: hundreds to thousands, with layered constraints
(sector, regulatory, ESG). Source: arXiv 2604.08180 (2026 review).

**Your angle:** explicitly report the largest n you can solve with each method
(QAOA-simulator vs. classical baseline) and show the point where quantum
solution quality or runtime starts to degrade. A clear "here's our scalability
frontier" chart is more credible than vague claims of speedup.

### Open problem B — Hardware noise vs. financial precision
NISQ-era devices have real accuracy limits; financial applications are
loss-sensitive to small errors. Source: Intel Market Research, May 2026.

**Your angle:** run the same problem on a noiseless simulator, a realistic
noise-model simulator, and (if accessible) real hardware. Quantify the
degradation. This directly demonstrates "robustness" from the brief's deliverables.

### Open problem C — Dynamic multi-period rebalancing under frictions
Multi-period optimization with transaction costs and integer constraints is
still an open challenge even classically. Source: arXiv 2502.05226 (2025).

**Your angle (stretch goal, optional):** if time allows, extend your QUBO from a
single-period allocation to a 2-3 period rebalancing problem with turnover
costs. Almost nobody else in the challenge will attempt this — it directly maps
to the brief's "turnover" and "cost sensitivity" tunable goals.

---

## Track 2: Call Center Staffing Optimization

### Open problem D — Real-time replanning under disruption
Most published work solves a static schedule well; live re-optimization when
conditions change mid-shift (volume spike, agent call-out) is largely unsolved.
Source: arXiv 2512.19340 (railway case study, 2026) — full-scale problems only
become tractable once shrunk to near-real-time subsets.

**Your angle:** build your staffing optimizer, then simulate a mid-day disruption
(e.g., 30% volume spike in one queue) and show how fast you can re-solve for a
patch schedule vs. a full re-solve. This maps directly to deliverable #6 in the
brief ("manager controls to prioritize... resilience").

### Open problem E — Embedding complexity for densely-coupled constraints
Highly connected constraint structures (agent × skill × shift × channel) don't
embed well on current quantum annealing hardware, forcing hybrid classical-
quantum decomposition. Source: arXiv 2509.04808 (AIS sports scheduling, 2025).

**Your angle:** be upfront about this in your write-up — propose a decomposition
strategy (e.g., solve per-skill-group sub-schedules with quantum, stitch
together classically) rather than pretending a single monolithic QUBO scales.

### Open problem F — Thin prior work specifically on call centers
Li et al. (2023) is one of the few papers modeling call center shift scheduling
as QUBO; it demonstrates competitive results but is limited in scope (likely
single-skill, single-channel).

**Your angle:** your differentiator can be extending toward multi-skill/
omnichannel — which the brief explicitly asks for ("agents across shifts,
skills, and communication channels") and which prior published work hasn't
fully covered. This is a genuine, defensible novelty claim.

### Open problem G — Honest state of the art
Current consensus (2026 QUBO/HPC scheduling benchmark): classical solvers
(CP-SAT, HEFT) remain the practical choice at today's problem scale; QUBO is a
framework for future hardware, not a current replacement. Source: arXiv
2605.25350.

**Your angle:** don't overclaim quantum advantage. Present your classical
baseline honestly as competitive-or-better at current scale, and frame your
quantum results as characterizing *where the crossover happens* as problem
size grows or hardware improves. This kind of intellectual honesty is what
distinguishes fellowship-caliber work from hype.

---

## Suggested narrative thread for your presentation
"We didn't just build a quantum solver — we mapped where quantum genuinely
helps today, where it doesn't yet, and what specific unsolved sub-problem
(dynamic rebalancing / real-time disruption replanning) we chose to push
further than existing published work." That story is memorable, defensible in
Q&A, and matches exactly what judges are scoring for.
