# Belief propagation Engine via Lightweight Inference on Factor graphs (BeLIEF)

A general-purpose Bayesian inference engine that computes `P(target | evidence)` over
factor graphs. Pure Python 3.11+, one dependency (`numpy`).

**What's here:** the inference core (`Model`, `Variable`, `Factor`, operators), a
production policy layer (`BayesianPolicy`), and a full production stack of six modules
(`SchedulerDB`, `ModelRegistry`, `DecisionLogger`, `EvidenceRecorder`,
`StreamingEngine`, `BayesianScheduler`) for autonomous agent use.

## Install

**One-time setup** (any machine):

```bash
# Clone the repo (first time only)
git clone git@github.com:hae-banko/bayesian-scheduler.git
cd bayesian-scheduler

# Install in editable mode — edits to source take effect immediately
pip install -e .
```

> **Editable mode (`-e`)** means the package is linked to your source tree.
> You can edit code and it works immediately — no `pip install` after every change.

**Alternatively, install directly from git (no clone):**

```bash
pip install git+ssh://git@github.com/hae-banko/bayesian-scheduler.git
```

> This installs a snapshot. For active development, use the editable clone above.

**Hermes Agent** — if the repo is already at `/mnt/h/fun/bayesian-engine`, just run:

```bash
pip install -e /mnt/h/fun/bayesian-engine
```

## Quick start — core inference

```python
from bayesian_engine import Model, Variable, Factor
from bayesian_engine.operators import table

# Build a model: P(rain | cloudy)
model = Model("weather")
model.add_variable(Variable("cloudy", domain=["yes", "no"], value="yes"))
model.add_variable(Variable("rain", domain=["yes", "no"]))
model.add_factor(Factor(
    inputs=["cloudy"],
    output="rain",
    weight_function=table({
        ("yes", "yes"): 0.8,
        ("yes", "no"):  0.2,
    }),
))

# Query
result = model.query("rain", evidence={"cloudy": "yes"})
print(result["rain"]["value"])        # 'yes'
print(result["rain"]["distribution"])  # [{'value': 'yes', 'probability': 0.8}, ...]
```

## Quick start — production (BayesianScheduler)

For the copresence use case (when should Reva surface herself?), use the
high-level scheduler which handles belief tracking, audit logging, and outcome
recording in one call:

```python
from bayesian_engine.production import BayesianScheduler

scheduler = BayesianScheduler(model_name="copresence")

# Inference with full audit trail
decision = scheduler.infer(
    target="action_class",
    evidence={
        "day_type": "weekday",
        "hour_block": "morning",
        "user_presence": "active_recently",
        "user_load": "free",
        "agent_need": "gentle_checkin",
    },
    session_id="hermes-session-abc",
    cron_job_id="copresence-cron-001",
)
# decision.action      → e.g. "CHECK_IN"
# decision.reason      → e.g. "argmax_above_threshold"
# decision.confidence  → e.g. 0.72

# Record what actually happened → updates future beliefs
scheduler.record_outcome(
    task_id="task-123",
    observed_action="CHECK_IN",
    outcome="positive",   # "positive" | "negative" | "neutral"
    observed_at="2026-05-07T18:00:00Z",
)

# Review past decisions
recent = scheduler.list_decisions(limit=10)
```

State lives in `~/.hermes/bayesian_scheduler/` (auto-created):
- `scheduler.db` — SQLite with full decision + outcome history
- `scheduler.streaming` — live belief state (particle filter, persisted)
- `scheduler.decisions` — JSONL audit log
- `scheduler.evidence` — outcome → evidence shaping

## Concepts

### Core
- **Variable** — Named node with a domain (discrete or continuous) and an
  optional current value.
- **Factor** — Weight function mapping N parent variables → 1 child variable
  output. Entries in the factor graph.
- **Model** — Named container owning variables and factors.
- **Inference** — `model.query(target, evidence)` runs variable elimination
  (exact) for acyclic graphs, or rejection sampling (approximate) for cyclic
  graphs.

### Production
- **StreamingEngine** — online Bayesian updating via particle filter
  (conjugate Dirichlet updates for CPT factors, SIR resampling for latent vars).
  Belief state is persisted across sessions.
- **SchedulerDB** — append-only SQLite store for every decision and outcome.
- **DecisionLogger** — dual-write to SQLite + JSONL; silent on failure.
- **EvidenceRecorder** — shapes observed outcomes into evidence dicts for
  `infer()` calls.
- **ModelRegistry** — tracks registered models and their file paths.
- **BayesianScheduler** — thin composition of all the above. Single entry point.

## Operators

| Operator | Returns | Use case |
|----------|---------|----------|
| `table(cpt)` | CPT lookup | Exact discrete probabilities |
| `sigmoid(w, sigma)` | `exp(-(out - σ(w·x))²/2σ²)` | Soft weighted constraint |
| `binary()` | 1.0 if all equal else 0.0 | Exact match |
| `gaussian(mu, sigma)` | `exp(-(x-μ)²/2σ²)` | Similarity/distance |
| `threshold(bound)` | 1.0 if first ≥ bound else 0.0 | Hard boundary |
| `product()` | ∏ inputs | Combine evidence |
| `sum()` | Σ inputs | Additive evidence |
| `max()` | max(inputs) | Winner-take-all |

## Design constraints

- **Stateless core** — the inference engine is a pure function of
  `(variables, factors, target, evidence)`.
- **Deterministic** — same inputs always produce the same output. Rejection
  sampling uses a fixed seed.
- **Streaming state** — belief tracking in `StreamingEngine` IS stateful
  (particle filter), but the core inference engine is stateless.
- **Target graph size** — ≤ 50 variables, ≤ 10 factors per variable for exact
  inference.

## Development

```bash
pip install -e ".[dev]"
/home/haeba/miniconda3/bin/python3 -m pytest tests/ -q   # 202 tests, all passing
ruff check .
```

> **Python interpreter:** always use `/home/haeba/miniconda3/bin/python3`.
> System Python (`/usr/bin/python3`) may not have `bayesian_engine` on `sys.path`.

## Production policy wrapper (lower-level)

For direct model loading without the full scheduler stack:

```python
from bayesian_engine.policy import BayesianPolicy

policy = BayesianPolicy.load("models/copresence.json")

decision = policy.decide(
    target="action_class",
    evidence={"day_type": "weekday", "hour_block": "morning"},
)
# PolicyDecision(action="SELF_MAINTAIN", reason="below_threshold",
#                confidence=0.10, trace={...})
```

The policy wrapper guarantees: invalid/missing models → safe fallback, low
confidence → suppressed, every decision → structured trace, deterministic
argmax-with-threshold selection.

## API overview

```python
# Core
Model(name)                           # Create a model
model.add_variable(variable)          # Register a variable
model.add_factor(factor)              # Register a factor
model.set_evidence({})                # Set variable values
model.query(target, evidence={})      # Run inference

# I/O
export_model(model, path)             # Serialize to JSON file
import_model(path)                    # Deserialize from JSON file
model_to_dict(model)                  # Serialize to dict
model_from_dict(data)                 # Deserialize from dict

# Policy
BayesianPolicy.load(path)             # Load model with validation
policy.decide(target, evidence)       # Safe production decision (fresh VE)
PolicyDecision(action, reason, confidence, trace)
select_action(distribution, threshold)  # Deterministic selector

# Production stack
BayesianScheduler(model_name, ...)     # Full production entry point
scheduler.infer(target, evidence, ...)  # Decision + belief update + audit
scheduler.record_outcome(...)          # Outcome → future evidence
scheduler.list_decisions(...)          # Query decision history
scheduler.get_beliefs(variable)        # Inspect current belief state
scheduler.reset()                      # Clear streaming beliefs
scheduler.list_models()               # List registered models
```

## Models

| Model | File | Purpose |
|-------|------|---------|
| Copresence (P0) | `models/copresence.json` | When should Reva surface herself? |

The copresence model has five input variables (day_type, hour_block,
user_presence, user_load, agent_need), two latent variables
(contact_window, action_class), and five action outputs (CHECK_IN,
BRIEF_ACK, FULL_RESPONSE, DEFER, SELF_MAINTAIN). 22 acceptance tests in
`tests/test_copresence_model.py`.

## License

MIT
