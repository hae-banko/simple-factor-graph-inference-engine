# Software Requirements Specification
## Bayesian Engine — General-Purpose Inference Module

**Status:** Design Iteration 1 — Decisions Recorded
**Date:** 2026-05-03
**Version:** 0.2-draft

---

## 1. Purpose

A minimal, general-purpose Bayesian inference engine that computes conditional
probabilities over observable and latent state variables. Designed for
reusability across multiple use cases — starting with autonomous agent
initiative scheduling, extensible to any system requiring probabilistic
reasoning over discrete or continuous variables.

## 2. Design Philosophy

The engine follows three constraints:

1. **Interpretable by inspection** — a human reading the factor graph or
   conditional probability tables should understand why any probability is what
   it is.

2. **Composable** — variables, factors, and queries can be added or removed
   without modifying existing code.

3. **Future-proof inference** — start with exact inference (variable
   elimination) for small-to-medium graphs; provide sampling fallback for
   cyclic or larger graphs.

### 2.1 Design Decisions (v0.1 → v0.2)

| Decision | Answer |
|----------|--------|
| **Dependencies** | `numpy` accepted for numerical operations |
| **Hidden variables** | Allowed — engines support latent variables |
| **Inference algorithm** | Variable elimination (primary), rejection sampling (fallback for cyclic graphs) |
| **Continuous variables** | Discretize at variable registration; factor-based approach preferred |

## 3. Core Concepts

### 3.1 Variable

A named variable with a domain (set of possible values):

- **Discrete** — enumerated values (e.g., `user_presence ∈ {present, absent, idle}`)
- **Ordinal** — ordered but finite (e.g., `mood_score ∈ [-1, 1]`)
- **Continuous** — real interval (e.g., `hour_of_day ∈ [0, 24)`)
- **Latent** — inferred variable not directly observed (e.g., `tolerance_greet ∈ [0, 1]`)

Each variable has a current value and a type that determines how its probability
contribution is computed.

### 3.2 Factor

A function that maps a subset of variables to a real-valued weight (not
necessarily normalized). Factors represent the "strength" of a particular
configuration.

Each factor has:

- A **parent variable set** (inputs)
- A **child variable** (output)
- A **weight function** `f: (parent_values) → ℝ` representing how strongly the
  parent configuration supports the child taking its current value

### 3.3 Factor Graph (Bi-partite)

Variables and factors form a bipartite graph:

- **Variable nodes** hold state
- **Factor nodes** connect input variables to output variables
- Each factor connects exactly 1 output variable to N ≥ 1 input variables

The engine does not enforce acyclicity — that is the user's responsibility.

### 3.4 Query

A query asks for the conditional probability of a target variable given evidence
on other variables:

    P(target | evidence = {v1: val1, v2: val2, ...})

### 3.5 Model

A named collection of variables and factors. A model is an independent graph —
you can have multiple models simultaneously (e.g., one for initiative
scheduling, one for tolerance beliefs).

## 4. Functional Requirements

### 4.1 Variable Registry

- Register a variable with name, type, domain, and model
- Set/update current value of a variable
- Get current value
- List all registered variables in a model

### 4.2 Factor Registry

- Register a factor (input variables, output variable, weight function, model)
- Remove a factor
- List factors for a given variable (incoming and outgoing)

### 4.3 Graph Management

- Build factor graph from registry
- Topological validation (cycle detection, advisory not blocking)
- Subgraph extraction (variables reachable from a query)

### 4.4 Inference

Given a query `P(target | evidence)`:

1. Fix evidence variables to their observed values
2. Build the factor graph for the relevant subgraph
3. Apply **variable elimination** — eliminate non-target, non-evidence
   variables by summing them out of the factor product
4. Normalize the resulting factor over the target domain
5. Return probability distribution

The inference algorithm must handle:

- **Variable elimination** — primary, exact inference for acyclic graphs
- **Rejection sampling** — fallback for graphs with cycles
- Graceful degradation: exact → approximate → error with diagnostic

**Dependency:** `numpy` for efficient array operations during factor
manipulation (factor product, sum-out, normalization).

### 4.5 Operators (Weight Functions)

The engine must ship with at least:

| Operator | Formula | Use case |
|----------|---------|----------|
| **sigmoid** | `σ(w·x)` | Soft constraint with weights |
| **table** | CPT lookup | Exact discrete probability tables |
| **binary** | `[x == y]` | Exact match / mismatch |
| **gaussian** | `exp(-(x-μ)²/2σ²)` | Similarity / distance |
| **threshold** | `[x ≥ bound]` | Hard boundary transitions |
| **product** | `∏ w_i` | Factor combination |
| **sum** | `∑ w_i` | Additive evidence |
| **max** | `max(w_i)` | Winner-take-all |

Users must be able to register custom callable operators.

### 4.6 Serialization

- Export entire model (variables + factors + graph) to JSON
- Import model from JSON
- Snapshot and restore model state

### 4.7 Observation Streaming

- Update evidence incrementally without recomputing the full graph
- Recompute only affected branches on variable change

## 5. Non-Functional Requirements

### 5.1 Simplicity

- Core inference logic: hundreds of lines, not thousands.
- One external dependency: `numpy` (stdlib otherwise).
- Developer reads the entire public API in five minutes.

### 5.2 Determinism

- Given the same variable values and factor weights, two queries must produce
  exactly the same result. No randomness except in explicit sampling operators.

### 5.3 Stateless Core

- The inference engine is a pure function of (model, evidence). No internal
  mutable state. State is held in the model object only.

### 5.4 Performance Boundaries

- Exact inference: graphs with ≤ 50 variables, ≤ 10 factors per variable
- Sampling: user-configurable iteration count, bounded wall-clock time
- Streaming updates: sub-millisecond for affected subgraph of ≤ 10 factors

## 6. API Sketch (Preliminary)

```python
import numpy as np
from bayesian_engine import Model, Variable, Factor, sigmoid

# Create a model
m = Model("reva_initiative")

# Register variables
m.add_variable(Variable(name="user_presence", domain=["present", "absent", "idle"], value="absent"))
m.add_variable(Variable(name="hour_of_day", domain=("continuous", 0, 24), value=13.0))
m.add_variable(Variable(name="greet_probability", domain=("continuous", 0, 1)))
m.add_variable(Variable(name="tolerance_greet", domain=("continuous", 0, 1), latent=True))

# Register factors
m.add_factor(Factor(
    inputs=["user_presence", "hour_of_day"],
    output="greet_probability",
    weight_function=sigmoid(w=np.array([2.0, -0.3])),
))
m.add_factor(Factor(
    inputs=["accepted_ratio", "response_time"],
    output="tolerance_greet",
    weight_function=sigmoid(w=np.array([1.5, -0.5])),
))

# Query
result = m.query(target="greet_probability", evidence={"user_presence": "present"})
# → {"greet_probability": {"value": 0.72, "distribution": ...}}
```

## 7. Architecture Decisions to Make (Open Questions)

These have been resolved in v0.2:

| # | Question | Decision |
|---|----------|----------|
| 1 | Variable elimination vs. product-of-factors? | **Variable elimination** (primary), rejection sampling (fallback) |
| 2 | Continuous variable support? | **Discretize** at registration; factor-based as primary approach |
| 3 | Factor graph data structure? | **Adjacency list** on variable objects |
| 4 | Cycle handling? | **Advisory detection** (warn, don't block) |
| 5 | Custom operator protocol? | **Protocol/ABC** for type safety |
| 6 | Serialization format? | **JSON** with versioned schema |
| 7 | Incremental vs. full recomputation? | **Incremental** (affected subgraph only) |
| 8 | Multi-model isolation? | **Strict isolation** — factors cannot cross model boundaries |
| 9 | Testing strategy? | **Property-based** for correctness; snapshot for regression |
| 10 | Minimum Python version? | **3.11+** |
| 11 | Dependencies? | **numpy** accepted |
| 12 | Hidden/latent variables? | **Allowed** — no-hidden-state constraint removed |
| 13 | Inference engine scope? | **Variable elimination** — Claude's job to implement |

## 8. First Use Case: Autonomous Agent Initiative

The first model this engine will drive:

**State variables:**
- `last_interaction` (minutes since last contact)
- `user_presence` (present / absent / idle)
- `hour_of_day` (0-24)
- `day_of_week` (0-6)
- `open_loops` (count of pending commitments)
- `outstanding_outputs` (count of undelivered results)
- `last_action_taken` (categorical: which action was last executed)
- `system_state` (normal / dnd / error)
- `mood_score` (derived from sentiment analysis, [-1, 1])

**Seed action targets** (each a binary/ordinal variable):
- `should_greet` → greet/re-engagement
- `should_check_in` → check on open loops
- `should_deliver` → deliver stored output
- `should_diary_nudge` → evening diary reminder
- `should_ambient` → ambient observation
- `should_silent` → do nothing (default, P >= 0.3)

**Bayesian Tolerance Layer** (separate model):
- `tolerance_greet` — latent belief state over Master's tolerance for greeting
- Evidence: `accepted_ratio`, `response_time`, `sentiment_of_reply`
- τ ≈ 48h decay to prior

## 9. Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1-draft | 2026-05-03 | Initial SRS draft — core concepts, requirements, open questions |
| 0.2-draft | 2026-05-03 | Design decisions recorded: numpy, latent variables, variable elimination |
