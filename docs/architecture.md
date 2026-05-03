# Architecture Document
## Bayesian Engine — Core Design Decisions

**Status:** Design Iteration 1 — Decisions Recorded
**Date:** 2026-05-03
**Version:** 0.2-draft

---

## 1. Package Structure

```
bayesian-engine/
├── src/
│   └── bayesian_engine/
│       ├── __init__.py
│       ├── core/
│       │   ├── model.py        # Model class — top-level container
│       │   ├── variable.py     # Variable definition (incl. latent flag)
│       │   ├── factor.py       # Factor definition
│       │   └── graph.py        # Factor graph (adjacency list)
│       ├── inference/
│       │   ├── base.py         # Inference protocol (ABC)
│       │   ├── elimination.py  # Variable elimination (primary)
│       │   ├── sampling.py     # Rejection / importance sampling
│       │   └── streaming.py    # Incremental update
│       ├── operators/
│       │   ├── __init__.py     # Public registry of built-in operators
│       │   ├── sigmoid.py
│       │   ├── table.py
│       │   ├── binary.py
│       │   ├── gaussian.py
│       │   ├── threshold.py
│       │   └── combinators.py  # product, sum, max
│       ├── io/
│       │   ├── serialize.py    # JSON import/export (versioned schema)
│       │   └── schema.py       # Validation schemas
│       └── utils/
│           └── types.py        # Domain types (Continuous, Discrete, Latent, etc.)
├── tests/
│   ├── test_variable.py
│   ├── test_factor.py
│   ├── test_graph.py
│   ├── test_inference.py       # Property-based + snapshot tests
│   ├── test_operators.py
│   ├── test_serialization.py
│   └── fixtures/
│       └── initiative_model.py  # Reusable model construction for tests
├── docs/
│   ├── srs.md
│   └── architecture.md
├── codex-scripts/              # Agent helper scripts (gitignored)
├── my_prompts/                 # Workflow system (git submodule)
├── .CLAUDE/                    # Claude Code configuration
├── CLAUDE.md                   # Project context
└── pyproject.toml
```

## 2. Data Flow

```
User code:
  model = Model("name")
  model.add_variable(Variable(name="x", domain=..., latent=False))
  model.add_variable(Variable(name="y", domain=..., latent=True))
  model.add_factor(Factor(inputs=[...], output="y", weight_function=...))
  model.set_evidence({"x": value})
  result = model.query(target="y", evidence={"x": value})

Internal flow for query():
  1. graph = build_factor_graph(model.variables, model.factors)
  2. subgraph = extract_relevant(graph, target, evidence)
  3. if subgraph is acyclic:
       result = variable_elimination(subgraph, target, evidence)
     else:
       result = rejection_sampling(subgraph, target, evidence, n=10000)
  4. return normalize(result)
```

## 3. Key Design Decision: Inference Engine

**Decision:** Variable elimination (primary), rejection sampling (fallback).

Variable elimination works by:
1. Ordering non-target, non-evidence variables for elimination
2. For each variable in order: multiply all factors involving that variable,
   then sum the variable out
3. The remaining factor is the unnormalized joint over the target

Elimination order matters for performance. For small graphs (~15 variables),
heuristic ordering (min-degree, min-fill) via greedy algorithm is sufficient.

For cyclic graphs (or those where variable elimination produces large
intermediate factors), fall back to rejection sampling with a configurable
iteration count (default: 10,000).

**Dependency:** `numpy` for factor storage (ndarray) and operations (product,
sum over axes, normalization).

## 4. Key Design Decision: Continuous Variable Handling

**Decision:** Discretize at variable registration. Factor-based approach (continuous
variables appear only as inputs, never as targets) is preferred when possible.

Continuous variables registered with `domain=("continuous", lo, hi)` are
discretized into N bins (default N=20, configurable) for inference of
continuous targets. For factor-based workflows, continuous variables act
as inputs whose values are evaluated by operator functions directly.

## 5. Key Design Decision: Factor Graph Data Structure

**Decision:** Adjacency list on variable objects.

Each variable stores:
- `incoming_factors: List[Factor]` — factors where this variable is the output
- `outgoing_factors: List[Factor]` — factors where this variable is an input

This gives O(1) lookups for the factor connectivity needed by variable
elimination (find all factors involving a variable) and streaming updates
(factors that depend on a changed variable).

## 6. Key Design Decision: Serialization

**Decision:** JSON with a documented schema. Versioned model files with a
`"version": 1` field for forward compatibility.

Schema includes:
- `version` (int) — schema version for migration
- `model_name` (str) — model identifier
- `variables` (list) — each with name, domain, latent flag, initial value
- `factors` (list) — each with inputs, output, operator type, parameters

## 7. Multi-Model Architecture

The engine supports multiple independent models. Each model is:

- A separate graph
- Owns its own variable and factor registries
- Queries are scoped to a single model
- **Strict isolation** — factors cannot cross model boundaries

This is relevant for the initiative use case: we may want:

- `initiative_model` — the state → action probability engine
- `tolerance_model` — the Bayesian belief over user tolerance (a model *about*
  the interaction, updated from evidence)

These could be separate models, or merged. The architecture supports both.

## 8. Versioning & Compatibility

- Version 0.x: rapid prototyping, API may change without notice
- Version 1.0: stable API, deprecation warnings for API changes
- Semantic versioning for library consumers
- Serialization schemas versioned independently of library version

## 9. Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1-draft | 2026-05-03 | Initial architecture document |
| 0.2-draft | 2026-05-03 | Decisions recorded: numpy, latent variables, variable elimination, adjacency list, strict model isolation |
