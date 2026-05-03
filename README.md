# Bayesian Engine

A general-purpose Bayesian inference engine that computes `P(target | evidence)`
over factor graphs. One dependency (`numpy`). Pure Python 3.11+.

## Install

```bash
pip install -e .
```

## Quick start

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

# Save / load
from bayesian_engine.io import export_model, import_model
export_model(model, "weather.json")
model2 = import_model("weather.json")
```

## Concepts

- **Variable** — Named node with a domain (discrete or continuous) and an
  optional current value.
- **Factor** — Weight function mapping N parent variables → 1 child variable
  output. Entries in the factor graph.
- **Model** — Named container owning variables and factors. Multi-model
  isolation: factors cannot cross model boundaries.
- **Inference** — `model.query(target, evidence)` runs variable elimination
  (exact) for acyclic graphs, or rejection sampling (approximate) for cyclic
  graphs. Returns a normalized probability distribution over the target
  variable's domain.

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

All operators are factory functions that return callable closures. Register
your own by providing any `Callable[..., float]`.

## Continuous variables

```python
model.add_variable(Variable("temperature", domain=("continuous", 0, 100)))
model.add_factor(Factor(
    inputs=["temperature"],
    output="alert",
    weight_function=threshold(75),
))
```

Continuous variables are discretized internally (default 20 bins, configurable).

## Design constraints

- **Stateless core** — the inference engine is a pure function of
  `(variables, factors, target, evidence)`.
- **Deterministic** — same inputs always produce the same output. Rejection
  sampling uses a fixed seed.
- **Target graph size** — ≤ 50 variables, ≤ 10 factors per variable for exact
  inference.

## Development

```bash
pip install -e ".[dev]"
pytest           # 62 tests
ruff check .     # lint
```

## Production policy wrapper

For safe autonomous agent use, load models through the policy layer which
fails closed to silence on any error:

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

The policy wrapper guarantees:
- Invalid/missing models → fallback to safe default
- Low-confidence actions → suppressed
- Every decision → structured JSON-serializable trace
- Deterministic argmax-with-threshold selection (no random sampling)

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
model_to_json(model)                  # Serialize to JSON string
model_from_json(text)                 # Deserialize from JSON string

# Policy
BayesianPolicy.load(path)             # Load model with validation
policy.decide(target, evidence)       # Safe production decision
PolicyDecision(action, reason, confidence, trace)
select_action(distribution, config)   # Deterministic selector
```

## License

MIT
