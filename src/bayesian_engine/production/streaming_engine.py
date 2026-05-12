"""Streaming belief-state engine: hybrid Dirichlet + particle filter for online Bayesian inference."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_engine.core.factor import Factor
from bayesian_engine.core.model import Model
from bayesian_engine.core.variable import Variable
from bayesian_engine.io.import_ import import_model
from bayesian_engine.utils.types import DiscreteDomain

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _belief_dir() -> Path:
    """Return the belief-persistence directory, ~/|~/.hermes/bayesian_scheduler/beliefs."""
    home = Path.home()
    return home / ".hermes" / "bayesian_scheduler" / "beliefs"


def _belief_path(model_name: str) -> Path:
    return _belief_dir() / f"{model_name}.json"


# ---------------------------------------------------------------------------
# Factor identification helpers
# ---------------------------------------------------------------------------

def _is_table_factor(factor: Factor) -> bool:
    """Return True when the factor uses the ``table`` CPT operator."""
    wf = factor.weight_function
    return getattr(wf, "__operator_name__", None) == "table"


def _factor_id(factor: Factor) -> str:
    """Stable hash of (factor.inputs tuple, factor.output) — used as dict key."""
    key = (tuple(factor.inputs), factor.output)
    return hashlib.sha256(str(key).encode()).hexdigest()[:16]


def _extract_cpt(factor: Factor) -> dict[tuple, float]:
    """Return the raw CPT dict from a table factor's weight function."""
    return factor.weight_function.__operator_params__["cpt"]


def _cpt_parent_configs(cpt: dict[tuple, float]) -> list[tuple]:
    """All unique parent-value tuples in a CPT (output value stripped)."""
    seen: set[tuple] = set()
    result: list[tuple] = []
    for key in cpt:
        parent_config = tuple(key[:-1])
        if parent_config not in seen:
            seen.add(parent_config)
            result.append(parent_config)
    return result


def _cpt_output_values(cpt: dict[tuple, float]) -> list[str]:
    """All unique output values across CPT entries."""
    return list({key[-1] for key in cpt})


# ---------------------------------------------------------------------------
# Dirichlet state helpers
# ---------------------------------------------------------------------------

def _build_initial_dirichlet_state(
    model: Model,
) -> dict[str, dict[str, dict[str, float]]]:
    """Build initial Dirichlet pseudo-count state for all table factors.

    Structure: {factor_id: {parent_config_str: {value: alpha}}}
    Initial alpha = 1.0 (Laplace / uniform prior).
    """
    state: dict[str, dict[str, dict[str, float]]] = {}
    for factor in model.factors:
        if not _is_table_factor(factor):
            continue
        cpt = _extract_cpt(factor)
        fid = _factor_id(factor)
        state[fid] = {}
        for parent_config in _cpt_parent_configs(cpt):
            config_key = ",".join(parent_config)
            output_values = _cpt_output_values(cpt)
            state[fid][config_key] = {val: 1.0 for val in output_values}
    return state


# ---------------------------------------------------------------------------
# Particle initialization — low-discrepancy (even spread over domain combos)
# ---------------------------------------------------------------------------

def _iter_domain_combinations(
    latent_vars: list[Variable],
) -> list[dict[str, str]]:
    """Yield every combination of latent variable assignments as a dict."""
    if not latent_vars:
        return [{}]
    first, rest = latent_vars[0], latent_vars[1:]
    rest_combos = _iter_domain_combinations(rest)
    results: list[dict[str, str]] = []
    for val in first.domain.values:
        for combo in rest_combos:
            c = dict(combo)
            c[first.name] = val
            results.append(c)
    return results


def _init_particles_low_discrepancy(
    latent_vars: list[Variable],
    n_particles: int,
) -> tuple[list[dict[str, str]], np.ndarray]:
    """Initialize particles with uniform spread over latent domain combinations.

    Args:
        latent_vars: Variables with latent=True.
        n_particles: Total particles to create.

    Returns:
        (particles list, weight ndarray of shape (n_particles,))
    """
    combinations = _iter_domain_combinations(latent_vars)
    n_combos = len(combinations)
    weights = np.full(n_particles, 1.0 / n_particles)

    if n_combos == 0:
        return [{} for _ in range(n_particles)], weights

    particles: list[dict[str, str]] = []
    per_combo = max(1, n_particles // n_combos)
    remainder = n_particles - per_combo * n_combos

    for i, combo in enumerate(combinations):
        count = per_combo + (1 if i < remainder else 0)
        for _ in range(count):
            particles.append(dict(combo))

    particles = particles[:n_particles]
    return particles, weights[:n_particles]


# ---------------------------------------------------------------------------
# StreamingEngine
# ---------------------------------------------------------------------------

class StreamingEngine:
    """Hybrid streaming belief-state engine.

    Maintains belief state across sequential calls using a hybrid approach:
    - **Dirichlet exact updates** for CPT (table) factors — conjugate update per tick
    - **Particle filter** for latent variables — importance-weighted resampling

    Parameters
    ----------
    model_path : str
        Path to the JSON model file.
    n_particles : int, default 200
        Number of particles for the latent-variable filter.
    seed : int, default 42
        RNG seed for resampling (fixed for determinism).
    """

    def __init__(
        self,
        model_path: str,
        n_particles: int = 200,
        seed: int = 42,
        learning_rate: float = 1.0,
    ) -> None:
        self._model_path = model_path
        self._model_name = Path(model_path).stem
        self._n_particles = n_particles
        self._seed = seed
        self._learning_rate = learning_rate
        self._rng = np.random.default_rng(seed)

        # Load model
        self._model = import_model(model_path)

        # Identify table vs. non-table factors
        self._table_factors: list[Factor] = []
        self._non_table_factors: list[Factor] = []
        for factor in self._model.factors:
            if _is_table_factor(factor):
                self._table_factors.append(factor)
            else:
                self._non_table_factors.append(factor)

        # Latent variables
        self._latent_vars = [
            v for v in self._model.variables.values() if v.latent
        ]

        # Try to load persisted state; fall back to fresh init on any error
        self._dirichlet_state: dict[str, dict[str, dict[str, float]]] = {}
        self._particles: list[dict[str, str]] = []
        self._weights: np.ndarray = np.array([])

        self._load_or_init()

    # ------------------------------------------------------------------
    # State initialization / persistence
    # ------------------------------------------------------------------

    def _load_or_init(self) -> None:
        """Load belief state from disk, or initialize fresh if unavailable."""
        path = _belief_path(self._model_name)
        try:
            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                self._dirichlet_state = data["dirichlet_state"]
                self._particles = data["particles"]
                self._weights = np.array(data["weights"])
                logger.info("Loaded belief state from %s", path)
                return  # loaded successfully
        except json.JSONDecodeError:
            logger.warning("Corrupted belief file %s — starting fresh.", path)
        except Exception as exc:
            logger.warning("Failed to load belief file %s (%s) — starting fresh.", path, exc)
        # Fallback: initialize fresh state
        self._init_fresh()

    def _init_fresh(self) -> None:
        """Initialize particles and Dirichlet state from the model priors."""
        self._dirichlet_state = _build_initial_dirichlet_state(self._model)
        self._particles, self._weights = _init_particles_low_discrepancy(
            self._latent_vars, self._n_particles
        )

    def _persist(self) -> None:
        """Write current belief state to disk (atomic write)."""
        path = _belief_path(self._model_name)
        try:
            _belief_dir().mkdir(parents=True, exist_ok=True)
            data = {
                "model_name": self._model_name,
                "saved_at": np.datetime64("now").astype(str),
                "n_particles": self._n_particles,
                "seed": self._seed,
                "dirichlet_state": self._dirichlet_state,
                "particles": self._particles,
                "weights": self._weights.tolist(),
            }
            tmp = path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(data, f)
            tmp.rename(path)
        except OSError as exc:
            logger.warning("Could not persist belief state to %s (%s) — continuing without persistence.", path, exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_and_query(
        self,
        target: str,
        new_evidence: dict[str, Any],
        return_distribution: bool = True,
    ) -> dict[str, Any]:
        """Atomically update beliefs with new evidence and query a variable.

        Args:
            target: Name of the variable to query (latent or observed).
            new_evidence: Mapping of variable names to newly observed values.
            return_distribution: If True, include full distribution in result.

        Returns
        -------
        Dict of the form:
            ``{target: {"value": <map_value>, "distribution": [...]}}``
        """
        # Set evidence on the model
        if new_evidence:
            self._model.set_evidence(new_evidence)

        # ---- Phase 1: Importance weight update --------------------------------
        self._weights = self._compute_importance_weights(new_evidence)

        # ---- Phase 2: Resample ------------------------------------------------
        self._resample()

        # ---- Phase 3: Query ----------------------------------------------------
        if target in self._model.variables:
            var = self._model.variables[target]
        else:
            raise KeyError(f"Variable '{target}' not found in model")

        if var.latent:
            result = self._query_latent(var)
        else:
            result = self._query_observed(var, new_evidence)

        # ---- Phase 4: Dirichlet update (table factors only) --------------------
        self._dirichlet_update(target)

        # ---- Phase 5: Persist -------------------------------------------------
        self._persist()

        return result

    def get_current_beliefs(self, variable: str) -> dict[str, Any]:
        """Return the current belief state for a variable.

        Args:
            variable: Name of the variable.

        Returns
        -------
        For particle-represented (latent) variables::

            {
                "type": "particles",
                "weights": [...],
                "distribution": {<value>: <probability>, ...}
            }

        For CPT (table-factor) variables::

            {
                "type": "cpt",
                "pseudo_counts": {<parent_config>: {<value>: <alpha>, ...}, ...},
                "probabilities": {<parent_config>: {<value>: <prob>, ...}, ...}
            }
        """
        if variable not in self._model.variables:
            raise KeyError(f"Variable '{variable}' not found in model")

        var = self._model.variables[variable]

        if var.latent:
            dist = self._compute_particle_distribution(variable)
            return {
                "type": "particles",
                "weights": self._weights.tolist(),
                "distribution": dist,
            }

        # Non-latent: find the table factor that produces it
        for factor in self._model.factors:
            if factor.output == variable and _is_table_factor(factor):
                fid = _factor_id(factor)
                cpt = _extract_cpt(factor)
                dirich = self._dirichlet_state.get(fid, {})

                # Compute probabilities from pseudo-counts
                probs: dict[str, dict[str, float]] = {}
                for config_str, alpha_dict in dirich.items():
                    total = sum(alpha_dict.values())
                    probs[config_str] = {
                        k: v / total for k, v in alpha_dict.items()
                    }

                return {
                    "type": "cpt",
                    "pseudo_counts": dirich,
                    "probabilities": probs,
                }

        # Fallback: return particle beliefs if variable is evidence
        return {
            "type": "particles",
            "weights": self._weights.tolist(),
            "distribution": self._compute_particle_distribution(variable),
        }

    def reset_beliefs(self) -> None:
        """Reset all beliefs to priors and delete persisted state."""
        self._init_fresh()
        path = _belief_path(self._model_name)
        try:
            if path.exists():
                path.unlink()
                logger.info("Deleted belief file %s", path)
        except OSError as exc:
            logger.warning("Could not delete belief file %s (%s)", path, exc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_importance_weights(
        self, evidence: dict[str, Any]
    ) -> np.ndarray:
        """Compute importance weight for each particle given current evidence.

        For table factors: P(output | parent_values) from CPT
        For non-table factors: P(output | parent_values) from weight function
        The product across all factors gives the importance weight.
        """
        n = len(self._particles)
        weights = np.ones(n, dtype=float)

        for factor in self._model.factors:
            cpt = _extract_cpt(factor) if _is_table_factor(factor) else None
            cpt_params = factor.weight_function.__operator_params__ if _is_table_factor(factor) else {}

            for i, particle in enumerate(self._particles):
                # Collect parent values from particle + evidence
                parent_vals: list[str] = []
                for input_name in factor.inputs:
                    if input_name in particle:
                        parent_vals.append(particle[input_name])
                    elif input_name in evidence:
                        parent_vals.append(evidence[input_name])
                    else:
                        # Parent not in particle or evidence — use model's current value
                        parent_vals.append(self._model.variables[input_name].value)

                if cpt is not None:
                    # Table factor: lookup in CPT
                    for output_val in _cpt_output_values(cpt):
                        key = tuple([*parent_vals, output_val])
                        prob = cpt.get(key, 0.0)
                        if factor.output in particle:
                            if particle[factor.output] == output_val:
                                weights[i] *= prob
                else:
                    # Non-table factor: call the weight function
                    for output_val in self._model.variables[factor.output].domain.values:
                        full_vals = [*parent_vals, output_val]
                        prob = factor.weight_function(*full_vals)
                        if factor.output in particle:
                            if particle[factor.output] == output_val:
                                weights[i] *= prob

        # Clip to avoid numerical issues
        weights = np.clip(weights, 1e-10, 1e10)

        # Normalize
        total = weights.sum()
        if total > 0:
            weights /= total
        else:
            weights = np.full(n, 1.0 / n)

        return weights

    def _resample(self) -> None:
        """Resample particles according to current weights (multinomial)."""
        n = len(self._particles)
        if n == 0:
            return
        indices = self._rng.choice(n, size=n, replace=True, p=self._weights)
        self._particles = [self._particles[i] for i in indices]
        self._weights = np.full(n, 1.0 / n)

    def _compute_particle_distribution(self, var_name: str) -> dict[str, float]:
        """Compute weighted distribution over values of var_name from particles."""
        domain = self._model.variables[var_name].domain.values
        dist: dict[str, float] = {v: 0.0 for v in domain}
        for particle, w in zip(self._particles, self._weights):
            if var_name in particle:
                dist[particle[var_name]] += w
        # Normalize
        total = sum(dist.values())
        if total > 0:
            dist = {k: v / total for k, v in dist.items()}
        return dist

    def _query_latent(self, var: Variable) -> dict[str, Any]:
        """Query a latent variable using particle distribution."""
        dist_dict = self._compute_particle_distribution(var.name)
        map_value = max(dist_dict, key=dist_dict.get)  # type: ignore[arg-type]

        distribution = [
            {"value": v, "probability": p}
            for v, p in sorted(dist_dict.items(), key=lambda x: -x[1])
        ]

        return {var.name: {"value": map_value, "distribution": distribution}}

    def _query_observed(self, var: Variable, evidence: dict[str, Any]) -> dict[str, Any]:
        """Query an observed variable — use the known value."""
        value = evidence.get(var.name, var.value)
        domain = var.domain.values
        if isinstance(var.domain, DiscreteDomain):
            domain_vals = var.domain.values
        else:
            domain_vals = domain

        distribution = [
            {"value": v, "probability": 1.0 if v == value else 0.0}
            for v in domain_vals
        ]
        return {var.name: {"value": value, "distribution": distribution}}

    def _dirichlet_update(self, target: str) -> None:
        """Update Dirichlet pseudo-counts based on the MAP value of the target factor.

        For table factors whose output is `target`, increment the pseudo-count
        corresponding to the MAP value under the current parent configuration.
        """
        if target not in self._model.variables:
            return

        target_var = self._model.variables[target]

        # Find table factors that output this variable
        for factor in self._table_factors:
            if factor.output != target:
                continue

            fid = _factor_id(factor)
            cpt = _extract_cpt(factor)
            dirich = self._dirichlet_state.setdefault(fid, {})

            # Get parent config from current model evidence/state
            parent_vals: list[str] = []
            for input_name in factor.inputs:
                if input_name in self._model.variables:
                    parent_vals.append(self._model.variables[input_name].value or "")

            parent_key = ",".join(parent_vals)

            # Determine MAP value from particles
            if target_var.latent:
                dist = self._compute_particle_distribution(target)
                map_val = max(dist, key=dist.get)  # type: ignore[arg-type]
            else:
                map_val = self._model.variables[target].value
                if map_val is None:
                    continue

            # Increment pseudo-count
            if parent_key not in dirich:
                output_vals = _cpt_output_values(cpt)
                dirich[parent_key] = {v: 1.0 for v in output_vals}

            if map_val in dirich[parent_key]:
                dirich[parent_key][map_val] += self._learning_rate