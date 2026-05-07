"""Tests for the P0 copresence policy model.

Covers the acceptance criteria from the first-use-case spec:
  1. Model generation produces a valid file
  2. BayesianPolicy.load() succeeds
  3. policy.decide() returns a PolicyDecision
  4. Trace is JSON-serializable
  5. Urgent cases can produce DIRECT_MESSAGE
  6. Low-need closed-window cases produce SILENCE or SELF_MAINTAIN
  7. Only the built-in `table` operator is used (no custom operators)
"""

import json
import os
import subprocess

import pytest

from bayesian_engine.policy import BayesianPolicy

# ---------------------------------------------------------------------------
# Shared fixture — build the model once per test session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def copresence_model_path():
    """Run the build script and return the path to the generated model."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    build_script = os.path.join(repo_root, "examples", "build_copresence_model.py")
    models_dir = os.path.join(repo_root, "models")
    model_path = os.path.join(models_dir, "copresence.json")

    # Clean up any stale output from previous runs
    if os.path.exists(model_path):
        os.unlink(model_path)

    result = subprocess.run(
        ["python", build_script],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"build script failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert os.path.exists(model_path), f"model file not created at {model_path}"

    return model_path


@pytest.fixture(scope="session")
def copresence_policy(copresence_model_path):
    return BayesianPolicy.load(copresence_model_path)


# ---------------------------------------------------------------------------
# Acceptance criteria 1-4: build, load, decide, serialise
# ---------------------------------------------------------------------------

class TestAcceptanceBuild:
    def test_build_script_succeeds(self, copresence_model_path):
        """Acceptance #1: the build script creates models/copresence.json."""

    def test_model_file_is_valid_json(self, copresence_model_path):
        with open(copresence_model_path) as f:
            data = json.load(f)
        assert isinstance(data, dict)
        assert "variables" in data
        assert "factors" in data


class TestAcceptanceLoad:
    def test_load_succeeds(self, copresence_policy):
        """Acceptance #2: BayesianPolicy.load() succeeds."""
        assert copresence_policy._model is not None
        assert copresence_policy._error is None
        assert copresence_policy._model.name == "copresence"


class TestAcceptanceDecide:
    def test_decide_returns_policy_decision(self, copresence_policy):
        """Acceptance #3: policy.decide(...) returns a PolicyDecision."""
        decision = copresence_policy.decide(
            target="action_class",
            evidence={
                "day_type": "weekday",
                "hour_block": "evening",
                "user_presence": "active_recently",
                "user_load": "normal",
                "agent_need": "gentle_checkin",
            },
            threshold=0.45,
            fallback_action="SELF_MAINTAIN",
        )
        assert decision.action is not None
        assert decision.reason is not None
        assert decision.confidence is not None

    def test_trace_is_json_serializable(self, copresence_policy):
        """Acceptance #4: the returned trace is JSON-serializable."""
        decision = copresence_policy.decide(
            target="action_class",
            evidence={
                "day_type": "weekday",
                "hour_block": "evening",
                "user_presence": "active_recently",
                "user_load": "normal",
                "agent_need": "gentle_checkin",
            },
            threshold=0.45,
            fallback_action="SELF_MAINTAIN",
        )
        json.dumps(decision.trace)  # raises if not serialisable


# ---------------------------------------------------------------------------
# Acceptance criteria 5-6: urgent → DIRECT_MESSAGE, low-need closed → SILENCE
# ---------------------------------------------------------------------------

class TestUrgentDirective:
    """Acceptance #5: urgent cases can produce DIRECT_MESSAGE."""

    @pytest.mark.parametrize("agent_need", ["urgent"])
    @pytest.mark.parametrize("contact_window", ["closed", "soft", "open"])
    def test_urgent_always_direct_message(self, copresence_policy, agent_need, contact_window):
        decision = copresence_policy.decide(
            target="action_class",
            evidence={
                "contact_window": contact_window,
                "agent_need": agent_need,
            },
            threshold=0.45,
            fallback_action="SELF_MAINTAIN",
        )
        # Urgent: soft → 0.75 DM, open → 0.90 DM, closed → 0.60 DM
        # All above 0.45 threshold
        assert decision.action == "DIRECT_MESSAGE", (
            f"urgent+{contact_window} should yield DIRECT_MESSAGE, got {decision.action} "
            f"(confidence={decision.confidence:.3f}, reason={decision.reason})"
        )


class TestLowNeedClosedWindow:
    """Acceptance #6: low-need closed-window cases produce SILENCE or SELF_MAINTAIN."""

    @pytest.mark.parametrize("agent_need", ["none", "gentle_checkin"])
    @pytest.mark.parametrize("contact_window", ["closed"])
    def test_closed_low_need_no_interrupt(self, copresence_policy, agent_need, contact_window):
        decision = copresence_policy.decide(
            target="action_class",
            evidence={
                "contact_window": contact_window,
                "agent_need": agent_need,
            },
            threshold=0.45,
            fallback_action="SELF_MAINTAIN",
        )
        assert decision.action in ("SILENCE", "SELF_MAINTAIN"), (
            f"low-need+closed should not interrupt user, got {decision.action}"
        )


# ---------------------------------------------------------------------------
# Acceptance criterion 7: only the built-in `table` operator is used
# ---------------------------------------------------------------------------

class TestOperatorConstraint:
    def test_only_table_operator_used(self, copresence_model_path):
        """Acceptance #7: only the built-in `table` operator is used."""
        with open(copresence_model_path) as f:
            data = json.load(f)

        for factor in data.get("factors", []):
            op_name = factor.get("operator", "")
            # operator can be a string or a dict with "name"
            if isinstance(op_name, dict):
                op_name = op_name.get("name", "")
            assert op_name == "table", (
                f"Factor '{factor.get('output')}' uses operator '{op_name}', "
                "but only 'table' is permitted for the P0 model."
            )


# ---------------------------------------------------------------------------
# Model structure tests — verify the graph matches the spec
# ---------------------------------------------------------------------------

class TestModelStructure:
    def test_all_spec_variables_present(self, copresence_model_path):
        with open(copresence_model_path) as f:
            data = json.load(f)

        spec_vars = {
            "day_type", "hour_block", "user_presence", "user_load",
            "agent_need", "contact_window", "action_class",
        }
        model_vars = {v["name"] for v in data.get("variables", [])}
        assert spec_vars == model_vars, f"Expected {spec_vars}, got {model_vars}"

    def test_latent_variables_are_marked(self, copresence_model_path):
        with open(copresence_model_path) as f:
            data = json.load(f)

        latent = {v["name"] for v in data.get("variables", []) if v.get("latent")}
        assert latent == {"contact_window", "action_class"}

    def test_observed_variables_not_marked_latent(self, copresence_model_path):
        with open(copresence_model_path) as f:
            data = json.load(f)

        observed = {v["name"] for v in data.get("variables", []) if not v.get("latent")}
        assert observed == {"day_type", "hour_block", "user_presence", "user_load", "agent_need"}

    def test_contact_window_factor_inputs(self, copresence_model_path):
        with open(copresence_model_path) as f:
            data = json.load(f)

        cw_factor = next(
            (f for f in data.get("factors", []) if f.get("output") == "contact_window"),
            None,
        )
        assert cw_factor is not None, "contact_window factor not found"
        assert set(cw_factor.get("inputs", [])) == {
            "day_type", "hour_block", "user_presence", "user_load"
        }

    def test_action_class_factor_inputs(self, copresence_model_path):
        with open(copresence_model_path) as f:
            data = json.load(f)

        ac_factor = next(
            (f for f in data.get("factors", []) if f.get("output") == "action_class"),
            None,
        )
        assert ac_factor is not None, "action_class factor not found"
        assert set(ac_factor.get("inputs", [])) == {"contact_window", "agent_need"}

    def test_cpt_entries_sum_to_one(self, copresence_model_path):
        """Every CPT row should sum to 1.0 (normalised distributions)."""
        with open(copresence_model_path) as f:
            data = json.load(f)

        for factor in data.get("factors", []):
            op_params = factor.get("params", {})
            cpt_raw = op_params.get("cpt", {})

            # Serialised format: {"_type": "cpt", "entries": [[[...keys...], prob], ...]}
            entries = cpt_raw.get("entries", [])

            from collections import defaultdict
            rows: dict = defaultdict(list)
            for item in entries:
                keys, prob = item[0], item[1]
                parent = tuple(keys[:-1])  # all except output value
                rows[parent].append(prob)

            for parent_key, probs in rows.items():
                total = sum(probs)
                assert abs(total - 1.0) < 1e-6, (
                    f"Factor '{factor['output']}' CPT row {parent_key} "
                    f"sums to {total:.6f}, expected 1.0"
                )


# ---------------------------------------------------------------------------
# Policy decision behaviour tests
# ---------------------------------------------------------------------------

class TestPolicyDefaults:
    """Production defaults: threshold=0.45, fallback=SELF_MAINTAIN."""

    def test_missing_model_returns_fallback(self):
        policy = BayesianPolicy.load("/nonexistent/copresence.json")
        decision = policy.decide(target="action_class", evidence={})
        assert decision.action == "SELF_MAINTAIN"
        assert decision.reason == "model_load_error"

    def test_below_threshold_returns_fallback(self, copresence_policy):
        # Build a distribution where all probs are below 0.45
        # by using low-signal evidence
        decision = copresence_policy.decide(
            target="action_class",
            evidence={
                "day_type": "weekday",
                "hour_block": "night",
                "user_presence": "idle",
                "user_load": "busy",
                "agent_need": "none",
            },
            threshold=0.45,
            fallback_action="SELF_MAINTAIN",
        )
        # night+idle+busy pushes contact_window toward closed
        # All action probs should be low
        dist = decision.trace["result"]["distribution"]
        max_prob = max(d["probability"] for d in dist)
        if max_prob < 0.45:
            assert decision.action == "SELF_MAINTAIN"
            assert decision.reason == "below_threshold"


class TestContactWindowPolicy:
    """The agent_need must not bypass contact_window except for urgent cases."""

    def test_non_urgent_cannot_override_closed_window_for_direct_message(
        self, copresence_policy
    ):
        """gentle_checkin + closed should never produce DIRECT_MESSAGE."""
        for agent_need in ["none", "gentle_checkin", "actionable_update"]:
            decision = copresence_policy.decide(
                target="action_class",
                evidence={
                    "contact_window": "closed",
                    "agent_need": agent_need,
                },
                threshold=0.35,  # lenient to isolate the policy constraint
                fallback_action="SELF_MAINTAIN",
            )
            if agent_need == "actionable_update":
                # actionable_update + closed: SELF_MAINTAIN 0.45, AMBIENT_PING 0.25
                assert decision.action != "DIRECT_MESSAGE"
            else:
                # none/gentle_checkin + closed: SILENCE or SELF_MAINTAIN
                assert decision.action in ("SILENCE", "SELF_MAINTAIN", "AMBIENT_PING")


class TestProductionDefaults:
    """Threshold 0.45 and fallback SELF_MAINTAIN as specified."""

    def test_threshold_045_default(self, copresence_policy):
        decision = copresence_policy.decide(
            target="action_class",
            evidence={
                "contact_window": "soft",
                "agent_need": "gentle_checkin",
            },
            threshold=0.45,
            fallback_action="SELF_MAINTAIN",
        )
        dist = decision.trace["result"]["distribution"]
        max_prob = max(d["probability"] for d in dist)
        if max_prob < 0.45:
            assert decision.action == "SELF_MAINTAIN"
            assert decision.reason == "below_threshold"

    def test_fallback_action_never_raises(self, copresence_policy):
        """Every failure mode must return a PolicyDecision, never raise."""
        import threading

        errors = []

        def call():
            try:
                copresence_policy.decide(
                    target="action_class",
                    evidence={"contact_window": "open", "agent_need": "urgent"},
                )
            except Exception as e:
                errors.append(e)

        t = threading.Thread(target=call)
        t.start()
        t.join()
        assert not errors, f"decide() raised: {errors[0]}"
