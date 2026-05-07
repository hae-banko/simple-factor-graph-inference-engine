"""Build script for the P0 copresence policy model.

Generates models/copresence.json — a hand-authored CPT model that decides
whether and how the agent should surface itself given user context and
agent need.

Usage:
    python examples/build_copresence_model.py
"""

from itertools import product
from pathlib import Path

from bayesian_engine import Factor, Model, Variable
from bayesian_engine.io import export_model
from bayesian_engine.operators import table

DAY_TYPES = ["weekday", "weekend"]

HOUR_BLOCKS = [
    "night",
    "morning",
    "workday",
    "afternoon",
    "evening",
    "late",
]

USER_PRESENCE = [
    "active_recently",
    "idle",
    "unknown",
]

USER_LOAD = [
    "free",
    "normal",
    "busy",
]

AGENT_NEED = [
    "none",
    "gentle_checkin",
    "actionable_update",
    "urgent",
]

CONTACT_WINDOW = [
    "closed",
    "soft",
    "open",
]

ACTION_CLASS = [
    "SILENCE",
    "SELF_MAINTAIN",
    "AMBIENT_PING",
    "CHECK_IN",
    "DIRECT_MESSAGE",
]


def normalized(scores):
    total = sum(max(value, 0.0) for value in scores.values())
    return {key: max(value, 0.0) / total for key, value in scores.items()}


def contact_distribution(day_type, hour_block, user_presence, user_load):
    score = 0.0

    score += {
        "night": -3.0,
        "morning": 1.0,
        "workday": 0.0,
        "afternoon": 0.8,
        "evening": 1.2,
        "late": -1.5,
    }[hour_block]

    score += {
        "active_recently": 1.5,
        "idle": -0.5,
        "unknown": -0.2,
    }[user_presence]

    score += {
        "free": 1.0,
        "normal": 0.0,
        "busy": -2.0,
    }[user_load]

    if day_type == "weekend" and hour_block in {"morning", "afternoon", "evening"}:
        score += 0.5

    return normalized({
        "closed": 2.0 - score,
        "soft": 2.0,
        "open": 1.0 + score,
    })


def action_distribution(contact_window, agent_need):
    distributions = {
        ("closed", "none"): {
            "SILENCE": 0.75,
            "SELF_MAINTAIN": 0.25,
            "AMBIENT_PING": 0.0,
            "CHECK_IN": 0.0,
            "DIRECT_MESSAGE": 0.0,
        },
        ("closed", "gentle_checkin"): {
            "SILENCE": 0.55,
            "SELF_MAINTAIN": 0.35,
            "AMBIENT_PING": 0.10,
            "CHECK_IN": 0.0,
            "DIRECT_MESSAGE": 0.0,
        },
        ("closed", "actionable_update"): {
            "SILENCE": 0.20,
            "SELF_MAINTAIN": 0.45,
            "AMBIENT_PING": 0.25,
            "CHECK_IN": 0.10,
            "DIRECT_MESSAGE": 0.0,
        },
        ("closed", "urgent"): {
            "SILENCE": 0.0,
            "SELF_MAINTAIN": 0.05,
            "AMBIENT_PING": 0.10,
            "CHECK_IN": 0.25,
            "DIRECT_MESSAGE": 0.60,
        },
        ("soft", "none"): {
            "SILENCE": 0.35,
            "SELF_MAINTAIN": 0.55,
            "AMBIENT_PING": 0.10,
            "CHECK_IN": 0.0,
            "DIRECT_MESSAGE": 0.0,
        },
        ("soft", "gentle_checkin"): {
            "SILENCE": 0.10,
            "SELF_MAINTAIN": 0.25,
            "AMBIENT_PING": 0.45,
            "CHECK_IN": 0.20,
            "DIRECT_MESSAGE": 0.0,
        },
        ("soft", "actionable_update"): {
            "SILENCE": 0.0,
            "SELF_MAINTAIN": 0.15,
            "AMBIENT_PING": 0.35,
            "CHECK_IN": 0.35,
            "DIRECT_MESSAGE": 0.15,
        },
        ("soft", "urgent"): {
            "SILENCE": 0.0,
            "SELF_MAINTAIN": 0.0,
            "AMBIENT_PING": 0.05,
            "CHECK_IN": 0.20,
            "DIRECT_MESSAGE": 0.75,
        },
        ("open", "none"): {
            "SILENCE": 0.15,
            "SELF_MAINTAIN": 0.65,
            "AMBIENT_PING": 0.20,
            "CHECK_IN": 0.0,
            "DIRECT_MESSAGE": 0.0,
        },
        ("open", "gentle_checkin"): {
            "SILENCE": 0.0,
            "SELF_MAINTAIN": 0.10,
            "AMBIENT_PING": 0.35,
            "CHECK_IN": 0.50,
            "DIRECT_MESSAGE": 0.05,
        },
        ("open", "actionable_update"): {
            "SILENCE": 0.0,
            "SELF_MAINTAIN": 0.05,
            "AMBIENT_PING": 0.20,
            "CHECK_IN": 0.35,
            "DIRECT_MESSAGE": 0.40,
        },
        ("open", "urgent"): {
            "SILENCE": 0.0,
            "SELF_MAINTAIN": 0.0,
            "AMBIENT_PING": 0.0,
            "CHECK_IN": 0.10,
            "DIRECT_MESSAGE": 0.90,
        },
    }

    return distributions[(contact_window, agent_need)]


def build_contact_window_cpt():
    cpt = {}

    for day_type, hour_block, user_presence, user_load in product(
        DAY_TYPES,
        HOUR_BLOCKS,
        USER_PRESENCE,
        USER_LOAD,
    ):
        distribution = contact_distribution(
            day_type=day_type,
            hour_block=hour_block,
            user_presence=user_presence,
            user_load=user_load,
        )

        for output, probability in distribution.items():
            cpt[(day_type, hour_block, user_presence, user_load, output)] = probability

    return cpt


def build_action_cpt():
    cpt = {}

    for contact_window, agent_need in product(CONTACT_WINDOW, AGENT_NEED):
        distribution = action_distribution(
            contact_window=contact_window,
            agent_need=agent_need,
        )

        for output, probability in distribution.items():
            cpt[(contact_window, agent_need, output)] = probability

    return cpt


def build_model():
    model = Model(
        "copresence",
        metadata={
            "model_version": "0.1.0",
            "description": (
                "P0 copresence policy model for deciding whether and how "
                "an agent should surface itself."
            ),
        },
    )

    for name, domain, latent in [
        ("day_type", DAY_TYPES, False),
        ("hour_block", HOUR_BLOCKS, False),
        ("user_presence", USER_PRESENCE, False),
        ("user_load", USER_LOAD, False),
        ("agent_need", AGENT_NEED, False),
        ("contact_window", CONTACT_WINDOW, True),
        ("action_class", ACTION_CLASS, True),
    ]:
        model.add_variable(Variable(name=name, domain=domain, latent=latent))

    model.add_factor(Factor(
        inputs=["day_type", "hour_block", "user_presence", "user_load"],
        output="contact_window",
        weight_function=table(build_contact_window_cpt()),
    ))

    model.add_factor(Factor(
        inputs=["contact_window", "agent_need"],
        output="action_class",
        weight_function=table(build_action_cpt()),
    ))

    return model


def main():
    output_path = Path("models/copresence.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    export_model(build_model(), output_path)

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
