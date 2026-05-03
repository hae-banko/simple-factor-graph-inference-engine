from bayesian_engine import Factor, Model, Variable
from bayesian_engine.operators import table


def build_model() -> Model:
    model = Model("minimal_autonomy")

    model.add_variable(Variable("day_type", domain=["weekday", "weekend"]))
    model.add_variable(Variable("hour_block", domain=["morning", "afternoon", "evening", "night"]))
    model.add_variable(Variable("should_message", domain=["yes", "no"]))

    model.add_factor(Factor(
        inputs=["day_type", "hour_block"],
        output="should_message",
        weight_function=table({
            ("weekday", "morning",   "yes"): 0.10,
            ("weekday", "morning",   "no"):  0.90,
            ("weekday", "afternoon", "yes"): 0.20,
            ("weekday", "afternoon", "no"):  0.80,
            ("weekday", "evening",   "yes"): 0.35,
            ("weekday", "evening",   "no"):  0.65,
            ("weekday", "night",     "yes"): 0.03,
            ("weekday", "night",     "no"):  0.97,

            ("weekend", "morning",   "yes"): 0.15,
            ("weekend", "morning",   "no"):  0.85,
            ("weekend", "afternoon", "yes"): 0.30,
            ("weekend", "afternoon", "no"):  0.70,
            ("weekend", "evening",   "yes"): 0.40,
            ("weekend", "evening",   "no"):  0.60,
            ("weekend", "night",     "yes"): 0.05,
            ("weekend", "night",     "no"):  0.95,
        }),
    ))

    return model


def most_likely(distribution: list[dict]) -> str:
    return max(distribution, key=lambda item: item["probability"])["value"]


def print_query(model: Model, day_type: str, hour_block: str) -> None:
    result = model.query(
        "should_message",
        evidence={
            "day_type": day_type,
            "hour_block": hour_block,
        },
    )

    distribution = result["should_message"]["distribution"]
    probs = {item["value"]: item["probability"] for item in distribution}

    print(f"day_type={day_type} | hour_block={hour_block}")
    print(f"P(should_message=yes) = {probs['yes']:.2f}")
    print(f"P(should_message=no)  = {probs['no']:.2f}")
    print(f"decision = {most_likely(distribution)}")
    print()


def main() -> None:
    model = build_model()

    for day_type in ["weekday", "weekend"]:
        for hour_block in ["morning", "afternoon", "evening", "night"]:
            print_query(model, day_type, hour_block)


if __name__ == "__main__":
    main()
