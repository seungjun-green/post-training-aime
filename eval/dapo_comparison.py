"""Compare completed, matched greedy AMC/MATH evaluation results."""


def comparison_rows(base, rl):
    for key in ["config", "execution", "protocol_digest", "dataset_suite", "selected_ids",
                "packages", "hardware", "runner_digest", "chat_template_digest"]:
        if base[key] != rl[key]:
            raise ValueError(f"Cannot compare mismatched {key}")
    if base["mode"] != "full" or rl["mode"] != "full":
        raise ValueError("Comparison table requires both full evaluations")
    if set(base["metrics"]) != {"amc23", "math_500"} or set(rl["metrics"]) != set(base["metrics"]):
        raise ValueError("Expected exactly AMC 2023 and MATH-500")
    rows = []
    for name, label, expected in [("amc23", "AMC 2023", 40), ("math_500", "MATH-500", 500)]:
        b, r = base["metrics"][name], rl["metrics"][name]
        for entry in [b, r]:
            if (entry["problems"] != expected or entry["responses"] != expected
                    or entry["samples_per_problem"] != 1):
                raise ValueError(f"Incomplete single-answer evaluation: {name}")
        rows.append({"Benchmark": label, "Problems": expected,
                     "Base accuracy (%)": 100 * b["avg@1"],
                     "DAPO-100 accuracy (%)": 100 * r["avg@1"],
                     "Difference (pp)": 100 * (r["avg@1"] - b["avg@1"]),
                     "Base mean tokens": b["response_length_tokens"]["all"],
                     "DAPO-100 mean tokens": r["response_length_tokens"]["all"]})
    return rows
