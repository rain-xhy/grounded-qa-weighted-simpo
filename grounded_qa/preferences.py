"""Construct a preference pair from two independently annotated answers."""
import math

from .claims import parse_claims

WEIGHTS = {"Q": 0.4, "E": 0.2, "C": 0.3, "B": 0.1}


def unit(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Expected a finite score between zero and one")
    return float(value)


def contribution(traces, count):
    if len(traces) != count or count == 0:
        raise ValueError("Each statement needs one ordered full/keep/drop trace")
    differences = []
    for trace in traces:
        full, keep, drop = (trace[key] for key in ("full", "keep", "drop"))
        if not all(type(v) in (int, float) and math.isfinite(v) for v in (full, keep, drop)):
            raise ValueError("Incomplete statement log probabilities")
        differences.append((full - drop) + (keep - full))
    mean = sum(differences) / len(differences)
    return 1 / (1 + math.exp(-max(-60, min(60, mean))))


def assess(candidate, sentences, condition):
    claims, errors = parse_claims(candidate["response"], sentences)
    if errors:
        raise ValueError(f"Invalid statement or citation structure: {errors}")
    annotations = candidate["statement_support"]
    if len(annotations) != len(claims) or any(type(v) is not bool for v in annotations):
        raise ValueError("Provide a verified support decision for every statement")
    if type(candidate["critical_error"]) is not bool:
        raise ValueError("Missing factual error check")
    values = {key: unit(candidate[key]) for key in ("Q", "E")}
    values["C"] = contribution(candidate["traces"], len(claims))
    if condition:
        values["B"] = unit(candidate["B"])
    elif candidate.get("B") is not None:
        raise ValueError("Natural questions have no behavior label")
    reward = sum(values[k] * WEIGHTS[k] for k in values) / sum(WEIGHTS[k] for k in values)
    positive = (values["Q"] >= .85 and values["E"] >= .90 and not candidate["critical_error"]
                and all(annotations) and all(claim.sentence_ids for claim in claims))
    return reward, positive, unit(candidate["confidence"])


def make_pair(record):
    """The input row groups two candidates under one frozen prompt and document."""
    first, second = record["candidates"]
    candidates = sorted(((assess(c, record["sentence_count"], record.get("expected_decision")), c)
                         for c in (first, second)), key=lambda item: item[0][0], reverse=True)
    (better, chosen), (worse, rejected) = candidates
    gap = better[0] - worse[0]
    if (not better[1] or min(better[2], worse[2]) < .8 or gap < .05
            or chosen["response"] == rejected["response"]):
        return None
    return {"id": record["id"], "prompt": record["prompt"], "chosen": chosen["response"],
            "rejected": rejected["response"], "reward_gap": gap, "sample_weight": 1 + .3 * gap}
