"""Freeze labels first, then measure the same ten cases at 0.45 and 0.40."""

import sys
sys.dont_write_bytecode = True

import argparse
import hashlib
import io
import json
import math
import platform
import statistics
from collections import Counter
from contextlib import nullcontext, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns
from unittest.mock import patch

from retrieval import build_decision_request
from threshold_wrapper import ThresholdClassifier, load_a6


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LOCK = HERE / "eval_cases.lock.json"
REPEATS = 30
WARMUPS = 3


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_cases():
    cases = json.loads((HERE / "eval-cases.json").read_text(encoding="utf-8"))
    assert len(cases) == 10, "Exactly ten cases are required."
    assert [case["id"] for case in cases] == [f"T{i:02d}" for i in range(1, 11)]
    assert Counter(case["case_type"] for case in cases) == {
        "normal": 3, "edge": 3, "fallback": 2, "difficult": 2}
    for case in cases:
        assert all(key in case for key in ("id", "case_type", "product_name", "description",
                                          "expected_category", "expected_fallback_behavior", "reason"))
        assert case["expected_fallback_behavior"] in ("none", "manual_review", "reject_invalid_input")
    return cases


def freeze():
    cases = load_cases()  # No fitting or prediction occurs in freeze mode.
    if LOCK.exists():
        raise RuntimeError("Dataset already frozen; do not overwrite expectations after observing results.")
    manifest = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_sha256": digest(HERE / "eval-cases.json"),
                "case_ids": [case["id"] for case in cases],
                "case_type_counts": dict(Counter(case["case_type"] for case in cases)),
                "a6_files": {path.name: digest(path) for path in ROOT.iterdir() if path.is_file()}}
    save_json(LOCK.name, manifest)
    message = ("EXPECTED RESULTS FROZEN BEFORE A7 EVALUATION\n"
               f"Dataset SHA-256: {manifest['dataset_sha256']}\n"
               f"UTC: {manifest['frozen_at_utc']}\n"
               "10 cases: 3 normal, 3 edge, 2 fallback, 2 difficult.\n"
               "No model was fitted or evaluated by this freeze command.\n")
    (HERE / "freeze_evidence.txt").write_text(message, encoding="utf-8")
    print(message, end="")


def check_integrity(manifest):
    assert digest(HERE / "eval-cases.json") == manifest["dataset_sha256"], "Frozen cases changed."
    for filename, expected in manifest["a6_files"].items():
        assert digest(ROOT / filename) == expected, f"A6 file changed: {filename}"


def timed_call(classifier, case):
    # Inject only T08's specified fault, identically for both versions. Entering
    # and restoring the mock is outside the measured classifier call.
    fault = (patch.object(classifier.model, "predict_proba", side_effect=RuntimeError("A7 simulated failure"))
             if case.get("simulate_model_failure") else nullcontext())
    with fault:
        start = perf_counter_ns()
        result = classifier.classify_product(case["product_name"], case["description"])
        elapsed_ms = (perf_counter_ns() - start) / 1_000_000
    return result, elapsed_ms


def expected_status(case):
    return {"none": "classified", "manual_review": "manual_review",
            "reject_invalid_input": "invalid_input"}[case["expected_fallback_behavior"]]


def percentile95(samples):
    # Nearest-rank p95: ceil(0.95 * N)-th sorted observation, starting at one.
    return sorted(samples)[math.ceil(0.95 * len(samples)) - 1]


def format_results(result):
    lines = [f"{result['version'].upper()} - threshold {result['threshold']:.2f}",
             "ID | Expected category / status | Actual category / status | Confidence | Result | Median ms"]
    for row in result["cases"]:
        lines.append(f"{row['id']} | {row['expected_category']} / {row['expected_status']} | "
                     f"{row['actual']['category']} / {row['actual']['status']} | "
                     f"{row['actual']['confidence']} | {row['result']} | {row['latency_ms']:.6f}")
    lines += [f"Accuracy: {result['correct_cases']}/10 = {result['accuracy_percent']:.1f}%",
              f"P95 latency: {result['p95_latency_ms']:.6f} ms ({result['timed_calls']} calls)",
              "API cost per task: N/A (local model; no paid API calls)"]
    return "\n".join(lines) + "\n"


def evaluate():
    cases = load_cases()
    manifest = json.loads(LOCK.read_text(encoding="utf-8"))
    check_integrity(manifest)
    import sklearn
    a6 = load_a6("a7_baseline_a6")
    assert a6.CONFIDENCE_THRESHOLD == 0.45
    baseline = a6.ProductClassifier()
    assert baseline.model is not None, "Original model training failed."
    improved = ThresholdClassifier(baseline.model, 0.40)
    assert improved.model is baseline.model

    # Verify the wrapper reproduces A6 exactly when given the original threshold.
    same_threshold = ThresholdClassifier(baseline.model, 0.45)
    for case in cases:
        assert timed_call(baseline, case)[0] == timed_call(same_threshold, case)[0]
    print("Preflight: wrapper at 0.45 matches original on all 10 cases; same fitted model reused.")
    print("Dataset SHA-256:", manifest["dataset_sha256"])
    print("Expectations were frozen before this run; no labels are changed by evaluation.")

    packet = build_decision_request(cases[0]["product_name"], cases[0]["description"])
    packet["actual_decision"] = baseline.classify_product(**packet["input"])
    packet["context_usage"] = "Retrieved application/reviewer metadata only; not passed into A6 model features."
    save_json("retrieval_demo.json", packet)
    print("\nACTUAL RETRIEVAL DEMONSTRATION")
    print(json.dumps(packet, indent=2))

    versions = {"baseline": baseline, "improved": improved}
    rows = {key: [] for key in versions}
    pooled = {key: [] for key in versions}
    for case in cases:
        for classifier in versions.values():
            for _ in range(WARMUPS):
                timed_call(classifier, case)
        observations = {key: [] for key in versions}
        predictions = {key: [] for key in versions}
        for repetition in range(REPEATS):
            # Alternate execution order to reduce consistent first/second bias.
            order = list(versions) if repetition % 2 == 0 else list(reversed(versions))
            for key in order:
                actual, elapsed = timed_call(versions[key], case)
                observations[key].append(elapsed)
                predictions[key].append(actual)
        for key in versions:
            actual = predictions[key][0]
            assert all(value == actual for value in predictions[key]), "Non-deterministic outputs observed."
            correct = actual["category"] == case["expected_category"] and actual["status"] == expected_status(case)
            rows[key].append({"id": case["id"], "case_type": case["case_type"],
                              "expected_category": case["expected_category"], "expected_status": expected_status(case),
                              "actual": actual, "result": "PASS" if correct else "FAIL",
                              "latency_ms": statistics.median(observations[key]),
                              "latency_samples_ms": observations[key]})
            pooled[key].extend(observations[key])

    reports = {}
    for key in versions:
        correct = sum(row["result"] == "PASS" for row in rows[key])
        reports[key] = {"version": key, "threshold": 0.45 if key == "baseline" else 0.40,
                        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
                        "dataset_sha256": manifest["dataset_sha256"], "python_version": platform.python_version(),
                        "sklearn_version": sklearn.__version__, "correct_cases": correct,
                        "accuracy_percent": correct / 10 * 100, "p95_latency_ms": percentile95(pooled[key]),
                        "timed_calls": len(pooled[key]), "repeats_per_case": REPEATS,
                        "warmups_per_case": WARMUPS, "api_cost_per_task": "N/A",
                        "timing_method": "perf_counter_ns around classifier call only; median per case; pooled nearest-rank p95; alternating version order; excludes training, imports, retrieval, logging, mock setup/restoration",
                        "cases": rows[key]}
        save_json(f"{key}_results.json", reports[key])
        text = format_results(reports[key])
        (HERE / f"{key}_results.txt").write_text(text, encoding="utf-8")
        print("\n" + text)

    print("CASE-BY-CASE COMPARISON")
    for before, after in zip(rows["baseline"], rows["improved"]):
        change = ("improved" if before["result"] == "FAIL" and after["result"] == "PASS" else
                  "worse" if before["result"] == "PASS" and after["result"] == "FAIL" else "unchanged")
        print(f"{before['id']}: {before['result']} -> {after['result']} ({change})")
    check_integrity(manifest)
    save_json("integrity_check.json", {"a6_files_unchanged": True, "dataset_unchanged": True,
                                      "files_checked": manifest["a6_files"],
                                      "dataset_sha256": manifest["dataset_sha256"]})
    print("\nINTEGRITY PASS: all original root A6 files and frozen evaluation cases unchanged.")
    print("API cost: N/A; all inference is local. Latencies are measured, not estimated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true", help="Lock cases BEFORE the first evaluation")
    args = parser.parse_args()
    if args.freeze:
        freeze()
    else:
        captured = io.StringIO()
        try:
            with redirect_stdout(captured):
                evaluate()
        finally:
            terminal_text = captured.getvalue()
            (HERE / "terminal_output.txt").write_text(terminal_text, encoding="utf-8")
            print(terminal_text, end="")
