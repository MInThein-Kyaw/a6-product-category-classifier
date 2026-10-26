"""Additional artifact and wrapper checks, separate from the ten-case evaluation."""

import sys
sys.dont_write_bytecode = True

import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock

from evaluate import check_integrity, load_cases, percentile95
from threshold_wrapper import ThresholdClassifier, load_a6


HERE = Path(__file__).resolve().parent


def main():
    captured = io.StringIO()
    with redirect_stdout(captured):
        manifest = json.loads((HERE / "eval_cases.lock.json").read_text())
        check_integrity(manifest)
        original_files = set(manifest["a6_files"])
        assert {p.name for p in HERE.parent.iterdir() if p.is_file()} == original_files
        print("PASS: original A6 file hashes and root file list are unchanged.")
        cases = load_cases()
        assert len(cases) == 10
        print("PASS: frozen dataset has exactly 3 normal, 3 edge, 2 fallback, 2 difficult cases.")
        for name in ("baseline", "improved"):
            result = json.loads((HERE / f"{name}_results.json").read_text())
            samples = [value for case in result["cases"] for value in case["latency_samples_ms"]]
            assert len(samples) == 300
            assert all(len(case["latency_samples_ms"]) == 30 for case in result["cases"])
            assert percentile95(samples) == result["p95_latency_ms"]
            assert sum(case["result"] == "PASS" for case in result["cases"]) == result["correct_cases"]
            assert result["dataset_sha256"] == manifest["dataset_sha256"]
            print(f"PASS: {name} counts, raw samples, accuracy numerator, dataset hash, and p95 agree.")

        # The following are explicitly simulated distributions. They are unit
        # checks of threshold boundaries, not additional real evaluation cases.
        labels = ["Electronics", "Clothing", "Home & Kitchen", "Books"]
        fake = Mock(classes_=labels)
        baseline_module = load_a6("a7_verification_a6")
        wrapper = ThresholdClassifier(fake)
        for maximum, old_status, new_status in (
            (0.39, "manual_review", "manual_review"),
            (0.40, "manual_review", "classified"),
            (0.42, "manual_review", "classified"),
            (0.45, "classified", "classified"),
        ):
            fake.predict_proba.return_value = [[maximum] + [(1 - maximum) / 3] * 3]
            original = baseline_module.ProductClassifier.classify_product(wrapper, "Device", "Description")
            after = wrapper.classify_product("Device", "Description")
            assert original["status"] == old_status and after["status"] == new_status
            print(f"PASS: SIMULATED confidence {maximum:.2f}: baseline {old_status}, wrapper {new_status}.")
        fake.predict_proba.return_value = [[float("nan"), 0.2, 0.3, 0.5]]
        assert wrapper.classify_product("Device", "Description")["reason"] == "Invalid model prediction."
        fake.predict_proba.side_effect = RuntimeError("Simulated fault")
        assert wrapper.classify_product("Device", "Description")["reason"] == "Model prediction failed."
        assert wrapper.classify_product(" ", "Description")["status"] == "invalid_input"
        wrapper.model = None
        assert wrapper.classify_product("Device", "Description")["reason"] == "Model unavailable."
        print("PASS: SIMULATED invalid probabilities, inference exception, invalid input, and unavailable model preserve fallback.")
        report = (HERE / "A7_Report.md").read_text(encoding="utf-8")
        assert sum(line.startswith("## ") for line in report.splitlines()) == 9
        assert len(json.loads((HERE / "knowledge-base.json").read_text())) >= 5
        print("PASS: report has nine sections and knowledge base has at least five records.")
        check_integrity(manifest)
        print("Verification complete. No evaluation results or expected labels were changed.")
    text = captured.getvalue()
    (HERE / "verification.txt").write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
