"""Reproduce Task 3 evidence; optionally render the architecture PNG.

This script never modifies classifier.py or the existing result files.
"""

import argparse
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from classifier import ProductClassifier


ROOT = Path(__file__).resolve().parent


def capture_evidence():
    classifier = ProductClassifier()
    if classifier.model is None:
        raise RuntimeError("A fitted model is required to demonstrate inference failure.")
    name = "Portable Bluetooth headphones"
    description = "Wireless audio headphones with microphone and rechargeable battery"
    expected_failure = {
        "category": "Manual Review", "confidence": None,
        "status": "manual_review", "reason": "Model prediction failed.",
    }
    output = io.StringIO()
    with redirect_stdout(output):
        print("TASK 3 - ACTUAL LOCAL EXECUTION EVIDENCE")
        print("HTTP requests are not executed; the proposed API is documentation only.")
        print()
        print("1. REAL MODEL: normal prediction")
        print("Input:", json.dumps({"product_name": name, "description": description}))
        normal = classifier.classify_product(name, description)
        print(json.dumps(normal, indent=2, allow_nan=False))
        normal_ok = normal["category"] == "Electronics" and normal["status"] == "classified"
        print("Expected: Electronics / classified")
        print("Result:", "PASS" if normal_ok else "FAIL")
        print()

        print("2. REAL MODEL: ambiguous input / low-confidence fallback")
        ambiguous = {
            "product_name": "Smart recipe tablet",
            "description": "Touchscreen tablet with digital cooking recipes and illustrated cookbook chapters",
        }
        print("Input:", json.dumps(ambiguous))
        low = classifier.classify_product(**ambiguous)
        print(json.dumps(low, indent=2, allow_nan=False))
        low_ok = (low["category"] == "Manual Review" and low["status"] == "manual_review"
                  and low["confidence"] is not None and low["confidence"] < 0.45)
        print("Expected: Manual Review / manual_review with confidence below 0.45")
        print("Result:", "PASS" if low_ok else "FAIL")
        print()

        print("3. SIMULATED MODEL FAILURE: temporary in-memory exception injection")
        print("Input:", json.dumps({"product_name": name, "description": description}))
        print("Injection: predict_proba raises RuntimeError('Simulated model failure').")
        # Patch only this fitted object's method. The context restores it afterwards.
        with patch.object(classifier.model, "predict_proba",
                          side_effect=RuntimeError("Simulated model failure")) as mocked:
            failure = classifier.classify_product(name, description)
            invoked = mocked.call_count == 1
        print(json.dumps(failure, indent=2, allow_nan=False))
        failure_ok = failure == expected_failure and invoked
        print("Expected:", json.dumps(expected_failure))
        print("Injected model method called:", invoked)
        print("Result:", "PASS" if failure_ok else "FAIL")
        print()

        print("4. REAL MODEL: prediction after removing the temporary mock")
        restored = classifier.classify_product(name, description)
        print(json.dumps(restored, indent=2, allow_nan=False))
        restored_ok = restored == normal and normal_ok
        print("Expected: original normal prediction restored")
        print("Result:", "PASS" if restored_ok else "FAIL")
        checks = [normal_ok, low_ok, failure_ok, restored_ok]
        print()
        print(f"Summary: {sum(checks)}/{len(checks)} PASS")

    report = output.getvalue()
    (ROOT / "task3_failure_evidence.txt").write_text(report, encoding="utf-8")
    print(report, end="")
    return all(checks)


def render_diagram():
    # matplotlib is only needed to regenerate the PNG, not to run the classifier.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(14, 12))
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")
    ax.set_xlim(-0.2, 14.4)
    ax.set_ylim(-0.3, 13.5)
    ax.axis("off")

    def box(x, y, width, height, text, color="#e0f2fe", dashed=False):
        ax.add_patch(FancyBboxPatch(
            (x - width / 2, y - height / 2), width, height,
            boxstyle="round,pad=0.08,rounding_size=0.12", linewidth=1.4,
            edgecolor="#475569", facecolor=color,
            linestyle="--" if dashed else "-", zorder=2))
        ax.text(x, y, text, ha="center", va="center", fontsize=10.5,
                color="#0f172a", linespacing=1.5, zorder=3)

    def arrow(points, label=None, label_at=None):
        for i in range(len(points) - 1):
            ax.add_patch(FancyArrowPatch(
                points[i], points[i + 1],
                arrowstyle="-|>" if i == len(points) - 2 else "-",
                mutation_scale=14, linewidth=1.3, color="#475569", zorder=1))
        if label:
            ax.text(*label_at, label, ha="center", va="center", fontsize=9,
                    color="#334155", bbox=dict(facecolor="#f8fafc", edgecolor="none", pad=2))

    ax.text(7.1, 13.15, "Product Category Classifier", ha="center", fontsize=20,
            weight="bold", color="#0f172a")
    ax.text(7.1, 12.65, "Task 3: platform integration and deterministic fallback",
            ha="center", fontsize=12, color="#475569")
    box(6, 11.7, 4.8, 0.8, "User submits product name + description", "#f1f5f9")
    box(6, 10.2, 4.8, 1.0,
        "Application adapter (PROPOSED HTTP API)\nPOST /classify-product\nCalls the existing Python function",
        "#ede9fe", True)
    box(6, 8.7, 4.3, 0.9, "Existing input validation\nNon-empty strings; trimmed length limits")
    box(1.4, 8.7, 2.55, 1.0, "Reject input\ninvalid_input JSON", "#fee2e2")
    box(6, 6.9, 4.3, 1.0, "Existing decision capability\nCheck model availability\nTF-IDF + Logistic Regression")
    box(6, 5.1, 4.3, 0.9, "Existing output checks\nAllowed labels + valid probabilities")
    box(6, 3.3, 4.3, 0.9, "Existing confidence check\nMaximum probability >= 0.45?")
    box(6, 1.4, 4.8, 1.1, "classified JSON -> category routing (PROPOSED)\nElectronics | Clothing | Home & Kitchen | Books",
        "#dcfce7", True)
    box(12, 5.1, 3.65, 1.6, "Deterministic fallback\ncategory: Manual Review\nstatus: manual_review\nConfidence: valid probability\nor null when unavailable", "#ffedd5")
    box(12, 2.3, 3.65, 1.0, "Application queues listing\nfor human review (PROPOSED)", "#ede9fe", True)

    arrow([(6, 11.22), (6, 10.78)])
    arrow([(6, 9.62), (6, 9.23)])
    arrow([(3.77, 8.7), (2.76, 8.7)], "Invalid", (3.28, 9.05))
    arrow([(6, 8.17), (6, 7.48)], "Valid", (6.4, 7.83))
    arrow([(6, 6.32), (6, 5.63)])
    arrow([(8.23, 6.9), (9.4, 6.9), (9.4, 5.7), (10.09, 5.7)],
          "Unavailable / model error", (10.9, 7.3))
    arrow([(8.23, 5.1), (10.09, 5.1)], "Invalid", (9.1, 5.43))
    arrow([(6, 4.57), (6, 3.83)], "Valid", (6.4, 4.2))
    arrow([(8.23, 3.3), (9.4, 3.3), (9.4, 4.5), (10.09, 4.5)],
          "Below 0.45", (9.3, 2.92))
    arrow([(6, 2.77), (6, 2.03)], "At least 0.45", (7.0, 2.4))
    arrow([(12, 4.22), (12, 2.88)], "Return review JSON", (12, 3.55))
    ax.text(7.1, 0.12,
            "Solid boxes: existing classifier behavior / input   |   Dashed boxes: proposed platform components\n"
            "Local inference is implemented. HTTP endpoint, catalog routing, and review queue are design proposals.",
            ha="center", va="center", fontsize=9, color="#475569", linespacing=1.6)
    destination = ROOT / "task3_architecture.png"
    fig.savefig(destination, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Diagram saved: {destination.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagram", action="store_true", help="Also regenerate the PNG (requires matplotlib)")
    args = parser.parse_args()
    passed = capture_evidence()
    if args.diagram:
        render_diagram()
    raise SystemExit(0 if passed else 1)
