"""Display genuine predictions and save the same readable output to a text file."""

import io
from contextlib import redirect_stdout
from pathlib import Path
from textwrap import fill

from classifier import ProductClassifier


# Expected categories are human-specified before prediction, not copied from results.
# These input pairs are distinct from the classifier's training examples.
PRODUCTS = [
    (
        "Portable Bluetooth headphones",
        "Wireless audio headphones with microphone and rechargeable battery",
        "Electronics",
    ),
    (
        "Casual cotton shirt",
        "Soft cotton shirt with long sleeves and collar for everyday wear",
        "Clothing",
    ),
    (
        "Large cooking saucepan",
        "Kitchen cooking pot with lid for preparing food",
        "Home & Kitchen",
    ),
    (
        "Space adventure novel",
        "Paperback fiction book with imaginative stories and adventure chapters",
        "Books",
    ),
    (
        "Compact wireless speaker",
        "Portable Bluetooth audio speaker with rechargeable battery for wireless music",
        "Electronics",
    ),
]

# Real inference on an ambiguous product demonstrates low-confidence fallback.
# No mocking, forced predictions, or changes to the model are used in this demo.
FALLBACK_PRODUCT = (
    "Smart recipe tablet",
    "Touchscreen tablet with digital cooking recipes and illustrated cookbook chapters",
    "Manual Review",
)


def display_prediction(classifier, number, product, fallback=False):
    name, description, expected = product
    result = classifier.classify_product(name, description)
    expected_status = "manual_review" if fallback else "classified"
    passed = result["category"] == expected and result["status"] == expected_status
    confidence = result["confidence"]
    score = "N/A" if confidence is None else f"{confidence:.6f} ({confidence:.2%})"

    print(f"Example {number}" + (" - Real low-confidence fallback" if fallback else ""))
    print("-" * 80)
    print(f"Product name      : {name}")
    print(fill(description, width=88, initial_indent="Description       : ",
               subsequent_indent=" " * 20))
    print(f"Expected category : {expected}")
    print(f"Actual category   : {result['category']}")
    print(f"Confidence score  : {score}")
    print(f"Status            : {result['status']}")
    print(f"Reason            : {result['reason']}")
    print(f"Result            : {'PASS' if passed else 'FAIL'}")
    print()
    return passed


def main():
    # The existing class trains once at construction and is reused for every input.
    classifier = ProductClassifier()
    output = io.StringIO()
    with redirect_stdout(output):
        print("PRODUCT CATEGORY CLASSIFIER - ACTUAL PREDICTIONS")
        print("Existing TF-IDF + Logistic Regression model; no simulated predictions.")
        print("PASS requires both the expected category and expected status.")
        print("Confidence is an estimated model probability, not guaranteed accuracy.")
        print()
        outcomes = [display_prediction(classifier, i, product)
                    for i, product in enumerate(PRODUCTS, start=1)]
        outcomes.append(display_prediction(classifier, len(PRODUCTS) + 1,
                                           FALLBACK_PRODUCT, fallback=True))
        print("=" * 80)
        print(f"Summary: {sum(outcomes)}/{len(outcomes)} PASS; "
              f"{len(outcomes) - sum(outcomes)} FAIL")

    # Save exactly the text produced by this run, even if an expectation fails.
    report = output.getvalue()
    destination = Path(__file__).resolve().with_name("prediction_results.txt")
    destination.write_text(report, encoding="utf-8")
    print(report, end="")
    return 0 if all(outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
