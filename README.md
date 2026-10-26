# Product Category Classifier

A small local machine-learning project for Assignment A6, Week 08. It classifies
English product names and descriptions into **Electronics**, **Clothing**,
**Home & Kitchen**, or **Books**, or requests **Manual Review**.

## Install and run in VS Code (Windows PowerShell)

1. Open VS Code, select **File > Open Folder**, and open `D:\Platform assignment 6`.
2. Select **Terminal > New Terminal** and use PowerShell.
3. Run these commands one at a time:

```powershell
Set-Location 'D:\Platform assignment 6'
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe classifier.py
.\.venv\Scripts\python.exe test_classifier.py
```

Python 3.13.14 and scikit-learn 1.9.1 were used for the recorded execution.
Dependency installation requires internet access. Calling the virtual environment's
Python directly avoids PowerShell activation-policy problems.
In VS Code, use **Ctrl+Shift+P > Python: Select Interpreter** and select
`.venv\Scripts\python.exe` if the Python extension is installed.

To run the tests again and save their complete output:

```powershell
.\.venv\Scripts\python.exe test_classifier.py | Tee-Object -FilePath test_results.txt
$LASTEXITCODE
```

Exit code `0` means success; a nonzero exit code means failure. Each case prints
its input, expected behavior, actual result, and PASS/FAIL. The unittest runner
prints the final test count and summary. Rerunning this command replaces the log
with a new genuine run. Tests also support unittest discovery:

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_classifier
```

Run a custom product:

```powershell
.\.venv\Scripts\python.exe classifier.py --product-name 'Casual cotton shirt' --description 'Soft cotton shirt with long sleeves and collar for everyday wear'
```

## How the model works

`classifier.py` contains 48 synthetic labeled training examples: 12 in each category.
Each example has a product name and description, joined into a single text input.
`TfidfVectorizer(ngram_range=(1, 2))` learns weighted word and word-pair features.
`LogisticRegression(C=4.0, random_state=42, max_iter=1000)` learns category weights.
Predictions come from this trained pipeline, without keyword-based category rules.

The default `classify_product(product_name, description)` function lazily trains
and caches one classifier. A `ProductClassifier()` object trains once when created
and can be reused. No model artifact, external API, key, or web server is needed.
The fitted model lives in memory, so a new process trains again. Install dependencies
before importing the module.

The highest estimated probability must reach **0.45** to classify automatically.
This is an illustrative threshold, not one validated on a representative evaluation
dataset. Confidence is the model's estimated probability, not a guarantee of correctness
or a calibrated production reliability score.

## Input and output contract

The function accepts two positional or named arguments: `product_name` and
`description`. Both must be strings containing non-whitespace text. Leading and
trailing whitespace is removed. After trimming, product names may contain at most
200 characters and descriptions at most 5,000 characters. Numbers, nulls, lists,
empty strings, and whitespace-only strings are rejected. Meaningless but non-empty
text is permitted by syntax validation and may be sent to review by the model.

Results are Python dictionaries serializable as JSON with exactly four fields:

| Field | Allowed values |
|---|---|
| `category` | One of the four categories for classification, `Manual Review` for fallback, or `null` for invalid input |
| `confidence` | Finite number in `[0, 1]` when valid prediction probabilities exist; otherwise `null` |
| `status` | `classified`, `manual_review`, or `invalid_input` |
| `reason` | Non-empty explanation of the outcome; not a generated product rationale |

Before accepting predictions, the code verifies all four category labels are present
exactly once, there is one probability row of length four, every probability is finite
and within `[0, 1]`, and the probabilities sum to one within absolute tolerance `1e-6`
(with Python's default relative tolerance). It then applies the confidence threshold.

## Example API integration

These examples describe a possible application integration contract. This project
implements the Python function and CLI; it does not implement an HTTP endpoint.
A future adapter could map `POST /classify-product` to `classify_product` and use
HTTP 400 for invalid input and HTTP 200 for classification or manual review.

Example request:

```http
POST /classify-product
Content-Type: application/json
```

```json
{
  "product_name": "Portable Bluetooth headphones",
  "description": "Wireless audio headphones with microphone and rechargeable battery"
}
```

Actual function response observed in the tests for that input:

```json
{
  "category": "Electronics",
  "confidence": 0.7183149479308034,
  "status": "classified",
  "reason": "Highest estimated category probability meets the 0.45 threshold."
}
```

Python integration:

```python
import json
from classifier import classify_product

result = classify_product(
    product_name="Portable Bluetooth headphones",
    description="Wireless audio headphones with microphone and rechargeable battery",
)
print(json.dumps(result, indent=2, allow_nan=False))
```

## Architecture and fallback

Open `architecture.mmd` with a Mermaid-capable viewer, or paste its contents into
a Mermaid code block in a Markdown editor that supports Mermaid.
The application validates input, requests a model decision, validates its output,
and routes the product to a category or human review. The diagram's queues and
routing represent downstream application actions; this project returns the decision
and does not create queues or change a catalog.

Fallback is deterministic:

| Condition | Result |
|---|---|
| Invalid input | Reject with `invalid_input`; category and confidence are null |
| Training failed or model unavailable | `Manual Review`, null confidence, `Model unavailable.` |
| Prediction raises an exception | `Manual Review`, null confidence, `Model prediction failed.` |
| Invalid labels, shape, or probabilities | `Manual Review`, null confidence, `Invalid model prediction.` |
| Maximum valid probability below 0.45 | `Manual Review`, retain maximum probability, explain low confidence |
| Valid prediction at or above 0.45 | Return its category and estimated probability |

Input validation happens before accessing the model. Internal exception details
are not returned to callers. The local model has no remote service to time out;
the unavailable-model test represents the equivalent dependency outage. A real
remote adapter would need timeout handling that returns the same review response.

## Observed tests and engineering reflection

The recorded run passed **18 unittest methods**, including additional cases within
the probability and length tests. `test_results.txt` contains the actual execution
evidence. Real model tests and simulated fault tests are labeled separately.

| Real-model input | Expected behavior | Actual category | Confidence | Result |
|---|---|---|---:|---|
| Portable Bluetooth headphones | Electronics | Electronics | 0.718315 | PASS |
| Casual cotton shirt | Clothing | Clothing | 0.797699 | PASS |
| Large cooking saucepan | Home & Kitchen | Home & Kitchen | 0.701209 | PASS |
| Space adventure novel | Books | Books | 0.748030 | PASS |
| Smart recipe tablet | Valid category or safe review | Manual Review | 0.376202 | PASS |
| Zqxv / Plmokn ytrewq | Manual Review | Manual Review | 0.261188 | PASS |

**What worked:** All four straightforward held-out product descriptions reached the
expected category. Invalid inputs were rejected, and simulated training/inference
failures and malformed outputs reached Manual Review.

**Unexpected decisions:** The hybrid recipe tablet was referred for review rather
than assigned to Electronics or Books. Its maximum estimated probability was only
0.376202. No wrong category was observed among the four straightforward examples;
this small test set cannot establish general accuracy. Simulated failures demonstrate
handling behavior rather than naturally occurring model faults.

**Production risks:** Only 48 synthetic examples cover a narrow English vocabulary.
Novel, multilingual, misleading, or mixed-category products may be misclassified,
including confidently. The threshold is uncalibrated. Broad exception handling safely
returns review but would need private diagnostic logging in production. Review queue
ownership, capacity, privacy controls, and monitoring would belong to the integrating app.

**Next improvements:** Collect representative labeled products, keep independent
training/validation/test splits, measure per-category precision/recall and review
rate, calibrate probabilities, and tune the threshold on validation data. Add human
feedback, model versioning, and drift monitoring before deployment.

## Files and report

| File | Purpose |
|---|---|
| `classifier.py` | Training examples, learned pipeline, validation, fallback, CLI |
| `test_classifier.py` | Automated tests and expected/actual evidence output |
| `requirements.txt` | Verified scikit-learn version; pip installs its dependencies |
| `test_results.txt` | Saved real test run |
| `architecture.mmd` | Mermaid integration diagram |
| `report_outline.md` | Assignment-template report draft with evidence references |

The report outline includes identity placeholders. Complete them and export a PDF
with the rendered architecture and relevant test evidence for submission. No PDF
is generated by this project.

References used: the provided `assignment_8.md` and `assignment8_template.md`,
scikit-learn's installed TF-IDF/Pipeline/LogisticRegression APIs, and Python's
standard-library unittest, json, and unittest.mock APIs. AI assistance was used to
write source code, synthetic training examples, tests, and documentation, and to run
the tests. Recorded outputs came from Python execution, not generated predictions.
