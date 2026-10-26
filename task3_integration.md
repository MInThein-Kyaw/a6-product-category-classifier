# Task 3 - Platform Integration and Fallback

## Architecture

![Product Category Classifier integration architecture](task3_architecture.png)

The PNG shows a product submission moving from the user through an application
adapter to the existing Python decision capability. The classifier validates input,
uses TF-IDF and Logistic Regression, checks prediction validity, and applies its
0.45 confidence threshold. It returns structured JSON for category routing, manual
review, or input rejection.

**Implemented:** Local Python classification, input validation, prediction validation,
confidence checks, and deterministic fallback in `classifier.py`.

**Proposed platform design:** The HTTP adapter, catalog updates, and human review
queue in the diagram are integration proposals. No HTTP endpoint, deployed service,
catalog update, or actual review queue is implemented by this project. Dashed boxes
mark proposed components. The application would route a listing only when status is
`classified`, queue it when status is `manual_review`, and ask the user to correct it
when status is `invalid_input`.

## Example API request and response - PROPOSED HTTP DESIGN

Proposed endpoint: **`POST /classify-product`**, with `Content-Type: application/json`.
The future adapter would parse a JSON object and pass its two required fields to
`classify_product(product_name, description)`. Initialize/reuse the classifier once
per application process rather than creating a new model for each request.

Request body:

```json
{
  "product_name": "Portable Bluetooth headphones",
  "description": "Wireless audio headphones with microphone and rechargeable battery"
}
```

Proposed HTTP response: **200 OK**. The following JSON body is the **actual local
function output** captured in `task3_failure_evidence.txt`; no HTTP call was made:

```json
{
  "category": "Electronics",
  "confidence": 0.7183149479308034,
  "status": "classified",
  "reason": "Highest estimated category probability meets the 0.45 threshold."
}
```

Proposed HTTP mapping: `classified` and `manual_review` both return HTTP 200 because
they are completed decision outcomes. `invalid_input` returns HTTP 400. A future
adapter must reject malformed JSON, non-object bodies, and missing required fields
with HTTP 400 before invoking the classifier. It should use the same four-field
invalid-input response shape. These HTTP checks and mappings are proposals; the
current Python function receives arguments directly.

## Input validation rules - existing behavior

| Check | Rule and outcome |
|---|---|
| Required arguments | Supply both `product_name` and `description` to the Python function; the proposed HTTP adapter would check missing fields |
| Types | Both values must be strings; reject null, numbers, lists, and other types |
| Empty content | Reject empty strings and strings containing only whitespace |
| Whitespace | Trim leading and trailing whitespace for length checks and model input |
| Product name length | At most 200 characters after trimming |
| Description length | At most 5,000 characters after trimming |
| Rejection | Return `invalid_input` with null category and confidence and an explanatory reason; skip model inference |

For example, an empty product name returns the following existing function response
(illustrative validation example; the new evidence run focuses on inference/fallback):

```json
{
  "category": null,
  "confidence": null,
  "status": "invalid_input",
  "reason": "product_name must be a non-empty string."
}
```

Non-empty text is not necessarily a recognizable product. Unrecognized or mixed
vocabulary is evaluated by the model and may need review.

## Output validation rules

Before accepting model probabilities, the existing classifier checks:

- Exactly four labels, matching Electronics, Clothing, Home & Kitchen, and Books.
- Exactly one prediction row with four probabilities.
- All probabilities are finite numbers in `[0, 1]`; NaN and infinity are invalid.
- Probabilities sum to one using `math.isclose(sum, 1.0, abs_tol=1e-6)` with its
  default relative tolerance.
- The highest probability must be **at least 0.45** for automatic classification.

The classifier constructs a result containing exactly these four fields:

| Field | Result contract |
|---|---|
| `category` | One of the four labels when classified; `Manual Review` on fallback; null for invalid input |
| `confidence` | Finite number between 0 and 1 when valid probabilities exist; otherwise null |
| `status` | `classified`, `manual_review`, or `invalid_input` |
| `reason` | Non-empty explanation of the outcome, without internal exception details |

These fields are constructed by the current implementation rather than parsed from
generated model text. A future HTTP adapter should verify the result contract before
serialization. Confidence is an estimated model probability; it is not a guarantee
of correctness. The 0.45 threshold has not been calibrated on production data.

## Deterministic fallback

| Trigger | Existing result | Proposed downstream action |
|---|---|---|
| Highest valid confidence below 0.45 | `Manual Review`, `manual_review`, retain valid confidence, reason explains low confidence | Queue for human review |
| Prediction raises an exception | `Manual Review`, `manual_review`, null confidence, `Model prediction failed.` | Queue for human review |
| Model unavailable, including caught training failure | `Manual Review`, `manual_review`, null confidence, `Model unavailable.` | Queue for human review |
| Invalid category labels or probabilities | `Manual Review`, `manual_review`, null confidence, `Invalid model prediction.` | Queue for human review |
| Invalid user input | `invalid_input`, null category/confidence | Ask user to correct the submission |

The fallback does not guess a category or retry with keyword rules. The current
implementation runs locally, so it has no remote service timeout. If an external
service is introduced, timeout/unavailability handling must be added in the adapter
and mapped to Manual Review; this is future design, not demonstrated functionality.

### Actual low-confidence example

Input:

```json
{
  "product_name": "Smart recipe tablet",
  "description": "Touchscreen tablet with digital cooking recipes and illustrated cookbook chapters"
}
```

Actual local output from the new Task 3 run:

```json
{
  "category": "Manual Review",
  "confidence": 0.37620155496730606,
  "status": "manual_review",
  "reason": "Confidence below 0.45; human review required."
}
```

## Evidence of simulated model failure

The separate script [`task3_demo.py`](task3_demo.py) trains the existing classifier,
checks a normal prediction, runs the ambiguous example above, and temporarily patches
the fitted model's `predict_proba` method to raise a `RuntimeError`. It calls the
unchanged classifier through its normal interface, so the returned fallback is
produced by the real exception handler. Only the failure is injected; the fallback
response is not fabricated. The patch is restored when the context exits.

Core demonstration code (executed by the script):

```python
from unittest.mock import patch

with patch.object(classifier.model, "predict_proba",
                  side_effect=RuntimeError("Simulated model failure")) as mocked:
    failure = classifier.classify_product(name, description)
    invoked = mocked.call_count == 1
```

For the same headphones input used in the API example, the recorded output was:

```json
{
  "category": "Manual Review",
  "confidence": null,
  "status": "manual_review",
  "reason": "Model prediction failed."
}
```

Full actual output is saved separately in
[`task3_failure_evidence.txt`](task3_failure_evidence.txt).

| Execution check | Actual result | Outcome |
|---|---|---|
| Real normal prediction | Electronics, confidence 0.7183149479308034 | PASS |
| Real ambiguous input | Manual Review, confidence 0.37620155496730606 | PASS |
| Simulated prediction exception | Manual Review, null confidence; injected method called once | PASS |
| Prediction after removing mock | Original Electronics result restored | PASS |

Recorded summary: **4/4 PASS**. This is an injected model-method failure, not evidence
of a real service outage or an executed HTTP endpoint.

## Reproduce the Task 3 artifacts

In VS Code's PowerShell terminal, from the project folder:

```powershell
.\.venv\Scripts\python.exe task3_demo.py
```

This prints the actual outcomes and updates only `task3_failure_evidence.txt`.
To regenerate the PNG too, use a Python environment containing matplotlib:

```powershell
.\.venv\Scripts\python.exe -m pip install matplotlib
.\.venv\Scripts\python.exe task3_demo.py --diagram
```

Matplotlib is optional and is only used for diagram rendering. The generated PNG is
already included. No changes are needed to the classifier's dependency list.

Existing `classifier.py`, `test_classifier.py`, `test_results.txt`, and
`prediction_results.txt` were left unchanged; Task 3 evidence is stored separately.
