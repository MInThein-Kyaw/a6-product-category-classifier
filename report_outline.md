# A6 - Individual Submission

**Student Name:** [Complete before submission]  
**Student ID:** [Complete before submission]  
**Project / Capability Name:** Product Category Classifier  
**Implementation Method:** Other - Python scikit-learn TF-IDF + Logistic Regression

## 1. Problem and Capability

**Problem:** A product catalog needs to route new listings into Electronics,
Clothing, Home & Kitchen, or Books. Sellers describe similar products in different
ways, so manually assigning every listing takes time. This capability predicts a
category and sends uncertain or failed decisions for manual review.

**Why AI?** A learned text classifier can combine weighted words and phrases rather
than requiring a separate hand-written category rule for every wording. It is still
limited by its small training set and does not understand products like a person.

**Input:** `product_name` and `description`, both non-empty strings. Example:
`Portable Bluetooth headphones` and
`Wireless audio headphones with microphone and rechargeable battery`.

**Expected Output:** JSON with `category`, `confidence`, `status`, and `reason`.
There are four automatic categories plus a Manual Review fallback. Invalid input
receives null category/confidence and `invalid_input` status.

## 2. Decision Implementation

**Decision Configuration / Code:** Attach `classifier.py` and `requirements.txt`.
The code trains a TF-IDF unigram/bigram vectorizer and Logistic Regression pipeline
on 48 synthetic labeled examples, 12 per category. Training is reproducible with
`random_state=42`; each classifier instance trains once. Confidence threshold: 0.45.

**Output Schema:** See README's field table and validation rules. Confidence is an
estimated probability, not a calibrated reliability guarantee. Successful results
contain a permitted category and probability at least 0.45; fallback has category
Manual Review and retains confidence only when valid probabilities are available.

### Test Results

These results come from the actual saved run on Python 3.13.14 / scikit-learn 1.9.1.
Complete descriptions and full-precision outputs appear in `test_results.txt`.

| # | Input (name; description in execution log) | Expected Decision | Actual Decision | Pass/Fail |
|---|---|---|---|---|
| 1 | Portable Bluetooth headphones | Electronics | Electronics, 0.718315 | PASS |
| 2 | Casual cotton shirt | Clothing | Clothing, 0.797699 | PASS |
| 3 | Large cooking saucepan | Home & Kitchen | Home & Kitchen, 0.701209 | PASS |
| 4 | Space adventure novel | Books | Books, 0.748030 | PASS |
| 5 | Smart recipe tablet | Valid classification or safe review | Manual Review, 0.376202 | PASS |
| 6 | Whitespace-only name | Reject invalid input | invalid_input | PASS |
| 7 | Zqxv; Plmokn ytrewq | Manual Review | Manual Review, 0.261188 | PASS |
| 8 | Headphones; simulated inference exception | Manual Review | Manual Review, null confidence | PASS |

**Evidence:** Attach `test_results.txt` and `test_classifier.py`. The suite passed
18 unittest methods, including real model inference, input rejection, and explicitly
simulated unavailable model, training failure, inference failure, invalid labels,
malformed probabilities, and low-confidence predictions. Four normal held-out
examples are a demonstration, not a statistically reliable evaluation dataset.

## 3. Platform Integration

**Architecture Diagram:** Render and insert `architecture.mmd`: user -> application
-> validation -> classifier -> output checks -> category routing or manual review.
Routing and review queues are the consuming application's responsibility.

**Example API Request:** Proposed `POST /classify-product`; no HTTP server is included.

```json
{
  "product_name": "Portable Bluetooth headphones",
  "description": "Wireless audio headphones with microphone and rechargeable battery"
}
```

**Example API Response:** Actual function output for this request:

```json
{
  "category": "Electronics",
  "confidence": 0.7183149479308034,
  "status": "classified",
  "reason": "Highest estimated category probability meets the 0.45 threshold."
}
```

**Validation Rules:** Both fields must be strings with text after trimming. Trimmed
name length must be at most 200 characters; description at most 5,000. Check category
labels, probability row shape, finite probabilities in [0,1], and their sum before
applying the confidence threshold. The result must serialize as strict JSON.

**Fallback Strategy:** Invalid input is rejected. Unavailable models, model errors,
invalid predictions, and confidence below 0.45 deterministically return Manual Review.
Failures without a usable distribution return null confidence. Low confidence retains
its valid probability. A production adapter could map invalid input to HTTP 400 and
both classification and review decisions to HTTP 200.

**Fallback Evidence:** In the log, the ambiguous tablet reaches review with probability
0.376202. A simulated inference exception returns:

```json
{
  "category": "Manual Review",
  "confidence": null,
  "status": "manual_review",
  "reason": "Model prediction failed."
}
```

## 4. Engineering Reflection

**What worked:** The four normal held-out examples reached the expected categories.
Input validation and simulated failure scenarios returned the documented safe results.

**Unexpected behavior or mistakes:** The recipe tablet was ambiguous enough to need
review (0.376202). No incorrect category appeared in the four straightforward tests.
This does not demonstrate broad accuracy or eliminate confident errors on new data.

**Production risks:** The synthetic dataset is small and English-only; vocabulary
changes and mixed products may cause errors. Probabilities and the review threshold
are uncalibrated. A real application needs diagnostic logging, privacy controls,
monitoring, and a staffed review queue.

**Next improvement:** Obtain representative labeled data, reserve independent
validation/test sets, measure precision/recall and review rates, calibrate probabilities,
and choose the threshold from validation results. Add human feedback and model versioning.

## 5. References

- Course assignment instructions: supplied `assignment_8.md` (Assignment A6, Week 08).
- Course report template: supplied `assignment8_template.md`.
- scikit-learn 1.9.1: TF-IDF, Pipeline, and Logistic Regression APIs.
- Python standard library: unittest, unittest.mock, json, and argparse.
- AI assistance: helped write implementation, synthetic data, tests, and documentation.
  Model predictions and saved test evidence came from executing the Python code.

## 6. Submission Checklist

- [x] Decision configuration or source code included
- [x] Five or more test cases completed
- [x] Actual execution evidence saved in test_results.txt
- [x] Integration architecture included as Mermaid source
- [x] Validation and fallback explained
- [x] Engineering reflection completed from observed tests
- [ ] Complete student name and ID
- [ ] Render architecture and include test evidence in the PDF report
- [ ] Export as A6_StudentID.pdf and submit with supporting files
