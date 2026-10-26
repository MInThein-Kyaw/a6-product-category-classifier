# A7 - Evaluate and Improve Your AI Capability

**Student Name:** [Enter student name]  
**Student ID:** [Enter student ID]  
**Project / Capability Name:** Product Category Classifier  
**A6 Project Reference:** Original `../classifier.py` and `../A6_Report.md` in Platform assignment 6  
**Implementation Method:** Other - local Python / scikit-learn  
**Course / Week:** Platform Development / 09 - Retrieval, Context and Evaluation

## 1. Grounding Context

**Knowledge Source:** Six synthetic product-category guidance records authored for
this project, stored in [knowledge-base.json](knowledge-base.json). They describe
the four A6 categories and important boundaries, such as a tablet versus a cookbook
and a printed garment versus a book. These are project guidance, not an external
authoritative catalog taxonomy.

**Knowledge Records:**

| ID / Category | Description | Keywords | Product Examples |
|---|---|---|---|
| K01 / Electronics | Audio devices use electronic components to play or capture sound. | headphones, speaker, audio, bluetooth, microphone, wireless, battery | Bluetooth headphones; Portable audio speaker |
| K02 / Electronics | Computing and connectivity hardware includes tablets, displays, and USB accessories. Classify a physical tablet as hardware even when it displays recipes. | computer, tablet, touchscreen, monitor, usb, adapter, router | Touchscreen tablet computer; USB network adapter |
| K03 / Clothing | Wearable garments belong to Clothing even when decorated with book or technology illustrations. | shirt, cotton, sleeves, trousers, jacket, wear, garment | Cotton long sleeve shirt; Book print T shirt |
| K04 / Home & Kitchen | Physical cookware and food preparation utensils belong to Home & Kitchen. A cookbook is a book rather than cookware. | saucepan, pan, pot, kitchen, cooking, lid, bowl | Stainless cooking saucepan; Mixing bowl |
| K05 / Home & Kitchen | Household textiles used on furniture or in rooms belong to Home & Kitchen; wearable fabric garments belong to Clothing. | bedsheet, towel, curtains, cushion, sofa, bedroom, linen | Bedroom curtains; Sofa cushion cover |
| K06 / Books | Printed reading and educational works belong to Books, including novels, poetry, textbooks, and cookbooks. Depictions of books printed on garments do not make the garment a book. | book, novel, paperback, fiction, poetry, textbook, cookbook, chapters | Paperback adventure novel; Illustrated recipe cookbook |

**Retrieval / Selection Method:** `retrieval.py` lowercases the description, extracts
alphanumeric word tokens, and counts unique overlaps with each record's keyword
tokens. It returns at most two records with positive overlap, sorted by score
descending and then record ID for ties. No matches return an empty list. This is
deterministic context retrieval, separate from the learned product classifier.

**Context Supplied to Model:** Actual implementation supplies a packet containing
`input` and `retrieved_context` to the application-level decision demonstration.
The existing A6 method receives only `packet["input"]`. It has no runtime context
parameter; the records are metadata available to an application or reviewer and
are **not consumed by the classifier's feature vectorizer**. Neither evaluation
uses runtime grounding or changes the classifier input text. A future integration
could train a model designed to consume retrieved records, but that is proposed
and was not implemented or measured in this controlled experiment.

For the headphones description in T01, actual retrieval selected K01 with score
5. Its selected description was:

> Audio devices use electronic components to play or capture sound.

Selected keywords: headphones, speaker, audio, bluetooth, microphone, wireless, battery.
Selected examples: Bluetooth headphones, Portable audio speaker.
The local classifier returned Electronics, confidence
0.7183149479308034. This prediction is **not evidence of a
grounding benefit**: context was retrieved but not included in model features.
The actual packet and decision are saved in [retrieval_demo.json](retrieval_demo.json).

## 2. Evaluation Dataset

The dataset contains exactly **10 cases: 3 normal, 3 edge, 2 fallback, 2 difficult**.
Expected categories follow the physical product identity; deliberately unsafe or
invalid conditions have explicit review/rejection targets. A case is correct only
when **both category and expected status match**. `none` maps to `classified`,
`manual_review` to `manual_review`, and `reject_invalid_input` to `invalid_input`.
Thus the required accuracy includes correct review and rejection behavior; it is
not an accuracy estimate based solely on classified products.

| ID | Case Type | Input | Expected Output | Expected Review/Fallback | Reason |
|---|---|---|---|---|---|
| T01 | normal | Name: "Portable Bluetooth headphones"<br>Description: Wireless audio headphones with microphone and rechargeable battery | Electronics | none | A physical wireless audio device is Electronics. |
| T02 | normal | Name: "Casual cotton shirt"<br>Description: Soft cotton shirt with long sleeves and collar for everyday wear | Clothing | none | A wearable cotton garment belongs to Clothing. |
| T03 | normal | Name: "Large cooking saucepan"<br>Description: Kitchen cooking pot with lid for preparing food | Home & Kitchen | none | Physical cookware belongs to Home & Kitchen. |
| T04 | edge | Name: "Novel"<br>Description: Paperback fiction book. | Books | none | The description is unusually short but still clearly identifies a printed novel. |
| T05 | edge | Name: "Zqxv"<br>Description: Plmokn ytrewq | Manual Review | manual_review | Unrecognizable vocabulary should be referred to a human rather than assigned a product category. |
| T06 | edge | Name: "USB accessory"<br>Description: Portable wireless adapter for a computer. | Electronics | none | A terse generic name still describes a physical computer connectivity accessory. |
| T07 | fallback | Name: "   "<br>Description: A useful household product | null | reject_invalid_input | A whitespace-only name must be rejected before inference. |
| T08 | fallback | Name: "Portable Bluetooth headphones"<br>Description: Wireless audio headphones with microphone and rechargeable battery<br>Fault: simulated prediction exception | Manual Review | manual_review | An intentionally injected prediction exception must trigger deterministic Manual Review, regardless of the product's normal category. |
| T09 | difficult | Name: "Smart recipe tablet"<br>Description: Touchscreen tablet with digital cooking recipes and illustrated cookbook chapters | Electronics | none | The physical item is a touchscreen tablet, so the semantic target is Electronics despite the cookbook wording. Abstention is safe but does not meet this strict automatic-routing target. |
| T10 | difficult | Name: "Literary print shirt"<br>Description: Cotton shirt with a printed book illustration and poetry text for casual wear | Clothing | none | The item is a wearable shirt; references to books and poetry describe its decoration. |

The input and expected results were saved and SHA-256 locked before the A7 baseline
and improved evaluations. The freeze command did not train or predict.
Freeze UTC: `2026-10-26T19:43:53.949902+00:00`. Dataset SHA-256:
`a2620f68b0fa35c699ec0c6e24068349ff9db4cd5addceb17a81819e4096d7e5`.
The same locked file was used for both evaluations, and no labels were changed.
Some products reuse earlier A6 demonstrations whose behavior was already known;
this is a reproducible coursework set, not a new blind or statistically independent
holdout study. T09 deliberately uses the semantic Electronics target, rather than
changing the expectation to match its previously observed abstention.

## 3. Baseline Evaluation

The baseline invokes the original A6 class with its unchanged **0.45** threshold.
It trains the original 48-example TF-IDF/Logistic Regression pipeline once. The run
used Python 3.13.14 and scikit-learn 1.9.1.
Confidence in the following table is the actual returned model probability, or null
where inference failed or input was rejected. Latency is per-case median.

| ID | Expected Output | Actual Output | Pass/Fail | Latency (ms) |
|---|---|---|---|---|
| T01 | Electronics; classified | Electronics; classified; confidence 0.7183149479308034 | PASS | 0.546750 |
| T02 | Clothing; classified | Clothing; classified; confidence 0.7976990227195476 | PASS | 0.562300 |
| T03 | Home & Kitchen; classified | Home & Kitchen; classified; confidence 0.7012092347176383 | PASS | 0.564200 |
| T04 | Books; classified | Books; classified; confidence 0.6901152674835483 | PASS | 0.553900 |
| T05 | Manual Review; manual_review | Manual Review; manual_review; confidence 0.2611879742278097 | PASS | 0.548750 |
| T06 | Electronics; classified | Electronics; classified; confidence 0.7207336452639604 | PASS | 0.559150 |
| T07 | null; invalid_input | null; invalid_input; confidence null | PASS | 0.000700 |
| T08 | Manual Review; manual_review | Manual Review; manual_review; confidence null | PASS | 0.016900 |
| T09 | Electronics; classified | Manual Review; manual_review; confidence 0.37620155496730606 | FAIL | 0.569600 |
| T10 | Clothing; classified | Clothing; classified; confidence 0.577241817684517 | PASS | 0.571950 |

**Baseline Accuracy:** 9 / 10 = **90.0%**.

**p95 Latency:** **0.730000 ms**, measured locally.

**Cost per Task:** **N/A**, because inference uses a local model with no paid API.
Hardware and operating costs were not measured; N/A does not mean computing is free.

**Measurement Method:** For each case/version, run three excluded warmups and then
30 timed repetitions. Time each call using `time.perf_counter_ns` and convert
nanoseconds to milliseconds. Alternate baseline/improved order on each repetition.
The table reports the median of each case's 30 calls; overall p95 is the nearest-rank
95th percentile across 300 timed calls per version (sorted observation 285).
All repeated outputs were checked for equality within each case/version.
Imports, training, retrieval, preflight checks, warmups, console output, and mock
setup/restoration are excluded. The injected exception and its handling are timed.
The workload includes quick invalid-input and failure paths, not just inference.
This is warm local-call latency, not HTTP, startup, or production load latency.
Raw samples are retained in [baseline_results.json](baseline_results.json).

## 4. One Controlled Improvement

**Selected Improvement:** Lower the classification confidence threshold only,
from **0.45 to 0.40**.

**Reason for Choosing It:** Earlier A6 results showed an ambiguous product being
referred for review. A lower threshold is a simple hypothesis for reducing
unnecessary abstentions. It was specified before this A7 comparison; it is an
improvement attempt, not a guaranteed accuracy increase.

**Before Configuration:** Original A6 `CONFIDENCE_THRESHOLD = 0.45`.

**After Configuration:** The A7-only wrapper's isolated in-memory copy uses
`CONFIDENCE_THRESHOLD = 0.40`. It calls the unchanged original decision method with
the exact same fitted model object. The numeric threshold in the human-readable
reason is updated accordingly. No training examples, vectorizer options, model
parameters, validation, fallback policy, evaluation cases, or root files change.
An extra preflight at 0.45 verified identical original/wrapper outputs on all ten
cases. Retrieval context is not used in either classification path.

Code: [threshold_wrapper.py](threshold_wrapper.py). Detailed comparison:
[improvement.md](improvement.md).

## 5. After-Improvement Evaluation

The same ten frozen cases and 30-repetition timing method were used at **0.40**.

| ID | Expected Output | Actual Output | Pass/Fail | Latency (ms) |
|---|---|---|---|---|
| T01 | Electronics; classified | Electronics; classified; confidence 0.7183149479308034 | PASS | 0.549000 |
| T02 | Clothing; classified | Clothing; classified; confidence 0.7976990227195476 | PASS | 0.564350 |
| T03 | Home & Kitchen; classified | Home & Kitchen; classified; confidence 0.7012092347176383 | PASS | 0.562550 |
| T04 | Books; classified | Books; classified; confidence 0.6901152674835483 | PASS | 0.558150 |
| T05 | Manual Review; manual_review | Manual Review; manual_review; confidence 0.2611879742278097 | PASS | 0.546750 |
| T06 | Electronics; classified | Electronics; classified; confidence 0.7207336452639604 | PASS | 0.564450 |
| T07 | null; invalid_input | null; invalid_input; confidence null | PASS | 0.001200 |
| T08 | Manual Review; manual_review | Manual Review; manual_review; confidence null | PASS | 0.018100 |
| T09 | Electronics; classified | Manual Review; manual_review; confidence 0.37620155496730606 | FAIL | 0.569100 |
| T10 | Clothing; classified | Clothing; classified; confidence 0.577241817684517 | PASS | 0.570700 |

**After Accuracy:** 9 / 10 = **90.0%**.

**After p95 Latency:** **0.658800 ms**, measured locally.

**Cost per Task:** **N/A**. Raw measurements:
[improved_results.json](improved_results.json).

## 6. Before / After Comparison

| Metric | Before (0.45) | After (0.40) | Change |
|---|---|---|---|
| Accuracy | 9/10 = 90.0% | 9/10 = 90.0% | +0.0 percentage points |
| p95 latency (measured) | 0.730000 ms | 0.658800 ms | -0.071200 ms |
| API cost per task | N/A - local model | N/A - local model | N/A |

**Cases That Improved:** None.

**Cases That Became Worse:** None.

**Cases That Stayed Unchanged:** T01, T02, T03, T04, T05, T06, T07, T08, T09, T10.

**Trade-offs and Conclusion:**

The attempted improvement did not increase accuracy. All categories, statuses,
and probabilities stayed the same. T09 remained Manual Review with confidence
0.37620155496730606, below both thresholds. Its physical product is a tablet, so the
frozen semantic expectation is Electronics/classified. Review is a safe abstention,
but fails that strict automatic-routing target. All other cases passed both versions.
No observed confidence falls in [0.40, 0.45), the interval in which this change could
alter routing. No expectations were revised after seeing these results.

The latency difference is an observation from one local session, not evidence that
the lower threshold causes a speed change. Both versions perform the same inference;
the wrapper adds a Python call and reason-text formatting. Scheduling, caching, and
measurement noise also affect timing. Additional process runs would be needed for
a reliable performance comparison.

Lowering a threshold may reduce reviews but can accept more incorrect predictions.
This dataset showed no review reduction, and provides no evidence supporting a
production switch. The model and either threshold remain unsuitable for production
without representative independent validation, calibration, monitoring, and human
review operations.

## 7. Engineering Reflection

**What I learned:** Systematic evaluation exposed a difference between clear product
examples and an ambiguous input. Both versions met nine expectations, but the
tablet remained a strict routing failure even though human review was a safe
response. Lowering the threshold did not fix the underlying uncertainty. Separating
retrieval metadata from model features also prevented attributing decisions to
grounding that the model never consumed.

**Why a few successful examples are insufficient:** Clear examples cover only a
small vocabulary and do not establish behavior on ambiguity, unfamiliar text, invalid
input, or dependency failure. This set includes those conditions but is still only
ten deliberately chosen cases, with some reused A6 examples. Thirty repeated runs
improve timing evidence, not the diversity or statistical size of the accuracy set.

**What I would evaluate next:** Use a larger, independently labeled representative
catalog set, including multilingual listings and overlapping categories. Measure
per-category precision/recall, calibration, wrong automatic routing, safe abstention,
and review workload separately. Select thresholds on validation data and evaluate
once on a held-out set. Test realistic outages, multiple sessions, cold-start time,
concurrency, and drift. Evaluate actual grounding separately before claiming a
retrieval benefit. Any HTTP adapter remains proposed and was not executed here.

## 8. Evidence and References

Actual artifacts, supplied in this folder:

- [knowledge-base.json](knowledge-base.json), [retrieval.py](retrieval.py), and
  [retrieval_demo.json](retrieval_demo.json): six records, retriever, actual selected context.
- [eval-cases.json](eval-cases.json), [eval_cases.lock.json](eval_cases.lock.json), and
  [freeze_evidence.txt](freeze_evidence.txt): ten cases, locked hash, pre-run freeze evidence.
- [evaluate.py](evaluate.py) and [threshold_wrapper.py](threshold_wrapper.py): evaluator
  and the one controlled threshold change.
- [baseline_results.json](baseline_results.json) / [baseline_results.txt](baseline_results.txt)
  and [improved_results.json](improved_results.json) / [improved_results.txt](improved_results.txt):
  measured predictions, timing, PASS/FAIL, and summaries.
- [terminal_output.txt](terminal_output.txt): actual captured evaluation console output,
  including the failed T09 case, comparison, and integrity verification.
- [integrity_check.json](integrity_check.json): SHA-256 verification that all original
  root A6 files and frozen cases remained unchanged.
- [verify.py](verify.py) and [verification.txt](verification.txt): separate artifact
  checks and explicitly simulated threshold-boundary/failure checks, not extra
  evaluation cases or additions to the ten-case accuracy denominator.
- [README.md](README.md), [improvement.md](improvement.md), and [write_reports.py](write_reports.py):
  reproduction commands, comparison, and report generation from saved measurements.

**References:** Teacher-provided `assignment_9.md` (A7, Week 09) and
`assignment9_template.md` (nine-section template); original A6
[classifier.py](../classifier.py) and [A6_Report.md](../A6_Report.md);
official scikit-learn
[TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
and [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
documentation; Python standard-library `time.perf_counter_ns`, `hashlib`, and
`unittest.mock` APIs. Context records are synthetic project guidance rather than
retrieved external facts. The diagram/API designs in A6 are not new A7 execution evidence.

**AI Assistance Disclosure:** OpenAI Codex assisted with the knowledge records,
evaluation-case preparation, Python scripts, execution, and report writing. Expected
results were frozen before the A7 evaluation runs. Predictions, PASS/FAIL results,
and latencies come from actual local execution. No screenshots, results, measured
costs, or API executions were fabricated. API cost is N/A; production trade-offs
are qualitative expectations, not measured production outcomes.

## 9. Submission Checklist

- [x] Same capability as Week 08; original root A6 files unchanged
- [x] At least five knowledge records (six supplied)
- [x] Ten evaluation cases with expected results frozen before testing
- [x] Baseline results completed using threshold 0.45
- [x] Exactly one improvement attempt: threshold 0.40
- [x] After-improvement results completed on the same cases and fitted model
- [x] Before/after comparison included, with no accuracy increase reported honestly
- [x] Actual execution evidence, reflection, references, and AI disclosure included
- [ ] Enter student name and ID
- [ ] Capture genuine screenshots listed in README if desired
- [ ] Export as A7_StudentID.pdf and submit with supporting A7 files
