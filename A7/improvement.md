# One Controlled Improvement: Confidence Threshold

## Configuration and controls

| Setting | Before | After |
|---|---|---|
| Classification threshold | 0.45 | 0.40 |
| Training examples | Original A6 48 examples | Unchanged |
| Model and vectorizer | Original fitted TF-IDF + Logistic Regression pipeline | Same fitted object |
| Validation and error handling | Original A6 method | Same method from isolated source namespace |
| Evaluation dataset | Frozen 10 cases | Same file and SHA-256 |
| Retrieved context in classifier features | Not used | Not used |

`threshold_wrapper.py` loads the original source into a separate in-memory namespace,
sets only its threshold constant, and runs the original decision method using the
baseline fitted model. The hardcoded threshold number in its explanation text is
updated from 0.45 to 0.40 so the explanation remains truthful. No A6 file is edited.
An additional preflight check at 0.45 matched the original outputs on all ten cases.

## Actual results

| Metric | Before (0.45) | After (0.40) | Change |
|---|---|---|---|
| Accuracy | 9/10 = 90.0% | 9/10 = 90.0% | +0.0 percentage points |
| p95 latency (measured) | 0.730000 ms | 0.658800 ms | -0.071200 ms |
| API cost per task | N/A - local model | N/A - local model | N/A |

| ID | Baseline | After | Outcome |
|---|---|---|---|
| T01 | PASS | PASS | Unchanged |
| T02 | PASS | PASS | Unchanged |
| T03 | PASS | PASS | Unchanged |
| T04 | PASS | PASS | Unchanged |
| T05 | PASS | PASS | Unchanged |
| T06 | PASS | PASS | Unchanged |
| T07 | PASS | PASS | Unchanged |
| T08 | PASS | PASS | Unchanged |
| T09 | FAIL | FAIL | Unchanged |
| T10 | PASS | PASS | Unchanged |

Cases improved: **None**. Cases worse: **None**.
Cases unchanged: **T01, T02, T03, T04, T05, T06, T07, T08, T09, T10**.

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

## Evidence and timing

Results come from `baseline_results.json`, `improved_results.json`, and
`terminal_output.txt`. Frozen dataset SHA-256: `a2620f68b0fa35c699ec0c6e24068349ff9db4cd5addceb17a81819e4096d7e5`.
Per-case table latency is the median of 30 timed calls after three warmups per
case/version. Overall p95 uses nearest rank across 300 timed calls per version:
sort the observations and select observation ceil(0.95 * 300) = 285 (one-based).
`perf_counter_ns` times only each classifier call. Imports, training, retrieval,
printing, warmups, preflight checks, and mock setup/restoration are excluded. Version
order alternates on each repetition. Invalid-input and injected-failure calls are
included as part of the ten-case workload; this is not a network or cold-start metric.
Raw latency samples are in the JSON files. API cost is N/A because there is no API;
hardware, electricity, and operating cost were not measured or estimated.
