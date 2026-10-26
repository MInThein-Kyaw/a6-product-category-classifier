"""Build the report and comparison from saved measurements; run no predictions."""

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def read(name):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def cell(value):
    if value is None:
        return "null"
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "|" + "---|" * len(headers),
        *["| " + " | ".join(cell(value) for value in row) + " |" for row in rows],
    ])


def result_table(report):
    return table(["ID", "Expected Output", "Actual Output", "Pass/Fail", "Latency (ms)"], [
        [row["id"], f"{cell(row['expected_category'])}; {row['expected_status']}",
         f"{cell(row['actual']['category'])}; {row['actual']['status']}; confidence {cell(row['actual']['confidence'])}",
         row["result"], f"{row['latency_ms']:.6f}"] for row in report["cases"]])


def main():
    baseline, improved = read("baseline_results.json"), read("improved_results.json")
    cases, records = read("eval-cases.json"), read("knowledge-base.json")
    lock, integrity = read("eval_cases.lock.json"), read("integrity_check.json")
    retrieval = read("retrieval_demo.json")
    assert baseline["dataset_sha256"] == improved["dataset_sha256"] == lock["dataset_sha256"]
    assert integrity["a6_files_unchanged"] and integrity["dataset_unchanged"]
    assert baseline["threshold"] == 0.45 and improved["threshold"] == 0.40
    changed_for_better, changed_for_worse, unchanged = [], [], []
    for before, after in zip(baseline["cases"], improved["cases"]):
        assert before["id"] == after["id"]
        if before["result"] == "FAIL" and after["result"] == "PASS":
            changed_for_better.append(before["id"])
        elif before["result"] == "PASS" and after["result"] == "FAIL":
            changed_for_worse.append(before["id"])
        else:
            unchanged.append(before["id"])
    better = ", ".join(changed_for_better) or "None"
    worse = ", ".join(changed_for_worse) or "None"
    same = ", ".join(unchanged) or "None"
    delta_accuracy = improved["accuracy_percent"] - baseline["accuracy_percent"]
    delta_latency = improved["p95_latency_ms"] - baseline["p95_latency_ms"]
    comparison = table(["Metric", "Before (0.45)", "After (0.40)", "Change"], [
        ["Accuracy", f"{baseline['correct_cases']}/10 = {baseline['accuracy_percent']:.1f}%",
         f"{improved['correct_cases']}/10 = {improved['accuracy_percent']:.1f}%", f"{delta_accuracy:+.1f} percentage points"],
        ["p95 latency (measured)", f"{baseline['p95_latency_ms']:.6f} ms",
         f"{improved['p95_latency_ms']:.6f} ms", f"{delta_latency:+.6f} ms"],
        ["API cost per task", "N/A - local model", "N/A - local model", "N/A"],
    ])
    outcomes = table(["ID", "Baseline", "After", "Outcome"], [
        [before["id"], before["result"], after["result"],
         "Improved" if before["id"] in changed_for_better else
         "Worse" if before["id"] in changed_for_worse else "Unchanged"]
        for before, after in zip(baseline["cases"], improved["cases"])])
    knowledge_table = table(["ID / Category", "Description", "Keywords", "Product Examples"], [
        [f"{record['id']} / {record['category']}", record["description"],
         ", ".join(record["keywords"]), "; ".join(record["product_examples"])] for record in records])
    dataset_table = table(["ID", "Case Type", "Input", "Expected Output", "Expected Review/Fallback", "Reason"], [
        [case["id"], case["case_type"],
         f"Name: {json.dumps(case['product_name'])}<br>Description: {case['description']}" +
         ("<br>Fault: simulated prediction exception" if case.get("simulate_model_failure") else ""),
         case["expected_category"], case["expected_fallback_behavior"], case["reason"]] for case in cases])

    interpretation = """The attempted improvement did not increase accuracy. All categories, statuses,
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
review operations."""

    improvement = f"""# One Controlled Improvement: Confidence Threshold

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

{comparison}

{outcomes}

Cases improved: **{better}**. Cases worse: **{worse}**.
Cases unchanged: **{same}**.

{interpretation}

## Evidence and timing

Results come from `baseline_results.json`, `improved_results.json`, and
`terminal_output.txt`. Frozen dataset SHA-256: `{lock['dataset_sha256']}`.
Per-case table latency is the median of 30 timed calls after three warmups per
case/version. Overall p95 uses nearest rank across 300 timed calls per version:
sort the observations and select observation ceil(0.95 * 300) = 285 (one-based).
`perf_counter_ns` times only each classifier call. Imports, training, retrieval,
printing, warmups, preflight checks, and mock setup/restoration are excluded. Version
order alternates on each repetition. Invalid-input and injected-failure calls are
included as part of the ten-case workload; this is not a network or cold-start metric.
Raw latency samples are in the JSON files. API cost is N/A because there is no API;
hardware, electricity, and operating cost were not measured or estimated.
"""
    (HERE / "improvement.md").write_text(improvement, encoding="utf-8")

    report = f"""# A7 - Evaluate and Improve Your AI Capability

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

{knowledge_table}

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
{retrieval['retrieved_context'][0]['score']}. Its selected description was:

> {retrieval['retrieved_context'][0]['description']}

Selected keywords: {', '.join(retrieval['retrieved_context'][0]['keywords'])}.
Selected examples: {', '.join(retrieval['retrieved_context'][0]['product_examples'])}.
The local classifier returned Electronics, confidence
{retrieval['actual_decision']['confidence']}. This prediction is **not evidence of a
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

{dataset_table}

The input and expected results were saved and SHA-256 locked before the A7 baseline
and improved evaluations. The freeze command did not train or predict.
Freeze UTC: `{lock['frozen_at_utc']}`. Dataset SHA-256:
`{lock['dataset_sha256']}`.
The same locked file was used for both evaluations, and no labels were changed.
Some products reuse earlier A6 demonstrations whose behavior was already known;
this is a reproducible coursework set, not a new blind or statistically independent
holdout study. T09 deliberately uses the semantic Electronics target, rather than
changing the expectation to match its previously observed abstention.

## 3. Baseline Evaluation

The baseline invokes the original A6 class with its unchanged **0.45** threshold.
It trains the original 48-example TF-IDF/Logistic Regression pipeline once. The run
used Python {baseline['python_version']} and scikit-learn {baseline['sklearn_version']}.
Confidence in the following table is the actual returned model probability, or null
where inference failed or input was rejected. Latency is per-case median.

{result_table(baseline)}

**Baseline Accuracy:** {baseline['correct_cases']} / 10 = **{baseline['accuracy_percent']:.1f}%**.

**p95 Latency:** **{baseline['p95_latency_ms']:.6f} ms**, measured locally.

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

{result_table(improved)}

**After Accuracy:** {improved['correct_cases']} / 10 = **{improved['accuracy_percent']:.1f}%**.

**After p95 Latency:** **{improved['p95_latency_ms']:.6f} ms**, measured locally.

**Cost per Task:** **N/A**. Raw measurements:
[improved_results.json](improved_results.json).

## 6. Before / After Comparison

{comparison}

**Cases That Improved:** {better}.

**Cases That Became Worse:** {worse}.

**Cases That Stayed Unchanged:** {same}.

**Trade-offs and Conclusion:**

{interpretation}

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
"""
    (HERE / "A7_Report.md").write_text(report, encoding="utf-8")
    print("Generated A7_Report.md and improvement.md from saved real results.")
    print(comparison)


if __name__ == "__main__":
    main()
