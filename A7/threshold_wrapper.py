"""Run unchanged A6 decision code in an isolated namespace with one new threshold."""

import importlib.util
import sys
from pathlib import Path


# Prevent imports of root A6 source from creating/changing root bytecode files.
sys.dont_write_bytecode = True
A6_SOURCE = Path(__file__).resolve().parent.parent / "classifier.py"


def load_a6(module_name):
    spec = importlib.util.spec_from_file_location(module_name, A6_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ThresholdClassifier:
    def __init__(self, fitted_model, threshold=0.40):
        if threshold not in (0.40, 0.45):
            raise ValueError("This experiment supports only thresholds 0.40 and 0.45.")
        self.model = fitted_model  # Reuse the exact baseline fitted pipeline.
        self.threshold = threshold
        self._a6 = load_a6("a7_threshold_a6")
        # This modifies an in-memory module copy, never the original file/module.
        self._a6.CONFIDENCE_THRESHOLD = threshold

    def classify_product(self, product_name, description):
        # The unchanged A6 method reads self.model. Calling it here reuses every
        # original input check, probability check, and exception/fallback branch.
        result = self._a6.ProductClassifier.classify_product(self, product_name, description)
        # A6's human-readable reasons hardcode 0.45. Update only that displayed
        # number so it accurately describes this configuration's actual threshold.
        result["reason"] = result["reason"].replace("0.45", f"{self.threshold:.2f}")
        return result
