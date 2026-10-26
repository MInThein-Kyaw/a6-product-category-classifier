"""Real inference and clearly labeled simulated fault tests, with evidence output."""

import json
import math
import sys
import unittest
from unittest.mock import Mock, patch

from classifier import CATEGORIES, TRAINING_DATA, ProductClassifier


class ClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classifier = ProductClassifier()

    def check_case(self, name, product_name, description, expected_status, expected_category=None):
        actual = self.classifier.classify_product(product_name, description)
        expected = {"status": expected_status, "category": expected_category}
        passed = False
        try:
            self.assertEqual(set(actual), {"category", "confidence", "status", "reason"})
            self.assertIsInstance(actual["reason"], str)
            self.assertTrue(actual["reason"])
            if expected_status == "classified_or_review":
                self.assertIn(actual["status"], ("classified", "manual_review"))
                self.assertIn(actual["category"], (*CATEGORIES, "Manual Review"))
            else:
                self.assertEqual(actual["status"], expected_status)
                self.assertEqual(actual["category"], expected_category)
            confidence = actual["confidence"]
            if confidence is not None:
                self.assertTrue(math.isfinite(confidence) and 0 <= confidence <= 1)
            if actual["status"] == "classified":
                self.assertIn(actual["category"], CATEGORIES)
                self.assertIsNotNone(confidence)
                self.assertGreaterEqual(confidence, 0.45)
            if actual["status"] == "manual_review":
                self.assertEqual(actual["category"], "Manual Review")
            if actual["status"] == "invalid_input":
                self.assertIsNone(confidence)
            json.dumps(actual, allow_nan=False)
            passed = True
        finally:
            print(json.dumps({"case": name, "input": {"product_name": product_name,
                             "description": description}, "expected": expected,
                             "actual": actual, "result": "PASS" if passed else "FAIL"}), flush=True)
        return actual

    def test_01_electronics(self):
        self.check_case("Real model: electronics", "Portable Bluetooth headphones",
                        "Wireless audio headphones with microphone and rechargeable battery", "classified", "Electronics")

    def test_02_clothing(self):
        self.check_case("Real model: clothing", "Casual cotton shirt",
                        "Soft cotton shirt with long sleeves and collar for everyday wear", "classified", "Clothing")

    def test_03_home(self):
        self.check_case("Real model: home", "Large cooking saucepan",
                        "Kitchen cooking pot with lid for preparing food", "classified", "Home & Kitchen")

    def test_04_books(self):
        self.check_case("Real model: books", "Space adventure novel",
                        "Paperback fiction book with imaginative stories and adventure chapters", "classified", "Books")

    def test_05_ambiguous(self):
        self.check_case("Real model: ambiguous hybrid", "Smart recipe tablet",
                        "Touchscreen tablet with digital cooking recipes and illustrated cookbook chapters",
                        "classified_or_review")

    def test_06_empty_name(self):
        self.check_case("Validation: whitespace name", "   ", "A useful product", "invalid_input")

    def test_07_empty_description(self):
        self.check_case("Validation: empty description", "A product", "", "invalid_input")

    def test_08_nonstring_name(self):
        self.check_case("Validation: numeric name", 123, "A useful product", "invalid_input")

    def test_09_nonstring_description(self):
        self.check_case("Validation: null description", "A product", None, "invalid_input")

    def test_10_unavailable(self):
        with patch.object(self.classifier, "model", None):
            self.check_case("Simulated: unavailable model", "Headphones", "Wireless audio", "manual_review", "Manual Review")

    def test_11_model_error(self):
        fake = Mock(classes_=list(CATEGORIES))
        fake.predict_proba.side_effect = RuntimeError("Simulated inference failure")
        with patch.object(self.classifier, "model", fake):
            self.check_case("Simulated: inference exception", "Headphones", "Wireless audio", "manual_review", "Manual Review")

    def test_12_invalid_category(self):
        fake = Mock(classes_=["Unknown", "Clothing", "Home & Kitchen", "Books"])
        fake.predict_proba.return_value = [[0.7, 0.1, 0.1, 0.1]]
        with patch.object(self.classifier, "model", fake):
            self.check_case("Simulated: invalid category", "Headphones", "Wireless audio", "manual_review", "Manual Review")

    def test_13_low_confidence(self):
        fake = Mock(classes_=list(CATEGORIES))
        fake.predict_proba.return_value = [[0.25, 0.25, 0.25, 0.25]]
        with patch.object(self.classifier, "model", fake):
            result = self.check_case("Simulated: low confidence", "Hybrid product", "Mixed features", "manual_review", "Manual Review")
            self.assertEqual(result["confidence"], 0.25)

    def test_14_invalid_probabilities(self):
        fake = Mock(classes_=list(CATEGORIES))
        for values in ([float("nan"), 0.1, 0.1, 0.1], [1.2, -0.2, 0, 0], [0.1] * 4, [0.5, 0.5]):
            with self.subTest(values=values), patch.object(self.classifier, "model", fake):
                fake.predict_proba.return_value = [values]
                self.check_case("Simulated: malformed probabilities", "A product", "Some features", "manual_review", "Manual Review")

    def test_15_length_limits(self):
        self.check_case("Validation: long name", "x" * 201, "Description", "invalid_input")
        self.check_case("Validation: long description", "Product", "x" * 5001, "invalid_input")

    def test_16_unseen_vocabulary(self):
        self.check_case("Real model: unseen vocabulary", "Zqxv", "Plmokn ytrewq", "manual_review", "Manual Review")

    def test_17_training_failure(self):
        with patch("classifier.Pipeline.fit", side_effect=RuntimeError("Simulated training failure")):
            unavailable = ProductClassifier()
        with patch.object(self, "classifier", unavailable):
            self.check_case("Simulated: training failure", "Headphones", "Wireless audio", "manual_review", "Manual Review")

    def test_18_training_data(self):
        counts = {category: len(examples) for category, examples in TRAINING_DATA.items()}
        for category in CATEGORIES:
            self.assertGreaterEqual(counts[category], 12)
        self.assertIsNotNone(self.classifier.model, "Real model must have trained successfully")
        print(json.dumps({"case": "Training data and real model", "expected": "12+ per category and fitted model",
                          "actual": counts, "result": "PASS"}), flush=True)


if __name__ == "__main__":
    # stdout keeps PowerShell from formatting unittest progress as native errors.
    runner = unittest.TextTestRunner(stream=sys.stdout, verbosity=2)
    unittest.main(testRunner=runner)
