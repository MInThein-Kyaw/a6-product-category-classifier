"""Small local product classifier. No network, API keys, or keyword routing."""

import argparse
import json
import math
from functools import lru_cache

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


CATEGORIES = ("Electronics", "Clothing", "Home & Kitchen", "Books")
CONFIDENCE_THRESHOLD = 0.45

# Synthetic training data: each tuple is (product name, description).
# These are training examples, not the held-out products used in tests.
TRAINING_DATA = {
    "Electronics": [
        ("Wireless headphones", "Bluetooth audio headphones with rechargeable battery and microphone"),
        ("Laptop computer", "Portable computer with processor memory keyboard and display"),
        ("Smartphone", "Mobile phone with touchscreen camera and wireless connectivity"),
        ("USB charger", "USB power adapter for charging phone and tablet batteries"),
        ("Bluetooth speaker", "Portable wireless audio speaker with rechargeable battery"),
        ("Digital camera", "Camera with optical lens sensor and digital photo storage"),
        ("Computer monitor", "High resolution display screen for desktop computers"),
        ("Wireless mouse", "Computer mouse with wireless receiver and optical sensor"),
        ("Mechanical keyboard", "USB computer keyboard with mechanical switches"),
        ("External SSD", "Portable computer storage drive with USB data connection"),
        ("Tablet device", "Touchscreen tablet computer with battery and WiFi"),
        ("WiFi router", "Wireless network router with ethernet ports for internet connectivity"),
    ],
    "Clothing": [
        ("Cotton T shirt", "Soft cotton shirt with short sleeves for casual wear"),
        ("Denim jeans", "Blue denim trousers with pockets and a comfortable fit"),
        ("Winter jacket", "Warm insulated jacket with long sleeves and zipper"),
        ("Summer dress", "Lightweight floral dress with a flowing skirt"),
        ("Running shorts", "Breathable sports shorts with elastic waistband"),
        ("Wool sweater", "Knitted wool sweater with long sleeves for warmth"),
        ("Cotton socks", "Soft breathable socks for everyday wear"),
        ("Formal shirt", "Long sleeve button shirt with collar for office wear"),
        ("Rain coat", "Waterproof hooded coat to wear in wet weather"),
        ("Pleated skirt", "Knee length fabric skirt with fitted waistband"),
        ("Fleece hoodie", "Warm hooded sweatshirt with sleeves and front pocket"),
        ("Linen trousers", "Lightweight linen pants with pockets for casual wear"),
    ],
    "Home & Kitchen": [
        ("Stainless saucepan", "Kitchen cooking pot with lid and heat resistant handle"),
        ("Ceramic plates", "Dinner plates for serving food at the dining table"),
        ("Chef knife", "Kitchen knife for cutting vegetables and preparing food"),
        ("Cutting board", "Wooden kitchen board for chopping vegetables"),
        ("Cotton bedsheet", "Soft bed linen sheet for bedroom mattress"),
        ("Bath towel", "Absorbent cotton towel for bathroom use"),
        ("Storage containers", "Kitchen containers with lids for storing food"),
        ("Frying pan", "Nonstick kitchen pan for cooking and frying food"),
        ("Ceramic mug", "Kitchen drinking cup with handle for tea and coffee"),
        ("Cushion cover", "Decorative fabric cover for living room sofa cushion"),
        ("Mixing bowl", "Kitchen bowl for mixing ingredients and preparing food"),
        ("Window curtains", "Fabric curtains for bedroom and living room windows"),
    ],
    "Books": [
        ("Mystery novel", "Paperback fiction book about a detective solving a mystery"),
        ("Science textbook", "Educational textbook with chapters exercises and science lessons"),
        ("Cooking recipe book", "Hardcover cookbook with recipes and illustrated cooking instructions"),
        ("World history book", "Nonfiction book with historical chapters and maps"),
        ("Children storybook", "Illustrated story book with bedtime tales for children"),
        ("Poetry collection", "Paperback book of poems and literary verse"),
        ("Python programming book", "Technical book with programming tutorials and code examples"),
        ("Travel guide book", "Printed guidebook with travel advice maps and destination chapters"),
        ("Author biography", "Hardcover nonfiction book about an author's life"),
        ("Fantasy novel", "Fiction paperback novel with magic adventure and imaginative stories"),
        ("Mathematics workbook", "Educational book with mathematics exercises and practice lessons"),
        ("Gardening handbook", "Reference book with chapters on growing plants and gardening"),
    ],
}


def _text(product_name, description):
    return f"{product_name.strip()} {description.strip()}"


def _result(category, confidence, status, reason):
    return dict(category=category, confidence=confidence, status=status, reason=reason)


def _review(reason, confidence=None):
    return _result("Manual Review", confidence, "manual_review", reason)


class ProductClassifier:
    """Train once on construction; retain failures so callers get safe fallback."""

    def __init__(self):
        self.model = None
        try:
            texts, labels = [], []
            for category, examples in TRAINING_DATA.items():
                for name, description in examples:
                    texts.append(_text(name, description))
                    labels.append(category)
            model = Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
                ("classifier", LogisticRegression(C=4.0, random_state=42, max_iter=1000)),
            ])
            model.fit(texts, labels)
            self.model = model
        except Exception:
            # Do not expose internal exception details in a public response.
            self.model = None

    def classify_product(self, product_name, description):
        for field, value, limit in (
            ("product_name", product_name, 200),
            ("description", description, 5000),
        ):
            if not isinstance(value, str) or not value.strip():
                return _result(None, None, "invalid_input", f"{field} must be a non-empty string.")
            if len(value.strip()) > limit:
                return _result(None, None, "invalid_input", f"{field} must be at most {limit} characters.")

        if self.model is None:
            return _review("Model unavailable.")

        try:
            # Check the complete distribution, not just its largest value.
            classes = list(self.model.classes_)
            rows = self.model.predict_proba([_text(product_name, description)])
            if len(rows) != 1:
                return _review("Invalid model prediction.")
            probabilities = [float(value) for value in rows[0]]
            if (len(classes) != 4 or set(classes) != set(CATEGORIES)
                    or len(probabilities) != 4
                    or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities)
                    or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-6)):
                return _review("Invalid model prediction.")
            best = max(range(len(probabilities)), key=probabilities.__getitem__)
            confidence = probabilities[best]
            if confidence < CONFIDENCE_THRESHOLD:
                return _review("Confidence below 0.45; human review required.", confidence)
            return _result(str(classes[best]), confidence, "classified",
                           "Highest estimated category probability meets the 0.45 threshold.")
        except Exception:
            return _review("Model prediction failed.")


@lru_cache(maxsize=1)
def _default_classifier():
    return ProductClassifier()


def classify_product(product_name, description):
    """Public entry point; lazily train and reuse one classifier instance."""
    return _default_classifier().classify_product(product_name, description)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product-name", default="Portable Bluetooth headphones")
    parser.add_argument("--description", default="Wireless audio headphones with microphone and rechargeable battery")
    args = parser.parse_args()
    print(json.dumps(classify_product(args.product_name, args.description), indent=2, allow_nan=False))
