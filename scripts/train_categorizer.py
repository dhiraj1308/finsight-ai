"""
Training script for the Categorizer.
Run with: python scripts/train_categorizer.py
Generates synthetic labeled data, trains the classifier,
asserts F1 >= 0.80, and saves the model.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ingestion.synthetic_generator import SyntheticGenerator
from categorization.categorizer import Categorizer

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "processed" / "categorizer.joblib"


def main():
    print("Generating synthetic training data...")
    gen = SyntheticGenerator()
    transactions = gen.generate(n=3000, seed=42)
    print(f"Generated {len(transactions)} labeled transactions.")

    print("Training categorizer...")
    cat = Categorizer()
    f1 = cat.train(transactions)
    print(f"Validation weighted F1 score: {f1:.4f}")

    if f1 < 0.80:
        print(f"WARNING: F1 score {f1:.4f} is below the required 0.80 threshold.")
        sys.exit(1)

    cat.save(OUTPUT_PATH)
    print(f"Model saved to: {OUTPUT_PATH}")
    print("Training complete.")


if __name__ == "__main__":
    main()