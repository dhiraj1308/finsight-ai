import random
import csv
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

from domain import Transaction


class SyntheticGenerator:
    """Generates realistic synthetic transaction data for development and testing."""

    # category -> (min_amount, max_amount)
    CATEGORIES: dict[str, tuple[float, float]] = {
        "Groceries": (5.0, 300.0),
        "Utilities": (20.0, 300.0),
        "Entertainment": (5.0, 150.0),
        "Dining": (5.0, 200.0),
        "Transport": (2.0, 100.0),
        "Healthcare": (10.0, 500.0),
        "Shopping": (10.0, 400.0),
        "Subscriptions": (5.0, 50.0),
    }

    # category -> list of realistic merchant names
    MERCHANTS: dict[str, list[str]] = {
        "Groceries": [
            "Whole Foods", "Trader Joe's", "Safeway", "Kroger", "Local Mart",
            # Indian merchants
            "FreshMart", "Reliance Fresh", "BigBasket", "More Supermarket", "DMart",
        ],
        "Utilities": [
            "City Power & Light", "Metro Water Co", "Gas Utility Inc", "ISP Broadband",
            # Indian merchants
            "Electricity Board", "Airtel", "Jio", "BESCOM", "TANGEDCO",
        ],
        "Entertainment": [
            "AMC Theatres", "Spotify", "Steam", "Netflix", "Local Arcade",
            # Indian merchants
            "Hotstar", "BookMyShow",
        ],
        "Dining": [
            "Chipotle", "Olive Garden", "Local Diner", "Sushi House", "Pizza Place",
            # Indian merchants
            "Zomato", "Swiggy", "Domino's", "Pizza Hut", "Starbucks India",
        ],
        "Transport": [
            "Uber", "Lyft", "Metro Transit", "Shell Gas Station", "Parking Garage",
            # Indian merchants
            "Ola", "Chennai Metro", "Rapido", "Indian Railways",
        ],
        "Healthcare": [
            "CVS Pharmacy", "Walgreens", "City Clinic", "Dental Care Co",
            # Indian merchants
            "Apollo Pharmacy", "PharmEasy", "1mg", "Apollo Hospitals",
        ],
        "Shopping": [
            "Amazon", "Target", "Best Buy", "Local Boutique", "IKEA",
            # Indian merchants
            "Myntra", "Flipkart", "Ajio", "Croma",
        ],
        "Subscriptions": [
            "Adobe Creative Cloud", "Notion", "GitHub", "Gym Membership",
            # Indian merchants
            "Amazon Prime", "YouTube Premium", "Spotify Premium",
        ],
    }

    def generate(
        self,
        n: int = 3000,
        start_date: date = date(2024, 1, 1),
        end_date: date = date(2024, 12, 31),
        seed: Optional[int] = None,
    ) -> list[Transaction]:
        if n < 1 or n > 100_000:
            raise ValueError(f"n must be between 1 and 100,000, got {n}")
        if start_date > end_date:
            raise ValueError(
                f"start_date ({start_date}) must not be after end_date ({end_date})"
            )

        rng = random.Random(seed)
        categories = list(self.CATEGORIES.keys())
        date_span_days = (end_date - start_date).days

        transactions: list[Transaction] = []
        for _ in range(n):
            category = rng.choice(categories)
            min_amt, max_amt = self.CATEGORIES[category]
            amount = round(rng.uniform(min_amt, max_amt), 2)
            merchant = rng.choice(self.MERCHANTS[category])
            txn_date = start_date + timedelta(days=rng.randint(0, date_span_days))

            transactions.append(
                Transaction(
                    date=txn_date,
                    merchant=merchant,
                    amount=amount,
                    category=category,
                )
            )

        return transactions

    def write_csv(self, transactions: list[Transaction], output_dir: Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "synthetic_transactions.csv"

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "merchant", "amount", "category"])
            for txn in transactions:
                writer.writerow(
                    [txn.date.isoformat(), txn.merchant, txn.amount, txn.category]
                )

        return output_path
