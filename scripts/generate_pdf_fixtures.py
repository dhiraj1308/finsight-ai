"""
One-time script to generate test PDF fixtures for PDFParser tests.
Run with: python scripts/generate_pdf_fixtures.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from pypdf import PdfReader, PdfWriter

FIXTURES_DIR = Path(__file__).parent.parent / "tests" / "fixtures"
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)


def create_valid_statement_pdf(output_path: Path) -> None:
    """Creates a PDF with a real-looking transaction table."""
    doc = SimpleDocTemplate(str(output_path), pagesize=letter)
    styles = getSampleStyleSheet()
    elements = [Paragraph("Bank Statement - March 2024", styles["Title"])]

    data = [
        ["Transaction Date", "Description", "Debit", "Category"],
        ["2024-03-15", "Whole Foods", "87.43", "Groceries"],
        ["2024-03-16", "Netflix", "15.99", "Entertainment"],
        ["2024-03-18", "Shell Gas", "45.20", "Transport"],
        ["2024-03-20", "Chipotle", "12.50", "Dining"],
    ]
    table = Table(data)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]
        )
    )
    elements.append(table)
    doc.build(elements)


def create_no_table_pdf(output_path: Path) -> None:
    """Creates a PDF with only prose text, no transaction table at all."""
    doc = SimpleDocTemplate(str(output_path), pagesize=letter)
    styles = getSampleStyleSheet()
    elements = [
        Paragraph("Welcome to Your Bank", styles["Title"]),
        Paragraph(
            "Thank you for being a valued customer. This document contains "
            "general information about our services and does not include "
            "any transaction details.",
            styles["Normal"],
        ),
    ]
    doc.build(elements)


def create_password_protected_pdf(source_path: Path, output_path: Path) -> None:
    """Takes an existing PDF and re-saves it with password protection."""
    reader = PdfReader(str(source_path))
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    writer.encrypt(user_password="test123")
    with open(output_path, "wb") as f:
        writer.write(f)


if __name__ == "__main__":
    valid_path = FIXTURES_DIR / "sample_bank_statement.pdf"
    no_table_path = FIXTURES_DIR / "no_transaction_table.pdf"
    protected_path = FIXTURES_DIR / "password_protected.pdf"

    create_valid_statement_pdf(valid_path)
    print(f"Created: {valid_path}")

    create_no_table_pdf(no_table_path)
    print(f"Created: {no_table_path}")

    create_password_protected_pdf(valid_path, protected_path)
    print(f"Created: {protected_path}")

    print("\nAll fixtures generated successfully.")