"""Debug script — inspect what pdfplumber sees in a PDF statement.

Usage (from project root, venv active):
    python scripts/debug_pdf.py <path-to-pdf> [password]

Example:
    python scripts/debug_pdf.py "C:/Users/P Dhiraj/Downloads/statement.pdf" PAND1805
"""
import sys
import io
import pdfplumber

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/debug_pdf.py <pdf_path> [password]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    password = sys.argv[2] if len(sys.argv) > 2 else None

    with open(pdf_path, "rb") as f:
        content = f.read()

    pdf = pdfplumber.open(io.BytesIO(content), password=password)

    with pdf:
        print(f"Total pages: {len(pdf.pages)}\n")
        for page_num, page in enumerate(pdf.pages, start=1):
            print(f"{'='*60}")
            print(f"PAGE {page_num}")
            print(f"{'='*60}")

            # Show tables found
            tables = page.extract_tables()
            print(f"  Tables found by extract_tables(): {len(tables)}")
            for t_idx, table in enumerate(tables):
                print(f"\n  Table {t_idx+1} ({len(table)} rows):")
                for r_idx, row in enumerate(table[:5]):  # show first 5 rows
                    print(f"    Row {r_idx}: {row}")
                if len(table) > 5:
                    print(f"    ... ({len(table)-5} more rows)")

            # Show raw text (first 1500 chars per page)
            text = page.extract_text() or ""
            print(f"\n  Raw text (first 1500 chars):\n")
            print(text[:1500])
            print()

if __name__ == "__main__":
    main()
