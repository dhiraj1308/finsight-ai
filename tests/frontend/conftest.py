"""
Test configuration for frontend tests.

Ensures src/ is on sys.path before any frontend test module is imported,
so that `import frontend.app` resolves correctly when stubs are injected
into sys.modules prior to the import.
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
