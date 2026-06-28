import sys
from pathlib import Path

# Add src/ to Python path so all imports work correctly
sys.path.insert(0, str(Path(__file__).parent))

from api.app import app  # noqa: F401 - imported for uvicorn