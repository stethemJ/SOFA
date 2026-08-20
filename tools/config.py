import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(os.environ.get("SOFA_OUTPUT_DIR", PROJECT_ROOT / "outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
