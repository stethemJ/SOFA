import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = Path(os.environ.get("SOFA_OUTPUT_DIR", PROJECT_ROOT / "outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SKY_MODEL_DEFAULT = "2008"  # "2008" or "2016" -- see foregrounds/tools/Sky.py
