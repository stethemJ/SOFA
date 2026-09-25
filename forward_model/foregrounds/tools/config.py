from pathlib import Path

from forward_model.tools.config import PROJECT_ROOT

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TE_MAP_PATH = PROJECT_ROOT / "data" / "te_map.fits"
EM_MAP_PATH = PROJECT_ROOT / "data" / "em_map.fits"
