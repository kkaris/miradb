import json
from pathlib import Path

EXTRACTION_METHODS_PATH = Path(__file__).parent / "extraction_methods.json"

with EXTRACTION_METHODS_PATH.open("r") as f:
    EXTRACTION_METHODS_INFO = json.load(f)
