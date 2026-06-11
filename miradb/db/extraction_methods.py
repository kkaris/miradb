import json
from pathlib import Path

EXTRACTION_METHODS_PATH = Path(__file__).parent / "extraction_methods.json"

with EXTRACTION_METHODS_PATH.open("r") as f:
    json_data = json.load(f)
    EXTRACTION_METHODS_INFO = json_data["extraction_methods"]
    EXTRACTION_METHODS_PRIORITY = json_data["extraction_method_priority"]
    EXTRACTION_METHOD_LABELS = {
        m["extraction_method"]: m["label"] for m in EXTRACTION_METHODS_INFO
    }
