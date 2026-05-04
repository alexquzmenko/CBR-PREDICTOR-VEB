from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from rf_project.config import load_yaml
from rf_project.modeling import infer_one


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--meeting-date", required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    result = infer_one(cfg, args.meeting_date)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
