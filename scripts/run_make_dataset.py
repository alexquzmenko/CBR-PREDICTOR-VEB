from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from rf_project.config import load_yaml
from rf_project.pipeline import run_make_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    manifest = run_make_dataset(cfg, args.version)
    manifest_path = Path(cfg["paths"]["processed_dir"]) / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Dataset built successfully: {manifest_path}")


if __name__ == "__main__":
    main()
