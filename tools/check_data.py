"""Check complete VOC/COCO lists, image-level labels and data files."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.config import load_config, check_resources, validate_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/voc.json')
    parser.add_argument('--lists-only', action='store_true')
    args = parser.parse_args()
    cfg = load_config(args.config)
    report = validate_splits(cfg) if args.lists_only else check_resources(cfg, data_only=True)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
