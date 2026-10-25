"""Set dataset, encoder and cached JEV resource paths."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.config import validate_splits, validate_role_bank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/voc.json')
    roots = parser.add_mutually_exclusive_group()
    roots.add_argument('--voc-root')
    roots.add_argument('--coco-root')
    for flag in ['clip-cache', 'dino-repo', 'dino-weights', 'role-bank']:
        parser.add_argument('--' + flag)
    args = parser.parse_args()
    config_file = (ROOT / args.config).resolve()
    cfg = json.loads(config_file.read_text(encoding='utf-8'))
    for dataset in ('voc', 'coco'):
        value = getattr(args, dataset + '_root')
        if value:
            if cfg.get('dataset', 'voc') != dataset:
                parser.error(f'--{dataset}-root requires --config configs/{dataset}.json')
            cfg['data'] = str(Path(value).expanduser().resolve())
    for flag, key in [('clip_cache', 'clip_cache'), ('dino_repo', 'dino_repo'),
                      ('dino_weights', 'dino_weights')]:
        value = getattr(args, flag)
        if value:
            cfg[key] = str(Path(value).expanduser().resolve())
    if args.role_bank:
        bank = Path(args.role_bank).expanduser().resolve()
        cfg['jev_role_bank'] = str(bank)
        cfg['jev_role_bank_sha256'] = hashlib.sha256(bank.read_bytes()).hexdigest()
        validate_role_bank(cfg)
    validate_splits(cfg)
    config_file.write_text(json.dumps(cfg, indent=2) + '\n', encoding='utf-8')
    print(config_file)


if __name__ == '__main__':
    main()
