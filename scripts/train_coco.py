"""Train RGSS using the complete COCO 2014 training split."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.config import load_config, validate_splits, check_resources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/coco.json')
    parser.add_argument('--run-dir', default='runs/rgss_coco')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve()
    cfg = load_config(config_path)
    if cfg['dataset'] != 'coco':
        raise ValueError('Expected a COCO configuration.')
    coverage = validate_splits(cfg)
    output = Path(args.run_dir).expanduser().resolve()
    command = [sys.executable, '-m', 'research.train', '--config', str(config_path), '--run-dir', str(output)]
    if args.dry_run:
        print(json.dumps({'cwd': str(ROOT), 'command': command, **coverage,
                          'iterations': cfg['iters'], 'batch_size': cfg['batch']}, indent=2))
        return
    if output.exists():
        raise FileExistsError(f'Choose a new training output directory: {output}')
    check_resources(cfg)
    env = os.environ.copy()
    for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
        env.setdefault(key, '4')
    subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == '__main__':
    main()
