"""Evaluate the trained model's CAM seeds on the official VOC training split."""
from pathlib import Path
import argparse
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.config import read_ids, load_config, check_resources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--checkpoint', default='best.pt')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    run, output = Path(args.run).expanduser().resolve(), Path(args.out).expanduser().resolve()
    ids = ROOT / 'datasets/voc/train.txt'
    command = [sys.executable, '-m', 'supplementary.cam_eval', '--code-root', str(ROOT),
               '--config', str(run / 'config.json'), '--checkpoint', str(run / args.checkpoint),
               '--ids', str(ids), '--visual-ids', str(ROOT / 'semantic_resources/visual_ids.json'),
               '--out', str(output), '--family', 'ours', '--name', 'RGSS']
    if args.dry_run:
        print(json.dumps({'cwd': str(ROOT), 'command': command, 'CAM_evaluation_images': len(read_ids(ids))}, indent=2))
        return
    if output.exists():
        raise FileExistsError(f'Choose a new CAM output directory: {output}')
    cfg = load_config(run / 'config.json')
    if cfg['dataset'] != 'voc':
        raise ValueError('CAM evaluation requires a VOC run.')
    check_resources(cfg)
    if not (run / args.checkpoint).is_file():
        raise FileNotFoundError(run / args.checkpoint)
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
