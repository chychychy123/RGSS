"""Evaluate a trained RGSS checkpoint on the complete COCO validation split."""
from pathlib import Path
import argparse
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.config import load_config, check_resources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--checkpoint', default='best.pt')
    parser.add_argument('--scales', nargs='+', type=float, default=[.7, 1., 1.2, 1.5])
    parser.add_argument('--no-crf', action='store_true')
    parser.add_argument('--no-flip', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    run, output = Path(args.run).expanduser().resolve(), Path(args.out).expanduser().resolve()
    command = [sys.executable, '-m', 'research.final_evaluate', '--run', str(run),
               '--checkpoint', args.checkpoint, '--out', str(output), '--recipe', 'excel_official',
               '--scales', *map(str, args.scales)]
    if not args.no_flip:
        command.append('--flip')
    if not args.no_crf:
        command.append('--crf')
    if args.dry_run:
        print(json.dumps({'cwd': str(ROOT), 'command': command, 'dataset': 'coco', 'validation_images': 40137}, indent=2))
        return
    if output.exists():
        raise FileExistsError(f'Choose a new evaluation output directory: {output}')
    cfg = load_config(run / 'config.json')
    if cfg['dataset'] != 'coco':
        raise ValueError('Expected a COCO run.')
    check_resources(cfg, for_evaluation=True)
    checkpoint = run / args.checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
