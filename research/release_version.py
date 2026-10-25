"""Identify source revisions for newly created training and evaluation records."""
from pathlib import Path
import json
import subprocess


def source_revision():
    root = Path(__file__).resolve().parents[1]
    if (root / '.git').is_dir():
        try:
            return subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                           text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return json.loads((root / 'SOURCE_VERSION.json').read_text(encoding='utf-8'))['release_id']
