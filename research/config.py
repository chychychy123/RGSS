"""Portable dataset configuration and complete VOC/COCO split validation."""
from pathlib import Path
import hashlib
import json
import re
from functools import lru_cache

ROOT = Path(__file__).resolve().parents[1]
PATH_KEYS = ('data', 'clip_cache', 'dino_repo', 'dino_weights', 'train_ids',
             'val_ids', 'class_labels', 'jev_role_bank', 'attribute_json')


def dataset_spec(config):
    dataset = config.get('dataset', 'voc')
    if dataset not in ('voc', 'coco'):
        raise ValueError(f'Unsupported dataset: {dataset}')
    return _dataset_spec(dataset)


@lru_cache(maxsize=2)
def _dataset_spec(dataset):
    return json.loads((ROOT / 'datasets' / dataset / 'metadata.json').read_text(encoding='utf-8'))


def normalize_config(config):
    config = dict(config)
    config.setdefault('dataset', 'voc')
    spec = dataset_spec(config)
    for key in ('num_classes', 'dataset_name', 'attribute_clusters', 'attribute_json',
                'train_images', 'val_images', 'val_masks'):
        config.setdefault(key, spec[key])
    for key in ('num_classes', 'dataset_name', 'attribute_clusters'):
        if config[key] != spec[key]:
            raise ValueError(f'{key} does not match the {config["dataset"]} dataset.')
    config.setdefault('class_labels', f'datasets/{config["dataset"]}/cls_labels_onehot.npy')
    for key in PATH_KEYS:
        if key in config and config[key]:
            path = Path(config[key]).expanduser()
            config[key] = str((ROOT / path).resolve() if not path.is_absolute() else path.resolve())
    return config


def load_config(path):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return normalize_config(json.loads(path.read_text(encoding='utf-8')))


def read_ids(path, dataset='voc'):
    ids = Path(path).read_text(encoding='utf-8').split()
    if not ids or len(set(ids)) != len(ids):
        raise ValueError(f'Empty list or duplicate image IDs: {path}')
    pattern = r'\d{4}_\d{6}' if dataset == 'voc' else r'COCO_(train|val)2014_\d{12}'
    if any(re.fullmatch(pattern, name) is None for name in ids):
        raise ValueError(f'Invalid {dataset} image identifier in {path}')
    return ids


def validate_splits(config, check_labels=True):
    cfg = normalize_config(config)
    spec = dataset_spec(cfg)
    train, val = (read_ids(cfg[key], cfg['dataset']) for key in ('train_ids', 'val_ids'))
    for ids, split in ((train, 'train'), (val, 'val')):
        canonical = read_ids(ROOT / spec[f'{split}_ids'], cfg['dataset'])
        if len(ids) != spec[f'{split}_count'] or set(ids) != set(canonical):
            raise ValueError(f'Use the complete {cfg["dataset"]} {split} split ({len(canonical):,} images).')
    numeric = lambda names: {n.rsplit('_', 1)[-1] for n in names}
    overlap = set(train) & set(val)
    if cfg['dataset'] == 'coco':
        overlap |= numeric(train) & numeric(val)
    if overlap:
        raise ValueError('Training and validation image IDs overlap.')
    if check_labels:
        import numpy as np
        labels = np.load(cfg['class_labels'], allow_pickle=True).item()
        for name in train:
            if name not in labels:
                raise ValueError(f'Missing image-level label: {name}')
            value = np.asarray(labels[name])
            if value.shape != (cfg['num_classes'] - 1,) or not np.isin(value, [0, 1]).all():
                raise ValueError(f'Invalid {cfg["num_classes"] - 1}-class image label: {name}')
    return {'dataset': cfg['dataset'], 'num_classes': cfg['num_classes'],
            'train_images': len(train), 'val_images': len(val),
            'overlapping_images': 0, 'training_split': spec['training_split']}


def validate_voc_splits(config, check_labels=True):
    if config.get('dataset', 'voc') != 'voc':
        raise ValueError('Expected a VOC configuration.')
    return validate_splits(config, check_labels)


def image_path(config, name, split='val'):
    spec = dataset_spec(config)
    template = config.get(f'{split}_images', spec[f'{split}_images'])
    return Path(config['data']) / template.format(name=name, image_id=name.rsplit('_', 1)[-1])


def mask_path(config, name):
    spec = dataset_spec(config)
    template = config.get('val_masks', spec['val_masks'])
    return Path(config['data']) / template.format(name=name, image_id=name.rsplit('_', 1)[-1])


def load_mask(config, name):
    import numpy as np
    from PIL import Image
    label = np.asarray(Image.open(mask_path(config, name)))
    classes = config.get('num_classes', dataset_spec(config)['num_classes'])
    if label.ndim != 2 or not np.isin(label, [*range(classes), 255]).all():
        raise ValueError(f'Expected indexed masks: background 0, classes 1..{classes-1}, ignore 255: {name}')
    return label


def validate_role_bank(config):
    """Reject missing caches and wrong class order before constructing the model."""
    import numpy as np
    cfg = normalize_config(config)
    path = Path(cfg['jev_role_bank'])
    if not path.is_file():
        raise FileNotFoundError(f'{path}: import the {cfg["dataset"]} JEV cache with tools/configure.py --config configs/{cfg["dataset"]}.json --role-bank /path/to/role_bank.json')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != cfg.get('jev_role_bank_sha256'):
        raise ValueError('JEV cache checksum mismatch; register it with tools/configure.py --role-bank.')
    bank = json.loads(path.read_text(encoding='utf-8'))
    if bank.get('classes') != dataset_spec(cfg)['classes']:
        raise ValueError('The JEV role cache class order does not match this dataset.')
    descriptors = bank.get('descriptors', [])
    if not descriptors or not all(isinstance(t, str) and t.strip() for t in descriptors):
        raise ValueError('The JEV cache must contain nonempty text descriptors.')
    shape = (cfg['num_classes'] ** 2, len(descriptors))
    for key in ('shared_weights', 'unique_weights'):
        values = np.asarray(bank.get(key), dtype=np.float32)
        if values.shape != shape or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f'Invalid {key}; expected finite nonnegative weights of shape {shape}.')
    return bank


def check_resources(config, data_only=False, for_evaluation=False):
    cfg = normalize_config(config)
    report = validate_splits(cfg, check_labels=not for_evaluation)
    paths = []
    splits = ('val',) if for_evaluation else ('train', 'val')
    for split in splits:
        paths.extend(image_path(cfg, n, split) for n in read_ids(cfg[f'{split}_ids'], cfg['dataset']))
    paths.extend(mask_path(cfg, n) for n in read_ids(cfg['val_ids'], cfg['dataset']))
    if cfg['dataset'] == 'voc':
        official = Path(cfg['data']) / 'ImageSets/Segmentation/val.txt'
        paths.append(official)
        if official.is_file() and read_ids(official) != read_ids(cfg['val_ids']):
            raise ValueError('VOC validation ID order differs from the official val.txt.')
    if not data_only:
        attr = f'{cfg["dataset_name"]}_desc_clip_ViT-B-16_gpt4.0_cluster_{cfg["attribute_clusters"]}_embedding_bank.pth'
        paths += [Path(cfg['clip_cache']) / 'ViT-B-16.pt', Path(cfg['dino_weights']),
                  Path(cfg['dino_repo']) / 'hubconf.py', Path(cfg['attribute_json']),
                  ROOT / 'attributes_text' / attr]
        if cfg.get('jev_roles', False) and not for_evaluation:
            validate_role_bank(cfg)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} resources are missing. First paths:\n' + '\n'.join(missing[:8]))
    return report
