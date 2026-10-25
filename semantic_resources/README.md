# Cached JEV roles

`voc_role_bank.json` contains the VOC cache. A COCO cache is not included in this release.

A cache JSON has four required fields:

- `classes`: foreground class names in exactly the order of `datasets/<dataset>/metadata.json` (ExCEL naming aliases are retained).
- `descriptors`: a list of nonempty descriptor strings, with length `D`.
- `shared_weights`: a nonnegative finite matrix with shape `C*C × D`.
- `unique_weights`: a nonnegative finite matrix with shape `C*C × D`.

Here `C` includes background: 21 for VOC and 81 for COCO. Pair `(anchor, alternative)` is stored at row `C * anchor + alternative`, with background ID 0. Weight columns follow `descriptors`; use the weights from the corresponding JEV preprocessing. The loader never generates or substitutes role assignments.

Register an existing cache and its SHA-256 checksum:

```bash
python tools/configure.py --config configs/coco.json --role-bank /path/to/coco_role_bank.json
```

The importer checks class order, matrix dimensions, finite nonnegative weights, and the checksum. Cached role evidence is used during training; checkpoint inference does not load a JEV cache.
