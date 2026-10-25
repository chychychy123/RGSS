# Attribution

- Semantic pipeline and inherited code: [ExCEL](https://github.com/zwyang6/ExCEL),
  CVPR 2025, and the user-provided ExCEL-derived implementation.
- Final inference reference: ExCEL commit
  2cd7f510d62793c76365268a2cc4da7cbe915e4d, tools/infer_seg_voc.py.
- COCO split lists, image-level labels, descriptors and 224-cluster embedding bank:
  supplied ExCEL source; category ordering follows its COCO loader.
- Frozen spatial encoder: [DINOv2](https://github.com/facebookresearch/dinov2),
  pinned to 7764ea0f912e53c92e82eb78a2a1631e92725fc8.
- CLIP components: [OpenAI CLIP](https://github.com/openai/CLIP) and inherited
  CLIP feature-processing code. The original OpenAI license is in `clip/LICENSE`.
- NVIDIA-derived segmentation heads retain their source headers and the
  [SegFormer source license](https://github.com/NVlabs/SegFormer/blob/master/LICENSE).
- ExCEL credits [SeCo](https://github.com/zwyang6/SeCo),
  [MoRe](https://github.com/zwyang6/MoRe) and
  [WeCLIP](https://github.com/zbf1991/WeCLIP).

Existing source comments and notices are retained. This archive does not assign
a new blanket license to third-party code. External repositories and pretrained
weights retain their upstream terms.
