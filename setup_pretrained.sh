#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p pretrained third_party
revision=7764ea0f912e53c92e82eb78a2a1631e92725fc8
if [ ! -d third_party/dinov2 ]; then
  git clone https://github.com/facebookresearch/dinov2 third_party/dinov2
  git -C third_party/dinov2 checkout "$revision"
elif [ "$(git -C third_party/dinov2 rev-parse HEAD)" != "$revision" ]; then
  printf '%s\n' "Use DINOv2 revision $revision, or configure your existing source path."
  exit 1
fi
wget -nc -P pretrained https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt
wget -nc -P pretrained https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth
printf '%s\n' \
  '5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f  pretrained/ViT-B-16.pt' \
  '0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73  pretrained/dinov2_vitb14_pretrain.pth' | sha256sum -c -
