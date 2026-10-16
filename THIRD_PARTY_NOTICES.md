# Licensing scope and third-party notices

The standard CC BY-NC 4.0 text is in [LICENSE.md](LICENSE.md). It applies to
original LiteNWM code and documentation in this repository, except material
identified under another license. Separately distributed project-trained
weights are covered only where their release explicitly states this license.
This notice does not assert ownership of third-party material or replace its
license. Existing copyright, attribution and license notices must be retained.

Commercial use of the project-owned material is not authorized by CC BY-NC
4.0 and requires separate permission from the relevant rights holders.
Third-party permissions must be obtained independently where necessary.

## External runtime components

| Component | Upstream source and license | Scope |
| --- | --- | --- |
| DINOv3 ViT-L/16 pretrained backbone | [Model and DINOv3 license](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m/blob/main/LICENSE.md) | Upstream pretrained weights and associated materials retain Meta's DINOv3 terms. They are not relicensed under this project's CC BY-NC declaration. |
| NoMaD / VisualNav Transformer | [Repository license](https://github.com/robodhruv/visualnav-transformer/blob/main/LICENSE) (MIT) | The minimal external source subset in the companion runtime archive retains its supplied LICENSE and author notices. Pretrained NoMaD assets remain upstream assets; consult the upstream release terms. |
| Diffusion Policy | [Repository](https://github.com/real-stanford/diffusion_policy) (MIT) | The minimal ConditionalUnet1D dependency subset in the companion archive retains its supplied LICENSE, including the Columbia Artificial Intelligence and Robotics Lab notice. |
| Unitree SDK 2 Python | [Repository](https://github.com/unitreerobotics/unitree_sdk2_python) | Installed separately; retain and comply with the license of the version installed. |
| ROS 2, PyTorch, torchvision, Transformers, Diffusers, OpenCV and TensorRT | Their respective upstream distributions | Installed separately, with their own licenses and notices. The LiteNWM license does not replace those terms. |

The source checkout excludes external pretrained weights. The author-prepared
companion archive contains external components as well as project-trained
assets; archive membership is not a grant of a uniform license. Preserve the
upstream licenses when preparing a public asset release, including a copy of
the DINOv3 agreement if distributing DINOv3 materials. Runtime file hashes are
recorded in [robot/assets_manifest.json](robot/assets_manifest.json); hashes
identify files, not redistribution permissions.

## Baselines and datasets

[Navigation World Models (NWM)](https://github.com/facebookresearch/nwm) is an
external research baseline. Its repository declares CC BY-NC 4.0 for code
and model weights. Other baselines keep their own terms; their mention in
research scripts does not relicense them or bundle their models here.

RECON, SCAND and SACSoN are external datasets. The project license does not
grant rights to their images, bags, trajectories or other underlying data.
Obtain data from the respective providers under their access and usage terms.
Included indices and provenance do not replace those terms.

## Paper and website

The [paper](https://arxiv.org/abs/2610.12368) and the independently hosted
[project website](https://litenwm.github.io/) retain their own applicable
terms. This repository's license does not relicense the paper, website
template, institutional logos, or external media.
