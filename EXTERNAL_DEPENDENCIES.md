# External dependencies

Project-owned material uses CC BY-NC 4.0 as described in README.md and
LICENSE.md. External components retain their upstream terms; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The Git source tree does not contain pretrained backbone weights or external
proposal/baseline models. The separately downloadable Go2 runtime archive
includes pinned DINOv3/NoMaD assets and their applicable third-party notices;
its license scope is described in THIRD_PARTY_NOTICES.md.

The mainline expects a locally obtained DINOv3-L/16 checkpoint compatible with
the identifier `facebook/dinov3-vitl16-pretrain-lvd1689m`. Users are
responsible for obtaining it and complying with its upstream license and
access terms.

RECON and SCAND data are also external. Only trajectory-level split metadata
and derived non-image metadata are included here.

The Go2 main-method runtime does not require OpenTrackVLA. The author-provided asset TAR supplies the pinned minimal NoMaD/VisualNav and diffusion-policy sources with their original licenses. ROS, the Unitree Python SDK, board PyTorch/torchvision and optional TensorRT must be installed for the target environment; see robot/README.md.
