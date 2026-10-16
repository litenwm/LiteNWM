"""Camera-calibration contract for high-resolution latent world models.

Processed JPEGs cannot recover physical intrinsics or camera-to-base
extrinsics.  This module therefore separates known image preprocessing from
physical calibration and represents missing calibration with explicit masks,
never guessed focal lengths.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import torch


CAMERA_FEATURE_DIM = 34


@dataclass(frozen=True)
class CameraCalibrationV1:
    source_width: int
    source_height: int
    intrinsic_matrix: tuple[float, ...] | None = None
    distortion_model: str | None = None
    distortion_coefficients: tuple[float, ...] = ()
    camera_to_base: tuple[float, ...] | None = None
    frame_id: str | None = None
    source: str = "unknown"

    @property
    def intrinsics_valid(self) -> bool:
        return (
            self.intrinsic_matrix is not None
            and len(self.intrinsic_matrix) == 9
            and np.isfinite(self.intrinsic_matrix).all()
        )

    @property
    def extrinsics_valid(self) -> bool:
        return (
            self.camera_to_base is not None
            and len(self.camera_to_base) == 16
            and np.isfinite(self.camera_to_base).all()
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "intrinsics_valid": self.intrinsics_valid,
            "extrinsics_valid": self.extrinsics_valid,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CameraCalibrationV1":
        return cls(
            source_width=int(value["source_width"]),
            source_height=int(value["source_height"]),
            intrinsic_matrix=(
                tuple(map(float, value["intrinsic_matrix"]))
                if value.get("intrinsic_matrix") is not None
                else None
            ),
            distortion_model=value.get("distortion_model"),
            distortion_coefficients=tuple(
                map(float, value.get("distortion_coefficients", ()))
            ),
            camera_to_base=(
                tuple(map(float, value["camera_to_base"]))
                if value.get("camera_to_base") is not None
                else None
            ),
            frame_id=value.get("frame_id"),
            source=str(value.get("source", "unknown")),
        )


def calibration_from_ros_camera_info(
    message: Any,
    *,
    source: str,
    camera_to_base: Sequence[float] | None = None,
) -> CameraCalibrationV1:
    matrix = getattr(message, "k", getattr(message, "K", None))
    distortion = getattr(message, "d", getattr(message, "D", ()))
    header = getattr(message, "header", None)
    frame_id = getattr(header, "frame_id", None)
    return CameraCalibrationV1(
        source_width=int(message.width),
        source_height=int(message.height),
        intrinsic_matrix=tuple(map(float, matrix)),
        distortion_model=str(message.distortion_model),
        distortion_coefficients=tuple(map(float, distortion)),
        camera_to_base=(
            tuple(map(float, camera_to_base))
            if camera_to_base is not None
            else None
        ),
        frame_id=str(frame_id) if frame_id else None,
        source=source,
    )


def imageops_fit_transform(
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    """Map source pixels to ImageOps.fit target pixels.

    The returned 3x3 affine captures center crop followed by anisotropic
    pixel-coordinate scaling.  With equal source/target aspect ratios it is a
    pure scale.
    """

    values = (source_width, source_height, target_width, target_height)
    if any(int(value) <= 0 for value in values):
        raise ValueError("source and target image dimensions must be positive")
    source_aspect = source_width / source_height
    target_aspect = target_width / target_height
    if source_aspect > target_aspect:
        crop_height = float(source_height)
        crop_width = crop_height * target_aspect
        offset_x = 0.5 * (source_width - crop_width)
        offset_y = 0.0
    else:
        crop_width = float(source_width)
        crop_height = crop_width / target_aspect
        offset_x = 0.0
        offset_y = 0.5 * (source_height - crop_height)
    scale_x = target_width / crop_width
    scale_y = target_height / crop_height
    return np.asarray(
        (
            (scale_x, 0.0, -scale_x * offset_x),
            (0.0, scale_y, -scale_y * offset_y),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float32,
    )


def transformed_intrinsic_matrix(
    calibration: CameraCalibrationV1,
    *,
    target_width: int,
    target_height: int,
) -> np.ndarray | None:
    if not calibration.intrinsics_valid:
        return None
    transform = imageops_fit_transform(
        calibration.source_width,
        calibration.source_height,
        target_width,
        target_height,
    )
    source_matrix = np.asarray(
        calibration.intrinsic_matrix,
        dtype=np.float32,
    ).reshape(3, 3)
    return transform @ source_matrix


def camera_feature_vector(
    calibration: CameraCalibrationV1,
    *,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    """Return 34 calibration features with explicit validity flags.

    Layout:
      0:5 normalized fx,fy,cx,cy,skew
      5:13 first eight distortion coefficients
      13:19 preprocessing affine top two rows
      19:22 camera-to-base translation
      22:31 camera-to-base rotation matrix
      31 source aspect / target aspect
      32 intrinsics-valid
      33 extrinsics-valid
    """

    features = np.zeros(CAMERA_FEATURE_DIM, dtype=np.float32)
    transform = imageops_fit_transform(
        calibration.source_width,
        calibration.source_height,
        target_width,
        target_height,
    )
    features[13:19] = transform[:2].reshape(-1)
    target_aspect = target_width / target_height
    features[31] = (
        calibration.source_width / calibration.source_height
    ) / target_aspect
    intrinsic = transformed_intrinsic_matrix(
        calibration,
        target_width=target_width,
        target_height=target_height,
    )
    if intrinsic is not None:
        features[:5] = (
            intrinsic[0, 0] / target_width,
            intrinsic[1, 1] / target_height,
            intrinsic[0, 2] / target_width,
            intrinsic[1, 2] / target_height,
            intrinsic[0, 1] / target_width,
        )
        coefficients = np.asarray(
            calibration.distortion_coefficients[:8],
            dtype=np.float32,
        )
        features[5 : 5 + len(coefficients)] = coefficients
        features[32] = 1.0
    if calibration.extrinsics_valid:
        extrinsic = np.asarray(
            calibration.camera_to_base,
            dtype=np.float32,
        ).reshape(4, 4)
        features[19:22] = extrinsic[:3, 3]
        features[22:31] = extrinsic[:3, :3].reshape(-1)
        features[33] = 1.0
    return features


def camera_feature_tensor(
    calibrations: Sequence[CameraCalibrationV1],
    *,
    target_width: int,
    target_height: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    values = np.stack(
        [
            camera_feature_vector(
                calibration,
                target_width=target_width,
                target_height=target_height,
            )
            for calibration in calibrations
        ],
        axis=0,
    )
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def unknown_camera_calibration(
    *,
    source_width: int,
    source_height: int,
    source: str,
) -> CameraCalibrationV1:
    return CameraCalibrationV1(
        source_width=source_width,
        source_height=source_height,
        source=source,
    )
