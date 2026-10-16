from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from recon_latents.camera_calibration_v1 import (
    CAMERA_FEATURE_DIM,
    CameraCalibrationV1,
    calibration_from_ros_camera_info,
    camera_feature_vector,
    imageops_fit_transform,
    unknown_camera_calibration,
)
from recon_latents.models_highres_dual_calibrated_v1 import (
    CalibratedHighResDualLatentWMConfig,
    CalibratedHighResMultiLayerDualLatentWM,
)


def test_equal_aspect_resize_transforms_intrinsics_exactly() -> None:
    calibration = CameraCalibrationV1(
        source_width=320,
        source_height=240,
        intrinsic_matrix=(200.0, 0.0, 160.0, 0.0, 210.0, 120.0, 0.0, 0.0, 1.0),
        distortion_model="plumb_bob",
        distortion_coefficients=(0.1, -0.2, 0.0, 0.0, 0.01),
        source="unit",
    )
    transform = imageops_fit_transform(320, 240, 448, 336)
    assert np.allclose(transform, np.diag((1.4, 1.4, 1.0)))
    features = camera_feature_vector(
        calibration,
        target_width=448,
        target_height=336,
    )
    assert features.shape == (CAMERA_FEATURE_DIM,)
    assert np.allclose(features[:4], (0.625, 0.875, 0.5, 0.5))
    assert features[32] == 1.0
    assert features[33] == 0.0


def test_ros_camera_info_contract() -> None:
    message = SimpleNamespace(
        width=640,
        height=480,
        k=(400.0, 0.0, 320.0, 0.0, 410.0, 240.0, 0.0, 0.0, 1.0),
        d=(0.1, -0.1, 0.0, 0.0, 0.01),
        distortion_model="plumb_bob",
        header=SimpleNamespace(frame_id="camera_optical"),
    )
    calibration = calibration_from_ros_camera_info(message, source="fake")
    assert calibration.intrinsics_valid
    assert not calibration.extrinsics_valid
    assert calibration.frame_id == "camera_optical"


def test_unknown_calibration_has_exactly_zero_model_effect() -> None:
    torch.manual_seed(8)
    config = CalibratedHighResDualLatentWMConfig(
        semantic_dim=12,
        geometry_dim=8,
        grid_height=2,
        grid_width=3,
        model_dim=16,
        heads=4,
        rollout_blocks=1,
    )
    model = CalibratedHighResMultiLayerDualLatentWM(config).eval()
    semantic = torch.randn(2, 4, 6, 12)
    geometry = torch.randn(2, 4, 6, 8)
    actions = torch.randn(2, 4, 3)
    unknown = unknown_camera_calibration(
        source_width=320,
        source_height=240,
        source="processed_scand_no_camera_info",
    )
    feature = torch.from_numpy(
        camera_feature_vector(
            unknown,
            target_width=448,
            target_height=336,
        )
    ).repeat(2, 1)
    changed_preprocessing = feature.clone()
    changed_preprocessing[:, 13:32] += 100.0
    first = model(semantic, geometry, actions, feature)
    second = model(semantic, geometry, actions, changed_preprocessing)
    assert not first["camera_conditioning_active"].any()
    assert torch.equal(
        first["predicted_semantic_tokens"],
        second["predicted_semantic_tokens"],
    )
    assert torch.equal(
        first["predicted_geometry_tokens"],
        second["predicted_geometry_tokens"],
    )
