"""Filesystem handshake for optional operator hints on four-frame scans.

The navigation process owns request creation and response validation.  The web
UI may only write a response for the currently published request.  No motion
command is represented by this protocol.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image


REQUEST_SCHEMA = "v70_scan_annotation_request_v1"
RESPONSE_SCHEMA = "v70_scan_annotation_response_v1"
HINT_SCHEMA = "v70_operator_visual_hint_v1"


def scan_annotation_required(*, enabled: bool, requests_created: int, max_rounds: int) -> bool:
    """Return whether this scan belongs to the bounded operator phase."""

    if not isinstance(requests_created, int) or requests_created < 0:
        raise ValueError("requests_created must be a nonnegative integer")
    if not isinstance(max_rounds, int) or not 1 <= max_rounds <= 10:
        raise ValueError("max_rounds must be in [1, 10]")
    return bool(enabled and requests_created < max_rounds)


def active_scan_required_for_plan(
    *,
    enabled: bool,
    annotation_enabled: bool,
    requests_created: int,
    max_rounds: int,
    scan_in_progress: bool,
) -> bool:
    """Limit active motion scans to the bounded annotation phase.

    A scan already in progress must be allowed to reach READY and feed its
    plan.  Once the final annotation scan has been consumed, later autonomous
    plans use the normal rolling four-frame history and must not start another
    physical right-to-left sweep.
    """

    if not isinstance(requests_created, int) or requests_created < 0:
        raise ValueError("requests_created must be a nonnegative integer")
    if not isinstance(max_rounds, int) or not 1 <= max_rounds <= 10:
        raise ValueError("max_rounds must be in [1, 10]")
    return bool(
        enabled
        and (
            scan_in_progress
            or not annotation_enabled
            or requests_created < max_rounds
        )
    )


def operator_direction_prior(directions: Sequence[int], *, required: int = 2) -> int:
    """Return a lateral prior only after consistent operator rounds."""

    if required < 2:
        raise ValueError("operator prior requires at least two confirmations")
    normalized = [int(value) for value in directions if int(value) in (-1, 1)]
    if len(normalized) < required:
        return 0
    recent = normalized[-required:]
    return recent[0] if all(value == recent[0] for value in recent) else 0


def autonomous_consensus_with_operator_prior(
    *,
    prior_direction: int,
    visual_bearing: Mapping[str, object],
    required_agreeing_frames: int = 2,
) -> dict[str, Any]:
    """Authorize a relaxed autonomous consensus without accepting opposition."""

    prior = int(prior_direction)
    if prior not in (-1, 0, 1):
        raise ValueError("prior_direction must be -1, 0, or 1")
    if not 2 <= required_agreeing_frames <= 4:
        raise ValueError("required_agreeing_frames must be in [2, 4]")
    frames = visual_bearing.get("frames", [])
    if not isinstance(frames, list):
        frames = []
    agreeing = 0
    opposing = 0
    lateral_valid = 0
    for frame in frames:
        if not isinstance(frame, Mapping) or not bool(frame.get("valid")):
            continue
        direction = int(frame.get("direction", 0))
        if direction not in (-1, 1):
            continue
        lateral_valid += 1
        if direction == prior:
            agreeing += 1
        elif prior != 0:
            opposing += 1
    authorized = bool(
        prior in (-1, 1)
        and agreeing >= required_agreeing_frames
        and opposing == 0
    )
    return {
        "authorized": authorized,
        "prior_direction": prior,
        "agreeing_frames": agreeing,
        "opposing_frames": opposing,
        "lateral_valid_frames": lateral_valid,
        "required_agreeing_frames": required_agreeing_frames,
        "reason": (
            "two_frame_agreement_with_operator_prior"
            if authorized
            else "operator_prior_consensus_not_met"
        ),
    }


def direct_operator_direction_override(hint_direction: int) -> bool:
    """Only a fresh visible annotation may impose a strict direction.

    A direction remembered from the bounded two-round annotation phase is a
    prior, not a permanent operator command.  Later plans must return to
    autonomous inference unless current visual evidence confirms that prior.
    """

    direction = int(hint_direction)
    if direction not in (-1, 0, 1):
        raise ValueError("hint_direction must be -1, 0, or 1")
    return direction in (-1, 1)


def confirmed_prior_after_annotation_phase(
    *,
    operator_hint_visible: bool,
    requests_created: int,
    max_rounds: int,
    prior_consensus: Mapping[str, object],
) -> bool:
    """Use a stored operator prior only with fresh two-frame confirmation."""

    if not isinstance(requests_created, int) or requests_created < 0:
        raise ValueError("requests_created must be a nonnegative integer")
    if not isinstance(max_rounds, int) or not 1 <= max_rounds <= 10:
        raise ValueError("max_rounds must be in [1, 10]")
    return bool(
        not operator_hint_visible
        and requests_created >= max_rounds
        and prior_consensus.get("authorized") is True
    )


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def create_scan_annotation_request(
    *,
    root: Path,
    request_id: str,
    images: Sequence[Image.Image],
    source_sha256: Sequence[str],
    yaw_offsets_rad: Sequence[float],
    timeout_s: float,
) -> dict[str, Any]:
    """Publish four immutable frames, then atomically expose their request."""

    if not request_id or "/" in request_id or request_id in {".", ".."}:
        raise ValueError("invalid annotation request id")
    if not (len(images) == len(source_sha256) == len(yaw_offsets_rad) == 4):
        raise ValueError("annotation request requires exactly four frames")
    timeout = float(timeout_s)
    if not math.isfinite(timeout) or not 10.0 <= timeout <= 900.0:
        raise ValueError("annotation timeout must be in [10, 900] seconds")

    destination = root.expanduser().resolve() / "requests" / request_id
    destination.mkdir(parents=True, exist_ok=False)
    frames: list[dict[str, Any]] = []
    for index, (image, digest, yaw_offset) in enumerate(
        zip(images, source_sha256, yaw_offsets_rad)
    ):
        if not isinstance(image, Image.Image):
            raise TypeError(f"images[{index}] is not a PIL image")
        yaw = float(yaw_offset)
        if not math.isfinite(yaw):
            raise ValueError(f"yaw_offsets_rad[{index}] must be finite")
        filename = f"history_{index}.jpg"
        image.convert("RGB").save(destination / filename, quality=92)
        frames.append(
            {
                "index": index,
                "filename": filename,
                "width": int(image.width),
                "height": int(image.height),
                "source_sha256": str(digest),
                "yaw_offset_rad": yaw,
            }
        )

    created = time.time()
    request: dict[str, Any] = {
        "schema": REQUEST_SCHEMA,
        "request_id": request_id,
        "created_unix": created,
        "expires_unix": created + timeout,
        "frames": frames,
        "response_required_before_planning": True,
        "motion_policy_while_waiting": "zero_velocity",
    }
    _atomic_json(destination / "request.json", request)
    _atomic_json(
        root.expanduser().resolve() / "current.json",
        {"schema": REQUEST_SCHEMA, "request_id": request_id},
    )
    return request


def validated_operator_hint(
    *,
    request: Mapping[str, object],
    response: Mapping[str, object],
    horizontal_fov_rad: float = 1.20,
) -> dict[str, Any]:
    """Validate a UI response and derive a signed bearing hint.

    Positive bearing/direction means left and negative means right, matching
    the V70 planner's lateral convention.  This remains advisory: downstream
    motion safety gates are unchanged.
    """

    request_id = str(request.get("request_id", ""))
    if response.get("schema") != RESPONSE_SCHEMA:
        raise ValueError("annotation response schema mismatch")
    if str(response.get("request_id", "")) != request_id:
        raise ValueError("annotation response request id mismatch")
    visible = response.get("visible")
    if not isinstance(visible, bool):
        raise ValueError("annotation visible must be boolean")
    base: dict[str, Any] = {
        "schema": HINT_SCHEMA,
        "request_id": request_id,
        "visible": visible,
        "source": "operator_scan_annotation",
    }
    if not visible:
        base.update({"direction": 0, "reason": "operator_marked_not_visible"})
        return base

    frame_index = response.get("frame_index")
    if not isinstance(frame_index, int) or not 0 <= frame_index < 4:
        raise ValueError("annotation frame index must be in [0, 3]")
    frames = request.get("frames")
    if not isinstance(frames, list) or len(frames) != 4:
        raise ValueError("annotation request frame contract invalid")
    frame = frames[frame_index]
    if not isinstance(frame, Mapping):
        raise ValueError("annotation request frame metadata invalid")
    bbox = response.get("bbox_norm")
    if not isinstance(bbox, Mapping):
        raise ValueError("visible annotation requires bbox_norm")
    values = {name: float(bbox[name]) for name in ("x", "y", "w", "h")}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("annotation bbox must be finite")
    if (
        values["x"] < 0.0
        or values["y"] < 0.0
        or values["w"] <= 0.0
        or values["h"] <= 0.0
        or values["x"] + values["w"] > 1.000001
        or values["y"] + values["h"] > 1.000001
        or values["w"] * values["h"] < 0.0004
    ):
        raise ValueError("annotation bbox is outside the frame or too small")

    hfov = float(horizontal_fov_rad)
    if not math.isfinite(hfov) or not 0.5 <= hfov <= 2.5:
        raise ValueError("horizontal fov must be in [0.5, 2.5] radians")
    center_x = values["x"] + 0.5 * values["w"]
    frame_yaw = float(frame["yaw_offset_rad"])
    bearing = frame_yaw + (0.5 - center_x) * hfov
    deadband = 0.06
    direction = 1 if bearing > deadband else (-1 if bearing < -deadband else 0)
    base.update(
        {
            "frame_index": frame_index,
            "bbox_norm": values,
            "bbox_center_x_norm": center_x,
            "frame_yaw_offset_rad": frame_yaw,
            "estimated_bearing_rad": bearing,
            "direction": direction,
            "direction_label": (
                "left" if direction > 0 else ("right" if direction < 0 else "center")
            ),
            "reason": "operator_confirmed_visible_target",
        }
    )
    return base


def read_operator_hint(*, root: Path, request: Mapping[str, object]) -> dict[str, Any] | None:
    request_id = str(request.get("request_id", ""))
    response_path = root.expanduser().resolve() / "requests" / request_id / "response.json"
    if not response_path.is_file():
        return None
    response = json.loads(response_path.read_text(encoding="utf-8"))
    if not isinstance(response, dict):
        raise ValueError("annotation response must be a JSON object")
    return validated_operator_hint(request=request, response=response)
