import json

from tools.scoring.run_train_dualdomain_cem_tail_v66_frozen import (
    CORRECTION,
    write_json_with_candidate_level_seal,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256


def test_training_result_records_candidate_level_correction(tmp_path) -> None:
    output = tmp_path / "result.json"
    write_json_with_candidate_level_seal(
        output, {"schema": "dualdomain_cem_tail_v66_training_result"}
    )
    value = json.loads(output.read_text(encoding="utf-8"))
    assert value["calibration_decision"].startswith("candidate-level safety")
    assert value["candidate_level_calibration_correction"] == str(CORRECTION)
    assert value["candidate_level_calibration_correction_sha256"] == file_sha256(
        CORRECTION
    )
