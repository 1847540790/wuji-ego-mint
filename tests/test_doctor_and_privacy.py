from __future__ import annotations

from mint.doctor import run_checks
from scripts.privacy_audit import RULES


def test_inference_profile_excludes_training_and_data_packages() -> None:
    names = {check.name for check in run_checks("inference")}
    assert "flask" in names
    assert "accelerate" not in names
    assert "ray" not in names
    assert "ultralytics" not in names


def test_full_profile_covers_training_and_data_runtime() -> None:
    names = {check.name for check in run_checks("full")}
    assert {"flask", "accelerate", "decord", "pyarrow", "ray", "ultralytics"} <= names


def test_privacy_rules_detect_representative_release_blockers() -> None:
    rules = {rule.name: rule.expression for rule in RULES}
    assert rules["private absolute path"].search("/" + "cpfs/private/project")
    assert rules["private object storage"].search("s3" + "://private-bucket/file")
    assert rules["credential assignment"].search("api_" + 'key="not-a-real-key"')
    assert rules["CJK text"].search("private text") is None
