from __future__ import annotations

import os
from pathlib import Path

import pytest

from mint.viewer.app import Job, ViewerService, create_app


def make_service(tmp_path: Path) -> ViewerService:
    samples = tmp_path / "samples"
    samples.mkdir()
    (samples / "ego4d-kitchen-demo.mp4").write_bytes(b"public-sample")
    return ViewerService(
        samples_dir=samples,
        artifacts_dir=tmp_path / "artifacts",
        checkpoint="checkpoint.safetensors",
        config="config.yaml",
        devices="cpu",
        max_frames=16,
        target_fps=8.0,
    )


def test_viewer_lists_only_opaque_sample_ids(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    client = create_app(service).test_client()

    response = client.get("/api/samples")
    payload = response.get_json()

    assert response.status_code == 200
    assert len(payload) == 1
    assert set(payload[0]) == {"id", "title", "collection", "size_mb"}
    assert "/" not in payload[0]["id"]
    assert "path" not in payload[0]


def test_viewer_has_no_ground_truth_routes(tmp_path: Path) -> None:
    app = create_app(make_service(tmp_path))
    rules = {rule.rule for rule in app.url_map.iter_rules()}

    assert not any("ground" in rule or "truth" in rule or "compare" in rule for rule in rules)
    status = app.test_client().get("/api/status").get_json()
    assert status["mode"] == "Prediction only"
    assert status["ground_truth"] is False


def test_artifact_route_rejects_unlisted_names(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    job = Job(id="job123", sample_id=next(iter(service.samples)), status="completed")
    service.jobs[job.id] = job
    target = service.artifacts_dir / job.id
    target.mkdir(parents=True)
    (target / "summary.json").write_text("{}", encoding="utf-8")
    (target / "secret.txt").write_text("private", encoding="utf-8")
    client = create_app(service).test_client()

    assert client.get(f"/artifacts/{job.id}/summary.json").status_code == 200
    assert client.get(f"/artifacts/{job.id}/secret.txt").status_code == 404


def test_sample_symlink_cannot_escape_approved_directory(tmp_path: Path) -> None:
    if not hasattr(os, "symlink"):
        pytest.skip("Symbolic links are unavailable on this platform")
    service = make_service(tmp_path)
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"outside")
    link = service.samples_dir / "linked.mp4"
    link.symlink_to(outside)
    service.samples = service._scan_samples()
    linked_id = next(sample_id for sample_id, sample in service.samples.items() if sample.path == link)

    with pytest.raises(PermissionError, match="escaped"):
        service.get_sample(linked_id)
