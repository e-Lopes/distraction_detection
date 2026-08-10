from pathlib import Path

import pytest
from fase_2.src.data.manifest import build_manifest_rows


def test_manifest_uses_anonymous_ids_and_relative_paths(monkeypatch, tmp_path: Path):
    videos = tmp_path / "fase_2" / "data" / "raw" / "videos"
    videos.mkdir(parents=True)
    (videos / "1.mp4").write_bytes(b"synthetic-not-a-real-video")

    monkeypatch.setattr(
        "fase_2.src.data.manifest.inspect_video",
        lambda _: {
            "fps": 17.0,
            "width": 1920,
            "height": 1080,
            "num_frames": 100,
            "duration_seconds": 100 / 17,
            "metadata_backend": "test",
            "metadata_limitation": "",
        },
    )
    rows = build_manifest_rows(
        [
            {
                "video_id": "video_01",
                "filename": "1.mp4",
                "session_id": "session_01",
                "annotation_version": "v1",
            }
        ],
        videos_dir=videos,
        repository_root=tmp_path,
    )
    assert rows[0]["video_id"] == "video_01"
    assert rows[0]["relative_path"] == "fase_2/data/raw/videos/1.mp4"
    assert str(tmp_path) not in rows[0]["relative_path"]


def test_manifest_rejects_video_outside_repository(monkeypatch, tmp_path: Path):
    outside = tmp_path.parent / "outside-video-test"
    outside.mkdir(exist_ok=True)
    (outside / "1.mp4").write_bytes(b"synthetic")
    monkeypatch.setattr("fase_2.src.data.manifest.inspect_video", lambda _: {})
    with pytest.raises(ValueError):
        build_manifest_rows(
            [
                {
                    "video_id": "video_01",
                    "filename": "1.mp4",
                    "session_id": "session_01",
                    "annotation_version": "v1",
                }
            ],
            videos_dir=outside,
            repository_root=tmp_path,
        )
