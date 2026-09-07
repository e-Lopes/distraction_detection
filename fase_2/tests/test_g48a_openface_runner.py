from fase_2.scripts.g48a_run_openface_all_frames import _csv_bool, _valid_openface_rows


def test_openface_image_rows_without_success_are_valid():
    rows = [{"confidence": "0.8", "x_0": "10"}]
    assert _valid_openface_rows(rows) == rows


def test_openface_tracking_rows_respect_success_flag():
    rows = [
        {"success": "0", "confidence": "0.9"},
        {"success": "1", "confidence": "0.7"},
    ]
    assert _valid_openface_rows(rows) == [rows[1]]


def test_csv_bool_handles_serialized_values():
    assert _csv_bool("True") and _csv_bool(1)
    assert not _csv_bool("False") and not _csv_bool(0)
