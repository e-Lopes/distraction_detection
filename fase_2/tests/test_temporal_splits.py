from dataclasses import replace

from fase_2.src.data.splits import (
    generate_leave_one_video_out,
    validate_split_blocks,
    window_subset,
)

VIDEO_FRAMES = {f"video_{number:02d}": 1_000 for number in range(1, 5)}


def test_generates_four_leave_one_video_out_folds():
    blocks = generate_leave_one_video_out(VIDEO_FRAMES, purge_gap_frames=150)
    assert {block.fold for block in blocks} == {1, 2, 3, 4}
    for fold in range(1, 5):
        fold_blocks = [block for block in blocks if block.fold == fold]
        tests = [block for block in fold_blocks if block.subset == "test"]
        assert len(tests) == 1
        assert tests[0].video_id == f"video_{fold:02d}"
        assert tests[0].start_frame == 0
        assert tests[0].end_frame == 999


def test_train_validation_are_contiguous_blocks_with_exact_purge_gap():
    blocks = generate_leave_one_video_out(VIDEO_FRAMES, purge_gap_frames=150)
    assert validate_split_blocks(blocks, purge_gap_frames=150) == []
    train = next(
        block
        for block in blocks
        if block.fold == 1 and block.video_id == "video_02" and block.subset == "train"
    )
    validation = next(
        block
        for block in blocks
        if block.fold == 1 and block.video_id == "video_02" and block.subset == "validation"
    )
    assert train.end_frame == 649
    assert validation.start_frame == 800
    assert validation.start_frame - train.end_frame - 1 == 150


def test_validator_rejects_temporal_overlap():
    blocks = generate_leave_one_video_out(VIDEO_FRAMES, purge_gap_frames=150)
    validation_index = next(
        index
        for index, block in enumerate(blocks)
        if block.fold == 1 and block.video_id == "video_02" and block.subset == "validation"
    )
    blocks[validation_index] = replace(blocks[validation_index], start_frame=600)
    errors = validate_split_blocks(blocks, purge_gap_frames=150)
    assert any("sobreposição" in error for error in errors)


def test_validator_rejects_insufficient_purge_gap():
    blocks = generate_leave_one_video_out(VIDEO_FRAMES, purge_gap_frames=150)
    validation_index = next(
        index
        for index, block in enumerate(blocks)
        if block.fold == 1 and block.video_id == "video_02" and block.subset == "validation"
    )
    blocks[validation_index] = replace(blocks[validation_index], start_frame=700)
    errors = validate_split_blocks(blocks, purge_gap_frames=150)
    assert any("purge gap insuficiente" in error for error in errors)


def test_window_crossing_split_or_purge_is_unassigned():
    blocks = generate_leave_one_video_out(VIDEO_FRAMES, purge_gap_frames=150)
    assert (
        window_subset(video_id="video_02", start_frame=640, end_frame=660, blocks=blocks, fold=1)
        is None
    )
    assert (
        window_subset(video_id="video_02", start_frame=800, end_frame=829, blocks=blocks, fold=1)
        == "validation"
    )


def test_overlapping_windows_from_same_segment_stay_in_same_subset():
    blocks = generate_leave_one_video_out(VIDEO_FRAMES, purge_gap_frames=150)
    first = window_subset(
        video_id="video_02", start_frame=100, end_frame=149, blocks=blocks, fold=1
    )
    overlapping = window_subset(
        video_id="video_02", start_frame=115, end_frame=164, blocks=blocks, fold=1
    )
    assert first == overlapping == "train"


def test_test_video_never_appears_in_development():
    blocks = generate_leave_one_video_out(VIDEO_FRAMES, purge_gap_frames=150)
    for fold in range(1, 5):
        test_video = next(
            block.video_id for block in blocks if block.fold == fold and block.subset == "test"
        )
        assert not any(
            block.fold == fold
            and block.video_id == test_video
            and block.subset in {"train", "validation"}
            for block in blocks
        )
