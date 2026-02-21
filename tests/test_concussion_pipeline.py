from backend.concussion_pipeline import ConcussionAnalyzer, LiveCollisionDetector, PoseSample, _estimate_impact_location


def test_live_collision_detector_start_and_end():
    detector = LiveCollisionDetector(threshold_ms=7.0, end_ratio=0.6, end_hold_ms=120.0)

    # Prime detector state.
    assert detector.update({"timestamp_ms": 0.0, "head_velocity_ms": 0.0}) is None

    start = detector.update({"timestamp_ms": 16.6, "head_velocity_ms": 8.1, "play_id": "p1", "player_id": "u7"})
    assert start is not None
    assert start["type"] == "collision_start"
    assert start["record_240fps"] is True

    # Stay below end threshold long enough to close event.
    assert detector.update({"timestamp_ms": 80.0, "head_velocity_ms": 2.0}) is None
    end = detector.update({"timestamp_ms": 220.0, "head_velocity_ms": 1.5})
    assert end is not None
    assert end["type"] == "collision_end"
    assert end["stop_after_ms"] == 1000


def test_analyze_pose_samples_high_risk_linear_velocity():
    analyzer = ConcussionAnalyzer(batch_size=60)
    samples: list[PoseSample] = []
    for frame_idx in range(30):
        x = float(frame_idx) * 0.1
        if frame_idx == 12:
            x += 6.0  # sudden impact displacement
        samples.append(
            PoseSample(
                head=(x, 20.0),
                neck=(x, 35.0),
                nose_conf=0.9,
                left_eye_conf=0.8,
                right_eye_conf=0.8,
                left_ear_conf=0.7,
                right_ear_conf=0.7,
            )
        )

    report = analyzer.analyze_pose_samples(
        samples=samples,
        fps=240.0,
        play_id="play-12",
        player_id="player-4",
        scale_m_per_pixel=0.01,
    )

    assert report["impact_detected"] is True
    assert report["risk_level"] == "HIGH"
    assert report["recommendation"] == "EVALUATE NOW"
    assert report["linear_velocity_ms"] > 7.0
    assert report["frame_of_impact"] >= 0


def test_impact_location_side_heuristic():
    location = _estimate_impact_location(
        PoseSample(
            head=(40.0, 20.0),
            neck=(38.0, 34.0),
            nose_conf=0.2,
            left_eye_conf=0.9,
            right_eye_conf=0.1,
            left_ear_conf=0.8,
            right_ear_conf=0.05,
        )
    )
    assert location == "SIDE"
