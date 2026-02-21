import numpy as np
from movement_quality import MovementQualityTracker

def test_concussion_and_fatigue():
    tracker = MovementQualityTracker(fps=30.0)
    
    # Simulate an intense sudden head movement (jerk)
    # 14 joints, dimension 2 or 3
    # Joints: 0 is left shoulder, 1 is right shoulder 
    
    # Frame 1: Base pose
    srp_joints_base = np.zeros((14, 2))
    srp_joints_base[0] = [0.0, 0.0]
    srp_joints_base[1] = [1.0, 0.0]
    
    # Feed base frames to stabilize
    for _ in range(5):
        tracker.process_frame(
            srp_joints=srp_joints_base,
            cluster_id=1,
            seg_info=None,
            representative_joints=None,
            fatigue_index=0.2, # 20% fatigue
            video_time=0.0,
            raw_joints=srp_joints_base
        )
    
    # Frame 6: Sudden huge jump in shoulder/head position (simulating head jerk)
    srp_joints_jerk = np.zeros((14, 2))
    srp_joints_jerk[0] = [0.0, 5.0]
    srp_joints_jerk[1] = [1.0, 5.0]
    
    tracker.process_frame(
        srp_joints=srp_joints_jerk,
        cluster_id=1,
        seg_info=None,
        representative_joints=None,
        fatigue_index=0.8, # 80% fatigue
        video_time=0.2,
        raw_joints=srp_joints_jerk
    )
    
    quality = tracker.get_frame_quality()
    
    print("Concussion Rating:", quality.get("concussion_rating"))
    print("Fatigue Rating:", quality.get("fatigue_rating"))
    
    assert quality["concussion_rating"] > 0.0, "Concussion rating should spike on sudden jerk"
    assert quality["fatigue_rating"] == 80.0, "Fatigue rating should reflect 80% fatigue"
    
if __name__ == "__main__":
    test_concussion_and_fatigue()
    print("Tests passed.")
