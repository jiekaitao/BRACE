using System;
using System.Collections.Generic;
using Newtonsoft.Json;

/// <summary>
/// Top-level response from the BRACE server (one per frame).
/// </summary>
[Serializable]
public class BraceResponse
{
    public int frame_index;
    public float video_time;
    public Dictionary<string, SubjectData> subjects;
    public List<int> active_track_ids;
    public TimingData timing;
}

/// <summary>
/// Per-subject data. Unselected subjects only have label/bbox/phase/selected.
/// Selected subjects include the full analysis payload.
/// </summary>
[Serializable]
public class SubjectData
{
    // --- Always present ---
    public string label;       // "S1", "S2", etc.
    public BBox bbox;          // Normalized [0,1] bounding box
    public string phase;       // "calibrating", "normal", "anomaly"
    public bool selected;      // true if this is the VR-selected subject

    // --- Only present when selected = true ---
    public int n_segments;
    public int n_clusters;
    public int cluster_id;
    public float consistency_score;
    public bool is_anomaly;
    public Dictionary<string, ClusterSummary> cluster_summary;
    public string identity_status;      // "unknown", "tentative", "confirmed"
    public float identity_confidence;
    public float velocity;
    public float rolling_velocity;
    public float fatigue_index;
    public float peak_velocity;
    public QualityData quality;
    public string alert_text;
}

/// <summary>
/// Normalized bounding box in image space.
/// y=0 is TOP of frame (BRACE convention). Unity viewport has y=0 at BOTTOM — flip Y when rendering.
/// </summary>
[Serializable]
public class BBox
{
    public float x1;  // Left   [0,1]
    public float y1;  // Top    [0,1]
    public float x2;  // Right  [0,1]
    public float y2;  // Bottom [0,1]
}

[Serializable]
public class ClusterSummary
{
    public int count;
    public float mean_score;
    public int anomaly_count;
    public string activity_label;
    public float composite_fatigue;
}

/// <summary>
/// Per-frame movement quality metrics from MovementQualityTracker.
/// All fields are optional — the server only sends them when available.
/// </summary>
[Serializable]
public class QualityData
{
    public float form_score;                // 0-100
    public MovementPhase movement_phase;
    public Biomechanics biomechanics;
    public List<InjuryRisk> injury_risks;
    public JointQuality joint_quality;
    public ActiveGuideline active_guideline;
    public FatigueTimeline fatigue_timeline;
    public float concussion_rating;
    public float fatigue_rating;
}

[Serializable]
public class MovementPhase
{
    public string label;       // "ascending", "descending", "transition", etc.
    public float progress;     // 0.0 - 1.0 within current phase
    public int cycle_count;    // Total rep count
}

[Serializable]
public class Biomechanics
{
    public float fppa_left;            // Frontal Plane Projection Angle (degrees)
    public float fppa_right;
    public float hip_drop;             // Pelvic obliquity (degrees)
    public float trunk_lean;           // Trunk deviation (degrees)
    public float asymmetry;            // Bilateral Asymmetry Index (%)
    public float curvature;
    public float jerk;
    public Dictionary<string, float> angular_velocities;
    public float anomaly_score;
    public float com_velocity;         // Center of Mass velocity
    public float com_sway;             // Center of Mass sway
}

[Serializable]
public class InjuryRisk
{
    public string joint;       // e.g., "right_knee", "pelvis", "trunk", "bilateral"
    public string risk;        // e.g., "acl_valgus", "hip_drop", "trunk_lean", "asymmetry"
    public string severity;    // "low", "medium", "high"
    public float value;        // Current metric value
    public float threshold;    // Threshold that was exceeded
}

[Serializable]
public class JointQuality
{
    public List<float> scores;      // 14 scores (one per feature joint), 0-100
    public List<int> degrading;     // Indices of joints with declining quality
    public List<float> deviations;  // Per-joint deviation values
}

[Serializable]
public class ActiveGuideline
{
    public string name;              // e.g., "squat"
    public string display_name;      // e.g., "Squat"
    public List<string> form_cues;   // Coaching cues
}

[Serializable]
public class FatigueTimeline
{
    public List<float> timestamps;
    public List<float> fatigue;
    public List<float> form_scores;
}

[Serializable]
public class TimingData
{
    public float decode_ms;
    public float pipeline_ms;
    public float identity_ms;
    public float analyzer_ms;
    public float total_ms;
}
