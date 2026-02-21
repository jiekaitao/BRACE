# Quest 3 VR Integration Guide for BRACE

This document provides comprehensive instructions for building a Meta Quest 3 Unity app that connects to the BRACE real-time movement analysis server. The Quest 3 captures passthrough camera frames, sends them to a GPU server over the network, and renders bounding boxes + biomechanics stats in mixed reality.

---

## 1. Project Setup

### Unity Version & Build Target
- **Unity 6** (6000.x LTS) recommended
- **Build Target**: Android
- **Minimum API Level**: 32 (Android 12L) — required for CPU camera image capture
- **Scripting Backend**: IL2CPP (required for Quest)
- **Target Architecture**: ARM64

### Required Packages

Install via Unity Package Manager:

| Package | ID | Purpose |
|---|---|---|
| Meta OpenXR | `com.unity.xr.meta-openxr` ≥2.2 | OpenXR backend + passthrough camera |
| OpenXR Plugin | `com.unity.xr.openxr` | Base OpenXR support |
| XR Management | `com.unity.xr.management` | XR plugin management |
| AR Foundation | `com.unity.xr.arfoundation` | Camera manager abstraction |
| NativeWebSocket | git URL (see below) | WebSocket client |
| Newtonsoft JSON | `com.unity.nuget.newtonsoft-json` | JSON deserialization |
| TextMeshPro | `com.unity.textmeshpro` | UI text rendering |

**NativeWebSocket install** — add to `Packages/manifest.json`:
```json
"com.endel.nativewebsocket": "https://github.com/endel/NativeWebSocket.git#upm"
```

### Project Settings

1. **XR Plug-in Management** → Android tab → Enable **OpenXR**
2. **OpenXR** → Android tab → Enable **Meta Quest: Camera (Passthrough)** feature
3. Click gear icon on Camera feature → Enable **Camera Image Support**
4. **Graphics**: Set to **Vulkan** (required for GPU camera capture)
5. **Player Settings**: Set **Minimum API Level** to 32

### Android Manifest Permissions

Add to `Assets/Plugins/Android/AndroidManifest.xml`:
```xml
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="horizonos.permission.HEADSET_CAMERA" />
<uses-permission android:name="com.oculus.permission.USE_SCENE" />
```

---

## 2. Camera Frame Capture

### Option A: WebCamTexture (Simplest)

```csharp
using UnityEngine;

public class FrameCapture : MonoBehaviour
{
    private WebCamTexture _webcam;
    private Texture2D _readTex;

    void Start()
    {
        // Find the Quest passthrough camera
        foreach (var device in WebCamTexture.devices)
        {
            if (device.isFrontFacing)
            {
                _webcam = new WebCamTexture(device.name, 1280, 960, 30);
                _webcam.Play();
                break;
            }
        }
    }

    /// <summary>
    /// Capture current frame as JPEG bytes.
    /// Call at your desired send rate (e.g., 15-30 fps).
    /// </summary>
    public byte[] CaptureJpeg(int quality = 65)
    {
        if (_webcam == null || !_webcam.isPlaying) return null;

        if (_readTex == null || _readTex.width != _webcam.width || _readTex.height != _webcam.height)
        {
            _readTex = new Texture2D(_webcam.width, _webcam.height, TextureFormat.RGBA32, false);
        }

        _readTex.SetPixels32(_webcam.GetPixels32());
        _readTex.Apply();

        // Downscale to 480p for bandwidth efficiency
        int targetH = 480;
        int targetW = Mathf.RoundToInt((float)_readTex.width / _readTex.height * targetH);
        RenderTexture rt = RenderTexture.GetTemporary(targetW, targetH);
        Graphics.Blit(_readTex, rt);
        RenderTexture prev = RenderTexture.active;
        RenderTexture.active = rt;
        Texture2D scaled = new Texture2D(targetW, targetH, TextureFormat.RGB24, false);
        scaled.ReadPixels(new Rect(0, 0, targetW, targetH), 0, 0);
        scaled.Apply();
        RenderTexture.active = prev;
        RenderTexture.ReleaseTemporary(rt);

        byte[] jpeg = scaled.EncodeToJPG(quality);
        Destroy(scaled);
        return jpeg;
    }

    void OnDestroy()
    {
        if (_webcam != null && _webcam.isPlaying) _webcam.Stop();
    }
}
```

### Option B: AR Foundation (Better Performance)

Use `ARCameraManager.TryAcquireLatestCpuImage()` for lower-latency capture. See Meta's [PassthroughCameraApiSamples](https://github.com/oculus-samples/Unity-PassthroughCameraApiSamples) for reference.

---

## 3. WebSocket Connection

### Connection URL

```
wss://baby-gator.tailea0e34.ts.net:8443/ws/analyze?mode=webcam&client=vr
```

- `mode=webcam` — live frame streaming mode
- `client=vr` — VR-optimized response format (minimal bandwidth)

For local network (no TLS):
```
ws://<server-ip>:8001/ws/analyze?mode=webcam&client=vr
```

### WebSocket Manager

```csharp
using System;
using System.Text;
using NativeWebSocket;
using Newtonsoft.Json;
using UnityEngine;

public class BraceWebSocket : MonoBehaviour
{
    [SerializeField] private string serverUrl = "wss://baby-gator.tailea0e34.ts.net:8443/ws/analyze?mode=webcam&client=vr";

    private WebSocket _ws;
    private FrameCapture _capture;

    // Latest parsed response (read by rendering code)
    public BraceResponse LatestResponse { get; private set; }

    // Frame send throttle
    private float _lastSendTime;
    private const float SEND_INTERVAL = 1f / 30f; // 30 fps max
    private int _inFlight;
    private const int MAX_IN_FLIGHT = 5;

    async void Start()
    {
        _capture = GetComponent<FrameCapture>();

        _ws = new WebSocket(serverUrl);
        _ws.OnOpen += () => Debug.Log("[BRACE] Connected");
        _ws.OnClose += (code) => Debug.Log($"[BRACE] Disconnected: {code}");
        _ws.OnError += (err) => Debug.LogError($"[BRACE] Error: {err}");
        _ws.OnMessage += OnMessage;

        await _ws.Connect();
    }

    void Update()
    {
#if !UNITY_WEBGL || UNITY_EDITOR
        _ws?.DispatchMessageQueue();
#endif

        // Send frames at throttled rate
        if (_ws?.State == WebSocketState.Open
            && Time.time - _lastSendTime >= SEND_INTERVAL
            && _inFlight < MAX_IN_FLIGHT)
        {
            SendFrame();
        }
    }

    private async void SendFrame()
    {
        byte[] jpeg = _capture.CaptureJpeg(65);
        if (jpeg == null) return;

        // Pack: [8-byte float64 timestamp (little-endian)] + [JPEG bytes]
        double videoTime = Time.timeAsDouble;
        byte[] timeBytes = BitConverter.GetBytes(videoTime);
        if (!BitConverter.IsLittleEndian) Array.Reverse(timeBytes);

        byte[] message = new byte[8 + jpeg.Length];
        Buffer.BlockCopy(timeBytes, 0, message, 0, 8);
        Buffer.BlockCopy(jpeg, 0, message, 8, jpeg.Length);

        _inFlight++;
        _lastSendTime = Time.time;
        await _ws.Send(message);
    }

    private void OnMessage(byte[] data)
    {
        _inFlight = Mathf.Max(0, _inFlight - 1);

        string json = Encoding.UTF8.GetString(data);
        try
        {
            LatestResponse = JsonConvert.DeserializeObject<BraceResponse>(json);
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[BRACE] Parse error: {e.Message}");
        }
    }

    /// <summary>
    /// Select a subject for detailed analysis data.
    /// Pass null to deselect.
    /// </summary>
    public async void SelectSubject(int? subjectId)
    {
        if (_ws?.State != WebSocketState.Open) return;

        string msg = subjectId.HasValue
            ? $"{{\"type\":\"select_subject\",\"subject_id\":{subjectId.Value}}}"
            : "{\"type\":\"select_subject\",\"subject_id\":null}";

        await _ws.SendText(msg);
    }

    async void OnDestroy()
    {
        if (_ws != null) await _ws.Close();
    }
}
```

---

## 4. C# Data Classes (VR Response Schema)

```csharp
using System;
using System.Collections.Generic;
using Newtonsoft.Json;

[Serializable]
public class BraceResponse
{
    public int frame_index;
    public float video_time;
    public Dictionary<string, SubjectData> subjects;
    public List<int> active_track_ids;
    public TimingData timing;
}

[Serializable]
public class SubjectData
{
    public string label;           // "S1", "S2", etc.
    public BBox bbox;              // Normalized [0,1] bounding box
    public string phase;           // "calibrating", "normal", "anomaly"
    public bool selected;          // true if this is the VR-selected subject

    // --- Fields below only present when selected = true ---
    public int n_segments;
    public int n_clusters;
    public int cluster_id;
    public float consistency_score;
    public bool is_anomaly;
    public Dictionary<string, ClusterSummary> cluster_summary;
    public string identity_status;  // "unknown", "tentative", "confirmed"
    public float identity_confidence;
    public float velocity;
    public float rolling_velocity;
    public float fatigue_index;
    public float peak_velocity;
    public QualityData quality;
    public string alert_text;
}

[Serializable]
public class BBox
{
    public float x1;  // Left edge [0,1]
    public float y1;  // Top edge [0,1]  — NOTE: y=0 is TOP (flip for Unity)
    public float x2;  // Right edge [0,1]
    public float y2;  // Bottom edge [0,1]
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

[Serializable]
public class QualityData
{
    public int form_score;           // 0-100
    public MovementPhase movement_phase;
    public Biomechanics biomechanics;
    public List<InjuryRisk> injury_risks;
    public JointQuality joint_quality;
    public ActiveGuideline active_guideline;
    public FatigueTimeline fatigue_timeline;
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
    public float fppa_left;    // Frontal Plane Projection Angle (degrees)
    public float fppa_right;
    public float hip_drop;     // Pelvic obliquity (degrees)
    public float trunk_lean;   // Trunk deviation (degrees)
    public float asymmetry;    // Bilateral Asymmetry Index (%)
}

[Serializable]
public class InjuryRisk
{
    public string joint;       // e.g., "right_knee"
    public string risk;        // e.g., "Knee Valgus"
    public string severity;    // "low", "medium", "high"
    public float value;        // Current angle/metric value
    public float threshold;    // Threshold that was exceeded
}

[Serializable]
public class JointQuality
{
    public List<int> scores;     // 14 scores (one per feature joint), 0-100
    public List<int> degrading;  // Indices of joints with declining quality
}

[Serializable]
public class ActiveGuideline
{
    public string name;            // e.g., "squat"
    public string display_name;    // e.g., "Squat"
    public List<string> form_cues; // Coaching cues
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
```

---

## 5. Bounding Box Rendering

### Coordinate Conversion

BRACE bounding boxes are normalized [0,1] in image space where **y=0 is the top** of the frame. Unity's viewport has **y=0 at the bottom**. You must flip Y.

```csharp
using UnityEngine;

public class BoundingBoxRenderer : MonoBehaviour
{
    [SerializeField] private float assumedDepth = 3f; // meters from camera
    [SerializeField] private Color unselectedColor = new Color(0.6f, 0.6f, 0.6f); // gray
    [SerializeField] private Color selectedColor = new Color(0.345f, 0.8f, 0.008f); // #58CC02 green
    [SerializeField] private Color hoverColor = new Color(1f, 0.84f, 0f); // gold highlight

    private Camera _cam;
    private Dictionary<string, BoxVisual> _boxes = new();

    // EMA smoothing factor (0 = no smoothing, 1 = no update)
    private const float SMOOTH_ALPHA = 0.3f;

    void Start()
    {
        _cam = Camera.main;
    }

    public void UpdateBoxes(Dictionary<string, SubjectData> subjects, List<int> activeIds)
    {
        HashSet<string> activeSet = new();
        foreach (var kvp in subjects)
        {
            activeSet.Add(kvp.Key);
            UpdateOrCreateBox(kvp.Key, kvp.Value);
        }

        // Remove boxes for subjects no longer tracked
        var toRemove = new List<string>();
        foreach (var kvp in _boxes)
        {
            if (!activeSet.Contains(kvp.Key))
            {
                Destroy(kvp.Value.gameObject);
                toRemove.Add(kvp.Key);
            }
        }
        foreach (var key in toRemove) _boxes.Remove(key);
    }

    private void UpdateOrCreateBox(string subjectId, SubjectData data)
    {
        if (!_boxes.TryGetValue(subjectId, out var box))
        {
            box = CreateBoxVisual(subjectId);
            _boxes[subjectId] = box;
        }

        // Convert normalized bbox to viewport coords (flip Y)
        // BRACE: y1=top, y2=bottom; Unity viewport: y=0 bottom, y=1 top
        Vector3 viewMin = new Vector3(data.bbox.x1, 1f - data.bbox.y2, assumedDepth);
        Vector3 viewMax = new Vector3(data.bbox.x2, 1f - data.bbox.y1, assumedDepth);

        Vector3 worldMin = _cam.ViewportToWorldPoint(viewMin);
        Vector3 worldMax = _cam.ViewportToWorldPoint(viewMax);

        Vector3 center = (worldMin + worldMax) / 2f;
        Vector3 size = worldMax - worldMin;

        // EMA smoothing to reduce jitter
        box.targetPosition = center;
        box.targetSize = size;
        box.transform.position = Vector3.Lerp(box.transform.position, center, SMOOTH_ALPHA);
        box.currentSize = Vector3.Lerp(box.currentSize, size, SMOOTH_ALPHA);

        // Update LineRenderer rectangle
        UpdateLineRenderer(box, box.currentSize);

        // Update color
        box.lineRenderer.startColor = data.selected ? selectedColor : unselectedColor;
        box.lineRenderer.endColor = box.lineRenderer.startColor;

        // Update label
        box.label.text = data.label;

        // Update collider for raycast hit-testing
        box.collider.size = box.currentSize;
    }

    private BoxVisual CreateBoxVisual(string id)
    {
        var go = new GameObject($"BBox_{id}");
        var bv = go.AddComponent<BoxVisual>();

        // LineRenderer for rectangle outline
        var lr = go.AddComponent<LineRenderer>();
        lr.positionCount = 5;
        lr.startWidth = 0.005f;
        lr.endWidth = 0.005f;
        lr.useWorldSpace = false;
        lr.loop = false;
        lr.material = new Material(Shader.Find("Sprites/Default"));
        bv.lineRenderer = lr;

        // BoxCollider for raycast selection
        var col = go.AddComponent<BoxCollider>();
        col.isTrigger = true;
        bv.collider = col;

        // TextMeshPro label above box
        var labelGo = new GameObject("Label");
        labelGo.transform.SetParent(go.transform, false);
        labelGo.transform.localPosition = new Vector3(0, 0.5f, 0); // Above box
        var tmp = labelGo.AddComponent<TMPro.TextMeshPro>();
        tmp.fontSize = 1f;
        tmp.alignment = TMPro.TextAlignmentOptions.Center;
        tmp.color = Color.white;
        bv.label = tmp;

        bv.subjectId = id;
        bv.currentSize = Vector3.one * 0.5f;

        return bv;
    }

    private void UpdateLineRenderer(BoxVisual box, Vector3 size)
    {
        float hw = size.x / 2f;
        float hh = size.y / 2f;
        box.lineRenderer.SetPosition(0, new Vector3(-hw, -hh, 0));
        box.lineRenderer.SetPosition(1, new Vector3(hw, -hh, 0));
        box.lineRenderer.SetPosition(2, new Vector3(hw, hh, 0));
        box.lineRenderer.SetPosition(3, new Vector3(-hw, hh, 0));
        box.lineRenderer.SetPosition(4, new Vector3(-hw, -hh, 0));

        // Position label above top edge
        box.label.transform.localPosition = new Vector3(0, hh + 0.05f, 0);
    }
}

public class BoxVisual : MonoBehaviour
{
    public string subjectId;
    public LineRenderer lineRenderer;
    public BoxCollider collider;
    public TMPro.TextMeshPro label;
    public Vector3 targetPosition;
    public Vector3 targetSize;
    public Vector3 currentSize;
}
```

---

## 6. Controller Raycast Selection

```csharp
using UnityEngine;

public class VRSelector : MonoBehaviour
{
    [SerializeField] private BraceWebSocket braceWs;
    [SerializeField] private BoundingBoxRenderer boxRenderer;

    // Reference to right controller (OVR or XR Interaction Toolkit)
    [SerializeField] private Transform rightController;
    [SerializeField] private LineRenderer laserPointer;

    private string _hoveredSubjectId;
    private string _selectedSubjectId;

    void Update()
    {
        // Raycast from controller
        Ray ray = new Ray(rightController.position, rightController.forward);

        if (Physics.Raycast(ray, out RaycastHit hit, 20f))
        {
            var box = hit.collider.GetComponent<BoxVisual>();
            if (box != null)
            {
                // Hover highlight
                _hoveredSubjectId = box.subjectId;
                box.lineRenderer.startWidth = 0.008f; // Thicker on hover
                box.lineRenderer.endWidth = 0.008f;

                // Check for trigger press (Meta Quest controller)
                if (OVRInput.GetDown(OVRInput.Button.PrimaryIndexTrigger, OVRInput.Controller.RTouch))
                {
                    OnSelectSubject(box.subjectId);
                }
            }
            else
            {
                ClearHover();

                // Click on empty space → deselect
                if (OVRInput.GetDown(OVRInput.Button.PrimaryIndexTrigger, OVRInput.Controller.RTouch))
                {
                    OnDeselectSubject();
                }
            }
        }
        else
        {
            ClearHover();
        }

        // Update laser pointer
        if (laserPointer != null)
        {
            laserPointer.SetPosition(0, rightController.position);
            laserPointer.SetPosition(1, rightController.position + rightController.forward * 10f);
        }
    }

    private void OnSelectSubject(string subjectId)
    {
        _selectedSubjectId = subjectId;
        if (int.TryParse(subjectId, out int sid))
        {
            braceWs.SelectSubject(sid);
        }
    }

    private void OnDeselectSubject()
    {
        _selectedSubjectId = null;
        braceWs.SelectSubject(null);
    }

    private void ClearHover()
    {
        _hoveredSubjectId = null;
    }
}
```

---

## 7. Stats Panel (Video Game Style)

When a subject is selected and `selected=true` with quality data, spawn a world-space Canvas panel:

```csharp
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

public class StatsPanel : MonoBehaviour
{
    [SerializeField] private GameObject panelPrefab; // World-space Canvas prefab

    private GameObject _activePanel;
    private string _activeSubjectId;
    private float _lastUpdateTime;
    private const float UPDATE_INTERVAL = 0.25f; // Update at 4Hz

    /// <summary>
    /// Show stats panel for a selected subject.
    /// Panel spawns 1.5m in front of the user, then stays static.
    /// </summary>
    public void ShowPanel(string subjectId, SubjectData data, Transform headTransform)
    {
        if (_activePanel != null && _activeSubjectId == subjectId)
        {
            // Already showing — just update data
            UpdatePanelData(data);
            return;
        }

        DismissPanel();

        _activeSubjectId = subjectId;
        _activePanel = Instantiate(panelPrefab);

        // Position 1.5m in front of user, facing them
        Vector3 spawnPos = headTransform.position + headTransform.forward * 1.5f;
        _activePanel.transform.position = spawnPos;
        _activePanel.transform.rotation = Quaternion.LookRotation(
            _activePanel.transform.position - headTransform.position
        );

        UpdatePanelData(data);
    }

    public void DismissPanel()
    {
        if (_activePanel != null)
        {
            Destroy(_activePanel);
            _activePanel = null;
            _activeSubjectId = null;
        }
    }

    public void UpdateIfNeeded(SubjectData data)
    {
        if (_activePanel == null) return;
        if (Time.time - _lastUpdateTime < UPDATE_INTERVAL) return;
        _lastUpdateTime = Time.time;
        UpdatePanelData(data);
    }

    private void UpdatePanelData(SubjectData data)
    {
        if (_activePanel == null || data.quality == null) return;

        // Find UI elements by name in the prefab hierarchy
        SetText("LabelText", data.label);
        SetText("PhaseText", data.quality.movement_phase?.label ?? data.phase);

        // Form Score (large, colored bar)
        int formScore = data.quality.form_score;
        SetText("FormScoreText", $"{formScore}");
        var formBar = _activePanel.transform.Find("FormScoreBar")?.GetComponent<Image>();
        if (formBar != null)
        {
            formBar.fillAmount = formScore / 100f;
            formBar.color = formScore >= 80 ? Color.green :
                            formScore >= 60 ? Color.yellow : Color.red;
        }

        // Rep counter
        if (data.quality.movement_phase != null)
        {
            SetText("RepCountText", $"Rep {data.quality.movement_phase.cycle_count}");
        }

        // Activity label
        if (data.cluster_summary != null)
        {
            foreach (var kvp in data.cluster_summary)
            {
                if (!string.IsNullOrEmpty(kvp.Value.activity_label))
                {
                    SetText("ActivityText", kvp.Value.activity_label);
                    break;
                }
            }
        }

        // Biomechanics
        if (data.quality.biomechanics != null)
        {
            var bio = data.quality.biomechanics;
            SetText("FPPAText", $"Knee Angle: L {bio.fppa_left:F1}° / R {bio.fppa_right:F1}°");
            SetText("HipDropText", $"Hip Drop: {bio.hip_drop:F1}°");
            SetText("TrunkLeanText", $"Trunk Lean: {bio.trunk_lean:F1}°");
            SetText("AsymmetryText", $"Asymmetry: {bio.asymmetry:F1}%");
        }

        // Injury risks
        if (data.quality.injury_risks != null && data.quality.injury_risks.Count > 0)
        {
            string risksText = "";
            foreach (var risk in data.quality.injury_risks)
            {
                string icon = risk.severity == "high" ? "!!" : "!";
                risksText += $"{icon} {risk.risk} ({risk.joint})\n";
            }
            SetText("InjuryRisksText", risksText.TrimEnd());
        }
        else
        {
            SetText("InjuryRisksText", "No risks detected");
        }

        // Fatigue
        SetText("FatigueText", $"Fatigue: {data.fatigue_index:P0}");

        // Alert
        if (!string.IsNullOrEmpty(data.alert_text))
        {
            SetText("AlertText", data.alert_text);
        }
    }

    private void SetText(string childName, string value)
    {
        var t = _activePanel.transform.Find(childName)?.GetComponent<TextMeshProUGUI>();
        if (t != null) t.text = value;
    }
}
```

### Panel Prefab Setup

Create a **World Space Canvas** prefab (`0.6m × 0.8m`):

1. **Canvas**: Render Mode = World Space, width=600, height=800, scale=(0.001, 0.001, 0.001)
2. **Background**: Dark semi-transparent panel (`Color(0.1, 0.1, 0.1, 0.85)`)
3. **Child TextMeshProUGUI elements** named:
   - `LabelText` — Subject label (e.g., "S1"), large font
   - `ActivityText` — Activity name (e.g., "Squat")
   - `PhaseText` — Current movement phase
   - `FormScoreText` — Form score number
   - `FormScoreBar` — Image with Fill method = Horizontal
   - `RepCountText` — Rep counter
   - `FPPAText`, `HipDropText`, `TrunkLeanText`, `AsymmetryText` — Biomechanics
   - `InjuryRisksText` — Injury risk warnings
   - `FatigueText` — Fatigue index
   - `AlertText` — Alert/coaching message
4. **X Button**: UI Button in top-right corner, calls `StatsPanel.DismissPanel()`

---

## 8. Putting It All Together

### Scene Hierarchy

```
XR Origin (OVRCameraRig)
  ├── TrackingSpace
  │   ├── CenterEyeAnchor (Main Camera)
  │   ├── LeftControllerAnchor
  │   └── RightControllerAnchor
  │       └── LaserPointer (LineRenderer)
  ├── OVRManager (Passthrough enabled)
  └── OVRPassthroughLayer (Underlay)

BraceManager (GameObject)
  ├── FrameCapture (component)
  ├── BraceWebSocket (component)
  ├── BoundingBoxRenderer (component)
  ├── VRSelector (component)
  └── StatsPanel (component)
```

### Main Update Loop

```csharp
using UnityEngine;

public class BraceManager : MonoBehaviour
{
    [SerializeField] private BraceWebSocket ws;
    [SerializeField] private BoundingBoxRenderer boxRenderer;
    [SerializeField] private StatsPanel statsPanel;
    [SerializeField] private Transform headTransform;

    void Update()
    {
        var response = ws.LatestResponse;
        if (response == null) return;

        // Update all bounding boxes
        boxRenderer.UpdateBoxes(response.subjects, response.active_track_ids);

        // Update stats panel for selected subject
        foreach (var kvp in response.subjects)
        {
            if (kvp.Value.selected && kvp.Value.quality != null)
            {
                statsPanel.ShowPanel(kvp.Key, kvp.Value, headTransform);
                statsPanel.UpdateIfNeeded(kvp.Value);
                break;
            }
        }
    }
}
```

---

## 9. VR Response Format Reference

### Unselected subject (~80 bytes):
```json
{
  "label": "S1",
  "bbox": {"x1": 0.3, "y1": 0.2, "x2": 0.7, "y2": 0.9},
  "phase": "normal",
  "selected": false
}
```

### Selected subject (~1.5KB):
```json
{
  "label": "S2",
  "bbox": {"x1": 0.2, "y1": 0.1, "x2": 0.6, "y2": 0.85},
  "phase": "normal",
  "selected": true,
  "n_segments": 5,
  "n_clusters": 3,
  "cluster_id": 2,
  "consistency_score": 0.92,
  "is_anomaly": false,
  "cluster_summary": {
    "2": {
      "count": 5,
      "mean_score": 0.91,
      "activity_label": "squat"
    }
  },
  "velocity": 1.23,
  "rolling_velocity": 1.15,
  "fatigue_index": 0.42,
  "peak_velocity": 2.87,
  "identity_status": "confirmed",
  "identity_confidence": 0.98,
  "quality": {
    "form_score": 87,
    "movement_phase": {"label": "descending", "progress": 0.65, "cycle_count": 4},
    "biomechanics": {
      "fppa_left": -3.2,
      "fppa_right": 5.1,
      "hip_drop": 2.3,
      "trunk_lean": 4.5,
      "asymmetry": 6.2
    },
    "injury_risks": [
      {
        "joint": "right_knee",
        "risk": "Knee Valgus",
        "severity": "medium",
        "value": 12.5,
        "threshold": 10
      }
    ],
    "joint_quality": {
      "scores": [95, 92, 88, 85, 90, 87, 91, 89, 78, 82, 93, 91, 88, 86],
      "degrading": [8, 9]
    },
    "active_guideline": {
      "name": "squat",
      "display_name": "Squat",
      "form_cues": ["Keep knees over toes"]
    },
    "fatigue_timeline": {
      "timestamps": [1.0, 2.0],
      "fatigue": [0.1, 0.2],
      "form_scores": [90, 87]
    }
  },
  "alert_text": "Watch your left knee!"
}
```

### Full frame response:
```json
{
  "frame_index": 42,
  "video_time": 1.4,
  "subjects": {
    "0": { "label": "S1", "bbox": {...}, "phase": "normal", "selected": false },
    "1": { "label": "S2", "bbox": {...}, "phase": "normal", "selected": true, "quality": {...}, ... }
  },
  "active_track_ids": [0, 1],
  "timing": {
    "decode_ms": 3.2,
    "pipeline_ms": 45.1,
    "identity_ms": 12.3,
    "analyzer_ms": 8.5,
    "total_ms": 69.1
  }
}
```

---

## 10. Latency Budget

| Stage | Latency |
|---|---|
| Camera capture + GPU→CPU readback | 40-60ms |
| JPEG encode + WebSocket send | 5-10ms |
| Network round-trip (Tailscale) | 40-100ms |
| GPU inference (YOLO + analysis) | 50-100ms |
| **Total end-to-end** | **~130-260ms** |

This is acceptable for bounding box overlays (not pixel-precise skeleton alignment). Use EMA smoothing on box positions to mask jitter.

---

## 11. Troubleshooting

| Issue | Solution |
|---|---|
| WebSocket won't connect | Check `INTERNET` permission in manifest; verify Tailscale is running on Quest |
| No camera frames | Ensure API level ≥ 32, Vulkan graphics, camera permission granted |
| Boxes flicker/jump | Increase EMA smoothing (lower `SMOOTH_ALPHA`); reduce frame send rate |
| High latency | Downscale to 480p; reduce JPEG quality; use `MAX_IN_FLIGHT=3` |
| JSON parse errors | Check `LatestResponse` for `type: "error"` messages from server |
| Black passthrough | Add `OVRPassthroughLayer` component, set to Underlay |
