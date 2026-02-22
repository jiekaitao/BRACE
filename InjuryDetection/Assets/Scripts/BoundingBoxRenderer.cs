using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Renders bounding boxes in world space for each tracked subject.
/// Reads BraceResponse from BraceWebSocket each frame, creates/updates/removes boxes.
/// Auto-finds BraceWebSocket if not assigned in Inspector.
/// </summary>
public class BoundingBoxRenderer : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private BraceWebSocket braceWs;

    [Header("Rendering")]
    [SerializeField] private float assumedDepth = 3f;
    [SerializeField] private float lineWidth = 0.005f;
    [SerializeField] private Color unselectedColor = new Color(0.6f, 0.6f, 0.6f, 0.9f);
    [SerializeField] private Color selectedColor = new Color(0.345f, 0.8f, 0.008f, 1f); // #58CC02
    [SerializeField] private Color calibratingColor = new Color(1f, 0.84f, 0f, 0.7f);   // gold

    [Header("Smoothing")]
    [Range(0.05f, 1f)]
    [SerializeField] private float smoothAlpha = 0.3f;

    private Camera _cam;
    private readonly Dictionary<string, SubjectBox> _boxes = new();
    private readonly List<string> _removeList = new();
    private Material _lineMaterial;

    /// <summary>Current number of bounding boxes displayed (for debug HUD).</summary>
    public int BoxCount => _boxes.Count;

    void Start()
    {
        // Auto-find BraceWebSocket
        if (braceWs == null)
            braceWs = GetComponent<BraceWebSocket>();
        if (braceWs == null)
            braceWs = FindObjectOfType<BraceWebSocket>();
        if (braceWs == null)
            Debug.LogError("[BRACE] BoundingBoxRenderer: No BraceWebSocket found!");

        // Find XR camera
        var rig = FindObjectOfType<OVRCameraRig>();
        if (rig != null && rig.centerEyeAnchor != null)
            _cam = rig.centerEyeAnchor.GetComponent<Camera>();
        if (_cam == null)
            _cam = Camera.main;
        if (_cam == null)
            Debug.LogError("[BRACE] BoundingBoxRenderer: No camera found!");

        // Create line material — use URP Unlit if available, fall back to Sprites/Default
        Shader shader = Shader.Find("Universal Render Pipeline/Unlit");
        if (shader == null)
            shader = Shader.Find("Sprites/Default");
        if (shader == null)
            shader = Shader.Find("Unlit/Color");

        if (shader != null)
        {
            _lineMaterial = new Material(shader);
            Debug.Log($"[BRACE] LineRenderer shader: {shader.name}");
        }
        else
        {
            Debug.LogError("[BRACE] No usable shader found for LineRenderer!");
        }
    }

    void Update()
    {
        if (braceWs == null || _cam == null) return;
        if (braceWs.LatestResponse == null) return;

        var response = braceWs.LatestResponse;
        if (response.subjects == null) return;

        _removeList.Clear();
        foreach (var key in _boxes.Keys)
            _removeList.Add(key);

        foreach (var kvp in response.subjects)
        {
            _removeList.Remove(kvp.Key);
            UpdateOrCreateBox(kvp.Key, kvp.Value);
        }

        foreach (var key in _removeList)
        {
            Destroy(_boxes[key].root);
            _boxes.Remove(key);
        }
    }

    void OnDestroy()
    {
        foreach (var kvp in _boxes)
            Destroy(kvp.Value.root);
        _boxes.Clear();

        if (_lineMaterial != null)
            Destroy(_lineMaterial);
    }

    // ------------------------------------------------------------------
    // Box lifecycle
    // ------------------------------------------------------------------

    private void UpdateOrCreateBox(string subjectId, SubjectData data)
    {
        if (data.bbox == null) return;

        if (!_boxes.TryGetValue(subjectId, out var box))
        {
            box = CreateBox(subjectId);
            _boxes[subjectId] = box;
            Debug.Log($"[BRACE] Created bounding box for subject {subjectId}");
        }

        // BRACE: y1=top, y2=bottom (y=0 at top)
        // Unity viewport: y=0 at bottom → flip Y
        Vector3 viewMin = new Vector3(data.bbox.x1, 1f - data.bbox.y2, assumedDepth);
        Vector3 viewMax = new Vector3(data.bbox.x2, 1f - data.bbox.y1, assumedDepth);

        Vector3 worldMin = _cam.ViewportToWorldPoint(viewMin);
        Vector3 worldMax = _cam.ViewportToWorldPoint(viewMax);

        Vector3 targetCenter = (worldMin + worldMax) / 2f;
        Vector3 targetSize = new Vector3(
            Mathf.Abs(worldMax.x - worldMin.x),
            Mathf.Abs(worldMax.y - worldMin.y),
            0.01f
        );

        if (box.initialized)
        {
            box.smoothCenter = Vector3.Lerp(box.smoothCenter, targetCenter, smoothAlpha);
            box.smoothSize = Vector3.Lerp(box.smoothSize, targetSize, smoothAlpha);
        }
        else
        {
            box.smoothCenter = targetCenter;
            box.smoothSize = targetSize;
            box.initialized = true;
        }

        box.root.transform.position = box.smoothCenter;

        Vector3 lookDir = box.smoothCenter - _cam.transform.position;
        if (lookDir.sqrMagnitude > 0.001f)
            box.root.transform.rotation = Quaternion.LookRotation(lookDir);

        UpdateLineRenderer(box.line, box.smoothSize);

        box.collider.size = new Vector3(box.smoothSize.x, box.smoothSize.y, 0.05f);

        Color color;
        if (data.selected)
            color = selectedColor;
        else if (data.phase == "calibrating")
            color = calibratingColor;
        else
            color = unselectedColor;

        box.line.startColor = color;
        box.line.endColor = color;

        box.label.text = data.label ?? subjectId;
        box.label.transform.localPosition = new Vector3(0, box.smoothSize.y / 2f + 0.04f, 0);

        box.subjectId = subjectId;
        box.latestData = data;
    }

    private SubjectBox CreateBox(string id)
    {
        var root = new GameObject($"BBox_{id}");
        var box = new SubjectBox { root = root };

        var lr = root.AddComponent<LineRenderer>();
        lr.positionCount = 5;
        lr.startWidth = lineWidth;
        lr.endWidth = lineWidth;
        lr.useWorldSpace = false;
        lr.loop = false;
        if (_lineMaterial != null)
            lr.material = new Material(_lineMaterial);
        lr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        lr.receiveShadows = false;
        box.line = lr;

        var col = root.AddComponent<BoxCollider>();
        col.isTrigger = true;
        col.size = Vector3.one * 0.3f;
        box.collider = col;

        var sbComp = root.AddComponent<SubjectBoxTag>();
        sbComp.box = box;

        // Label
        var labelGo = new GameObject("Label");
        labelGo.transform.SetParent(root.transform, false);
        labelGo.transform.localPosition = new Vector3(0, 0.2f, 0);

        var tm = labelGo.AddComponent<TextMesh>();
        tm.fontSize = 48;
        tm.characterSize = 0.02f;
        tm.anchor = TextAnchor.LowerCenter;
        tm.alignment = TextAlignment.Center;
        tm.color = Color.white;

        Font font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        if (font == null)
            font = Resources.GetBuiltinResource<Font>("Arial.ttf");
        if (font != null)
        {
            tm.font = font;
            var labelRenderer = labelGo.GetComponent<MeshRenderer>();
            if (labelRenderer != null)
                labelRenderer.material = font.material;
        }

        box.label = tm;
        box.subjectId = id;

        return box;
    }

    private void UpdateLineRenderer(LineRenderer lr, Vector3 size)
    {
        float hw = size.x / 2f;
        float hh = size.y / 2f;
        lr.SetPosition(0, new Vector3(-hw, -hh, 0));
        lr.SetPosition(1, new Vector3( hw, -hh, 0));
        lr.SetPosition(2, new Vector3( hw,  hh, 0));
        lr.SetPosition(3, new Vector3(-hw,  hh, 0));
        lr.SetPosition(4, new Vector3(-hw, -hh, 0));
    }
}

/// <summary>Per-subject bounding box state.</summary>
public class SubjectBox
{
    public GameObject root;
    public LineRenderer line;
    public BoxCollider collider;
    public TextMesh label;
    public string subjectId;
    public SubjectData latestData;

    public Vector3 smoothCenter;
    public Vector3 smoothSize;
    public bool initialized;
}

/// <summary>Tag component so raycasts can find the SubjectBox data.</summary>
public class SubjectBoxTag : MonoBehaviour
{
    public SubjectBox box;
}
