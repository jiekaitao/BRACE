using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Renders bounding boxes in world space for each tracked subject.
/// Reads BraceResponse from BraceWebSocket each frame, creates/updates/removes boxes.
/// Each box has a BoxCollider for controller raycast selection (Task 6).
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

    void Start()
    {
        // Use the XR main camera
        var rig = FindObjectOfType<OVRCameraRig>();
        _cam = rig != null ? rig.centerEyeAnchor.GetComponent<Camera>() : Camera.main;

        _lineMaterial = new Material(Shader.Find("Sprites/Default"));
    }

    void Update()
    {
        if (braceWs == null || braceWs.LatestResponse == null) return;

        var response = braceWs.LatestResponse;
        if (response.subjects == null) return;

        // Track which subjects are still present
        _removeList.Clear();
        foreach (var key in _boxes.Keys)
            _removeList.Add(key);

        // Update or create boxes
        foreach (var kvp in response.subjects)
        {
            _removeList.Remove(kvp.Key);
            UpdateOrCreateBox(kvp.Key, kvp.Value);
        }

        // Remove stale boxes
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
        }

        // --- Convert normalized bbox to world position ---
        // BRACE: y1 = top, y2 = bottom (y=0 at top)
        // Unity viewport: y=0 at bottom, y=1 at top → flip Y
        Vector3 viewMin = new Vector3(data.bbox.x1, 1f - data.bbox.y2, assumedDepth);
        Vector3 viewMax = new Vector3(data.bbox.x2, 1f - data.bbox.y1, assumedDepth);

        Vector3 worldMin = _cam.ViewportToWorldPoint(viewMin);
        Vector3 worldMax = _cam.ViewportToWorldPoint(viewMax);

        Vector3 targetCenter = (worldMin + worldMax) / 2f;
        Vector3 targetSize = new Vector3(
            Mathf.Abs(worldMax.x - worldMin.x),
            Mathf.Abs(worldMax.y - worldMin.y),
            0.01f // thin depth for collider
        );

        // EMA smoothing
        if (box.initialized)
        {
            box.smoothCenter = Vector3.Lerp(box.smoothCenter, targetCenter, smoothAlpha);
            box.smoothSize = Vector3.Lerp(box.smoothSize, targetSize, smoothAlpha);
        }
        else
        {
            // First frame — snap instantly
            box.smoothCenter = targetCenter;
            box.smoothSize = targetSize;
            box.initialized = true;
        }

        // Apply position
        box.root.transform.position = box.smoothCenter;

        // Face the camera
        Vector3 lookDir = box.smoothCenter - _cam.transform.position;
        if (lookDir.sqrMagnitude > 0.001f)
            box.root.transform.rotation = Quaternion.LookRotation(lookDir);

        // Update rectangle outline
        UpdateLineRenderer(box.line, box.smoothSize);

        // Update collider
        box.collider.size = new Vector3(box.smoothSize.x, box.smoothSize.y, 0.05f);

        // Update color
        Color color;
        if (data.selected)
            color = selectedColor;
        else if (data.phase == "calibrating")
            color = calibratingColor;
        else
            color = unselectedColor;

        box.line.startColor = color;
        box.line.endColor = color;

        // Update label text and position
        box.label.text = data.label ?? subjectId;
        box.label.transform.localPosition = new Vector3(0, box.smoothSize.y / 2f + 0.04f, 0);

        // Store subject data for raycast selection
        box.subjectId = subjectId;
        box.latestData = data;
    }

    private SubjectBox CreateBox(string id)
    {
        var root = new GameObject($"BBox_{id}");
        var box = new SubjectBox { root = root };

        // LineRenderer — rectangle outline (5 points, closed loop)
        var lr = root.AddComponent<LineRenderer>();
        lr.positionCount = 5;
        lr.startWidth = lineWidth;
        lr.endWidth = lineWidth;
        lr.useWorldSpace = false;
        lr.loop = false;
        lr.material = new Material(_lineMaterial);
        lr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        lr.receiveShadows = false;
        box.line = lr;

        // BoxCollider — trigger for controller raycast
        var col = root.AddComponent<BoxCollider>();
        col.isTrigger = true;
        col.size = Vector3.one * 0.3f;
        box.collider = col;

        // SubjectBox component on the root so raycasts can find it
        var sbComp = root.AddComponent<SubjectBoxTag>();
        sbComp.box = box;

        // Label — 3D TextMesh above the box
        var labelGo = new GameObject("Label");
        labelGo.transform.SetParent(root.transform, false);
        labelGo.transform.localPosition = new Vector3(0, 0.2f, 0);

        var tm = labelGo.AddComponent<TextMesh>();
        tm.fontSize = 48;
        tm.characterSize = 0.02f;
        tm.anchor = TextAnchor.LowerCenter;
        tm.alignment = TextAlignment.Center;
        tm.color = Color.white;
        tm.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        if (tm.font == null)
            tm.font = Resources.GetBuiltinResource<Font>("Arial.ttf");

        // TextMesh needs a MeshRenderer with the font material
        var labelRenderer = labelGo.GetComponent<MeshRenderer>();
        if (labelRenderer != null && tm.font != null)
            labelRenderer.material = tm.font.material;

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

/// <summary>
/// Per-subject bounding box state. Stored in the renderer's dictionary
/// and referenced via SubjectBoxTag on the GameObject.
/// </summary>
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

/// <summary>
/// Tiny MonoBehaviour on each bounding box GameObject so that
/// Physics.Raycast hits can look up the SubjectBox data.
/// </summary>
public class SubjectBoxTag : MonoBehaviour
{
    public SubjectBox box;
}
