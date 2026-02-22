using UnityEngine;
using UnityEngine.UI;
using TMPro;

/// <summary>
/// Head-locked stats panel positioned below eye level.
/// Shows real-time BRACE analysis for the selected subject.
/// Follows the user's head movement so it stays visible but out of the way.
/// Click a bounding box to show that player's stats; click empty space to hide.
/// </summary>
public class InfoPanel : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private BraceWebSocket braceWs;

    [Header("Display")]
    [SerializeField] private float hudDistance = 2f;

    [Tooltip("Vertical offset below eye level (positive = further down)")]
    [SerializeField] private float downOffset = 1.1f;

    [Tooltip("Horizontal offset to the right (positive = further right)")]
    [SerializeField] private float rightOffset = 0f;

    [Tooltip("TMP font size for title")]
    [SerializeField] private float titleFontSize = 80f;

    [Tooltip("TMP font size for body text")]
    [SerializeField] private float bodyFontSize = 64f;

    [Tooltip("World scale of the canvas")]
    [SerializeField] private float canvasWorldScale = 0.004f;

    [Tooltip("Panel size in canvas units")]
    [SerializeField] private Vector2 panelSize = new Vector2(1100, 900);

    [Tooltip("Panel background alpha (0..1)")]
    [Range(0f, 1f)]
    [SerializeField] private float backgroundAlpha = 0.8f;

    [Tooltip("Higher = crisper in world space")]
    [SerializeField] private float dynamicPixelsPerUnit = 40f;

    private Canvas _canvas;
    private TextMeshProUGUI _titleText;
    private TextMeshProUGUI _bodyText;
    private Transform _cameraAnchor;

    private string _selectedSubjectId;
    private float _lastUpdateTime;
    private const float UPDATE_INTERVAL = 0.25f;

    private void Start()
    {
        // Auto-find BraceWebSocket
        if (braceWs == null)
            braceWs = FindAnyObjectByType<BraceWebSocket>();

        var rig = FindAnyObjectByType<OVRCameraRig>();
        if (rig != null)
            _cameraAnchor = rig.centerEyeAnchor;
        else
            _cameraAnchor = Camera.main != null ? Camera.main.transform : transform;

        CreatePanel();
        Hide();

        ControllerRaycast.OnSubjectSelected += OnSubjectSelected;
    }

    private void OnDestroy()
    {
        ControllerRaycast.OnSubjectSelected -= OnSubjectSelected;
    }

    private void Update()
    {
        if (_canvas == null || _cameraAnchor == null) return;
        if (_selectedSubjectId == null) return;

        // Reposition: below eye level, centered
        Vector3 forward = _cameraAnchor.forward;
        Vector3 down = -_cameraAnchor.up * downOffset;

        _canvas.transform.position = _cameraAnchor.position + forward * hudDistance + down;

        // Face the camera (billboard)
        _canvas.transform.rotation = Quaternion.LookRotation(
            _canvas.transform.position - _cameraAnchor.position
        );

        // Update content at 4Hz
        if (Time.time - _lastUpdateTime < UPDATE_INTERVAL) return;
        _lastUpdateTime = Time.time;

        if (braceWs == null) return;
        var response = braceWs.LatestResponse;
        if (response?.subjects == null) return;
        if (!response.subjects.TryGetValue(_selectedSubjectId, out var data)) return;

        UpdateContent(data);
    }

    private void OnSubjectSelected(SubjectBox box)
    {
        if (box == null)
        {
            _selectedSubjectId = null;
            Hide();
            return;
        }

        _selectedSubjectId = box.subjectId;

        _titleText.text = box.latestData?.label ?? box.subjectId;
        if (box.latestData != null)
            UpdateContent(box.latestData);
        else
            _bodyText.text = "Waiting for data...";

        _canvas.gameObject.SetActive(true);
    }

    private void UpdateContent(SubjectData data)
    {
        // Title
        string activity = GetActivityLabel(data);
        _titleText.text = activity != null
            ? $"{data.label}  -  {activity}"
            : data.label ?? "";

        // Body
        var sb = new System.Text.StringBuilder();

        // Selection status (helps diagnose if server got the select message)
        string selStr = data.selected
            ? "<color=#00ff00>SELECTED</color>"
            : "<color=#ffcc00>PENDING</color>";
        sb.AppendLine($"Status: {selStr}");

        // Phase
        if (data.quality?.movement_phase != null)
        {
            var mp = data.quality.movement_phase;
            sb.AppendLine($"Phase: {Capitalize(mp.label)}   Rep {mp.cycle_count}");
        }
        else
        {
            sb.AppendLine($"Phase: {Capitalize(data.phase)}");
        }

        // Top-level stats (available even without quality data)
        if (data.velocity > 0.001f || data.rolling_velocity > 0.001f)
            sb.AppendLine($"Velocity: {data.velocity:F2}  Rolling: {data.rolling_velocity:F2}");

        if (data.consistency_score > 0.001f)
            sb.AppendLine($"Consistency: {data.consistency_score:F2}");

        if (data.n_segments > 0)
            sb.AppendLine($"Segments: {data.n_segments}  Clusters: {data.n_clusters}");

        // Form score (requires quality)
        if (data.quality != null)
        {
            sb.AppendLine();
            int form = Mathf.RoundToInt(data.quality.form_score);
            string formColor = form >= 80 ? "#00ff00" : form >= 60 ? "#ffcc00" : "#ff3333";
            string formLabel = form >= 80 ? "Good" : form >= 60 ? "Fair" : "Poor";
            sb.AppendLine($"Form: <color={formColor}>{form}/100  ({formLabel})</color>");

            if (data.quality.concussion_rating > 0.01f)
            {
                string concColor = data.quality.concussion_rating > 50f ? "#ff3333" : data.quality.concussion_rating > 20f ? "#ffcc00" : "#00ff00";
                sb.AppendLine($"Concussion Risk: <color={concColor}>{data.quality.concussion_rating:F0}/100</color>");
            }
        }

        // Biomechanics (requires quality)
        if (data.quality?.biomechanics != null)
        {
            var bio = data.quality.biomechanics;
            sb.AppendLine();
            sb.AppendLine("<color=#88ccff>-- Biomechanics --</color>");
            sb.AppendLine($"Knee (FPPA):  L {bio.fppa_left:F1}\u00b0   R {bio.fppa_right:F1}\u00b0");
            sb.AppendLine($"Hip Drop: {bio.hip_drop:F1}\u00b0    Trunk Lean: {bio.trunk_lean:F1}\u00b0");
            sb.AppendLine($"Asymmetry: {bio.asymmetry:F1}%");
        }

        // Injury risks (requires quality)
        if (data.quality?.injury_risks != null && data.quality.injury_risks.Count > 0)
        {
            sb.AppendLine();
            sb.AppendLine("<color=#ff6666>-- Injury Risks --</color>");
            foreach (var risk in data.quality.injury_risks)
            {
                string sevColor = risk.severity == "high" ? "#ff3333" : "#ffcc00";
                sb.AppendLine($"<color={sevColor}>{FormatRiskName(risk.risk)}</color> ({risk.joint})  [{risk.severity}]");
            }
        }

        // Fatigue (top-level field, not inside quality)
        if (data.fatigue_index > 0.01f)
        {
            sb.AppendLine();
            string fatColor = data.fatigue_index > 0.6f ? "#ff3333" : data.fatigue_index > 0.3f ? "#ffcc00" : "#00ff00";
            sb.AppendLine($"Fatigue: <color={fatColor}>{data.fatigue_index:P0}</color>");
        }

        // Coaching tips (requires quality)
        if (data.quality?.active_guideline?.form_cues != null
            && data.quality.active_guideline.form_cues.Count > 0)
        {
            sb.AppendLine();
            sb.AppendLine("<color=#88ccff>-- Tips --</color>");
            foreach (var cue in data.quality.active_guideline.form_cues)
                sb.AppendLine($"  {cue}");
        }

        if (!string.IsNullOrEmpty(data.alert_text))
        {
            sb.AppendLine();
            sb.AppendLine($"<color=#ffcc00>>> {data.alert_text}</color>");
        }

        // Show message when quality data hasn't arrived yet
        if (data.quality == null)
        {
            sb.AppendLine();
            if (!data.selected)
                sb.AppendLine("<color=#ffcc00>Tap subject to load detailed stats...</color>");
            else
                sb.AppendLine("<color=#ffcc00>Collecting analysis data...</color>");
        }

        _bodyText.text = sb.ToString();
    }

    private static string GetActivityLabel(SubjectData data)
    {
        if (data.quality?.active_guideline != null)
            return data.quality.active_guideline.display_name;

        if (data.cluster_summary != null)
        {
            foreach (var kvp in data.cluster_summary)
            {
                if (!string.IsNullOrEmpty(kvp.Value.activity_label))
                    return Capitalize(kvp.Value.activity_label);
            }
        }
        return null;
    }

    private static string Capitalize(string s)
    {
        if (string.IsNullOrEmpty(s)) return s;
        return char.ToUpper(s[0]) + s.Substring(1);
    }

    private static string FormatRiskName(string raw)
    {
        if (string.IsNullOrEmpty(raw)) return raw;
        return raw.Replace("_", " ")
                   .Replace("acl", "ACL")
                   .Replace("Acl", "ACL");
    }

    public void Hide()
    {
        if (_canvas != null)
            _canvas.gameObject.SetActive(false);
    }

    private void CreatePanel()
    {
        var canvasObj = new GameObject("InfoPanelCanvas");
        canvasObj.transform.SetParent(transform, false);

        _canvas = canvasObj.AddComponent<Canvas>();
        _canvas.renderMode = RenderMode.WorldSpace;

        var scaler = canvasObj.AddComponent<CanvasScaler>();
        scaler.uiScaleMode = CanvasScaler.ScaleMode.ConstantPixelSize;
        scaler.dynamicPixelsPerUnit = dynamicPixelsPerUnit;

        var rt = _canvas.GetComponent<RectTransform>();
        rt.sizeDelta = panelSize;
        rt.localScale = Vector3.one * canvasWorldScale;

        var raycaster = canvasObj.AddComponent<GraphicRaycaster>();
        raycaster.enabled = false;

        // Semi-transparent background
        var bgObj = new GameObject("BG");
        bgObj.transform.SetParent(canvasObj.transform, false);

        var bgRect = bgObj.AddComponent<RectTransform>();
        bgRect.anchorMin = Vector2.zero;
        bgRect.anchorMax = Vector2.one;
        bgRect.offsetMin = Vector2.zero;
        bgRect.offsetMax = Vector2.zero;

        var bgImage = bgObj.AddComponent<Image>();
        bgImage.color = new Color(0.05f, 0.05f, 0.1f, backgroundAlpha);

        // Title (TextMeshPro)
        var titleObj = new GameObject("Title");
        titleObj.transform.SetParent(canvasObj.transform, false);

        _titleText = titleObj.AddComponent<TextMeshProUGUI>();
        _titleText.enableWordWrapping = false;
        _titleText.richText = true;
        _titleText.fontSize = titleFontSize;
        _titleText.fontStyle = FontStyles.Bold;
        _titleText.color = new Color(0.345f, 0.8f, 0.008f); // green #58CC02
        _titleText.alignment = TextAlignmentOptions.Center;
        _titleText.extraPadding = true;
        _titleText.isOrthographic = true;

        var titleRect = _titleText.GetComponent<RectTransform>();
        titleRect.anchorMin = new Vector2(0, 0.88f);
        titleRect.anchorMax = new Vector2(1, 0.98f);
        titleRect.offsetMin = new Vector2(18, 0);
        titleRect.offsetMax = new Vector2(-18, 0);

        // Body (TextMeshPro)
        var bodyObj = new GameObject("Body");
        bodyObj.transform.SetParent(canvasObj.transform, false);

        _bodyText = bodyObj.AddComponent<TextMeshProUGUI>();
        _bodyText.enableWordWrapping = true;
        _bodyText.richText = true;
        _bodyText.fontSize = bodyFontSize;
        _bodyText.color = Color.white;
        _bodyText.alignment = TextAlignmentOptions.TopLeft;
        _bodyText.extraPadding = true;
        _bodyText.isOrthographic = true;

        var bodyRect = _bodyText.GetComponent<RectTransform>();
        bodyRect.anchorMin = new Vector2(0, 0.02f);
        bodyRect.anchorMax = new Vector2(1, 0.86f);
        bodyRect.offsetMin = new Vector2(24, 14);
        bodyRect.offsetMax = new Vector2(-24, -14);
    }
}
