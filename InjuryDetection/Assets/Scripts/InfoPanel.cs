using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// World-space stats panel showing real-time BRACE analysis for the selected subject.
/// Spawns in front of the user on selection, live-updates at 4Hz from WebSocket data.
/// Auto-finds BraceWebSocket if not assigned in Inspector.
/// </summary>
public class InfoPanel : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private BraceWebSocket braceWs;

    [Header("Layout")]
    [SerializeField] private float displayDistance = 1.5f;
    [SerializeField] private int titleFontSize = 40;
    [SerializeField] private int bodyFontSize = 28;

    private Canvas _canvas;
    private Text _titleText;
    private Text _bodyText;
    private Transform _cameraAnchor;

    private string _selectedSubjectId;
    private float _lastUpdateTime;
    private const float UPDATE_INTERVAL = 0.25f;

    private void Start()
    {
        // Auto-find BraceWebSocket
        if (braceWs == null)
            braceWs = FindObjectOfType<BraceWebSocket>();

        var rig = FindObjectOfType<OVRCameraRig>();
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
        if (_selectedSubjectId == null || braceWs == null) return;
        if (Time.time - _lastUpdateTime < UPDATE_INTERVAL) return;

        var response = braceWs.LatestResponse;
        if (response?.subjects == null) return;
        if (!response.subjects.TryGetValue(_selectedSubjectId, out var data)) return;
        if (!data.selected) return;

        _lastUpdateTime = Time.time;
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

        var ct = _canvas.transform;
        ct.position = _cameraAnchor.position + _cameraAnchor.forward * displayDistance;
        ct.rotation = Quaternion.LookRotation(ct.position - _cameraAnchor.position);

        _titleText.text = box.latestData?.label ?? box.subjectId;
        if (box.latestData != null)
            UpdateContent(box.latestData);
        else
            _bodyText.text = "Waiting for data...";

        _canvas.gameObject.SetActive(true);
    }

    private void UpdateContent(SubjectData data)
    {
        string activity = GetActivityLabel(data);
        _titleText.text = activity != null
            ? $"{data.label}  -  {activity}"
            : data.label ?? "";

        var sb = new System.Text.StringBuilder();

        if (data.quality?.movement_phase != null)
        {
            var mp = data.quality.movement_phase;
            sb.AppendLine($"Phase: {Capitalize(mp.label)}   Rep {mp.cycle_count}");
        }
        else
        {
            sb.AppendLine($"Phase: {Capitalize(data.phase)}");
        }

        if (data.quality != null)
        {
            int form = Mathf.RoundToInt(data.quality.form_score);
            string formLabel = form >= 80 ? "Good" : form >= 60 ? "Fair" : "Poor";
            sb.AppendLine($"Form: {form}/100  ({formLabel})");
        }

        sb.AppendLine();

        if (data.quality?.biomechanics != null)
        {
            var bio = data.quality.biomechanics;
            sb.AppendLine("-- Biomechanics --");
            sb.AppendLine($"Knee Angle (FPPA):  L {bio.fppa_left:F1}   R {bio.fppa_right:F1}");
            sb.AppendLine($"Hip Drop: {bio.hip_drop:F1}    Trunk Lean: {bio.trunk_lean:F1}");
            sb.AppendLine($"Asymmetry: {bio.asymmetry:F1}%");
        }

        if (data.quality?.injury_risks != null && data.quality.injury_risks.Count > 0)
        {
            sb.AppendLine();
            sb.AppendLine("-- Injury Risks --");
            foreach (var risk in data.quality.injury_risks)
            {
                string icon = risk.severity == "high" ? "!!" : "! ";
                sb.AppendLine($"{icon} {FormatRiskName(risk.risk)} ({risk.joint})  [{risk.severity}]");
            }
        }

        if (data.fatigue_index > 0.01f)
        {
            sb.AppendLine();
            sb.AppendLine($"Fatigue: {data.fatigue_index:P0}");
        }

        if (data.quality?.active_guideline?.form_cues != null
            && data.quality.active_guideline.form_cues.Count > 0)
        {
            sb.AppendLine();
            sb.AppendLine("-- Tips --");
            foreach (var cue in data.quality.active_guideline.form_cues)
                sb.AppendLine($"  {cue}");
        }

        if (!string.IsNullOrEmpty(data.alert_text))
        {
            sb.AppendLine();
            sb.AppendLine($">> {data.alert_text}");
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
        canvasObj.transform.SetParent(transform);
        _canvas = canvasObj.AddComponent<Canvas>();
        _canvas.renderMode = RenderMode.WorldSpace;

        var scaler = canvasObj.AddComponent<CanvasScaler>();
        scaler.dynamicPixelsPerUnit = 10f;
        canvasObj.AddComponent<GraphicRaycaster>();

        var rt = _canvas.GetComponent<RectTransform>();
        rt.sizeDelta = new Vector2(650, 800);
        rt.localScale = Vector3.one * 0.001f;

        var bgObj = new GameObject("Background");
        bgObj.transform.SetParent(canvasObj.transform, false);
        var bgRect = bgObj.AddComponent<RectTransform>();
        bgRect.anchorMin = Vector2.zero;
        bgRect.anchorMax = Vector2.one;
        bgRect.offsetMin = Vector2.zero;
        bgRect.offsetMax = Vector2.zero;
        var bgImage = bgObj.AddComponent<Image>();
        bgImage.color = new Color(0.05f, 0.05f, 0.05f, 0.85f);

        var titleObj = new GameObject("Title");
        titleObj.transform.SetParent(canvasObj.transform, false);
        _titleText = titleObj.AddComponent<Text>();
        _titleText.font = GetFont();
        _titleText.fontSize = titleFontSize;
        _titleText.fontStyle = FontStyle.Bold;
        _titleText.color = new Color(0.345f, 0.8f, 0.008f);
        _titleText.alignment = TextAnchor.MiddleCenter;
        var titleRect = _titleText.GetComponent<RectTransform>();
        titleRect.anchorMin = new Vector2(0, 0.88f);
        titleRect.anchorMax = new Vector2(1, 0.98f);
        titleRect.offsetMin = new Vector2(20, 0);
        titleRect.offsetMax = new Vector2(-20, 0);

        var bodyObj = new GameObject("Body");
        bodyObj.transform.SetParent(canvasObj.transform, false);
        _bodyText = bodyObj.AddComponent<Text>();
        _bodyText.font = _titleText.font;
        _bodyText.fontSize = bodyFontSize;
        _bodyText.color = Color.white;
        _bodyText.alignment = TextAnchor.UpperLeft;
        _bodyText.lineSpacing = 1.1f;
        var bodyRect = _bodyText.GetComponent<RectTransform>();
        bodyRect.anchorMin = new Vector2(0, 0.02f);
        bodyRect.anchorMax = new Vector2(1, 0.86f);
        bodyRect.offsetMin = new Vector2(30, 0);
        bodyRect.offsetMax = new Vector2(-30, 0);
    }

    private static Font GetFont()
    {
        var font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        if (font == null)
            font = Resources.GetBuiltinResource<Font>("Arial.ttf");
        return font;
    }
}
