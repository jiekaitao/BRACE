using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// Always-visible debug overlay showing pipeline status.
/// Attach to the same GameObject as BraceWebSocket, FrameCapture, and BoundingBoxRenderer,
/// or it will auto-find them. Shows a small panel in the lower-left of your VR view.
/// </summary>
public class DebugHUD : MonoBehaviour
{
    [Header("References (auto-found if empty)")]
    [SerializeField] private BraceWebSocket braceWs;
    [SerializeField] private FrameCapture frameCapture;
    [SerializeField] private BoundingBoxRenderer boxRenderer;

    [Header("Display")]
    [SerializeField] private bool showHUD = true;
    [SerializeField] private float hudDistance = 2f;
    [SerializeField] private int fontSize = 28;

    private Canvas _canvas;
    private Text _text;
    private Transform _cameraAnchor;
    private float _lastUpdateTime;

    void Start()
    {
        // Auto-find components
        if (braceWs == null) braceWs = FindAnyObjectByType<BraceWebSocket>();
        if (frameCapture == null) frameCapture = FindAnyObjectByType<FrameCapture>();
        if (boxRenderer == null) boxRenderer = FindAnyObjectByType<BoundingBoxRenderer>();

        // Find camera
        var rig = FindAnyObjectByType<OVRCameraRig>();
        if (rig != null)
            _cameraAnchor = rig.centerEyeAnchor;
        else
            _cameraAnchor = Camera.main != null ? Camera.main.transform : transform;

        CreateHUD();
    }

    void Update()
    {
        if (!showHUD || _text == null || _cameraAnchor == null) return;

        // Update content at 4Hz
        if (Time.time - _lastUpdateTime < 0.25f) return;
        _lastUpdateTime = Time.time;

        // Reposition: lower-left of view, always follows head
        Vector3 forward = _cameraAnchor.forward;
        Vector3 down = -_cameraAnchor.up * 0.4f;
        Vector3 left = -_cameraAnchor.right * 0.35f;
        _canvas.transform.position = _cameraAnchor.position + forward * hudDistance + down + left;
        _canvas.transform.rotation = Quaternion.LookRotation(
            _canvas.transform.position - _cameraAnchor.position
        );

        // Build status text
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("== BRACE DEBUG ==");

        // WebSocket status
        if (braceWs != null)
        {
            string wsStatus = braceWs.IsConnected ? "<color=green>CONNECTED</color>" : "<color=red>DISCONNECTED</color>";
            sb.AppendLine($"WS: {wsStatus}");
            sb.AppendLine($"URL: {TruncateUrl(braceWs.ServerUrl)}");
            if (!braceWs.IsConnected && !string.IsNullOrEmpty(braceWs.LastError))
            {
                string err = braceWs.LastError;
                if (err.Length > 80) err = err.Substring(0, 80) + "...";
                sb.AppendLine($"<color=red>Err: {err}</color>");
            }
            sb.AppendLine($"Recv: {braceWs.FramesReceived}  InFlight: {braceWs.InFlight}");
        }
        else
        {
            sb.AppendLine("<color=red>WS: NOT FOUND</color>");
        }

        // Camera status
        if (frameCapture != null)
        {
            string camStatus = frameCapture.IsCapturing ? "<color=green>CAPTURING</color>" : "<color=yellow>WAITING</color>";
            sb.AppendLine($"Cam: {camStatus} ({frameCapture.CameraName})");
            sb.AppendLine($"Sent: {frameCapture.FramesSent}");
        }
        else
        {
            sb.AppendLine("<color=red>Cam: NOT FOUND</color>");
        }

        // Bounding boxes
        if (boxRenderer != null)
        {
            sb.AppendLine($"Boxes: {boxRenderer.BoxCount}");
        }

        // Latest response info
        if (braceWs != null && braceWs.LatestResponse != null)
        {
            var r = braceWs.LatestResponse;
            int subjectCount = r.subjects != null ? r.subjects.Count : 0;
            sb.AppendLine($"Subjects: {subjectCount}");
            if (r.timing != null)
                sb.AppendLine($"Server: {r.timing.total_ms:F0}ms");
        }

        _text.text = sb.ToString();
    }

    private void CreateHUD()
    {
        var canvasObj = new GameObject("DebugHUDCanvas");
        canvasObj.transform.SetParent(transform);
        _canvas = canvasObj.AddComponent<Canvas>();
        _canvas.renderMode = RenderMode.WorldSpace;

        var scaler = canvasObj.AddComponent<CanvasScaler>();
        scaler.dynamicPixelsPerUnit = 10f;

        var rt = _canvas.GetComponent<RectTransform>();
        rt.sizeDelta = new Vector2(500, 350);
        rt.localScale = Vector3.one * 0.001f;

        // Semi-transparent background
        var bgObj = new GameObject("BG");
        bgObj.transform.SetParent(canvasObj.transform, false);
        var bgRect = bgObj.AddComponent<RectTransform>();
        bgRect.anchorMin = Vector2.zero;
        bgRect.anchorMax = Vector2.one;
        bgRect.offsetMin = Vector2.zero;
        bgRect.offsetMax = Vector2.zero;
        var bgImage = bgObj.AddComponent<Image>();
        bgImage.color = new Color(0, 0, 0, 0.7f);

        // Text
        var textObj = new GameObject("Text");
        textObj.transform.SetParent(canvasObj.transform, false);
        _text = textObj.AddComponent<Text>();
        _text.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        if (_text.font == null)
            _text.font = Resources.GetBuiltinResource<Font>("Arial.ttf");
        _text.fontSize = fontSize;
        _text.color = Color.white;
        _text.alignment = TextAnchor.UpperLeft;
        _text.supportRichText = true;
        var textRect = _text.GetComponent<RectTransform>();
        textRect.anchorMin = Vector2.zero;
        textRect.anchorMax = Vector2.one;
        textRect.offsetMin = new Vector2(15, 10);
        textRect.offsetMax = new Vector2(-15, -10);
    }

    private static string TruncateUrl(string url)
    {
        if (string.IsNullOrEmpty(url)) return "N/A";
        // Show just host:port
        try
        {
            int schemeEnd = url.IndexOf("://");
            if (schemeEnd >= 0)
            {
                string rest = url.Substring(schemeEnd + 3);
                int pathStart = rest.IndexOf('/');
                if (pathStart > 0)
                    return rest.Substring(0, pathStart);
                return rest;
            }
        }
        catch { }
        return url.Length > 40 ? url.Substring(0, 40) + "..." : url;
    }
}
