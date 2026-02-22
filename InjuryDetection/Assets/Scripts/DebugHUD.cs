using UnityEngine;
using UnityEngine.UI;
using TMPro;

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

    [Tooltip("Vertical offset below eye level (positive = further down)")]
    [SerializeField] private float downOffset = 1.1f;

    [Tooltip("Horizontal offset to the left (positive = further left)")]
    [SerializeField] private float leftOffset = 0.55f;

    [Tooltip("TMP font size. TMP stays crisp at high sizes in VR.")]
    [SerializeField] private float fontSize = 36f;

    [Tooltip("World scale of the canvas. Increase if HUD is too small in headset.")]
    [SerializeField] private float canvasWorldScale = 0.0018f;

    [Tooltip("Higher = crisper in world space, but slightly more expensive.")]
    [SerializeField] private float dynamicPixelsPerUnit = 40f;

    [Tooltip("Panel size in canvas units (not meters).")]
    [SerializeField] private Vector2 panelSize = new Vector2(650, 500);

    [Tooltip("Panel background alpha (0..1).")]
    [Range(0f, 1f)]
    [SerializeField] private float backgroundAlpha = 0.75f;

    private Canvas _canvas;
    private TextMeshProUGUI _text;
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

        // Reposition: below eye level, offset left (side-by-side with InfoPanel on right)
        Vector3 forward = _cameraAnchor.forward;
        Vector3 down = -_cameraAnchor.up * downOffset;
        Vector3 left = -_cameraAnchor.right * leftOffset;

        _canvas.transform.position = _cameraAnchor.position + forward * hudDistance + down + left;

        // Face the camera (billboard)
        _canvas.transform.rotation = Quaternion.LookRotation(
            _canvas.transform.position - _cameraAnchor.position
        );

        // Build status text
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("== BRACE DEBUG ==");

        // WebSocket status
        if (braceWs != null)
        {
            string wsStatus = braceWs.IsConnected ? "<color=#00ff00>CONNECTED</color>" : "<color=#ff3333>DISCONNECTED</color>";
            sb.AppendLine($"WS: {wsStatus}");
            sb.AppendLine($"URL: {braceWs.ServerUrl}");

            if (!braceWs.IsConnected)
            {
                if (!string.IsNullOrEmpty(braceWs.LastError))
                {
                    string err = braceWs.LastError;
                    if (err.Length > 120) err = err.Substring(0, 120) + "...";
                    sb.AppendLine($"<color=#ff3333>Error: {err}</color>");
                }
                else
                {
                    sb.AppendLine("<color=#ffcc00>Trying to connect...</color>");
                }
            }

            sb.AppendLine($"Recv: {braceWs.FramesReceived}  InFlight: {braceWs.InFlight}");
        }
        else
        {
            sb.AppendLine("<color=#ff3333>WS: MISSING</color>");
            sb.AppendLine("<color=#ffcc00>Add BraceWebSocket component</color>");
        }

        // Camera status
        if (frameCapture != null)
        {
            string camStatus = frameCapture.IsCapturing ? "<color=#00ff00>CAPTURING</color>" : "<color=#ffcc00>WAITING</color>";
            sb.AppendLine($"Cam: {camStatus} [{frameCapture.CameraIndex}] ({frameCapture.CameraName})");
            sb.AppendLine($"Perm: {frameCapture.PermissionStatus}  Res: {frameCapture.Resolution}");
            sb.AppendLine($"Sent: {frameCapture.FramesSent}  Rot: {frameCapture.RotationAngle}° Mirror: {frameCapture.IsMirrored}");

            // Brightness diagnostic — detect black frames
            float bright = frameCapture.AvgBrightness;
            if (frameCapture.IsCapturing && frameCapture.FramesSent > 0)
            {
                string brightColor = bright < 5f ? "#ff3333" : bright < 30f ? "#ffcc00" : "#00ff00";
                sb.AppendLine($"Brightness: <color={brightColor}>{bright:F0}</color>/255  NonBlack: {frameCapture.NonBlackPixels}/{frameCapture.TotalSampled}");

                if (bright < 5f)
                    sb.AppendLine("<color=#ff3333>FRAMES ARE BLACK!</color>");
            }

            // List all available cameras
            var devices = WebCamTexture.devices;
            if (devices.Length > 0)
            {
                sb.Append("Cams: ");
                for (int i = 0; i < devices.Length; i++)
                {
                    string marker = (i.ToString() == frameCapture.CameraIndex.Split('/')[0]) ? ">" : " ";
                    sb.Append($"{marker}[{i}]{devices[i].name} ");
                }
                sb.AppendLine();
            }

            if (!frameCapture.IsCapturing && frameCapture.CameraName == "NOT FOUND")
                sb.AppendLine("<color=#ffcc00>No camera - check permissions</color>");
            else if (!frameCapture.IsCapturing && frameCapture.CameraName == "PERM DENIED")
                sb.AppendLine("<color=#ff3333>Camera permission denied! Go to Quest Settings > Apps > InjuryDetection > Permissions</color>");
            else if (!frameCapture.IsCapturing && frameCapture.CameraName == "none")
                sb.AppendLine("<color=#ffcc00>Camera not initialized yet</color>");
        }
        else
        {
            sb.AppendLine("<color=#ff3333>Cam: MISSING</color>");
            sb.AppendLine("<color=#ffcc00>Add FrameCapture component</color>");
        }

        // Bounding boxes
        if (boxRenderer != null)
        {
            sb.AppendLine($"Boxes: {boxRenderer.BoxCount}  FOV: {boxRenderer.FovSource}");
        }

        // Latest response info
        if (braceWs != null && braceWs.LatestResponse != null)
        {
            var r = braceWs.LatestResponse;
            int subjectCount = r.subjects != null ? r.subjects.Count : 0;
            sb.AppendLine($"Subjects: {subjectCount}");

            if (r.timing != null)
                sb.AppendLine($"Server: {r.timing.total_ms:F0}ms");

            if (subjectCount == 0 && braceWs.FramesReceived > 10)
                sb.AppendLine("<color=#ffcc00>No people detected in frame</color>");
        }
        else if (braceWs != null && braceWs.IsConnected && braceWs.FramesReceived == 0)
        {
            sb.AppendLine("<color=#ffcc00>Connected, awaiting first response</color>");
        }

        _text.text = sb.ToString();
    }

    private void CreateHUD()
    {
        var canvasObj = new GameObject("DebugHUDCanvas");
        canvasObj.transform.SetParent(transform, false);

        _canvas = canvasObj.AddComponent<Canvas>();
        _canvas.renderMode = RenderMode.WorldSpace;

        var scaler = canvasObj.AddComponent<CanvasScaler>();
        scaler.uiScaleMode = CanvasScaler.ScaleMode.ConstantPixelSize;
        scaler.dynamicPixelsPerUnit = dynamicPixelsPerUnit;

        var rt = _canvas.GetComponent<RectTransform>();
        rt.sizeDelta = panelSize;
        rt.localScale = Vector3.one * canvasWorldScale;

        // Optional but helps world-space UI behave nicely
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

        // Text (TextMeshPro)
        var textObj = new GameObject("Text");
        textObj.transform.SetParent(canvasObj.transform, false);

        _text = textObj.AddComponent<TextMeshProUGUI>();
        _text.enableWordWrapping = false;
        _text.richText = true;
        _text.fontSize = fontSize;
        _text.color = Color.white;
        _text.alignment = TextAlignmentOptions.TopLeft;

        // These help with crispness in world space
        _text.extraPadding = true;
        _text.isOrthographic = true;

        var textRect = _text.GetComponent<RectTransform>();
        textRect.anchorMin = Vector2.zero;
        textRect.anchorMax = Vector2.one;
        textRect.offsetMin = new Vector2(18, 14);
        textRect.offsetMax = new Vector2(-18, -14);
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