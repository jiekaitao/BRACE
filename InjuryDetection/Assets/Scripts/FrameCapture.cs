using UnityEngine;
#if UNITY_ANDROID && !UNITY_EDITOR
using UnityEngine.Android;
#endif

/// <summary>
/// Captures Quest 3 passthrough camera frames as JPEG bytes.
/// Downscales to 480p and feeds frames to BraceWebSocket at the throttled rate.
/// Handles camera rotation/mirror correction for proper YOLO detection.
/// Auto-finds BraceWebSocket if not assigned in Inspector.
/// </summary>
public class FrameCapture : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private BraceWebSocket braceWs;

    [Header("Capture Settings")]
    [SerializeField] private int targetHeight = 480;
    [SerializeField] [Range(1, 100)] private int jpegQuality = 65;

    private WebCamTexture _webcam;
    private Texture2D _captureTex;
    private Texture2D _scaledTex;
    private RenderTexture _rt;
    private RenderTexture _correctedRt; // for rotation/mirror correction
    private Material _correctionMat;     // shader material for UV transform
    private int _scaledH;
    private bool _ready;
    private int _framesSent;
    private bool _cameraInfoLogged;
    private float _avgBrightness;
    private int _nonBlackPixels;
    private int _totalSampled;

    /// <summary>Camera device name (for debug HUD).</summary>
    public string CameraName { get; private set; } = "none";

    /// <summary>True if webcam is capturing.</summary>
    public bool IsCapturing => _webcam != null && _webcam.isPlaying && _webcam.width > 32;

    /// <summary>Total frames sent to server (for debug HUD).</summary>
    public int FramesSent => _framesSent;

    /// <summary>Camera rotation angle (for debug HUD).</summary>
    public int RotationAngle => _webcam != null ? _webcam.videoRotationAngle : 0;

    /// <summary>Camera vertically mirrored (for debug HUD).</summary>
    public bool IsMirrored => _webcam != null && _webcam.videoVerticallyMirrored;

    /// <summary>Average brightness of last captured frame (0-255). 0 = all black.</summary>
    public float AvgBrightness => _avgBrightness;

    /// <summary>Number of non-black pixels in last sample.</summary>
    public int NonBlackPixels => _nonBlackPixels;

    /// <summary>Total pixels sampled for brightness check.</summary>
    public int TotalSampled => _totalSampled;

    /// <summary>Camera resolution string (for debug HUD).</summary>
    public string Resolution => _webcam != null && _webcam.width > 32 ? $"{_webcam.width}x{_webcam.height}" : "N/A";

    void Start()
    {
        // Auto-find BraceWebSocket
        if (braceWs == null)
            braceWs = GetComponent<BraceWebSocket>();
        if (braceWs == null)
            braceWs = FindAnyObjectByType<BraceWebSocket>();

        if (braceWs == null)
            Debug.LogError("[BRACE] FrameCapture: No BraceWebSocket found!");

        RequestCameraPermissionAndStart();
    }

    void Update()
    {
        if (!_ready || braceWs == null) return;
        if (!braceWs.ReadyToSend()) return;
        if (_webcam == null || !_webcam.isPlaying) return;
        if (!_webcam.didUpdateThisFrame) return;

        // Log camera properties once after first real frame
        if (!_cameraInfoLogged && _webcam.width > 32)
        {
            _cameraInfoLogged = true;
            Debug.Log($"[BRACE] Camera info: rotation={_webcam.videoRotationAngle}° mirrored={_webcam.videoVerticallyMirrored} res={_webcam.width}x{_webcam.height}");
        }

        byte[] jpeg = CaptureJpeg();
        if (jpeg != null)
        {
            braceWs.SendFrame(jpeg);
            _framesSent++;
        }
    }

    void OnDestroy()
    {
        if (_webcam != null && _webcam.isPlaying)
            _webcam.Stop();

        if (_rt != null)
            RenderTexture.ReleaseTemporary(_rt);

        if (_correctedRt != null)
        {
            RenderTexture.ReleaseTemporary(_correctedRt);
            _correctedRt = null;
        }

        if (_scaledTex != null)
            Destroy(_scaledTex);

        if (_captureTex != null)
            Destroy(_captureTex);

        if (_correctionMat != null)
            Destroy(_correctionMat);
    }

    // ------------------------------------------------------------------
    // Permission + Camera init
    // ------------------------------------------------------------------

    private void RequestCameraPermissionAndStart()
    {
#if UNITY_ANDROID && !UNITY_EDITOR
        // Quest 3 (Android 12+) requires runtime permission for headset camera
        if (!Permission.HasUserAuthorizedPermission(Permission.Camera))
        {
            Debug.Log("[BRACE] Requesting camera permission...");
            var callbacks = new PermissionCallbacks();
            callbacks.PermissionGranted += (perm) =>
            {
                Debug.Log($"[BRACE] Permission granted: {perm}");
                StartCamera();
            };
            callbacks.PermissionDenied += (perm) =>
            {
                Debug.LogError($"[BRACE] Permission DENIED: {perm}");
                CameraName = "PERM DENIED";
            };
            callbacks.PermissionDeniedAndDontAskAgain += (perm) =>
            {
                Debug.LogError($"[BRACE] Permission DENIED permanently: {perm}");
                CameraName = "PERM DENIED";
            };
            Permission.RequestUserPermission(Permission.Camera, callbacks);
            return;
        }
        Debug.Log("[BRACE] Camera permission already granted");
#endif
        StartCamera();
    }

    private void StartCamera()
    {
        WebCamDevice[] devices = WebCamTexture.devices;
        Debug.Log($"[BRACE] Camera devices found: {devices.Length}");

        if (devices.Length == 0)
        {
            Debug.LogWarning("[BRACE] No camera devices found!");
            CameraName = "NOT FOUND";
            return;
        }

        // Log all devices for debugging
        for (int i = 0; i < devices.Length; i++)
            Debug.Log($"[BRACE]   Camera[{i}]: \"{devices[i].name}\" front={devices[i].isFrontFacing}");

        // Prefer front-facing (Quest passthrough), fall back to first available
        string deviceName = devices[0].name;
        foreach (var device in devices)
        {
            if (device.isFrontFacing)
            {
                deviceName = device.name;
                break;
            }
        }

        CameraName = deviceName;
        Debug.Log($"[BRACE] Using camera: {deviceName}");
        _webcam = new WebCamTexture(deviceName, 1280, 960, 30);
        _webcam.Play();

        _scaledH = targetHeight;
        _ready = true;
    }

    // ------------------------------------------------------------------
    // JPEG capture with rotation/mirror correction
    // ------------------------------------------------------------------

    private byte[] CaptureJpeg()
    {
        int w = _webcam.width;
        int h = _webcam.height;

        // Skip if camera hasn't initialized yet (returns 16x16 placeholder)
        if (w < 32 || h < 32) return null;

        // Read pixels from WebCamTexture
        if (_captureTex == null || _captureTex.width != w || _captureTex.height != h)
        {
            Debug.Log($"[BRACE] Camera resolution: {w}x{h}");
            _captureTex = new Texture2D(w, h, TextureFormat.RGBA32, false);
        }

        Color32[] pixels = _webcam.GetPixels32();
        _captureTex.SetPixels32(pixels);
        _captureTex.Apply();

        // Sample brightness from ~100 evenly spaced pixels to detect black frames
        int step = Mathf.Max(1, pixels.Length / 100);
        float brightnessSum = 0;
        int nonBlack = 0;
        int sampled = 0;
        for (int i = 0; i < pixels.Length; i += step)
        {
            float lum = (pixels[i].r + pixels[i].g + pixels[i].b) / 3f;
            brightnessSum += lum;
            if (lum > 5f) nonBlack++;
            sampled++;
        }
        _avgBrightness = sampled > 0 ? brightnessSum / sampled : 0;
        _nonBlackPixels = nonBlack;
        _totalSampled = sampled;

        // Determine output dimensions accounting for rotation
        int angle = _webcam.videoRotationAngle;
        bool mirrored = _webcam.videoVerticallyMirrored;
        bool swapDims = (angle == 90 || angle == 270);

        // Source dimensions after rotation
        int srcW = swapDims ? h : w;
        int srcH = swapDims ? w : h;

        // Downscale target
        int outH = _scaledH;
        int outW = Mathf.RoundToInt((float)srcW / srcH * outH);

        // Step 1: Blit capture texture to a corrected RT with rotation/mirror
        if (_correctedRt != null && (_correctedRt.width != srcW || _correctedRt.height != srcH))
        {
            RenderTexture.ReleaseTemporary(_correctedRt);
            _correctedRt = null;
        }
        if (_correctedRt == null)
            _correctedRt = RenderTexture.GetTemporary(srcW, srcH, 0, RenderTextureFormat.ARGB32);

        // Build UV transform matrix for rotation + mirror correction
        if (angle != 0 || mirrored)
        {
            if (_correctionMat == null)
            {
                Shader shader = Shader.Find("Hidden/BlitCopy");
                if (shader == null) shader = Shader.Find("Unlit/Texture");
                _correctionMat = new Material(shader);
            }

            // Compute UV transform: rotate around (0.5, 0.5), then optionally mirror Y
            Matrix4x4 uvMatrix = GetUVTransform(angle, mirrored);

            // Use GPU to transform
            RenderTexture prev = RenderTexture.active;
            RenderTexture.active = _correctedRt;
            GL.Clear(true, true, Color.black);

            GL.PushMatrix();
            GL.LoadOrtho();

            _correctionMat.mainTexture = _captureTex;
            _correctionMat.SetPass(0);

            // Draw a quad with transformed UVs
            GL.Begin(GL.QUADS);
            Vector4 bl = uvMatrix * new Vector4(0, 0, 0, 1);
            Vector4 br = uvMatrix * new Vector4(1, 0, 0, 1);
            Vector4 tr = uvMatrix * new Vector4(1, 1, 0, 1);
            Vector4 tl = uvMatrix * new Vector4(0, 1, 0, 1);

            GL.TexCoord2(bl.x, bl.y); GL.Vertex3(0, 0, 0);
            GL.TexCoord2(br.x, br.y); GL.Vertex3(1, 0, 0);
            GL.TexCoord2(tr.x, tr.y); GL.Vertex3(1, 1, 0);
            GL.TexCoord2(tl.x, tl.y); GL.Vertex3(0, 1, 0);
            GL.End();

            GL.PopMatrix();
            RenderTexture.active = prev;
        }
        else
        {
            // No correction needed — just blit directly
            Graphics.Blit(_captureTex, _correctedRt);
        }

        // Step 2: Downscale corrected RT to output size
        if (_rt != null && (_rt.width != outW || _rt.height != outH))
        {
            RenderTexture.ReleaseTemporary(_rt);
            _rt = null;
        }
        if (_rt == null)
            _rt = RenderTexture.GetTemporary(outW, outH, 0, RenderTextureFormat.ARGB32);

        Graphics.Blit(_correctedRt, _rt);

        if (_scaledTex == null || _scaledTex.width != outW || _scaledTex.height != outH)
        {
            if (_scaledTex != null) Destroy(_scaledTex);
            _scaledTex = new Texture2D(outW, outH, TextureFormat.RGB24, false);
        }

        RenderTexture prev2 = RenderTexture.active;
        RenderTexture.active = _rt;
        _scaledTex.ReadPixels(new Rect(0, 0, outW, outH), 0, 0);
        _scaledTex.Apply();
        RenderTexture.active = prev2;

        return _scaledTex.EncodeToJPG(jpegQuality);
    }

    /// <summary>
    /// Build a UV transform matrix that rotates around (0.5, 0.5) by the given angle
    /// and optionally mirrors vertically. This ensures the output JPEG has people upright.
    /// </summary>
    private static Matrix4x4 GetUVTransform(int angleDeg, bool mirrorY)
    {
        // Translate to origin, rotate, translate back
        Matrix4x4 toOrigin = Matrix4x4.Translate(new Vector3(-0.5f, -0.5f, 0));
        Matrix4x4 fromOrigin = Matrix4x4.Translate(new Vector3(0.5f, 0.5f, 0));

        // Rotation (counterclockwise to undo camera rotation)
        float rad = -angleDeg * Mathf.Deg2Rad;
        float cos = Mathf.Cos(rad);
        float sin = Mathf.Sin(rad);
        Matrix4x4 rot = Matrix4x4.identity;
        rot.m00 = cos;  rot.m01 = -sin;
        rot.m10 = sin;  rot.m11 = cos;

        Matrix4x4 result = fromOrigin * rot * toOrigin;

        if (mirrorY)
        {
            // Flip Y: v → 1 - v
            Matrix4x4 flip = Matrix4x4.identity;
            flip.m11 = -1f;
            flip.m13 = 1f;
            result = flip * result;
        }

        return result;
    }
}
