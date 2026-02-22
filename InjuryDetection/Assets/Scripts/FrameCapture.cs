using UnityEngine;
#if UNITY_ANDROID && !UNITY_EDITOR
using UnityEngine.Android;
#endif

/// <summary>
/// Captures Quest 3 passthrough camera frames as JPEG bytes.
/// Downscales to 480p and feeds frames to BraceWebSocket at the throttled rate.
/// Handles HorizonOS camera permission, rotation/mirror correction, and camera cycling.
/// Auto-finds BraceWebSocket if not assigned in Inspector.
/// </summary>
public class FrameCapture : MonoBehaviour
{
    // HorizonOS permission required for Quest headset cameras
    private const string HEADSET_CAMERA_PERM = "horizonos.permission.HEADSET_CAMERA";

    [Header("References")]
    [SerializeField] private BraceWebSocket braceWs;

    [Header("Capture Settings")]
    [SerializeField] private int targetHeight = 480;
    [SerializeField] [Range(1, 100)] private int jpegQuality = 65;

    [Header("Black Frame Detection")]
    [Tooltip("If brightness stays below this for blackFrameThreshold frames, try next camera")]
    [SerializeField] private int blackFrameLimit = 10;

    private WebCamTexture _webcam;
    private Texture2D _captureTex;
    private Texture2D _scaledTex;
    private RenderTexture _rt;
    private int _scaledH;
    private bool _ready;
    private int _framesSent;
    private bool _cameraInfoLogged;
    private float _avgBrightness;
    private int _nonBlackPixels;
    private int _totalSampled;
    private int _blackFrameCount;
    private int _currentCameraIndex;
    private int _totalCameras;
    private bool _triedAllCameras;
    private string _permissionStatus = "unknown";

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

    /// <summary>Permission status (for debug HUD).</summary>
    public string PermissionStatus => _permissionStatus;

    /// <summary>Current camera index / total (for debug HUD).</summary>
    public string CameraIndex => $"{_currentCameraIndex}/{_totalCameras}";

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

    /// <summary>Switch to a specific camera index. Call from outside or via controller.</summary>
    public void SwitchToCamera(int index)
    {
        if (index >= 0 && index < _totalCameras && index != _currentCameraIndex)
        {
            Debug.Log($"[BRACE] Manual switch to camera {index}");
            _triedAllCameras = false;
            StartCameraAtIndex(index);
        }
    }

    /// <summary>Cycle to the next camera.</summary>
    public void CycleNextCamera()
    {
        int next = (_currentCameraIndex + 1) % Mathf.Max(1, _totalCameras);
        Debug.Log($"[BRACE] Cycling to camera {next}/{_totalCameras}");
        _triedAllCameras = false;
        StartCameraAtIndex(next);
    }

    void Update()
    {
        // Manual camera switch: Right controller B button or Left controller Y button
        if (OVRInput.GetDown(OVRInput.Button.Two, OVRInput.Controller.RTouch) ||
            OVRInput.GetDown(OVRInput.Button.Two, OVRInput.Controller.LTouch))
        {
            CycleNextCamera();
        }

        if (!_ready) return;
        if (_webcam == null || !_webcam.isPlaying) return;
        if (!_webcam.didUpdateThisFrame) return;

        // Log camera properties once after first real frame
        if (!_cameraInfoLogged && _webcam.width > 32)
        {
            _cameraInfoLogged = true;
            Debug.Log($"[BRACE] Camera info: rotation={_webcam.videoRotationAngle}° mirrored={_webcam.videoVerticallyMirrored} res={_webcam.width}x{_webcam.height}");
        }

        // Always sample brightness (independent of WebSocket readiness)
        SampleBrightness();

        // Auto-cycle if black frames persist
        if (!_triedAllCameras && _avgBrightness < 2f)
        {
            _blackFrameCount++;
            if (_blackFrameCount >= blackFrameLimit)
            {
                Debug.LogWarning($"[BRACE] Camera {_currentCameraIndex} black for {_blackFrameCount} frames, trying next...");
                TryNextCamera();
                return;
            }
        }
        else
        {
            _blackFrameCount = 0; // reset if we get a good frame
        }

        // Send frame to server (throttled)
        if (braceWs != null && braceWs.ReadyToSend())
        {
            byte[] jpeg = CaptureJpeg();
            if (jpeg != null)
            {
                braceWs.SendFrame(jpeg);
                _framesSent++;
            }
        }
    }

    void OnDestroy()
    {
        if (_webcam != null && _webcam.isPlaying)
            _webcam.Stop();

        if (_rt != null)
            RenderTexture.ReleaseTemporary(_rt);

        if (_scaledTex != null)
            Destroy(_scaledTex);

        if (_captureTex != null)
            Destroy(_captureTex);
    }

    // ------------------------------------------------------------------
    // Permission + Camera init
    // ------------------------------------------------------------------

    private void RequestCameraPermissionAndStart()
    {
#if UNITY_ANDROID && !UNITY_EDITOR
        // Quest 3 requires horizonos.permission.HEADSET_CAMERA (NOT android.permission.CAMERA)
        bool hasHeadsetPerm = Permission.HasUserAuthorizedPermission(HEADSET_CAMERA_PERM);
        bool hasAndroidPerm = Permission.HasUserAuthorizedPermission(Permission.Camera);

        Debug.Log($"[BRACE] Permission check: HEADSET_CAMERA={hasHeadsetPerm}, android.CAMERA={hasAndroidPerm}");

        if (!hasHeadsetPerm)
        {
            _permissionStatus = "requesting";
            Debug.Log("[BRACE] Requesting horizonos.permission.HEADSET_CAMERA...");

            var callbacks = new PermissionCallbacks();
            callbacks.PermissionGranted += (perm) =>
            {
                Debug.Log($"[BRACE] Permission granted: {perm}");
                _permissionStatus = "granted";
                // Also request android.permission.CAMERA as fallback
                if (!Permission.HasUserAuthorizedPermission(Permission.Camera))
                {
                    Permission.RequestUserPermission(Permission.Camera);
                }
                StartCamera();
            };
            callbacks.PermissionDenied += (perm) =>
            {
                Debug.LogWarning($"[BRACE] Permission denied: {perm}, trying android.permission.CAMERA...");
                _permissionStatus = "headset_denied";
                // Fall back to standard Android camera permission
                RequestAndroidCameraPermission();
            };
            callbacks.PermissionDeniedAndDontAskAgain += (perm) =>
            {
                Debug.LogWarning($"[BRACE] Permission denied permanently: {perm}, trying android.permission.CAMERA...");
                _permissionStatus = "headset_denied";
                RequestAndroidCameraPermission();
            };
            Permission.RequestUserPermission(HEADSET_CAMERA_PERM, callbacks);
            return;
        }

        _permissionStatus = "granted";
        Debug.Log("[BRACE] HEADSET_CAMERA permission already granted");

        // Ensure android.permission.CAMERA is also granted
        if (!hasAndroidPerm)
        {
            Permission.RequestUserPermission(Permission.Camera);
        }
#else
        _permissionStatus = "editor";
#endif
        StartCamera();
    }

#if UNITY_ANDROID && !UNITY_EDITOR
    private void RequestAndroidCameraPermission()
    {
        if (Permission.HasUserAuthorizedPermission(Permission.Camera))
        {
            _permissionStatus = "android_only";
            StartCamera();
            return;
        }

        var callbacks = new PermissionCallbacks();
        callbacks.PermissionGranted += (perm) =>
        {
            Debug.Log($"[BRACE] Android camera permission granted: {perm}");
            _permissionStatus = "android_only";
            StartCamera();
        };
        callbacks.PermissionDenied += (perm) =>
        {
            Debug.LogError($"[BRACE] All camera permissions DENIED");
            _permissionStatus = "ALL DENIED";
            CameraName = "PERM DENIED";
        };
        callbacks.PermissionDeniedAndDontAskAgain += (perm) =>
        {
            Debug.LogError($"[BRACE] All camera permissions DENIED permanently");
            _permissionStatus = "ALL DENIED";
            CameraName = "PERM DENIED";
        };
        Permission.RequestUserPermission(Permission.Camera, callbacks);
    }
#endif

    private void StartCamera()
    {
        WebCamDevice[] devices = WebCamTexture.devices;
        _totalCameras = devices.Length;
        Debug.Log($"[BRACE] Camera devices found: {devices.Length}");

        if (devices.Length == 0)
        {
            Debug.LogWarning("[BRACE] No camera devices found!");
            CameraName = "NOT FOUND";
            return;
        }

        // Log all devices for debugging
        for (int i = 0; i < devices.Length; i++)
            Debug.Log($"[BRACE]   Camera[{i}]: \"{devices[i].name}\" front={devices[i].isFrontFacing} available={devices[i].availableResolutions?.Length ?? -1}");

        StartCameraAtIndex(_currentCameraIndex);
    }

    private void StartCameraAtIndex(int index)
    {
        WebCamDevice[] devices = WebCamTexture.devices;
        if (index >= devices.Length)
        {
            Debug.LogError("[BRACE] No more cameras to try!");
            _triedAllCameras = true;
            return;
        }

        // Stop previous camera if any
        if (_webcam != null && _webcam.isPlaying)
            _webcam.Stop();

        _currentCameraIndex = index;
        string deviceName = devices[index].name;
        CameraName = deviceName;
        _cameraInfoLogged = false;
        _blackFrameCount = 0;
        _framesSent = 0;

        Debug.Log($"[BRACE] Starting camera[{index}]: \"{deviceName}\" front={devices[index].isFrontFacing}");
        _webcam = new WebCamTexture(deviceName, 1280, 960, 30);
        _webcam.Play();

        // Reset textures for new camera
        if (_captureTex != null) { Destroy(_captureTex); _captureTex = null; }

        _scaledH = targetHeight;
        _ready = true;
    }

    private void TryNextCamera()
    {
        int next = _currentCameraIndex + 1;
        if (next >= _totalCameras)
        {
            Debug.LogError("[BRACE] Tried all cameras — none producing image data. Check camera permissions in Quest Settings.");
            _triedAllCameras = true;
            return;
        }
        StartCameraAtIndex(next);
    }

    // ------------------------------------------------------------------
    // Brightness sampling (runs every frame, independent of JPEG capture)
    // ------------------------------------------------------------------

    private void SampleBrightness()
    {
        if (_webcam == null || _webcam.width < 32) return;

        Color32[] pixels = _webcam.GetPixels32();
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
    }

    // ------------------------------------------------------------------
    // JPEG capture
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

        _captureTex.SetPixels32(_webcam.GetPixels32());
        _captureTex.Apply();

        // Determine output dimensions accounting for rotation
        int angle = _webcam.videoRotationAngle;
        bool mirrored = _webcam.videoVerticallyMirrored;
        bool swapDims = (angle == 90 || angle == 270);

        int srcW = swapDims ? h : w;
        int srcH = swapDims ? w : h;
        int outH = _scaledH;
        int outW = Mathf.RoundToInt((float)srcW / srcH * outH);

        // Downscale via GPU blit (rotation is 0 and mirror is false on Quest, so just blit)
        if (_rt != null && (_rt.width != outW || _rt.height != outH))
        {
            RenderTexture.ReleaseTemporary(_rt);
            _rt = null;
        }
        if (_rt == null)
            _rt = RenderTexture.GetTemporary(outW, outH, 0, RenderTextureFormat.ARGB32);

        Graphics.Blit(_captureTex, _rt);

        if (_scaledTex == null || _scaledTex.width != outW || _scaledTex.height != outH)
        {
            if (_scaledTex != null) Destroy(_scaledTex);
            _scaledTex = new Texture2D(outW, outH, TextureFormat.RGB24, false);
        }

        RenderTexture prev = RenderTexture.active;
        RenderTexture.active = _rt;
        _scaledTex.ReadPixels(new Rect(0, 0, outW, outH), 0, 0);
        _scaledTex.Apply();
        RenderTexture.active = prev;

        return _scaledTex.EncodeToJPG(jpegQuality);
    }
}
