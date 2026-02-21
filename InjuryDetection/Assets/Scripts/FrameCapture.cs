using UnityEngine;

/// <summary>
/// Captures Quest 3 passthrough camera frames as JPEG bytes.
/// Downscales to 480p and feeds frames to BraceWebSocket at the throttled rate.
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
    private int _scaledW;
    private int _scaledH;
    private bool _ready;

    void Start()
    {
        if (braceWs == null)
            braceWs = GetComponent<BraceWebSocket>();

        StartCamera();
    }

    void Update()
    {
        if (!_ready || braceWs == null) return;
        if (!braceWs.ReadyToSend()) return;
        if (!_webcam.didUpdateThisFrame) return;

        byte[] jpeg = CaptureJpeg();
        if (jpeg != null)
            braceWs.SendFrame(jpeg);
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
    // Camera init
    // ------------------------------------------------------------------

    private void StartCamera()
    {
        WebCamDevice[] devices = WebCamTexture.devices;
        if (devices.Length == 0)
        {
            Debug.LogWarning("[BRACE] No camera devices found");
            return;
        }

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

        Debug.Log($"[BRACE] Using camera: {deviceName}");
        _webcam = new WebCamTexture(deviceName, 1280, 960, 30);
        _webcam.Play();

        // Compute scaled dimensions preserving aspect ratio
        _scaledH = targetHeight;
        _scaledW = Mathf.RoundToInt((float)_webcam.requestedWidth / _webcam.requestedHeight * _scaledH);

        _ready = true;
    }

    // ------------------------------------------------------------------
    // JPEG capture
    // ------------------------------------------------------------------

    private byte[] CaptureJpeg()
    {
        if (_webcam == null || !_webcam.isPlaying) return null;

        int w = _webcam.width;
        int h = _webcam.height;

        // Skip if camera hasn't initialized yet (returns 16x16 placeholder)
        if (w < 32 || h < 32) return null;

        // Read pixels from WebCamTexture
        if (_captureTex == null || _captureTex.width != w || _captureTex.height != h)
            _captureTex = new Texture2D(w, h, TextureFormat.RGBA32, false);

        _captureTex.SetPixels32(_webcam.GetPixels32());
        _captureTex.Apply();

        // Downscale via GPU blit
        int outW = Mathf.RoundToInt((float)w / h * _scaledH);
        int outH = _scaledH;

        if (_rt != null && (_rt.width != outW || _rt.height != outH))
        {
            RenderTexture.ReleaseTemporary(_rt);
            _rt = null;
        }
        if (_rt == null)
            _rt = RenderTexture.GetTemporary(outW, outH, 0, RenderTextureFormat.ARGB32);

        Graphics.Blit(_captureTex, _rt);

        // Read back scaled result
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
