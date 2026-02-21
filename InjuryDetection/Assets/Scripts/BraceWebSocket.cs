using System;
using System.Text;
using NativeWebSocket;
using Newtonsoft.Json;
using UnityEngine;

/// <summary>
/// Manages WebSocket connection to the BRACE server.
/// Sends binary frames (8-byte LE timestamp + JPEG) and receives JSON analysis responses.
/// </summary>
public class BraceWebSocket : MonoBehaviour
{
    [Header("Connection")]
    [SerializeField] private string serverUrl = "ws://192.168.1.100:8001/ws/analyze?mode=webcam&client=vr";

    [Header("Frame Throttle")]
    [SerializeField] private int maxFps = 30;
    [SerializeField] private int maxInFlight = 5;

    private WebSocket _ws;
    private float _lastSendTime;
    private int _inFlight;
    private bool _connected;

    /// <summary>Latest parsed response from the server. Read by rendering components each frame.</summary>
    public BraceResponse LatestResponse { get; private set; }

    /// <summary>True when the WebSocket is open and ready to send.</summary>
    public bool IsConnected => _connected;

    /// <summary>Number of frames awaiting server response.</summary>
    public int InFlight => _inFlight;

    /// <summary>Minimum interval between frame sends, derived from maxFps.</summary>
    private float SendInterval => 1f / maxFps;

    // ------------------------------------------------------------------
    // Lifecycle
    // ------------------------------------------------------------------

    async void Start()
    {
        _ws = new WebSocket(serverUrl);

        _ws.OnOpen += () =>
        {
            _connected = true;
            _inFlight = 0;
            Debug.Log("[BRACE] WebSocket connected");
        };

        _ws.OnClose += (code) =>
        {
            _connected = false;
            Debug.Log($"[BRACE] WebSocket closed: {code}");
        };

        _ws.OnError += (err) =>
        {
            Debug.LogError($"[BRACE] WebSocket error: {err}");
        };

        _ws.OnMessage += OnMessage;

        Debug.Log($"[BRACE] Connecting to {serverUrl}");
        await _ws.Connect();
    }

    void Update()
    {
#if !UNITY_WEBGL || UNITY_EDITOR
        _ws?.DispatchMessageQueue();
#endif
    }

    async void OnDestroy()
    {
        if (_ws != null)
        {
            _connected = false;
            await _ws.Close();
        }
    }

    // ------------------------------------------------------------------
    // Sending frames
    // ------------------------------------------------------------------

    /// <summary>
    /// Returns true if enough time has passed and in-flight slots are available.
    /// Call this before CaptureJpeg to avoid unnecessary work.
    /// </summary>
    public bool ReadyToSend()
    {
        return _connected
            && Time.time - _lastSendTime >= SendInterval
            && _inFlight < maxInFlight;
    }

    /// <summary>
    /// Send a JPEG frame to the server for analysis.
    /// Binary format: [8-byte float64 LE timestamp] + [JPEG bytes].
    /// </summary>
    public async void SendFrame(byte[] jpeg)
    {
        if (!_connected || jpeg == null || jpeg.Length == 0) return;

        // Pack timestamp (little-endian float64) + JPEG
        double videoTime = Time.timeAsDouble;
        byte[] timeBytes = BitConverter.GetBytes(videoTime);
        if (!BitConverter.IsLittleEndian)
            Array.Reverse(timeBytes);

        byte[] message = new byte[8 + jpeg.Length];
        Buffer.BlockCopy(timeBytes, 0, message, 0, 8);
        Buffer.BlockCopy(jpeg, 0, message, 8, jpeg.Length);

        _inFlight++;
        _lastSendTime = Time.time;

        try
        {
            await _ws.Send(message);
        }
        catch (Exception e)
        {
            _inFlight = Mathf.Max(0, _inFlight - 1);
            Debug.LogWarning($"[BRACE] Send failed: {e.Message}");
        }
    }

    // ------------------------------------------------------------------
    // Subject selection (VR → server)
    // ------------------------------------------------------------------

    /// <summary>
    /// Tell the server which subject to send detailed analysis for.
    /// Pass null to deselect (all subjects get minimal data).
    /// </summary>
    public async void SelectSubject(int? subjectId)
    {
        if (!_connected) return;

        string msg = subjectId.HasValue
            ? $"{{\"type\":\"select_subject\",\"subject_id\":{subjectId.Value}}}"
            : "{\"type\":\"select_subject\",\"subject_id\":null}";

        try
        {
            await _ws.SendText(msg);
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[BRACE] SelectSubject failed: {e.Message}");
        }
    }

    // ------------------------------------------------------------------
    // Receiving responses
    // ------------------------------------------------------------------

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
            Debug.LogWarning($"[BRACE] JSON parse error: {e.Message}");
        }
    }
}
