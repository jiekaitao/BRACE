using System;
using System.Text;
using NativeWebSocket;
using Newtonsoft.Json;
using UnityEngine;

/// <summary>
/// Manages WebSocket connection to the BRACE server.
/// Sends binary frames (8-byte LE timestamp + 4-byte LE selection + JPEG) and receives JSON analysis responses.
/// Auto-reconnects on disconnect.
///
/// Binary frame format v2:
///   [8-byte LE double: video_time][4-byte LE int32: selected_subject_id][JPEG bytes]
///   Selection values: -2 = no change, -1 = deselect, >= 0 = subject ID.
/// </summary>
public class BraceWebSocket : MonoBehaviour
{
    [Header("Connection")]
    [SerializeField] private string serverUrl = "wss://ws.braceml.com/ws/analyze?mode=webcam&client=vr";

    [Header("Frame Throttle")]
    [SerializeField] private int maxFps = 30;
    [SerializeField] private int maxInFlight = 5;

    [Header("Reconnect")]
    [SerializeField] private float reconnectDelay = 3f;

    private WebSocket _ws;
    private float _lastSendTime;
    private int _inFlight;
    private bool _connected;
    private int _framesReceived;
    private float _reconnectTimer;
    private bool _shouldReconnect;

    // Subject selection embedded in binary frames (-2 = no change)
    private int _pendingSelect = NO_CHANGE;
    private const int NO_CHANGE = -2;
    private const int DESELECT = -1;

    // Reuse settings to avoid GC pressure at 30fps on Quest 3
    private static readonly JsonSerializerSettings _jsonSettings = new JsonSerializerSettings
    {
        FloatParseHandling = FloatParseHandling.Double,       // Accept NaN/Infinity
        MissingMemberHandling = MissingMemberHandling.Ignore, // Ignore unknown fields
        NullValueHandling = NullValueHandling.Ignore,         // Skip null → value type
    };

    /// <summary>Latest parsed response from the server.</summary>
    public BraceResponse LatestResponse { get; private set; }

    /// <summary>True when the WebSocket is open and ready to send.</summary>
    public bool IsConnected => _connected;

    /// <summary>Number of frames awaiting server response.</summary>
    public int InFlight => _inFlight;

    /// <summary>Total frames received from server (for debug HUD).</summary>
    public int FramesReceived => _framesReceived;

    /// <summary>The URL we are connecting to (for debug HUD).</summary>
    public string ServerUrl => serverUrl;

    /// <summary>Last error message (for debug HUD).</summary>
    public string LastError { get; private set; } = "";

    /// <summary>Current pending selection value (for debug HUD). -2=none, -1=deselect, >=0=subject.</summary>
    public int PendingSelect => _pendingSelect;

    private float SendInterval => 1f / maxFps;

    // ------------------------------------------------------------------
    // Lifecycle
    // ------------------------------------------------------------------

    void Start()
    {
        Connect();
    }

    void Update()
    {
#if !UNITY_WEBGL || UNITY_EDITOR
        _ws?.DispatchMessageQueue();
#endif

        // Auto-reconnect
        if (_shouldReconnect && !_connected)
        {
            _reconnectTimer -= Time.deltaTime;
            if (_reconnectTimer <= 0f)
            {
                _shouldReconnect = false;
                Debug.Log("[BRACE] Reconnecting...");
                Connect();
            }
        }
    }

    async void OnDestroy()
    {
        _shouldReconnect = false;
        if (_ws != null)
        {
            _connected = false;
            await _ws.Close();
        }
    }

    private async void Connect()
    {
        Debug.Log($"[BRACE] Connecting to {serverUrl}");

        _ws = new WebSocket(serverUrl);

        _ws.OnOpen += () =>
        {
            _connected = true;
            _inFlight = 0;
            _shouldReconnect = false;
            Debug.Log("[BRACE] WebSocket CONNECTED");
        };

        _ws.OnClose += (code) =>
        {
            _connected = false;
            Debug.Log($"[BRACE] WebSocket CLOSED: {code}");
            ScheduleReconnect();
        };

        _ws.OnError += (err) =>
        {
            LastError = err;
            Debug.LogError($"[BRACE] WebSocket ERROR: {err}");
            ScheduleReconnect();
        };

        _ws.OnMessage += OnMessage;

        try
        {
            await _ws.Connect();
        }
        catch (Exception e)
        {
            LastError = e.Message;
            Debug.LogError($"[BRACE] Connect exception: {e.Message}");
            ScheduleReconnect();
        }
    }

    private void ScheduleReconnect()
    {
        if (!_shouldReconnect)
        {
            _shouldReconnect = true;
            _reconnectTimer = reconnectDelay;
            Debug.Log($"[BRACE] Will reconnect in {reconnectDelay}s");
        }
    }

    // ------------------------------------------------------------------
    // Sending frames
    // ------------------------------------------------------------------

    public bool ReadyToSend()
    {
        return _connected
            && Time.time - _lastSendTime >= SendInterval
            && _inFlight < maxInFlight;
    }

    /// <summary>
    /// Send a camera frame with embedded subject selection.
    /// Binary format v2: [8B timestamp][4B int32 selection][JPEG]
    /// </summary>
    public async void SendFrame(byte[] jpeg)
    {
        if (!_connected || jpeg == null || jpeg.Length == 0) return;

        double videoTime = Time.timeAsDouble;
        byte[] timeBytes = BitConverter.GetBytes(videoTime);
        if (!BitConverter.IsLittleEndian)
            Array.Reverse(timeBytes);

        // Consume the pending selection (send once, then revert to NO_CHANGE)
        int sel = _pendingSelect;
        byte[] selBytes = BitConverter.GetBytes(sel);
        if (!BitConverter.IsLittleEndian)
            Array.Reverse(selBytes);

        // After sending the selection once, revert to no-change so we don't
        // keep re-sending it every frame
        if (sel != NO_CHANGE)
            _pendingSelect = NO_CHANGE;

        // v2 format: [8B time][4B selection][JPEG]
        byte[] message = new byte[12 + jpeg.Length];
        Buffer.BlockCopy(timeBytes, 0, message, 0, 8);
        Buffer.BlockCopy(selBytes, 0, message, 8, 4);
        Buffer.BlockCopy(jpeg, 0, message, 12, jpeg.Length);

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
    // Subject selection
    // ------------------------------------------------------------------

    /// <summary>
    /// Select a subject. The selection is embedded in the next binary frame
    /// (bypasses SendText which doesn't work reliably on Quest 3 Android).
    /// </summary>
    public void SelectSubject(int? subjectId)
    {
        _pendingSelect = subjectId.HasValue ? subjectId.Value : DESELECT;
        Debug.Log($"[BRACE] SelectSubject queued: {_pendingSelect}");
    }

    // ------------------------------------------------------------------
    // Receiving responses
    // ------------------------------------------------------------------

    /// <summary>Last JSON parse error message (for DebugHUD).</summary>
    public string LastParseError { get; private set; } = "";

    /// <summary>Size in bytes of the last received message (for DebugHUD).</summary>
    public int LastMessageSize { get; private set; }

    /// <summary>Count of JSON parse failures (for DebugHUD).</summary>
    public int ParseErrors { get; private set; }

    private void OnMessage(byte[] data)
    {
        _inFlight = Mathf.Max(0, _inFlight - 1);
        LastMessageSize = data.Length;

        string json = Encoding.UTF8.GetString(data);
        try
        {
            LatestResponse = JsonConvert.DeserializeObject<BraceResponse>(json, _jsonSettings);
            _framesReceived++;
            LastParseError = "";  // Clear on success
        }
        catch (Exception e)
        {
            ParseErrors++;
            LastParseError = e.Message;
            Debug.LogWarning($"[BRACE] JSON parse error #{ParseErrors}: {e.Message}\nJSON size: {data.Length} bytes\nFirst 500 chars: {json.Substring(0, Mathf.Min(500, json.Length))}");
        }
    }
}
