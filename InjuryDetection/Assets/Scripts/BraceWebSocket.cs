using System;
using System.Text;
using NativeWebSocket;
using Newtonsoft.Json;
using UnityEngine;

/// <summary>
/// Manages WebSocket connection to the BRACE server.
/// Sends binary frames (8-byte LE timestamp + JPEG) and receives JSON analysis responses.
/// Auto-reconnects on disconnect.
/// </summary>
public class BraceWebSocket : MonoBehaviour
{
    [Header("Connection")]
    [SerializeField] private string serverUrl = "ws://192.168.1.100:8001/ws/analyze?mode=webcam&client=vr";

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

    public async void SendFrame(byte[] jpeg)
    {
        if (!_connected || jpeg == null || jpeg.Length == 0) return;

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
    // Subject selection
    // ------------------------------------------------------------------

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
            _framesReceived++;
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[BRACE] JSON parse error: {e.Message}\nFirst 200 chars: {json.Substring(0, Mathf.Min(200, json.Length))}");
        }
    }
}
