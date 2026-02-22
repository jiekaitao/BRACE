import Foundation
import Combine
import os

private let log = Logger(subsystem: "com.brace.capture", category: "WebSocket")

/// WebSocket client using URLSessionWebSocketTask.
/// Speaks the same binary protocol as the BRACE web frontend:
///   Send: [8-byte Float64 LE timestamp] + [JPEG bytes]
///   Recv: JSON text (MultiSubjectFrameResponse)
final class BRACEWebSocket: ObservableObject {

    // MARK: - Published State

    enum ConnectionState: String {
        case disconnected, connecting, connected
    }

    @Published var state: ConnectionState = .disconnected
    @Published var fpsIn: Double = 0
    @Published var fpsOut: Double = 0
    @Published var rttMs: Double = 0
    @Published var lastFrame: MultiSubjectFrameResponse?
    @Published var landmarks: [Landmark] = Array(repeating: Landmark(), count: 33)

    // MARK: - Configuration

    var serverURL: String = "wss://ws.braceml.com/ws/analyze?mode=webcam&fps=240&client=ios" {
        didSet { reconnect() }
    }

    // MARK: - Backpressure

    private let maxInFlight = 2
    private var framesInFlight = 0

    /// Returns true if a frame can be sent (backpressure not exceeded).
    var canSend: Bool { framesInFlight < maxInFlight && state == .connected }

    // MARK: - Private

    private var task: URLSessionWebSocketTask?
    private var urlSession: URLSession?
    private var retryDelay: TimeInterval = 1.0
    private let maxRetryDelay: TimeInterval = 10.0
    private var isStopped = false

    // RTT tracking
    private var sendTimesQueue: [CFAbsoluteTime] = []

    // FPS tracking
    private var sentCount = 0
    private var recvCount = 0
    private var fpsTimer = CFAbsoluteTimeGetCurrent()

    // MARK: - Public API

    func connect() {
        isStopped = false
        retryDelay = 1.0
        doConnect()
    }

    func disconnect() {
        isStopped = true
        task?.cancel(with: .normalClosure, reason: nil)
        task = nil
        state = .disconnected
    }

    /// Send a JPEG frame with the BRACE binary protocol.
    func sendFrame(jpegData: Data, timestamp: Double) {
        guard canSend else { return }

        // Build binary message: 8-byte Float64 LE + JPEG
        var ts = timestamp
        var header = Data(count: 8)
        header.withUnsafeMutableBytes { ptr in
            ptr.storeBytes(of: ts, as: Double.self)
        }
        let message = header + jpegData

        framesInFlight += 1
        sentCount += 1
        sendTimesQueue.append(CFAbsoluteTimeGetCurrent())

        task?.send(.data(message)) { [weak self] error in
            if let error {
                log.error("Send error: \(error.localizedDescription)")
                DispatchQueue.main.async {
                    self?.framesInFlight = max(0, (self?.framesInFlight ?? 1) - 1)
                }
            }
        }
    }

    // MARK: - Private

    private func doConnect() {
        guard !isStopped else { return }

        state = .connecting

        guard let url = URL(string: serverURL) else {
            log.error("Invalid URL: \(self.serverURL)")
            return
        }

        let config = URLSessionConfiguration.default
        config.waitsForConnectivity = true
        urlSession = URLSession(configuration: config)

        let wsTask = urlSession!.webSocketTask(with: url)
        self.task = wsTask
        wsTask.resume()

        // Start receive loop
        receiveMessage()

        // Reset state
        framesInFlight = 0
        sendTimesQueue.removeAll()
        sentCount = 0
        recvCount = 0
        fpsTimer = CFAbsoluteTimeGetCurrent()

        state = .connected
        retryDelay = 1.0
        log.info("Connected to \(self.serverURL)")
    }

    private func receiveMessage() {
        task?.receive { [weak self] result in
            guard let self else { return }

            switch result {
            case .success(let message):
                DispatchQueue.main.async {
                    self.handleMessage(message)
                }
                self.receiveMessage() // continue receiving
            case .failure(let error):
                log.error("Receive error: \(error.localizedDescription)")
                DispatchQueue.main.async {
                    self.handleDisconnect()
                }
            }
        }
    }

    private func handleMessage(_ message: URLSessionWebSocketTask.Message) {
        framesInFlight = max(0, framesInFlight - 1)
        recvCount += 1

        // RTT tracking
        if !sendTimesQueue.isEmpty {
            let sendTime = sendTimesQueue.removeFirst()
            let rtt = (CFAbsoluteTimeGetCurrent() - sendTime) * 1000.0
            rttMs = rtt
        }

        // FPS tracking (1-second window)
        let now = CFAbsoluteTimeGetCurrent()
        let elapsed = now - fpsTimer
        if elapsed >= 1.0 {
            fpsIn = Double(recvCount) / elapsed
            fpsOut = Double(sentCount) / elapsed
            recvCount = 0
            sentCount = 0
            fpsTimer = now
        }

        // Parse JSON
        switch message {
        case .string(let text):
            parseResponse(text)
        case .data(let data):
            if let text = String(data: data, encoding: .utf8) {
                parseResponse(text)
            }
        @unknown default:
            break
        }
    }

    private func parseResponse(_ text: String) {
        guard let data = text.data(using: .utf8) else { return }

        do {
            let frame = try JSONDecoder().decode(MultiSubjectFrameResponse.self, from: data)
            let expanded = expandFirstSubjectLandmarks(frame)
            lastFrame = frame
            landmarks = expanded
        } catch {
            log.debug("Parse skip: \(error.localizedDescription)")
        }
    }

    private func expandFirstSubjectLandmarks(_ frame: MultiSubjectFrameResponse) -> [Landmark] {
        guard let firstSubject = frame.subjects.values.first,
              let sparse = firstSubject.landmarks else {
            return Array(repeating: Landmark(), count: 33)
        }
        return expandLandmarks(sparse)
    }

    private func handleDisconnect() {
        state = .disconnected

        guard !isStopped else { return }

        log.info("Reconnecting in \(self.retryDelay)s")
        DispatchQueue.global().asyncAfter(deadline: .now() + retryDelay) { [weak self] in
            DispatchQueue.main.async {
                self?.doConnect()
            }
        }
        retryDelay = min(retryDelay * 1.5, maxRetryDelay)
    }

    private func reconnect() {
        disconnect()
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
            self?.connect()
        }
    }
}
