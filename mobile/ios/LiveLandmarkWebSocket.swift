import Foundation

final class LiveLandmarkWebSocket {
    private let session: URLSession
    private(set) var endpoint: URL
    private var socketTask: URLSessionWebSocketTask?
    private let onCollisionSignal: (CollisionControlMessage) -> Void
    private let onStatus: ((String) -> Void)?
    private let onMessage: ((String) -> Void)?

    init(
        endpoint: URL,
        onCollisionSignal: @escaping (CollisionControlMessage) -> Void,
        onStatus: ((String) -> Void)? = nil,
        onMessage: ((String) -> Void)? = nil
    ) {
        self.session = URLSession(configuration: .default)
        self.endpoint = endpoint
        self.onCollisionSignal = onCollisionSignal
        self.onStatus = onStatus
        self.onMessage = onMessage
    }

    func updateEndpoint(_ endpoint: URL) {
        self.endpoint = endpoint
        onStatus?("Updated live stream endpoint: \(endpoint.absoluteString)")
    }

    func connect() {
        socketTask = session.webSocketTask(with: endpoint)
        socketTask?.resume()
        onStatus?("WebSocket connected")
        receiveLoop()
        sendPing()
    }

    func disconnect() {
        socketTask?.cancel(with: .goingAway, reason: nil)
        socketTask = nil
        onStatus?("WebSocket disconnected")
    }

    func sendPing() {
        guard let socketTask else { return }
        socketTask.send(.string("{\"type\":\"ping\"}")) { [weak self] error in
            if let error {
                self?.onStatus?("WebSocket ping failed: \(error.localizedDescription)")
            }
        }
    }

    func sendFrame(
        playID: String,
        playerID: String,
        timestampMs: Double,
        headX: Double,
        headY: Double,
        shoulderWidthPx: Double
    ) {
        guard let socketTask else { return }
        let payload: [String: Any] = [
            "play_id": playID,
            "player_id": playerID,
            "timestamp_ms": timestampMs,
            "head": ["x": headX, "y": headY],
            "shoulder_width_px": shoulderWidthPx
        ]
        guard let data = try? JSONSerialization.data(withJSONObject: payload, options: []),
              let text = String(data: data, encoding: .utf8) else {
            return
        }
        socketTask.send(.string(text)) { [weak self] error in
            if let error {
                self?.onStatus?("WebSocket send failed: \(error.localizedDescription)")
            }
        }
    }

    private func receiveLoop() {
        socketTask?.receive { [weak self] result in
            guard let self else { return }
            switch result {
            case .failure(let error):
                self.onStatus?("WebSocket receive failed: \(error.localizedDescription)")
                return
            case .success(let message):
                let text: String?
                switch message {
                case .string(let messageText):
                    text = messageText
                case .data(let data):
                    text = String(data: data, encoding: .utf8)
                @unknown default:
                    text = nil
                }
                if let text {
                    self.onMessage?(text)
                    if let data = text.data(using: .utf8),
                       let signal = try? JSONDecoder().decode(CollisionControlMessage.self, from: data) {
                        self.onCollisionSignal(signal)
                    }
                }
                self.receiveLoop()
            }
        }
    }
}
