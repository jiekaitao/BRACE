import AVFoundation
import Foundation

@MainActor
final class DemoViewModel: ObservableObject {
    @Published var backendBaseURL: String = "https://ws.braceml.com"
    @Published var liveStreamOverrideURL: String = ""
    @Published var playID: String = "demo-play-001"
    @Published var playerID: String = "player-24"
    @Published var coachEmail: String = "coach@example.com"

    @Published var isTripodRunning = false
    @Published var isRecording = false
    @Published var isSocketConnected = false
    @Published var syntheticFeedEnabled = false
    @Published var showSettings = false

    @Published var captureSession: AVCaptureSession?
    @Published var queueDepth: Int = 0
    @Published var lastReport: UploadReport?
    @Published var logs: [String] = []

    private var manager: SidelineCaptureManager?
    private var managerBaseURL: String?
    private var liveSocket: LiveLandmarkWebSocket?
    private var syntheticTask: Task<Void, Never>?

    func startTripodMode() {
        do {
            try ensureManager()
            guard let manager else { return }
            try manager.startTripodAutoMode(
                playID: playID.trimmingCharacters(in: .whitespacesAndNewlines),
                playerID: playerID.trimmingCharacters(in: .whitespacesAndNewlines),
                coachEmail: normalizedCoachEmail
            )
            isTripodRunning = true
            isRecording = true
            appendLog("Tripod auto mode started")
        } catch {
            appendLog("Failed to start tripod mode: \(error.localizedDescription)")
        }
    }

    func stopTripodMode() {
        manager?.stopTripodAutoMode()
        isTripodRunning = false
        isRecording = false
        stopSyntheticFeed()
        appendLog("Tripod auto mode stopped")
    }

    func connectLiveStream() {
        do {
            try ensureManager()
            let endpoint = try buildLiveStreamURL()
            liveSocket = LiveLandmarkWebSocket(
                endpoint: endpoint,
                onCollisionSignal: { [weak self] message in
                    guard let self else { return }
                    self.manager?.handleCollisionMessage(message)
                    self.appendLog("Collision signal: \(message.type)")
                },
                onStatus: { [weak self] status in
                    self?.appendLog(status)
                },
                onMessage: { [weak self] message in
                    guard let self else { return }
                    if message.count < 300 {
                        self.appendLog("WS: \(message)")
                    }
                }
            )
            liveSocket?.connect()
            isSocketConnected = true
            appendLog("Connected live stream to \(endpoint.absoluteString)")
        } catch {
            appendLog("Failed to connect live stream: \(error.localizedDescription)")
        }
    }

    func disconnectLiveStream() {
        liveSocket?.disconnect()
        liveSocket = nil
        isSocketConnected = false
        stopSyntheticFeed()
        appendLog("Disconnected live stream")
    }

    func sendPlayActive() {
        manager?.ingestPlaySignal(
            TripodPlaySignal(
                movingPlayers: 10,
                nearestDistanceMeters: 0.9,
                whistleDetected: false,
                collisionActive: true
            )
        )
        appendLog("Manual signal: play active")
    }

    func sendPlayIdle() {
        manager?.ingestPlaySignal(
            TripodPlaySignal(
                movingPlayers: 0,
                nearestDistanceMeters: 4.0,
                whistleDetected: false,
                collisionActive: false
            )
        )
        appendLog("Manual signal: play idle")
    }

    func sendWhistle() {
        manager?.ingestPlaySignal(
            TripodPlaySignal(
                movingPlayers: 0,
                nearestDistanceMeters: nil,
                whistleDetected: true,
                collisionActive: false
            )
        )
        appendLog("Manual signal: whistle")
    }

    func flushQueue() {
        manager?.flushUploadQueue()
        appendLog("Requested queued upload flush")
    }

    func toggleSyntheticFeed() {
        syntheticFeedEnabled.toggle()
        if syntheticFeedEnabled {
            startSyntheticFeed()
        } else {
            stopSyntheticFeed()
        }
    }

    private func startSyntheticFeed() {
        guard syntheticTask == nil else { return }
        guard liveSocket != nil else {
            syntheticFeedEnabled = false
            appendLog("Connect live stream before starting synthetic feed")
            return
        }
        appendLog("Synthetic landmark feed started")
        syntheticTask = Task { [weak self] in
            guard let self else { return }
            var frame = 0
            while !Task.isCancelled {
                let nowMs = Date().timeIntervalSince1970 * 1000.0
                let phase = frame % 360
                let collisionBurst = phase >= 120 && phase < 138
                let baseX = 320.0 + sin(Double(frame) * 0.08) * 3.0
                let headX = collisionBurst ? baseX + 140.0 : baseX
                let headY = 180.0

                liveSocket?.sendFrame(
                    playID: playID,
                    playerID: playerID,
                    timestampMs: nowMs,
                    headX: headX,
                    headY: headY,
                    shoulderWidthPx: 90.0
                )

                if collisionBurst {
                    manager?.ingestPlaySignal(
                        TripodPlaySignal(movingPlayers: 12, nearestDistanceMeters: 0.8, whistleDetected: false, collisionActive: true)
                    )
                } else {
                    manager?.ingestPlaySignal(
                        TripodPlaySignal(movingPlayers: 6, nearestDistanceMeters: 1.6, whistleDetected: false, collisionActive: false)
                    )
                }

                if phase == 260 {
                    manager?.ingestPlaySignal(
                        TripodPlaySignal(movingPlayers: 0, nearestDistanceMeters: nil, whistleDetected: true, collisionActive: false)
                    )
                }

                frame += 1
                try? await Task.sleep(nanoseconds: 16_000_000)
            }
        }
    }

    private func stopSyntheticFeed() {
        syntheticTask?.cancel()
        syntheticTask = nil
        syntheticFeedEnabled = false
    }

    private func ensureManager() throws {
        let base = try uploadBaseURL()
        if manager == nil || managerBaseURL != base.absoluteString {
            manager = SidelineCaptureManager(uploadBaseURL: base)
            managerBaseURL = base.absoluteString
            wireManagerCallbacks()
            captureSession = manager?.captureSessionForPreview()
            appendLog("Capture manager initialized with \(base.absoluteString)")
        }
    }

    private func wireManagerCallbacks() {
        manager?.onStatusUpdate = { [weak self] status in
            self?.appendLog("Capture: \(status)")
        }
        manager?.onUploadReport = { [weak self] report in
            self?.lastReport = report
            self?.appendLog(
                "Report received: \(report.riskLevel), v=\(String(format: "%.2f", report.linearVelocityMs)) m/s, r=\(String(format: "%.1f", report.rotationalVelocityDegs)) deg/s"
            )
        }
        manager?.onUploadFailed = { [weak self] error in
            self?.appendLog("Upload error: \(error)")
        }
        manager?.onQueueDepthChanged = { [weak self] depth in
            self?.queueDepth = depth
        }
    }

    private func uploadBaseURL() throws -> URL {
        try Self.validatedUploadBaseURL(from: backendBaseURL)
    }

    private func buildLiveStreamURL() throws -> URL {
        let override = liveStreamOverrideURL.trimmingCharacters(in: .whitespacesAndNewlines)
        if !override.isEmpty {
            guard let wsURL = URL(string: override),
                  let scheme = wsURL.scheme?.lowercased(),
                  scheme == "ws" || scheme == "wss" else {
                throw NSError(domain: "DemoViewModel", code: 101, userInfo: [NSLocalizedDescriptionKey: "Live stream URL must start with ws:// or wss://"])
            }
            return wsURL
        }

        let base = try uploadBaseURL()
        return try Self.defaultLiveStreamURL(fromUploadBaseURL: base)
    }

    private var normalizedCoachEmail: String? {
        let trimmed = coachEmail.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? nil : trimmed
    }

    private func appendLog(_ line: String) {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss"
        let stamp = formatter.string(from: Date())
        logs.insert("[\(stamp)] \(line)", at: 0)
        if logs.count > 200 {
            logs = Array(logs.prefix(200))
        }
    }

    static func validatedUploadBaseURL(from input: String) throws -> URL {
        let trimmed = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let url = URL(string: trimmed), let scheme = url.scheme?.lowercased(),
              scheme == "http" || scheme == "https" else {
            throw NSError(domain: "DemoViewModel", code: 100, userInfo: [NSLocalizedDescriptionKey: "Backend URL must start with http:// or https://"])
        }
        return url
    }

    static func defaultLiveStreamURL(fromUploadBaseURL uploadBaseURL: URL) throws -> URL {
        guard var components = URLComponents(url: uploadBaseURL, resolvingAgainstBaseURL: false) else {
            throw NSError(domain: "DemoViewModel", code: 102, userInfo: [NSLocalizedDescriptionKey: "Invalid backend URL"])
        }
        components.scheme = (components.scheme == "https") ? "wss" : "ws"
        components.path = "/live-stream"
        guard let wsURL = components.url else {
            throw NSError(domain: "DemoViewModel", code: 103, userInfo: [NSLocalizedDescriptionKey: "Could not build live stream URL"])
        }
        return wsURL
    }
}
