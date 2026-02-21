import AVFoundation
import Foundation
import Network

#if canImport(UIKit)
import UIKit
#endif

struct CollisionControlMessage: Codable {
    let type: String
    let record240fps: Bool?
    let stopAfterMs: Int?
    let eventID: String?

    enum CodingKeys: String, CodingKey {
        case type
        case record240fps = "record_240fps"
        case stopAfterMs = "stop_after_ms"
        case eventID = "event_id"
    }
}

struct UploadReport: Codable {
    let playID: String
    let playerID: String
    let impactDetected: Bool
    let riskLevel: String
    let linearVelocityMs: Double
    let rotationalVelocityDegs: Double
    let impactDurationMs: Double
    let impactLocation: String
    let frameOfImpact: Int
    let recommendation: String

    enum CodingKeys: String, CodingKey {
        case playID = "play_id"
        case playerID = "player_id"
        case impactDetected = "impact_detected"
        case riskLevel = "risk_level"
        case linearVelocityMs = "linear_velocity_ms"
        case rotationalVelocityDegs = "rotational_velocity_degs"
        case impactDurationMs = "impact_duration_ms"
        case impactLocation = "impact_location"
        case frameOfImpact = "frame_of_impact"
        case recommendation
    }
}

struct TripodPlaySignal {
    let movingPlayers: Int
    let nearestDistanceMeters: Double?
    let whistleDetected: Bool
    let collisionActive: Bool

    init(
        movingPlayers: Int = 0,
        nearestDistanceMeters: Double? = nil,
        whistleDetected: Bool = false,
        collisionActive: Bool = false
    ) {
        self.movingPlayers = movingPlayers
        self.nearestDistanceMeters = nearestDistanceMeters
        self.whistleDetected = whistleDetected
        self.collisionActive = collisionActive
    }
}

final class ClipUploadClient {
    private let baseURL: URL
    private let session: URLSession

    init(baseURL: URL, session: URLSession = .shared) {
        self.baseURL = baseURL
        self.session = session
    }

    func uploadClip(
        fileURL: URL,
        playID: String,
        playerID: String,
        coachEmail: String?
    ) async throws -> UploadReport {
        let endpoint = baseURL.appendingPathComponent("upload-clip")
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"

        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        request.httpBody = try makeMultipartBody(
            fileURL: fileURL,
            playID: playID,
            playerID: playerID,
            coachEmail: coachEmail,
            boundary: boundary
        )

        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse, (200...299).contains(http.statusCode) else {
            throw NSError(domain: "ClipUploadClient", code: 1)
        }
        return try JSONDecoder().decode(UploadReport.self, from: data)
    }

    private func makeMultipartBody(
        fileURL: URL,
        playID: String,
        playerID: String,
        coachEmail: String?,
        boundary: String
    ) throws -> Data {
        var body = Data()
        let lineBreak = "\r\n"

        func appendField(name: String, value: String) {
            body.append("--\(boundary)\(lineBreak)".data(using: .utf8)!)
            body.append("Content-Disposition: form-data; name=\"\(name)\"\(lineBreak)\(lineBreak)".data(using: .utf8)!)
            body.append("\(value)\(lineBreak)".data(using: .utf8)!)
        }

        appendField(name: "play_id", value: playID)
        appendField(name: "player_id", value: playerID)
        if let coachEmail, !coachEmail.isEmpty {
            appendField(name: "coach_email", value: coachEmail)
        }

        let fileData = try Data(contentsOf: fileURL)
        body.append("--\(boundary)\(lineBreak)".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"file\"; filename=\"\(fileURL.lastPathComponent)\"\(lineBreak)".data(using: .utf8)!)
        body.append("Content-Type: video/mp4\(lineBreak)\(lineBreak)".data(using: .utf8)!)
        body.append(fileData)
        body.append(lineBreak.data(using: .utf8)!)
        body.append("--\(boundary)--\(lineBreak)".data(using: .utf8)!)

        return body
    }
}

final class SidelineCaptureManager: NSObject {
    private struct RecordedSegment {
        let url: URL
        let startedAt: Date
        let endedAt: Date
    }

    private let captureSession = AVCaptureSession()
    private let movieOutput = AVCaptureMovieFileOutput()
    private var cameraInput: AVCaptureDeviceInput?
    private var stopTimer: DispatchWorkItem?
    private var segmentStopTimer: DispatchWorkItem?
    private var forceCutTimer: DispatchWorkItem?

    private let uploadClient: ClipUploadClient
    private let uploadQueue = TripodUploadQueue()

    private var networkMonitor: NWPathMonitor?
    private let networkMonitorQueue = DispatchQueue(label: "brace.sideline.network-monitor")
    private var networkAvailable = true
    private var queueWorkerTask: Task<Void, Never>?

    private var activePlayID: String = ""
    private var activePlayerID: String = ""
    private var activeCoachEmail: String?

    private var whistleReceived = false
    private var latestRecordedClip: URL?

    private var tripodAutoModeEnabled = false
    private var currentSegmentStart: Date?
    private var segmentHistory: [RecordedSegment] = []
    private var playActive = false
    private var playStartTime: Date?
    private var lastMotionTimestamp: Date?
    private var pendingFinalizeAt: Date?

    private let segmentDurationSeconds: TimeInterval = 8.0
    private let preRollSeconds: TimeInterval = 2.0
    private let postRollSeconds: TimeInterval = 1.0
    private let playEndInactivitySeconds: TimeInterval = 1.2
    private let maxRetainedSegmentSeconds: TimeInterval = 90.0

    var onStatusUpdate: ((String) -> Void)?
    var onUploadReport: ((UploadReport) -> Void)?
    var onUploadFailed: ((String) -> Void)?
    var onQueueDepthChanged: ((Int) -> Void)?

    init(uploadBaseURL: URL) {
        self.uploadClient = ClipUploadClient(baseURL: uploadBaseURL)
        super.init()
    }

    func pendingUploadCount() async -> Int {
        await uploadQueue.pendingCount()
    }

    func flushUploadQueue() {
        Task { await processUploadQueueOnce() }
    }

    func captureSessionForPreview() -> AVCaptureSession {
        captureSession
    }

    func configure240FPSCapture(playID: String, playerID: String, coachEmail: String? = nil) throws {
        activePlayID = playID
        activePlayerID = playerID
        activeCoachEmail = coachEmail

        captureSession.beginConfiguration()
        captureSession.sessionPreset = .high

        let discovery = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInWideAngleCamera],
            mediaType: .video,
            position: .back
        )
        guard let camera = discovery.devices.first else {
            throw NSError(domain: "SidelineCapture", code: 11, userInfo: [NSLocalizedDescriptionKey: "Back camera unavailable"])
        }

        let formats = camera.formats.filter { format in
            let dims = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
            let supports240 = format.videoSupportedFrameRateRanges.contains { $0.maxFrameRate >= 240.0 }
            return supports240 && dims.width >= 1280
        }
        guard let best240Format = formats.max(by: {
            let d0 = CMVideoFormatDescriptionGetDimensions($0.formatDescription)
            let d1 = CMVideoFormatDescriptionGetDimensions($1.formatDescription)
            return d0.width * d0.height < d1.width * d1.height
        }) else {
            throw NSError(domain: "SidelineCapture", code: 12, userInfo: [NSLocalizedDescriptionKey: "No 240 FPS format found"])
        }

        try camera.lockForConfiguration()
        camera.activeFormat = best240Format
        camera.activeVideoMinFrameDuration = CMTime(value: 1, timescale: 240)
        camera.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: 240)
        camera.unlockForConfiguration()

        if let existing = cameraInput {
            captureSession.removeInput(existing)
        }

        cameraInput = try AVCaptureDeviceInput(device: camera)
        if let cameraInput, captureSession.canAddInput(cameraInput) {
            captureSession.addInput(cameraInput)
        }
        if captureSession.outputs.contains(movieOutput) == false && captureSession.canAddOutput(movieOutput) {
            captureSession.addOutput(movieOutput)
        }

        captureSession.commitConfiguration()
        captureSession.startRunning()
    }

    func startTripodAutoMode(playID: String, playerID: String, coachEmail: String? = nil) throws {
        try configure240FPSCapture(playID: playID, playerID: playerID, coachEmail: coachEmail)
        try enableTripodForegroundPowerMode()
        tripodAutoModeEnabled = true
        playActive = false
        playStartTime = nil
        lastMotionTimestamp = nil
        pendingFinalizeAt = nil

        startNetworkMonitor()
        startQueueWorker()
        if !movieOutput.isRecording {
            startNextSegmentRecording()
        }
        emitStatus("Tripod auto mode started")
    }

    func stopTripodAutoMode() {
        tripodAutoModeEnabled = false
        playActive = false
        playStartTime = nil
        lastMotionTimestamp = nil
        pendingFinalizeAt = nil
        segmentStopTimer?.cancel()
        segmentStopTimer = nil
        forceCutTimer?.cancel()
        forceCutTimer = nil
        queueWorkerTask?.cancel()
        queueWorkerTask = nil
        stopNetworkMonitor()
        disableTripodForegroundPowerMode()
        if movieOutput.isRecording {
            movieOutput.stopRecording()
        }
        emitStatus("Tripod auto mode stopped")
    }

    func ingestPlaySignal(_ signal: TripodPlaySignal) {
        if !tripodAutoModeEnabled {
            if signal.whistleDetected {
                handlePlayWhistle()
            }
            return
        }

        let now = Date()
        if signal.whistleDetected {
            requestPlayFinalization(at: now)
            return
        }

        let proximityActive = (signal.nearestDistanceMeters ?? .greatestFiniteMagnitude) <= 1.8
        let playLikelyActive = signal.collisionActive || signal.movingPlayers >= 2 || proximityActive

        if playLikelyActive {
            if !playActive {
                playActive = true
                playStartTime = now
            }
            lastMotionTimestamp = now
            return
        }

        if playActive, let lastMotionTimestamp, now.timeIntervalSince(lastMotionTimestamp) >= playEndInactivitySeconds {
            requestPlayFinalization(at: now)
        }
    }

    func handleCollisionMessage(_ message: CollisionControlMessage) {
        if tripodAutoModeEnabled {
            if message.type == "collision_start" {
                ingestPlaySignal(TripodPlaySignal(collisionActive: true))
            } else if message.type == "collision_end" {
                requestPlayFinalization(at: Date())
            }
            return
        }

        if message.type == "collision_start", message.record240fps == true {
            start240FPSRecording()
            return
        }
        if message.type == "collision_end" {
            let stopAfter = max(message.stopAfterMs ?? 1000, 0)
            scheduleStopRecording(afterMs: stopAfter)
        }
    }

    func handlePlayWhistle() {
        if tripodAutoModeEnabled {
            ingestPlaySignal(TripodPlaySignal(whistleDetected: true))
            return
        }
        whistleReceived = true
        guard let latestRecordedClip else { return }
        Task {
            try? await compressAndUploadClip(latestRecordedClip)
        }
    }

    private func start240FPSRecording() {
        stopTimer?.cancel()
        stopTimer = nil
        guard !movieOutput.isRecording else { return }
        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("play-\(UUID().uuidString).mov")
        movieOutput.startRecording(to: outputURL, recordingDelegate: self)
        emitStatus("Started 240 FPS collision recording")
    }

    private func startNextSegmentRecording() {
        guard tripodAutoModeEnabled else { return }
        guard !movieOutput.isRecording else { return }
        currentSegmentStart = Date()
        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("segment-\(UUID().uuidString).mov")
        movieOutput.startRecording(to: outputURL, recordingDelegate: self)
        scheduleSegmentStop(after: segmentDurationSeconds)
        emitStatus("Started segment recording at 240 FPS")
    }

    private func scheduleStopRecording(afterMs: Int) {
        stopTimer?.cancel()
        let workItem = DispatchWorkItem { [weak self] in
            self?.stopRecording()
        }
        stopTimer = workItem
        DispatchQueue.main.asyncAfter(deadline: .now() + .milliseconds(afterMs), execute: workItem)
    }

    private func scheduleSegmentStop(after seconds: TimeInterval) {
        segmentStopTimer?.cancel()
        let workItem = DispatchWorkItem { [weak self] in
            self?.stopRecording()
        }
        segmentStopTimer = workItem
        DispatchQueue.main.asyncAfter(deadline: .now() + seconds, execute: workItem)
    }

    private func scheduleForcedCut(at time: Date) {
        forceCutTimer?.cancel()
        let interval = max(time.timeIntervalSinceNow, 0.0)
        let workItem = DispatchWorkItem { [weak self] in
            self?.stopRecording()
        }
        forceCutTimer = workItem
        DispatchQueue.main.asyncAfter(deadline: .now() + interval, execute: workItem)
    }

    private func stopRecording() {
        guard movieOutput.isRecording else { return }
        movieOutput.stopRecording()
        emitStatus("Stopped recording segment")
    }

    private func requestPlayFinalization(at now: Date) {
        guard playStartTime != nil else { return }
        playActive = false
        let finalizeTime = now.addingTimeInterval(postRollSeconds)
        pendingFinalizeAt = finalizeTime
        scheduleForcedCut(at: finalizeTime)
    }

    private func finalizePlayIfReady(currentSegmentEnd: Date) {
        guard let finalizeAt = pendingFinalizeAt, currentSegmentEnd >= finalizeAt else { return }
        guard let playStartTime else { return }
        pendingFinalizeAt = nil

        let clipStart = playStartTime.addingTimeInterval(-preRollSeconds)
        let clipEnd = finalizeAt
        let segments = segmentHistory.filter {
            $0.startedAt <= clipEnd && $0.endedAt >= clipStart
        }
        guard !segments.isEmpty else {
            self.playStartTime = nil
            self.lastMotionTimestamp = nil
            return
        }

        self.playStartTime = nil
        self.lastMotionTimestamp = nil

        Task {
            do {
                let merged = try await composePlayClip(from: segments)
                let compressed = try await transcodeToMP4(inputURL: merged)
                try? FileManager.default.removeItem(at: merged)
                if let pending = await uploadQueue.enqueue(
                    clipURL: compressed,
                    playID: activePlayID,
                    playerID: activePlayerID,
                    coachEmail: activeCoachEmail
                ) {
                    _ = pending
                    emitStatus("Queued play clip for upload")
                    await reportQueueDepth()
                    await processUploadQueueOnce()
                } else {
                    emitUploadFailure("Failed to enqueue play clip upload")
                }
            } catch {
                // Keep running capture loop. Upload queue will handle retries for enqueued files.
                emitUploadFailure("Failed to finalize play clip: \(error.localizedDescription)")
            }
        }
    }

    private func appendSegment(url: URL, startedAt: Date, endedAt: Date) {
        segmentHistory.append(RecordedSegment(url: url, startedAt: startedAt, endedAt: endedAt))
        pruneOldSegments()
    }

    private func pruneOldSegments() {
        let cutoff = Date().addingTimeInterval(-maxRetainedSegmentSeconds)
        var retained: [RecordedSegment] = []
        for segment in segmentHistory {
            if segment.endedAt >= cutoff {
                retained.append(segment)
            } else {
                try? FileManager.default.removeItem(at: segment.url)
            }
        }
        segmentHistory = retained
    }

    private func processUploadQueueOnce() async {
        guard networkAvailable else { return }
        let due = await uploadQueue.dueItems()
        for item in due {
            let clipURL = URL(fileURLWithPath: item.filePath)
            do {
                let report = try await uploadClient.uploadClip(
                    fileURL: clipURL,
                    playID: item.playID,
                    playerID: item.playerID,
                    coachEmail: item.coachEmail
                )
                await uploadQueue.markSuccess(id: item.id)
                emitStatus("Uploaded clip \(item.id)")
                emitUploadReport(report)
                await reportQueueDepth()
            } catch {
                await uploadQueue.markFailure(id: item.id)
                emitUploadFailure("Upload retry failed for clip \(item.id): \(error.localizedDescription)")
                await reportQueueDepth()
            }
        }
    }

    private func startQueueWorker() {
        queueWorkerTask?.cancel()
        queueWorkerTask = Task { [weak self] in
            guard let self else { return }
            while !Task.isCancelled {
                await self.processUploadQueueOnce()
                try? await Task.sleep(nanoseconds: 2_000_000_000)
            }
        }
    }

    private func startNetworkMonitor() {
        guard networkMonitor == nil else { return }
        let monitor = NWPathMonitor()
        monitor.pathUpdateHandler = { [weak self] path in
            guard let self else { return }
            self.networkAvailable = path.status == .satisfied
            self.emitStatus(path.status == .satisfied ? "Network reachable" : "Network unreachable; uploads queued")
            if path.status == .satisfied {
                Task { await self.processUploadQueueOnce() }
            }
        }
        monitor.start(queue: networkMonitorQueue)
        networkMonitor = monitor
    }

    private func stopNetworkMonitor() {
        networkMonitor?.cancel()
        networkMonitor = nil
    }

    private func compressAndUploadClip(_ clipURL: URL) async throws {
        let compressed = try await transcodeToMP4(inputURL: clipURL)
        do {
            let report = try await uploadClient.uploadClip(
                fileURL: compressed,
                playID: activePlayID,
                playerID: activePlayerID,
                coachEmail: activeCoachEmail
            )
            try? FileManager.default.removeItem(at: compressed)
            try? FileManager.default.removeItem(at: clipURL)
            emitStatus("Uploaded clip and deleted local files")
            emitUploadReport(report)
            await reportQueueDepth()
        } catch {
            _ = await uploadQueue.enqueue(
                clipURL: compressed,
                playID: activePlayID,
                playerID: activePlayerID,
                coachEmail: activeCoachEmail
            )
            try? FileManager.default.removeItem(at: clipURL)
            emitStatus("Upload failed, queued for retry")
            await processUploadQueueOnce()
            await reportQueueDepth()
        }
    }

    private func composePlayClip(from segments: [RecordedSegment]) async throws -> URL {
        let sorted = segments.sorted { $0.startedAt < $1.startedAt }
        guard let first = sorted.first else {
            throw NSError(domain: "SidelineCapture", code: 41, userInfo: [NSLocalizedDescriptionKey: "No segments to compose"])
        }

        if sorted.count == 1 {
            let copiedURL = FileManager.default.temporaryDirectory
                .appendingPathComponent("play-merged-\(UUID().uuidString).mov")
            if FileManager.default.fileExists(atPath: copiedURL.path) {
                try? FileManager.default.removeItem(at: copiedURL)
            }
            try FileManager.default.copyItem(at: first.url, to: copiedURL)
            return copiedURL
        }

        let composition = AVMutableComposition()
        guard let videoTrack = composition.addMutableTrack(
            withMediaType: .video,
            preferredTrackID: kCMPersistentTrackID_Invalid
        ) else {
            throw NSError(domain: "SidelineCapture", code: 42, userInfo: [NSLocalizedDescriptionKey: "Cannot allocate composition video track"])
        }

        let audioTrack = composition.addMutableTrack(
            withMediaType: .audio,
            preferredTrackID: kCMPersistentTrackID_Invalid
        )

        var cursor = CMTime.zero
        for segment in sorted {
            let asset = AVAsset(url: segment.url)
            let duration = asset.duration
            guard duration.isNumeric && duration.seconds > 0 else { continue }
            let timeRange = CMTimeRange(start: .zero, duration: duration)

            if let sourceVideo = asset.tracks(withMediaType: .video).first {
                try videoTrack.insertTimeRange(timeRange, of: sourceVideo, at: cursor)
            }
            if let sourceAudio = asset.tracks(withMediaType: .audio).first, let audioTrack {
                try? audioTrack.insertTimeRange(timeRange, of: sourceAudio, at: cursor)
            }
            cursor = CMTimeAdd(cursor, duration)
        }

        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("play-merged-\(UUID().uuidString).mov")
        guard let export = AVAssetExportSession(asset: composition, presetName: AVAssetExportPresetPassthrough) else {
            throw NSError(domain: "SidelineCapture", code: 43, userInfo: [NSLocalizedDescriptionKey: "Cannot create export session"])
        }
        export.outputURL = outputURL
        export.outputFileType = .mov

        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            export.exportAsynchronously {
                switch export.status {
                case .completed:
                    continuation.resume()
                case .failed, .cancelled:
                    continuation.resume(throwing: export.error ?? NSError(domain: "SidelineCapture", code: 44))
                default:
                    continuation.resume(throwing: NSError(domain: "SidelineCapture", code: 45))
                }
            }
        }

        return outputURL
    }

    private func transcodeToMP4(inputURL: URL) async throws -> URL {
        let asset = AVAsset(url: inputURL)
        guard let export = AVAssetExportSession(asset: asset, presetName: AVAssetExportPresetMediumQuality) else {
            throw NSError(domain: "SidelineCapture", code: 21, userInfo: [NSLocalizedDescriptionKey: "Cannot create MP4 export session"])
        }

        let outputURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("clip-\(UUID().uuidString).mp4")
        export.outputURL = outputURL
        export.outputFileType = .mp4
        export.shouldOptimizeForNetworkUse = true

        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            export.exportAsynchronously {
                switch export.status {
                case .completed:
                    continuation.resume()
                case .failed, .cancelled:
                    continuation.resume(throwing: export.error ?? NSError(domain: "SidelineCapture", code: 22))
                default:
                    continuation.resume(throwing: NSError(domain: "SidelineCapture", code: 23))
                }
            }
        }
        return outputURL
    }

    private func enableTripodForegroundPowerMode() throws {
        #if canImport(UIKit)
        if UIApplication.shared.applicationState != .active {
            throw NSError(
                domain: "SidelineCapture",
                code: 31,
                userInfo: [NSLocalizedDescriptionKey: "Tripod auto mode requires app in foreground. Use Guided Access for sideline sessions."]
            )
        }
        UIApplication.shared.isIdleTimerDisabled = true
        #endif
    }

    private func disableTripodForegroundPowerMode() {
        #if canImport(UIKit)
        UIApplication.shared.isIdleTimerDisabled = false
        #endif
    }

    private func emitStatus(_ message: String) {
        DispatchQueue.main.async { [weak self] in
            self?.onStatusUpdate?(message)
        }
    }

    private func emitUploadFailure(_ message: String) {
        DispatchQueue.main.async { [weak self] in
            self?.onUploadFailed?(message)
            self?.onStatusUpdate?("Error: \(message)")
        }
    }

    private func emitUploadReport(_ report: UploadReport) {
        DispatchQueue.main.async { [weak self] in
            self?.onUploadReport?(report)
        }
    }

    private func reportQueueDepth() async {
        let count = await uploadQueue.pendingCount()
        DispatchQueue.main.async { [weak self] in
            self?.onQueueDepthChanged?(count)
        }
    }
}

extension SidelineCaptureManager: AVCaptureFileOutputRecordingDelegate {
    func fileOutput(
        _ output: AVCaptureFileOutput,
        didFinishRecordingTo outputFileURL: URL,
        from connections: [AVCaptureConnection],
        error: Error?
    ) {
        guard error == nil else {
            emitUploadFailure("Recording failed: \(error?.localizedDescription ?? "unknown error")")
            return
        }

        if tripodAutoModeEnabled {
            let startedAt = currentSegmentStart ?? Date().addingTimeInterval(-segmentDurationSeconds)
            let endedAt = Date()
            appendSegment(url: outputFileURL, startedAt: startedAt, endedAt: endedAt)
            finalizePlayIfReady(currentSegmentEnd: endedAt)
            startNextSegmentRecording()
            return
        }

        latestRecordedClip = outputFileURL
        if whistleReceived {
            Task {
                try? await compressAndUploadClip(outputFileURL)
            }
        }
    }
}
