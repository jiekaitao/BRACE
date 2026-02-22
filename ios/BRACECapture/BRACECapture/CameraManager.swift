import AVFoundation
import UIKit
import os

private let log = Logger(subsystem: "com.brace.capture", category: "Camera")

/// Manages AVCaptureSession at up to 120fps, downscales to 480p JPEG,
/// and provides frames to the WebSocket sender via a callback.
final class CameraManager: NSObject, ObservableObject {

    // MARK: - Published State
    @Published var isRunning = false
    @Published var captureFPS: Double = 0
    @Published var actualFormat: String = ""

    // MARK: - Frame Callback
    /// Called on the capture queue with (jpegData, captureTimestamp).
    var onFrame: ((Data, Double) -> Void)?

    // MARK: - Private
    let session = AVCaptureSession()
    private let captureQueue = DispatchQueue(label: "com.brace.captureQueue", qos: .userInteractive)
    private var videoOutput: AVCaptureVideoDataOutput?
    private let targetHeight: Int = 480
    private let jpegQuality: CGFloat = 0.65

    // FPS tracking
    private var frameCount = 0
    private var fpsTimer: Date = .now

    // CIContext for efficient downscale + JPEG encode
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    // MARK: - Setup

    func configure() {
        captureQueue.async { [weak self] in
            self?._configure()
        }
    }

    private func _configure() {
        session.beginConfiguration()
        defer { session.commitConfiguration() }

        // Front camera
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front) else {
            log.error("No front camera available")
            return
        }

        // Add input
        do {
            let input = try AVCaptureDeviceInput(device: device)
            if session.canAddInput(input) {
                session.addInput(input)
            }
        } catch {
            log.error("Camera input error: \(error.localizedDescription)")
            return
        }

        // Find best format supporting >= 120fps
        var bestFormat: AVCaptureDevice.Format?
        var bestRange: AVFrameRateRange?
        var bestWidth: Int32 = 0

        for format in device.formats {
            let dims = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
            let mediaType = CMFormatDescriptionGetMediaSubType(format.formatDescription)
            // Prefer 420v (kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange)
            guard mediaType == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange ||
                  mediaType == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange else { continue }

            for range in format.videoSupportedFrameRateRanges {
                if range.maxFrameRate >= 120 {
                    // Pick the format with the largest width that still supports 120fps
                    // but don't go above 1920 to save processing
                    if dims.width > bestWidth && dims.width <= 1920 {
                        bestFormat = format
                        bestRange = range
                        bestWidth = dims.width
                    }
                }
            }
        }

        // Fallback: pick highest fps format available
        if bestFormat == nil {
            for format in device.formats {
                let dims = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
                for range in format.videoSupportedFrameRateRanges {
                    if bestRange == nil || range.maxFrameRate > bestRange!.maxFrameRate {
                        bestFormat = format
                        bestRange = range
                        bestWidth = dims.width
                    }
                }
            }
        }

        guard let selectedFormat = bestFormat, let selectedRange = bestRange else {
            log.error("No suitable camera format found")
            return
        }

        do {
            try device.lockForConfiguration()
            device.activeFormat = selectedFormat

            let targetFPS = min(selectedRange.maxFrameRate, 120.0)
            let timescale = CMTimeScale(targetFPS)
            device.activeVideoMinFrameDuration = CMTime(value: 1, timescale: timescale)
            device.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: timescale)

            device.unlockForConfiguration()

            let dims = CMVideoFormatDescriptionGetDimensions(selectedFormat.formatDescription)
            let formatStr = "\(dims.width)x\(dims.height) @ \(Int(targetFPS))fps"
            log.info("Camera configured: \(formatStr)")
            DispatchQueue.main.async { [weak self] in
                self?.actualFormat = formatStr
            }
        } catch {
            log.error("Device config error: \(error.localizedDescription)")
        }

        // Video output
        let output = AVCaptureVideoDataOutput()
        output.alwaysDiscardsLateVideoFrames = true
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        output.setSampleBufferDelegate(self, queue: captureQueue)

        if session.canAddOutput(output) {
            session.addOutput(output)
        }
        videoOutput = output

        // Mirror front camera
        if let connection = output.connection(with: .video) {
            if connection.isVideoMirroringSupported {
                connection.isVideoMirrored = true
            }
        }
    }

    // MARK: - Start / Stop

    func start() {
        captureQueue.async { [weak self] in
            guard let self, !self.session.isRunning else { return }
            self.session.startRunning()
            DispatchQueue.main.async {
                self.isRunning = true
            }
        }
    }

    func stop() {
        captureQueue.async { [weak self] in
            guard let self, self.session.isRunning else { return }
            self.session.stopRunning()
            DispatchQueue.main.async {
                self.isRunning = false
            }
        }
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {

    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        // FPS tracking
        frameCount += 1
        let now = Date()
        let elapsed = now.timeIntervalSince(fpsTimer)
        if elapsed >= 1.0 {
            let fps = Double(frameCount) / elapsed
            DispatchQueue.main.async { [weak self] in
                self?.captureFPS = fps
            }
            frameCount = 0
            fpsTimer = now
        }

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        // Downscale to 480p height, preserve aspect ratio
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let srcWidth = ciImage.extent.width
        let srcHeight = ciImage.extent.height
        let scale = CGFloat(targetHeight) / srcHeight
        let destWidth = Int(srcWidth * scale)
        let destHeight = targetHeight

        let scaled = ciImage.transformed(by: CGAffineTransform(scaleX: scale, y: scale))

        // Encode to JPEG
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
        guard let jpegData = ciContext.jpegRepresentation(
            of: scaled,
            colorSpace: colorSpace,
            options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: jpegQuality]
        ) else { return }

        let timestamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds

        onFrame?(jpegData, timestamp)
    }
}
