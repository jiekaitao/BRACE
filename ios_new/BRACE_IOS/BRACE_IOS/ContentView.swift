import SwiftUI
import AVFoundation

struct ContentView: View {
    @StateObject private var camera = CameraManager()
    @StateObject private var ws = BRACEWebSocket()
    @AppStorage("serverURL") private var serverURL = "wss://ws.braceml.com/ws/analyze?mode=webcam&fps=240&client=ios"
    @State private var showSettings = false
    @State private var selectedJointIndex = 25 // L Knee default
    @State private var zoomScale: CGFloat = 1.0
    @State private var lastZoomScale: CGFloat = 1.0

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            VStack(spacing: 0) {
                // Camera preview + skeleton overlay (~65%)
                GeometryReader { geo in
                    ZStack {
                        CameraPreview(session: camera.session)
                            .ignoresSafeArea(edges: .top)

                        SkeletonOverlayView(
                            subjects: ws.lastFrame?.subjects ?? [:],
                            selectedSubjectId: ws.selectedSubjectId,
                            cameraWidth: camera.frameWidth,
                            cameraHeight: camera.frameHeight
                        )
                    }
                    .scaleEffect(zoomScale)
                    .gesture(
                        MagnificationGesture()
                            .onChanged { value in
                                zoomScale = max(1.0, min(lastZoomScale * value, 5.0))
                            }
                            .onEnded { value in
                                lastZoomScale = zoomScale
                                if zoomScale < 1.05 { zoomScale = 1.0; lastZoomScale = 1.0 }
                            }
                    )
                    .simultaneousGesture(
                        TapGesture(count: 2).onEnded {
                            // Double-tap to reset zoom & deselect
                            withAnimation(.easeOut(duration: 0.2)) {
                                zoomScale = 1.0
                                lastZoomScale = 1.0
                            }
                            ws.selectSubject(nil)
                        }
                    )
                    .simultaneousGesture(
                        SpatialTapGesture().onEnded { value in
                            handleTapToSelect(at: value.location, in: geo.size)
                        }
                    )
                    .clipped()
                    .overlay(alignment: .top) {
                        // Status overlay
                        HStack {
                            connectionBadge
                            Spacer()
                            fpsLabel
                        }
                        .padding(.horizontal, 12)
                        .padding(.top, 8)
                    }
                    .overlay(alignment: .bottom) {
                        // Subject info bar
                        if let subject = selectedOrFirstSubject {
                            subjectInfoBar(subject)
                        }
                    }
                }
                .frame(maxWidth: .infinity)
                .layoutPriority(1)

                // Divider
                Rectangle()
                    .fill(Color.gray.opacity(0.3))
                    .frame(height: 1)

                // Acceleration chart (~35%)
                AccelerationChartView(
                    landmarks: ws.landmarks,
                    selectedJointIndex: $selectedJointIndex
                )
                .frame(maxWidth: .infinity)
                .frame(height: UIScreen.main.bounds.height * 0.32)
                .padding(.vertical, 8)
                .background(Color.black)
            }

            // Settings button
            VStack {
                Spacer()
                HStack {
                    Spacer()
                    Button {
                        showSettings = true
                    } label: {
                        Image(systemName: "gearshape.fill")
                            .font(.title3)
                            .foregroundStyle(.white.opacity(0.7))
                            .padding(12)
                            .background(.ultraThinMaterial, in: Circle())
                    }
                    .padding(.trailing, 16)
                    .padding(.bottom, 16)
                }
            }
        }
        .sheet(isPresented: $showSettings) {
            SettingsView(serverURL: $serverURL, ws: ws)
        }
        .onAppear {
            requestCameraPermission()
        }
    }

    // MARK: - Tap-to-Select

    private func handleTapToSelect(at point: CGPoint, in size: CGSize) {
        guard let frame = ws.lastFrame else { return }

        // Map tap from view space back to normalised [0,1] camera space,
        // accounting for resizeAspectFill crop.
        let cw = camera.frameWidth
        let ch = camera.frameHeight
        let vw = size.width
        let vh = size.height
        let camAspect = cw / max(ch, 1)
        let viewAspect = vw / max(vh, 1)

        let fillScale: CGFloat
        let offsetX: CGFloat
        let offsetY: CGFloat
        if camAspect < viewAspect {
            fillScale = vw / cw
            offsetX = 0
            offsetY = (ch * fillScale - vh) / 2.0
        } else {
            fillScale = vh / ch
            offsetX = (cw * fillScale - vw) / 2.0
            offsetY = 0
        }

        let nx = Double((point.x + offsetX) / (cw * fillScale))
        let ny = Double((point.y + offsetY) / (ch * fillScale))

        var bestId: String?
        var bestDist: Double = .infinity

        for (subjectId, subject) in frame.subjects {
            guard let bbox = subject.bbox else { continue }

            // Check if tap is inside bounding box
            if nx >= bbox.x1 && nx <= bbox.x2 && ny >= bbox.y1 && ny <= bbox.y2 {
                // Pick the closest to bbox centre
                let cx = (bbox.x1 + bbox.x2) / 2.0
                let cy = (bbox.y1 + bbox.y2) / 2.0
                let dist = (nx - cx) * (nx - cx) + (ny - cy) * (ny - cy)
                if dist < bestDist {
                    bestDist = dist
                    bestId = subjectId
                }
            }
        }

        if let bestId {
            // Toggle: tap same subject again to deselect
            if ws.selectedSubjectId == bestId {
                ws.selectSubject(nil)
            } else {
                ws.selectSubject(bestId)
            }
        }
    }

    // MARK: - Helpers

    private var selectedOrFirstSubject: SubjectData? {
        guard let frame = ws.lastFrame else { return nil }
        if let sid = ws.selectedSubjectId, let s = frame.subjects[sid] {
            return s
        }
        return frame.subjects.values.first
    }

    // MARK: - Subviews

    private var connectionBadge: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(ws.state == .connected ? Color.green : (ws.state == .connecting ? Color.yellow : Color.red))
                .frame(width: 8, height: 8)

            Text(ws.state.rawValue.capitalized)
                .font(.system(.caption2, design: .monospaced))
                .foregroundStyle(.white)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background(.ultraThinMaterial, in: Capsule())
    }

    private var fpsLabel: some View {
        VStack(alignment: .trailing, spacing: 2) {
            Text("CAP \(Int(camera.captureFPS))fps")
                .font(.system(.caption2, design: .monospaced))
                .foregroundStyle(.white.opacity(0.7))
            Text("OUT \(Int(ws.fpsOut)) | IN \(Int(ws.fpsIn))")
                .font(.system(.caption2, design: .monospaced))
                .foregroundStyle(.white.opacity(0.7))
            Text("RTT \(Int(ws.rttMs))ms")
                .font(.system(.caption2, design: .monospaced))
                .foregroundStyle(.white.opacity(0.7))
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 8))
    }

    private func subjectInfoBar(_ subject: SubjectData) -> some View {
        HStack(spacing: 12) {
            let phaseColor: Color = subject.phase == "anomaly" ? .red : (subject.phase == "normal" ? .green : .yellow)
            Text(subject.phase.uppercased())
                .font(.system(.caption2, design: .monospaced, weight: .bold))
                .foregroundStyle(.white)
                .padding(.horizontal, 8)
                .padding(.vertical, 3)
                .background(phaseColor.opacity(0.8), in: Capsule())

            if let formScore = subject.quality?.form_score {
                Text("Form \(Int(formScore))")
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.white)
            }

            Text("Seg \(subject.n_segments) | Cl \(subject.n_clusters)")
                .font(.system(.caption2, design: .monospaced))
                .foregroundStyle(.white.opacity(0.7))

            Spacer()

            // Subject count indicator
            if let frame = ws.lastFrame, frame.subjects.count > 1 {
                Text("\(frame.subjects.count) ppl")
                    .font(.system(.caption2, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.5))
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(.ultraThinMaterial)
    }

    // MARK: - Permissions & Lifecycle

    private func requestCameraPermission() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            startPipeline()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                if granted {
                    DispatchQueue.main.async { startPipeline() }
                }
            }
        default:
            break
        }
    }

    private func startPipeline() {
        camera.configure()
        camera.start()

        ws.serverURL = serverURL
        ws.connect()

        // Wire camera frames to WebSocket (called on capture queue, sendFrame is thread-safe)
        camera.onFrame = { [weak ws] jpegData, timestamp in
            ws?.sendFrame(jpegData: jpegData, timestamp: timestamp)
        }
    }
}

// MARK: - Camera Preview (UIViewRepresentable)

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> UIView {
        let view = PreviewView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspectFill
        // Back camera — no mirroring
        return view
    }

    func updateUIView(_ uiView: UIView, context: Context) {}

    class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    }
}

// MARK: - Settings View

struct SettingsView: View {
    @Binding var serverURL: String
    @ObservedObject var ws: BRACEWebSocket
    @Environment(\.dismiss) private var dismiss
    @State private var editURL: String = ""

    var body: some View {
        NavigationStack {
            Form {
                Section("Server") {
                    TextField("WebSocket URL", text: $editURL)
                        .font(.system(.body, design: .monospaced))
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()

                    Button("Apply & Reconnect") {
                        serverURL = editURL
                        ws.serverURL = editURL
                    }
                    .disabled(editURL == serverURL)
                }

                Section("Connection") {
                    LabeledContent("Status", value: ws.state.rawValue.capitalized)
                    LabeledContent("FPS Out", value: String(format: "%.0f", ws.fpsOut))
                    LabeledContent("FPS In", value: String(format: "%.0f", ws.fpsIn))
                    LabeledContent("RTT", value: String(format: "%.0f ms", ws.rttMs))
                }

                Section("Camera") {
                    LabeledContent("Format", value: ws.state == .connected ? "Active" : "---")
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
            .onAppear {
                editURL = serverURL
            }
        }
    }
}
