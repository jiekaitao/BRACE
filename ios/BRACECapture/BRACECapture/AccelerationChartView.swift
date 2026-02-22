import SwiftUI

/// Joint name mapping for the 14 feature joints (indices in the 33-joint MediaPipe layout).
let kJointNames: [(index: Int, name: String)] = [
    (11, "L Shoulder"), (12, "R Shoulder"),
    (13, "L Elbow"),    (14, "R Elbow"),
    (15, "L Wrist"),    (16, "R Wrist"),
    (23, "L Hip"),      (24, "R Hip"),
    (25, "L Knee"),     (26, "R Knee"),
    (27, "L Ankle"),    (28, "R Ankle"),
    (31, "L Foot"),     (32, "R Foot"),
]

/// A real-time chart that computes and displays acceleration magnitude
/// for a selected joint, derived from successive landmark positions.
struct AccelerationChartView: View {
    let landmarks: [Landmark]
    @Binding var selectedJointIndex: Int

    // History buffers
    @State private var positionHistory: [(x: Double, y: Double, t: CFAbsoluteTime)] = []
    @State private var velocityHistory: [(mag: Double, t: CFAbsoluteTime)] = []
    @State private var accelerationHistory: [Double] = []

    private let maxSamples = 200
    private let emaAlpha = 0.3

    var body: some View {
        VStack(spacing: 4) {
            // Joint selector
            HStack {
                Text("Joint:")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                Picker("Joint", selection: $selectedJointIndex) {
                    ForEach(kJointNames, id: \.index) { joint in
                        Text(joint.name).tag(joint.index)
                    }
                }
                .pickerStyle(.menu)
                .tint(.green)
                .onChange(of: selectedJointIndex) { _, _ in
                    // Reset history on joint change
                    positionHistory.removeAll()
                    velocityHistory.removeAll()
                    accelerationHistory.removeAll()
                }

                Spacer()

                if let lastAccel = accelerationHistory.last {
                    Text(String(format: "%.1f u/s\u{00B2}", lastAccel))
                        .font(.system(.caption, design: .monospaced))
                        .foregroundStyle(.green)
                }
            }
            .padding(.horizontal, 12)

            // Chart
            Canvas { context, size in
                guard !accelerationHistory.isEmpty else { return }

                let w = size.width
                let h = size.height
                let count = accelerationHistory.count

                // Auto-scale Y
                let maxVal = max(accelerationHistory.max() ?? 1.0, 0.1)
                let yScale = (h - 8) / maxVal

                // Gradient fill
                let points: [CGPoint] = accelerationHistory.enumerated().map { i, val in
                    let x = w * CGFloat(i) / CGFloat(max(count - 1, 1))
                    let y = h - val * yScale - 4
                    return CGPoint(x: x, y: y)
                }

                // Fill area
                var fillPath = Path()
                fillPath.move(to: CGPoint(x: points[0].x, y: h))
                for pt in points {
                    fillPath.addLine(to: pt)
                }
                fillPath.addLine(to: CGPoint(x: points.last!.x, y: h))
                fillPath.closeSubpath()

                let gradient = Gradient(colors: [
                    Color.green.opacity(0.4),
                    Color.green.opacity(0.05),
                ])
                context.fill(
                    fillPath,
                    with: .linearGradient(gradient, startPoint: CGPoint(x: 0, y: 0), endPoint: CGPoint(x: 0, y: h))
                )

                // Line
                var linePath = Path()
                linePath.move(to: points[0])
                for pt in points.dropFirst() {
                    linePath.addLine(to: pt)
                }
                context.stroke(linePath, with: .color(.green), lineWidth: 1.5)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .onChange(of: landmarks) { _, newLandmarks in
            updateHistory(newLandmarks)
        }
    }

    private func updateHistory(_ lms: [Landmark]) {
        guard selectedJointIndex < lms.count else { return }
        let lm = lms[selectedJointIndex]
        guard lm.visibility >= 0.3 else { return }

        let now = CFAbsoluteTimeGetCurrent()

        // Append position
        positionHistory.append((x: lm.x, y: lm.y, t: now))
        if positionHistory.count > maxSamples + 5 {
            positionHistory.removeFirst()
        }

        // Compute velocity from last two positions
        if positionHistory.count >= 2 {
            let curr = positionHistory[positionHistory.count - 1]
            let prev = positionHistory[positionHistory.count - 2]
            let dt = curr.t - prev.t
            if dt > 0.001 {
                let dx = curr.x - prev.x
                let dy = curr.y - prev.y
                let velMag = sqrt(dx * dx + dy * dy) / dt
                velocityHistory.append((mag: velMag, t: now))
                if velocityHistory.count > maxSamples + 5 {
                    velocityHistory.removeFirst()
                }
            }
        }

        // Compute acceleration from last two velocities
        if velocityHistory.count >= 2 {
            let curr = velocityHistory[velocityHistory.count - 1]
            let prev = velocityHistory[velocityHistory.count - 2]
            let dt = curr.t - prev.t
            if dt > 0.001 {
                let rawAccel = abs(curr.mag - prev.mag) / dt

                // EMA smoothing
                let smoothed: Double
                if let last = accelerationHistory.last {
                    smoothed = emaAlpha * rawAccel + (1.0 - emaAlpha) * last
                } else {
                    smoothed = rawAccel
                }

                accelerationHistory.append(smoothed)
                if accelerationHistory.count > maxSamples {
                    accelerationHistory.removeFirst()
                }
            }
        }
    }
}
