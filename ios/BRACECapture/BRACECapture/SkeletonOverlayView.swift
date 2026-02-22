import SwiftUI

/// Draws a skeleton overlay on top of the camera preview.
/// Landmarks are in normalised [0,1] coordinates; the view maps them
/// to its own size. The front camera is already mirrored by CameraManager.
struct SkeletonOverlayView: View {
    let landmarks: [Landmark]
    let visibilityThreshold: Double = 0.3

    // Colours
    private let boneColor = Color.green.opacity(0.8)
    private let jointColor = Color.white
    private let lowVisColor = Color.red.opacity(0.3)

    var body: some View {
        Canvas { context, size in
            let w = size.width
            let h = size.height

            // Draw bones
            for (a, b) in kBoneConnections {
                guard a < landmarks.count, b < landmarks.count else { continue }
                let lmA = landmarks[a]
                let lmB = landmarks[b]

                guard lmA.visibility >= visibilityThreshold,
                      lmB.visibility >= visibilityThreshold else { continue }

                let pA = CGPoint(x: lmA.x * w, y: lmA.y * h)
                let pB = CGPoint(x: lmB.x * w, y: lmB.y * h)

                var path = Path()
                path.move(to: pA)
                path.addLine(to: pB)

                context.stroke(path, with: .color(boneColor), lineWidth: 3)
            }

            // Draw joints
            for (idx, lm) in landmarks.enumerated() {
                guard kSendIndices.contains(idx) else { continue }
                guard lm.visibility > 0.01 else { continue }

                let pt = CGPoint(x: lm.x * w, y: lm.y * h)
                let radius: CGFloat = lm.visibility >= visibilityThreshold ? 5 : 3
                let color = lm.visibility >= visibilityThreshold ? jointColor : lowVisColor

                let rect = CGRect(
                    x: pt.x - radius,
                    y: pt.y - radius,
                    width: radius * 2,
                    height: radius * 2
                )
                context.fill(Path(ellipseIn: rect), with: .color(color))
            }
        }
    }
}
