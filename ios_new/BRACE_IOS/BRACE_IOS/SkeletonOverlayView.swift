import SwiftUI

/// Draws skeleton overlays for all subjects on top of the camera preview.
/// Landmarks and bboxes are in normalised [0,1] coordinates (camera image space).
/// This view accounts for AVCaptureVideoPreviewLayer's resizeAspectFill crop
/// by computing the same scale+offset transform from camera space → view space.
struct SkeletonOverlayView: View {
    let subjects: [String: SubjectData]
    let selectedSubjectId: String?
    /// Camera frame dimensions (after portrait rotation) — needed to compute aspect-fill offset.
    let cameraWidth: CGFloat
    let cameraHeight: CGFloat
    let visibilityThreshold: Double = 0.3

    private let selectedBoneColor = Color.green.opacity(0.85)
    private let unselectedBoneColor = Color.blue.opacity(0.35)
    private let selectedJointColor = Color.white
    private let unselectedJointColor = Color.white.opacity(0.4)
    private let lowVisColor = Color.red.opacity(0.25)
    private let selectedBboxColor = Color.green
    private let unselectedBboxColor = Color.yellow.opacity(0.5)

    var body: some View {
        Canvas { context, size in
            let vw = size.width
            let vh = size.height

            // Compute resizeAspectFill transform: same as AVCaptureVideoPreviewLayer
            let camAspect = cameraWidth / max(cameraHeight, 1)
            let viewAspect = vw / max(vh, 1)

            let fillScale: CGFloat
            let offsetX: CGFloat
            let offsetY: CGFloat

            if camAspect < viewAspect {
                // Camera is narrower → scale to fill width, crop top/bottom
                fillScale = vw / cameraWidth
                let displayedH = cameraHeight * fillScale
                offsetX = 0
                offsetY = (displayedH - vh) / 2.0
            } else {
                // Camera is wider → scale to fill height, crop left/right
                fillScale = vh / cameraHeight
                let displayedW = cameraWidth * fillScale
                offsetX = (displayedW - vw) / 2.0
                offsetY = 0
            }

            // Map normalised [0,1] camera coordinate to view coordinate
            func mapPoint(_ nx: Double, _ ny: Double) -> CGPoint {
                let px = CGFloat(nx) * cameraWidth * fillScale - offsetX
                let py = CGFloat(ny) * cameraHeight * fillScale - offsetY
                return CGPoint(x: px, y: py)
            }

            for (subjectId, subject) in subjects {
                let isSelected = selectedSubjectId == nil || selectedSubjectId == subjectId
                let boneColor = isSelected ? selectedBoneColor : unselectedBoneColor
                let jointColor = isSelected ? selectedJointColor : unselectedJointColor
                let boneWidth: CGFloat = isSelected ? 2 : 1

                // Draw bounding box
                if let bbox = subject.bbox {
                    let topLeft = mapPoint(bbox.x1, bbox.y1)
                    let bottomRight = mapPoint(bbox.x2, bbox.y2)
                    let bboxRect = CGRect(
                        x: topLeft.x,
                        y: topLeft.y,
                        width: bottomRight.x - topLeft.x,
                        height: bottomRight.y - topLeft.y
                    )
                    let bboxPath = Path(roundedRect: bboxRect, cornerRadius: 3)
                    let bboxColor = isSelected ? selectedBboxColor : unselectedBboxColor
                    context.stroke(bboxPath, with: .color(bboxColor.opacity(0.6)), lineWidth: isSelected ? 1.5 : 0.75)

                    // Label
                    let label = Text(subject.label)
                        .font(.system(size: 10, weight: .medium, design: .monospaced))
                        .foregroundColor(.white)
                    context.draw(label, at: CGPoint(x: bboxRect.minX + 3, y: bboxRect.minY - 8), anchor: .leading)
                }

                // Draw skeleton (only if landmarks present)
                guard let sparse = subject.landmarks else { continue }
                let landmarks = expandLandmarks(sparse)

                // Draw bones
                for (a, b) in kBoneConnections {
                    guard a < landmarks.count, b < landmarks.count else { continue }
                    let lmA = landmarks[a]
                    let lmB = landmarks[b]

                    guard lmA.visibility >= visibilityThreshold,
                          lmB.visibility >= visibilityThreshold else { continue }

                    let pA = mapPoint(lmA.x, lmA.y)
                    let pB = mapPoint(lmB.x, lmB.y)

                    var path = Path()
                    path.move(to: pA)
                    path.addLine(to: pB)

                    context.stroke(path, with: .color(boneColor), lineWidth: boneWidth)
                }

                // Draw joints
                for (idx, lm) in landmarks.enumerated() {
                    guard kSendIndices.contains(idx) else { continue }
                    guard lm.visibility > 0.01 else { continue }

                    let pt = mapPoint(lm.x, lm.y)
                    let radius: CGFloat = lm.visibility >= visibilityThreshold ? (isSelected ? 3 : 2) : 1.5
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
}
