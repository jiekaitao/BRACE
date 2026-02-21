import SwiftUI

struct CaptureView: View {
    @ObservedObject var viewModel: DemoViewModel

    var body: some View {
        VStack(spacing: 0) {
            // Camera preview — dominant
            ZStack(alignment: .topLeading) {
                CameraPreview(session: viewModel.captureSession)
                    .frame(maxWidth: .infinity)
                    .frame(height: UIScreen.main.bounds.height * 0.50)
                    .clipped()

                // Recording status pill
                HStack(spacing: 6) {
                    Circle()
                        .fill(viewModel.isRecording ? Color.red : Color.gray)
                        .frame(width: 10, height: 10)
                    Text(viewModel.isRecording ? "REC 240 FPS" : "STANDBY")
                        .font(.system(.caption, design: .monospaced, weight: .bold))
                        .foregroundStyle(.white)
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(.ultraThinMaterial, in: Capsule())
                .padding(12)
            }

            ScrollView {
                VStack(spacing: 16) {
                    // Primary action button
                    Button {
                        if viewModel.isTripodRunning {
                            viewModel.stopTripodMode()
                        } else {
                            viewModel.startTripodMode()
                        }
                    } label: {
                        HStack {
                            Image(systemName: viewModel.isTripodRunning ? "stop.circle.fill" : "record.circle")
                                .font(.title2)
                            Text(viewModel.isTripodRunning ? "Stop Capture" : "Start Capture")
                                .fontWeight(.semibold)
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 14)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(viewModel.isTripodRunning ? .red : .blue)

                    // Play signal controls
                    HStack(spacing: 10) {
                        Button("Play Active") { viewModel.sendPlayActive() }
                            .buttonStyle(.bordered)
                            .tint(.green)
                        Button("Play Idle") { viewModel.sendPlayIdle() }
                            .buttonStyle(.bordered)
                        Button("Whistle") { viewModel.sendWhistle() }
                            .buttonStyle(.bordered)
                            .tint(.orange)
                    }
                    .font(.subheadline)

                    // Upload queue
                    if viewModel.queueDepth > 0 {
                        HStack {
                            Image(systemName: "arrow.up.circle")
                            Text("\(viewModel.queueDepth) clip\(viewModel.queueDepth == 1 ? "" : "s") pending upload")
                                .font(.subheadline)
                            Spacer()
                            Button("Flush") { viewModel.flushQueue() }
                                .font(.subheadline)
                                .buttonStyle(.bordered)
                        }
                        .padding(12)
                        .background(Color(.systemGray6))
                        .clipShape(RoundedRectangle(cornerRadius: 10))
                    }

                    // Risk report card
                    if let report = viewModel.lastReport {
                        RiskReportCard(report: report)
                    }
                }
                .padding()
            }
        }
    }
}

// MARK: - Risk Report Card

private struct RiskReportCard: View {
    let report: UploadReport

    private var riskColor: Color {
        switch report.riskLevel.uppercased() {
        case "HIGH": return .red
        case "MODERATE": return .orange
        default: return .green
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header
            HStack {
                Image(systemName: report.impactDetected ? "exclamationmark.triangle.fill" : "checkmark.shield.fill")
                    .foregroundStyle(riskColor)
                    .font(.title2)
                VStack(alignment: .leading, spacing: 2) {
                    Text(report.riskLevel)
                        .font(.headline)
                        .foregroundStyle(riskColor)
                    Text(report.recommendation)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }

            Divider()

            // Metrics grid
            LazyVGrid(columns: [
                GridItem(.flexible()),
                GridItem(.flexible())
            ], alignment: .leading, spacing: 8) {
                MetricRow(label: "Linear Velocity", value: String(format: "%.2f m/s", report.linearVelocityMs))
                MetricRow(label: "Rotational Velocity", value: String(format: "%.1f deg/s", report.rotationalVelocityDegs))
                MetricRow(label: "Impact Duration", value: String(format: "%.1f ms", report.impactDurationMs))
                MetricRow(label: "Impact Location", value: report.impactLocation)
                MetricRow(label: "Frame of Impact", value: "\(report.frameOfImpact)")
                MetricRow(label: "Player", value: report.playerID)
            }
        }
        .padding()
        .background(riskColor.opacity(0.08))
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(riskColor.opacity(0.3), lineWidth: 1)
        )
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }
}

private struct MetricRow: View {
    let label: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label)
                .font(.system(.caption2))
                .foregroundStyle(.secondary)
            Text(value)
                .font(.system(.callout, design: .monospaced, weight: .medium))
        }
    }
}
