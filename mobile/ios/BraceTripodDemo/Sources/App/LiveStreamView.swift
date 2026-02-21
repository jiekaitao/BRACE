import SwiftUI

struct LiveStreamView: View {
    @ObservedObject var viewModel: DemoViewModel

    var body: some View {
        ScrollView {
            VStack(spacing: 16) {
                // Connection status
                GroupBox {
                    HStack(spacing: 8) {
                        Circle()
                            .fill(viewModel.isSocketConnected ? Color.green : Color.gray)
                            .frame(width: 10, height: 10)
                        Text(viewModel.isSocketConnected ? "Connected" : "Disconnected")
                            .font(.subheadline)
                            .foregroundStyle(viewModel.isSocketConnected ? .primary : .secondary)
                        Spacer()
                    }
                } label: {
                    Label("WebSocket Status", systemImage: "antenna.radiowaves.left.and.right")
                }

                // Live stream URL override
                GroupBox {
                    TextField("ws://... or wss://... (optional)", text: $viewModel.liveStreamOverrideURL)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .textFieldStyle(.roundedBorder)
                        .font(.system(.callout, design: .monospaced))
                } label: {
                    Label("Endpoint Override", systemImage: "link")
                }

                // Controls
                HStack(spacing: 12) {
                    Button {
                        viewModel.connectLiveStream()
                    } label: {
                        HStack {
                            Image(systemName: "play.fill")
                            Text("Connect")
                        }
                        .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(viewModel.isSocketConnected)

                    Button {
                        viewModel.disconnectLiveStream()
                    } label: {
                        HStack {
                            Image(systemName: "stop.fill")
                            Text("Disconnect")
                        }
                        .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    .disabled(!viewModel.isSocketConnected)
                }

                // Synthetic feed
                Button {
                    viewModel.toggleSyntheticFeed()
                } label: {
                    HStack {
                        Image(systemName: viewModel.syntheticFeedEnabled ? "waveform.circle.fill" : "waveform.circle")
                        Text(viewModel.syntheticFeedEnabled ? "Stop Synthetic Feed" : "Start Synthetic Feed")
                    }
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .tint(viewModel.syntheticFeedEnabled ? .orange : .blue)

                // Event log
                GroupBox {
                    if viewModel.logs.isEmpty {
                        Text("No events yet")
                            .foregroundStyle(.secondary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    } else {
                        LazyVStack(alignment: .leading, spacing: 4) {
                            ForEach(Array(viewModel.logs.enumerated()), id: \.offset) { _, line in
                                Text(line)
                                    .font(.system(.caption, design: .monospaced))
                                    .frame(maxWidth: .infinity, alignment: .leading)
                            }
                        }
                    }
                } label: {
                    Label("Event Log", systemImage: "list.bullet.rectangle")
                }
            }
            .padding()
        }
    }
}
