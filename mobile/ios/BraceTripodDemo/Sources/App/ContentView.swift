import SwiftUI

struct ContentView: View {
    @ObservedObject var viewModel: DemoViewModel

    var body: some View {
        NavigationStack {
            TabView {
                CaptureView(viewModel: viewModel)
                    .tabItem {
                        Label("Capture", systemImage: "video.fill")
                    }
                    .tag(0)

                LiveStreamView(viewModel: viewModel)
                    .tabItem {
                        Label("Live Stream", systemImage: "antenna.radiowaves.left.and.right")
                    }
                    .tag(1)
            }
            .navigationTitle("BRACE Sideline")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        viewModel.showSettings = true
                    } label: {
                        Image(systemName: "gearshape")
                    }
                }
            }
            .sheet(isPresented: $viewModel.showSettings) {
                SettingsSheet(viewModel: viewModel)
            }
        }
    }
}

// MARK: - Settings Sheet

private struct SettingsSheet: View {
    @ObservedObject var viewModel: DemoViewModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                Section("Backend") {
                    TextField("Upload base URL", text: $viewModel.backendBaseURL)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .font(.system(.callout, design: .monospaced))
                }

                Section("Session") {
                    TextField("Play ID", text: $viewModel.playID)
                    TextField("Player ID", text: $viewModel.playerID)
                    TextField("Coach Email", text: $viewModel.coachEmail)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .keyboardType(.emailAddress)
                }

                Section("Status") {
                    LabeledContent("Capture") {
                        Text(viewModel.isTripodRunning ? "Running" : "Stopped")
                            .foregroundStyle(viewModel.isTripodRunning ? .green : .secondary)
                    }
                    LabeledContent("WebSocket") {
                        Text(viewModel.isSocketConnected ? "Connected" : "Disconnected")
                            .foregroundStyle(viewModel.isSocketConnected ? .green : .secondary)
                    }
                    LabeledContent("Pending Uploads") {
                        Text("\(viewModel.queueDepth)")
                    }
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}
