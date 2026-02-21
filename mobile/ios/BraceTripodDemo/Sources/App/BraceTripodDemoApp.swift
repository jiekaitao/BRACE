import SwiftUI

@main
struct BraceTripodDemoApp: App {
    @StateObject private var viewModel = DemoViewModel()

    var body: some Scene {
        WindowGroup {
            ContentView(viewModel: viewModel)
        }
    }
}
