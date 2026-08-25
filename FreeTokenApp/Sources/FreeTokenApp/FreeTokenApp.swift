import SwiftUI
import AppKit

@main
struct FreeTokenApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @StateObject private var state = AppState()

    var body: some Scene {
        WindowGroup("SaveToken — local inference") {
            ContentView()
                .environmentObject(state)
                .frame(minWidth: 960, minHeight: 640)
                .onAppear { state.start() }
        }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        // Bring the window forward even when launched as an unbundled binary.
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }
}
