import XCTest
@testable import FreeTokenApp

/// First-run / offline behavior, portability guarantees, and settings
/// persistence. Portability rule: no command or constant may contain a
/// maintainer-specific absolute path.
final class OfflineAndReleaseTests: XCTestCase {
    private func tempStore() -> SettingsStore {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("savetoken-tests-\(UUID().uuidString)",
                                    isDirectory: true)
        return SettingsStore(directory: dir)
    }

    @MainActor
    func testBackendUnavailableProducesOfflineState() async {
        // Port 59999 has no listener: connection refused immediately.
        let state = AppState(client: ServerClient(port: 59999),
                             settingsStore: tempStore())
        await state.refresh()
        guard case .offline(let reason) = state.status else {
            return XCTFail("expected offline, got \(state.status)")
        }
        XCTAssertTrue(reason.contains("127.0.0.1:59999"))
        XCTAssertTrue(state.connectionLabel.hasPrefix("Offline"))
        XCTAssertFalse(state.isOnline)
        XCTAssertTrue(state.models.isEmpty)
        XCTAssertNil(state.health)
    }

    @MainActor
    func testSendIsNoOpWhileOffline() async {
        let state = AppState(client: ServerClient(port: 59999),
                             settingsStore: tempStore())
        await state.refresh()
        state.composer = "hello"
        state.send()
        XCTAssertFalse(state.isGenerating)
        XCTAssertTrue(state.messages.isEmpty)
    }

    func testBackendHelpConstants() {
        XCTAssertEqual(BackendHelp.host, "127.0.0.1")  // loopback only, always
        XCTAssertEqual(BackendHelp.port, 8321)
        XCTAssertEqual(BackendHelp.pollIntervalSeconds, 5)
    }

    func testProviderCommandsArePortable() {
        // Without a configured workspace the command uses a placeholder —
        // never a maintainer path.
        let generic = BackendProvider.saveToken.startCommand()
        XCTAssertEqual(generic,
            "cd \"/path/to/SaveToken\" && python3 -m freetoken.server --port 8321")
        // With a user-configured workspace it uses exactly that folder.
        let configured = BackendProvider.saveToken.startCommand(
            workspacePath: "/tmp/some-checkout", port: 8321)
        XCTAssertEqual(configured,
            "cd \"/tmp/some-checkout\" && python3 -m freetoken.server --port 8321")
        XCTAssertEqual(BackendProvider.ollama.startCommand(), "ollama serve")
        XCTAssertEqual(BackendProvider.saveToken.launchCommand(), "open SaveToken.app")
        XCTAssertEqual(BackendProvider.ollama.launchCommand(), "open -a Ollama")
        // Portability invariant across every variant.
        for provider in BackendProvider.allCases {
            for ws in [nil, "", "/anywhere/at/all"] {
                let cmd = provider.startCommand(workspacePath: ws)
                XCTAssertFalse(cmd.contains("/Users/"),
                               "maintainer path leaked in: \(cmd)")
                XCTAssertFalse(cmd.contains("/home/"),
                               "absolute home path leaked in: \(cmd)")
            }
        }
    }

    func testServerClientNeverLeavesLoopback() {
        for port in [8321, 1, 65535] {
            let c = ServerClient(port: port)
            XCTAssertEqual(c.baseURL.host, "127.0.0.1")
            XCTAssertEqual(c.baseURL.scheme, "http")
            XCTAssertTrue(c.baseURL.absoluteString.hasPrefix("http://127.0.0.1:"))
        }
    }

    func testOllamaProviderUsesLoopbackPort() {
        let c = ServerClient(provider: .ollama)
        XCTAssertEqual(c.provider, .ollama)
        XCTAssertEqual(c.baseURL.absoluteString, "http://127.0.0.1:11434")
        XCTAssertEqual(c.endpointLabel, "http://127.0.0.1:11434 (loopback only)")
    }

    func testSettingsDefaultsRoundTripAndCorruptFile() {
        let store = tempStore()
        // Missing file → defaults.
        var s = store.load()
        XCTAssertEqual(s, AppSettings())
        XCTAssertEqual(s.provider, .saveToken)
        XCTAssertFalse(s.hasWorkspace)
        // Round trip.
        s.provider = .ollama
        s.workspacePath = "/tmp/my-checkout"
        s.saveTokenPort = 9000
        s.selectedOllamaModelID = "qwen3.8:27b-mlx"
        s.sshToolsEnabled = true
        XCTAssertTrue(store.save(s))
        let loaded = store.load()
        XCTAssertEqual(loaded, s)
        // Corrupt file → defaults, no crash.
        try? Data("not json".utf8).write(to: store.fileURL)
        XCTAssertEqual(store.load(), AppSettings())
        try? FileManager.default.removeItem(at: store.directory)
    }

    @MainActor
    func testProviderSelectionPersistsToSettings() async {
        let store = tempStore()
        let state = AppState(client: ServerClient(provider: .saveToken),
                             settingsStore: store)
        state.selectProvider(.ollama)
        XCTAssertEqual(store.load().provider, .ollama)
        state.selectProvider(.saveToken)
        XCTAssertEqual(store.load().provider, .saveToken)
        try? FileManager.default.removeItem(at: store.directory)
    }

    @MainActor
    func testWorkspaceSettingDrivesStartCommand() {
        let state = AppState(client: ServerClient(provider: .saveToken),
                             settingsStore: tempStore())
        XCTAssertTrue(state.currentStartCommand.contains("/path/to/SaveToken"))
        state.updateWorkspace("/tmp/my-checkout")
        XCTAssertEqual(state.currentStartCommand,
            "cd \"/tmp/my-checkout\" && python3 -m freetoken.server --port 8321")
        XCTAssertTrue(state.settings.hasWorkspace)
    }

    @MainActor
    func testOllamaModelAndSSHPreferencesPersist() {
        let store = tempStore()
        var initial = AppSettings()
        initial.provider = .ollama
        initial.selectedOllamaModelID = "qwen3.8:27b-mlx"
        initial.sshToolsEnabled = true
        XCTAssertTrue(store.save(initial))

        let state = AppState(client: ServerClient(provider: .ollama), settingsStore: store)
        XCTAssertEqual(state.settings.selectedOllamaModelID, "qwen3.8:27b-mlx")
        XCTAssertTrue(state.sshToolsEnabled)

        state.setSSHToolsEnabled(false)
        XCTAssertFalse(store.load().sshToolsEnabled)
        try? FileManager.default.removeItem(at: store.directory)
    }
}
