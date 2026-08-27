import Foundation

/// Per-user settings, stored outside the app bundle and outside any source
/// checkout: ~/Library/Application Support/SaveToken/settings.json.
/// Nothing here ever contains credentials; only UI preferences and the
/// user-chosen location of their SaveToken checkout.
struct AppSettings: Codable, Equatable {
    var provider: BackendProvider = .saveToken
    /// Directory that contains the `freetoken` Python package. Empty string
    /// means "not configured yet" — the UI then shows portable instructions
    /// instead of machine-specific paths.
    var workspacePath: String = ""
    var saveTokenPort: Int = BackendProvider.saveToken.port
    var ollamaPort: Int = BackendProvider.ollama.port
    /// Last locally installed Ollama model selected by the user. The model
    /// is only a name; weights remain managed by Ollama and are never copied
    /// into SaveToken.
    var selectedOllamaModelID: String? = nil
    /// Whether the local Ollama model may request an SSH command. Every
    /// requested command still goes through the in-app approval dialog.
    var sshToolsEnabled: Bool = false
    /// Empty/nil means macOS chooses the voice automatically for the text.
    var speechVoiceIdentifier: String? = nil
    /// AVSpeechUtterance rate. The UI keeps this inside a comfortable range.
    var speechRate: Double = 0.5
    /// Speak each completed assistant response without an extra click.
    var autoSpeakReplies: Bool = false
    /// When enabled, each prompt is augmented with DuckDuckGo Instant Answer results.
    var webSearchEnabled: Bool = false

    private enum CodingKeys: String, CodingKey {
        case provider
        case workspacePath
        case saveTokenPort
        case ollamaPort
        case selectedOllamaModelID
        case sshToolsEnabled
        case speechVoiceIdentifier
        case speechRate
        case autoSpeakReplies
        case webSearchEnabled
    }

    init() {}

    /// Decode older settings files that predate model and SSH preferences.
    /// Missing keys intentionally fall back to safe defaults.
    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        provider = try values.decodeIfPresent(BackendProvider.self, forKey: .provider) ?? .saveToken
        workspacePath = try values.decodeIfPresent(String.self, forKey: .workspacePath) ?? ""
        saveTokenPort = try values.decodeIfPresent(Int.self, forKey: .saveTokenPort) ?? BackendProvider.saveToken.port
        ollamaPort = try values.decodeIfPresent(Int.self, forKey: .ollamaPort) ?? BackendProvider.ollama.port
        selectedOllamaModelID = try values.decodeIfPresent(String.self, forKey: .selectedOllamaModelID)
        sshToolsEnabled = try values.decodeIfPresent(Bool.self, forKey: .sshToolsEnabled) ?? false
        speechVoiceIdentifier = try values.decodeIfPresent(String.self, forKey: .speechVoiceIdentifier)
        let decodedSpeechRate = try values.decodeIfPresent(Double.self, forKey: .speechRate) ?? 0.5
        speechRate = min(max(decodedSpeechRate, 0.1), 1.0)
        autoSpeakReplies = try values.decodeIfPresent(Bool.self, forKey: .autoSpeakReplies) ?? false
        webSearchEnabled = try values.decodeIfPresent(Bool.self, forKey: .webSearchEnabled) ?? false
    }

    var hasWorkspace: Bool {
        !workspacePath.trimmingCharacters(in: .whitespaces).isEmpty
    }
}

final class SettingsStore {
    static let folderName = "SaveToken"
    static let fileName = "settings.json"

    let directory: URL

    /// Default location: ~/Library/Application Support/SaveToken.
    /// Tests (and future sandboxed builds) may inject another directory.
    init(directory: URL? = nil) {
        if let directory {
            self.directory = directory
        } else {
            let base = FileManager.default.urls(
                for: .applicationSupportDirectory, in: .userDomainMask).first
                ?? FileManager.default.homeDirectoryForCurrentUser
            self.directory = base.appendingPathComponent(Self.folderName,
                                                         isDirectory: true)
        }
    }

    var fileURL: URL {
        directory.appendingPathComponent(Self.fileName)
    }

    /// Loads settings, falling back to defaults when the file is missing,
    /// unreadable, or corrupt. Never throws into the UI.
    func load() -> AppSettings {
        guard let data = try? Data(contentsOf: fileURL) else { return AppSettings() }
        return (try? JSONDecoder().decode(AppSettings.self, from: data)) ?? AppSettings()
    }

    /// Saves atomically (temp file + rename), creating the directory if
    /// needed. Returns false if persistence failed (settings still usable
    /// in-memory).
    @discardableResult
    func save(_ settings: AppSettings) -> Bool {
        do {
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            let data = try JSONEncoder().encode(settings)
            let tmp = fileURL.appendingPathExtension("tmp")
            try data.write(to: tmp, options: .atomic)
            _ = try? FileManager.default.replaceItemAt(fileURL, withItemAt: tmp)
            if FileManager.default.fileExists(atPath: tmp.path) {
                try FileManager.default.moveItem(at: tmp, to: fileURL)
            }
            return true
        } catch {
            return false
        }
    }
}
