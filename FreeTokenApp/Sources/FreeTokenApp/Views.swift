import SwiftUI
import AppKit

struct ContentView: View {
    @EnvironmentObject var state: AppState

    var body: some View {
        VStack(spacing: 0) {
            HeaderBar()
            Divider()
            HSplitView {
                SidebarView()
                    .frame(minWidth: 280, idealWidth: 320, maxWidth: 420)
                ChatView()
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            Divider()
            DisclaimerBar()
        }
    }
}

// MARK: header / status

struct HeaderBar: View {
    @EnvironmentObject var state: AppState

    private var dotColor: Color {
        switch state.status {
        case .online: return .green
        case .connecting: return .yellow
        case .offline: return .red
        }
    }

    private var localityText: String {
        if state.provider == .ollama, state.activeModel?.isOllamaCloud == true {
            return "\(state.endpointLabel) — remote Ollama cloud model"
        }
        return "\(state.endpointLabel) — fully local"
    }

    var body: some View {
        HStack(spacing: 8) {
            Circle().fill(dotColor).frame(width: 10, height: 10)
            Text("\(state.provider.displayName) · \(state.connectionLabel)")
                .font(.subheadline)
            Text("(\(localityText))")
                .font(.caption)
                .foregroundColor(.secondary)
            Spacer()
            Button {
                Task { await state.refresh() }
            } label: {
                Image(systemName: "arrow.clockwise")
            }
            .help("Re-check server status")
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
    }
}

// MARK: sidebar: model picker + diagnostics

struct SidebarView: View {
    @EnvironmentObject var state: AppState

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                Picker("Backend", selection: Binding(
                    get: { state.provider },
                    set: { state.selectProvider($0) })) {
                    ForEach(BackendProvider.allCases) { provider in
                        Text(provider.displayName).tag(provider)
                    }
                }
                .pickerStyle(.segmented)
                .disabled(state.isGenerating || state.isSwitching)

                SettingsView()

                HStack {
                    Text("Models").font(.headline)
                    Spacer()
                    Button("Refresh") { Task { await state.refresh() } }
                        .font(.caption)
                }
                if state.models.isEmpty {
                    Text(state.isOnline
                         ? "No models in catalog."
                         : "Server offline — catalog unavailable.")
                        .foregroundColor(.secondary)
                        .font(.callout)
                }
                ForEach(state.models) { model in
                    ModelRow(model: model)
                }
                if let banner = state.banner {
                    BannerView(banner: banner)
                }
                Divider()
                SSHPanel()
                Divider()
                DiagnosticsView()
            }
            .padding(12)
        }
    }
}

struct ModelRow: View {
    @EnvironmentObject var state: AppState
    let model: ModelInfo

    private func residentText() -> String {
        if state.provider == .ollama {
            if model.isOllamaCloud {
                return "remote inference through Ollama cloud; no local weights"
            }
            if let mb = model.expected_resident_mb {
                return String(format: "downloaded ~%.0f MB (Ollama; loads on use)", Double(mb))
            }
            return "managed by Ollama; resident memory measured by Ollama"
        }
        if model.isActive, let gb = model.weights_resident_gb {
            return String(format: "resident %.2f GB (measured at load)", gb)
        }
        if let mb = model.expected_resident_mb {
            return "expected ~\(mb) MB (estimate)"
        }
        return "resident memory unknown"
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(model.id)
                    .font(.system(.body, design: .monospaced).weight(.semibold))
                Spacer()
                if model.isActive {
                    Text("ACTIVE")
                        .font(.caption2.bold())
                        .padding(.horizontal, 6).padding(.vertical, 2)
                        .background(Color.green.opacity(0.2))
                        .foregroundColor(.green)
                        .cornerRadius(4)
                }
                if state.provider == .ollama && model.isOllamaCloud {
                    Text("CLOUD")
                        .font(.caption2.bold())
                        .padding(.horizontal, 6).padding(.vertical, 2)
                        .background(Color.orange.opacity(0.2))
                        .foregroundColor(.orange)
                        .cornerRadius(4)
                }
            }
            Text("rev \(model.revision ?? "?") · \(model.format ?? "?") · \(model.dtype ?? "?")")
                .font(.caption).foregroundColor(.secondary)
            Text(residentText()).font(.caption)
            if state.provider == .ollama {
                Text(model.isOllamaCloud
                     ? "context limit \(model.context_limit.map { "\($0)" } ?? "?") · prompts leave this Mac through Ollama"
                     : "context limit \(model.context_limit.map { "\($0)" } ?? "?") · loaded by Ollama on first request")
                    .font(.caption).foregroundColor(.secondary)
            } else {
                Text("context limit \(model.context_limit.map { "\($0)" } ?? "?") · cap \(model.active_cap.map { "\($0)" } ?? "?") (default \(model.default_cap.map { "\($0)" } ?? "?"), extended \(model.extended_cap.map { "\($0)" } ?? "?")\(model.allow_extended == true ? ", extended enabled" : "")")
                    .font(.caption).foregroundColor(.secondary)
            }
            if (model.research_only ?? false) {
                Text("research-only").font(.caption2).foregroundColor(.orange)
            }
            if !model.isActive {
                Button(state.provider == .ollama ? "Use this model" : "Make active (verify + load)") {
                    state.switchTo(modelID: model.id)
                }
                .disabled(state.isGenerating || state.isSwitching)
                .help(state.switchBlockedReason ?? (state.provider == .ollama
                    ? (model.isOllamaCloud
                       ? "Select this remote Ollama cloud model."
                       : "Select this locally installed Ollama model.")
                    : "Explicitly switch to this model. Checksum-verified before loading."))
            }
            if state.provider == .saveToken && state.isSwitching && !model.isActive {
                ProgressView().controlSize(.small)
            }
        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(model.isActive ? Color.accentColor.opacity(0.08) : Color.gray.opacity(0.06))
        )
    }
}

struct BannerView: View {
    let banner: Banner

    var body: some View {
        Text(banner.text)
            .font(.callout)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(8)
            .background(
                RoundedRectangle(cornerRadius: 6)
                    .fill(banner.kind == .error ? Color.red.opacity(0.12) : Color.blue.opacity(0.10))
            )
            .foregroundColor(banner.kind == .error ? .red : .primary)
            .textSelection(.enabled)
    }
}

struct SSHPanel: View {
    @EnvironmentObject var state: AppState
    @State private var expanded = true

    var body: some View {
        DisclosureGroup("SSH access", isExpanded: $expanded) {
            VStack(alignment: .leading, spacing: 8) {
                Text("Uses aliases from ~/.ssh/config. Passwords and private keys never enter SaveToken.")
                    .font(.caption)
                    .foregroundColor(.secondary)

                if state.sshAliases.isEmpty {
                    Text("No SSH aliases found. Add a Host entry to ~/.ssh/config, then refresh.")
                        .font(.caption)
                        .foregroundColor(.orange)
                } else {
                    Picker("Host", selection: Binding(
                        get: { state.selectedSSHAlias ?? "" },
                        set: { state.selectedSSHAlias = $0.isEmpty ? nil : $0 })) {
                        ForEach(state.sshAliases) { host in
                            Text(host.alias).tag(host.alias)
                        }
                    }

                    HStack(spacing: 6) {
                        Button {
                            state.checkSSHConnection()
                        } label: {
                            Label("Check connection", systemImage: "checkmark.shield")
                        }
                        .disabled(state.isSSHRunning)
                        Button {
                            state.refreshSSHAliases()
                        } label: {
                            Image(systemName: "arrow.clockwise")
                        }
                        .help("Reload SSH aliases from ~/.ssh/config")
                    }
                }

                Text("Remote command").font(.caption).foregroundColor(.secondary)
                TextEditor(text: $state.sshCommand)
                    .font(.system(.caption, design: .monospaced))
                    .frame(minHeight: 48, maxHeight: 92)
                    .padding(4)
                    .background(RoundedRectangle(cornerRadius: 6).fill(Color.gray.opacity(0.06)))

                Button {
                    state.requestSSHCommand()
                } label: {
                    Label("Run with confirmation", systemImage: "terminal")
                }
                .disabled(state.selectedSSHAlias == nil || state.isSSHRunning)

                if state.provider == .ollama {
                    Toggle("Allow Ollama model to request SSH", isOn: Binding(
                        get: { state.sshToolsEnabled },
                        set: { state.setSSHToolsEnabled($0) }))
                        .disabled(state.isGenerating)
                    Text(state.sshToolsEnabled
                         ? "Enabled for server/SSH prompts; every requested command still requires your approval."
                         : "Off for normal chats. Enable it when you want the model to inspect a server.")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }

                if state.isSSHRunning {
                    HStack(spacing: 6) {
                        ProgressView().controlSize(.small)
                        Text("SSH command running…").font(.caption)
                    }
                }

                if let result = state.sshLastResult {
                    VStack(alignment: .leading, spacing: 3) {
                        Text("\(result.alias) $ \(result.command)")
                            .font(.system(.caption, design: .monospaced).weight(.semibold))
                            .textSelection(.enabled)
                        Text(result.output)
                            .font(.system(.caption, design: .monospaced))
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(6)
                            .background(RoundedRectangle(cornerRadius: 6).fill(Color.black.opacity(0.06)))
                    }
                }
            }
            .padding(.top, 6)
        }
        .onAppear { state.refreshSSHAliases() }
        .alert(item: $state.sshConfirmation) { confirmation in
            let source = confirmation.source == .model
                ? "The local Ollama model requested this command."
                : "Review this remote command before running it."
            return Alert(
                title: Text("Confirm SSH command"),
                message: Text("\(source)\n\nHost: \(confirmation.alias)\n\n\(confirmation.command)"),
                primaryButton: .destructive(Text("Run command")) {
                    state.confirmSSHCommand()
                },
                secondaryButton: .cancel {
                    state.cancelSSHCommand()
                })
        }
    }
}

// MARK: settings (persisted in ~/Library/Application Support/SaveToken)

struct SettingsView: View {
    @EnvironmentObject var state: AppState
    @State private var expanded = false

    var body: some View {
        DisclosureGroup("Settings", isExpanded: $expanded) {
            VStack(alignment: .leading, spacing: 8) {
                if state.provider == .saveToken {
                    Text("SaveToken checkout folder (contains the freetoken package). Used only to build the copyable startup command — never sent anywhere.")
                        .font(.caption2).foregroundColor(.secondary)
                    TextField("/path/to/SaveToken", text: Binding(
                        get: { state.settings.workspacePath },
                        set: { state.updateWorkspace($0) }))
                        .textFieldStyle(.roundedBorder)
                        .font(.system(.caption, design: .monospaced))
                    Text(state.settings.hasWorkspace
                         ? "Configured — startup commands use this folder."
                         : "Not configured — instructions show a placeholder.")
                        .font(.caption2)
                        .foregroundColor(state.settings.hasWorkspace ? .green : .secondary)
                }
                SpeechSettingsView(speech: state.speech)
                setupStatusRow("Ollama on 127.0.0.1:11434", detected: state.ollamaDetected)
                setupStatusRow("\(state.provider.displayName) backend",
                               detected: state.isOnline ? true : false)
                Text("Settings persist per user; endpoints stay loopback-only.")
                    .font(.caption2).foregroundColor(.secondary)
            }
            .padding(.top, 4)
        }
    }

    private func setupStatusRow(_ label: String, detected: Bool?) -> some View {
        HStack(spacing: 6) {
            Circle()
                .fill(detected == nil ? Color.gray : (detected! ? Color.green : Color.red))
                .frame(width: 8, height: 8)
            Text("\(label): \(detected == nil ? "checking…" : (detected! ? "detected" : "not detected"))")
                .font(.caption)
        }
    }
}

struct SpeechSettingsView: View {
    @EnvironmentObject var state: AppState
    @ObservedObject var speech: SpeechService

    var body: some View {
        Divider()
        Text("On-device speech").font(.caption.bold())
        Picker("Voice", selection: Binding(
            get: { state.settings.speechVoiceIdentifier ?? "" },
            set: { state.setSpeechVoiceIdentifier($0.isEmpty ? nil : $0) })) {
            Text("Automatic (macOS)").tag("")
            ForEach(speech.voices) { voice in
                Text(voice.label).tag(voice.id)
            }
        }
        .pickerStyle(.menu)

        HStack {
            Text("Speed").font(.caption)
            Slider(value: Binding(
                get: { state.settings.speechRate },
                set: { state.setSpeechRate($0) }), in: 0.3...0.65)
            Text(String(format: "%.2f", state.settings.speechRate))
                .font(.caption.monospacedDigit())
                .frame(width: 32, alignment: .trailing)
        }

        Toggle("Speak completed replies automatically", isOn: Binding(
            get: { state.settings.autoSpeakReplies },
            set: { state.setAutoSpeakReplies($0) }))
            .font(.caption)

        Button {
            if speech.isSpeaking {
                speech.stop()
            } else {
                state.previewSpeech()
            }
        } label: {
            Label(speech.isSpeaking ? "Stop voice test" : "Test voice",
                  systemImage: speech.isSpeaking ? "stop.fill" : "speaker.wave.2.fill")
        }
        .disabled(speech.isExporting)

        Text("Uses voices installed in macOS. Speech playback and WAV export stay on this Mac.")
            .font(.caption2)
            .foregroundColor(.secondary)
    }
}

// MARK: first-run / offline help

struct CopyableCommand: View {
    let label: String
    let command: String

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label).font(.caption).foregroundColor(.secondary)
            HStack(alignment: .top, spacing: 8) {
                Text(command)
                    .font(.system(.caption, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(8)
                    .background(RoundedRectangle(cornerRadius: 6).fill(Color.black.opacity(0.06)))
                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(command, forType: .string)
                } label: {
                    Image(systemName: "doc.on.doc")
                }
                .help("Copy command to clipboard")
            }
        }
    }
}

struct OfflinePanel: View {
    @EnvironmentObject var state: AppState

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 8) {
                Image(systemName: "server.rack")
                    .font(.title2).foregroundColor(.orange)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Backend not running").font(.headline)
                    Text("\(state.provider.displayName) is offline. It connects only to \(state.endpointLabel) on this Mac.")
                        .font(.caption).foregroundColor(.secondary)
                }
            }
            Text("Start the local backend, then press Retry (or wait — the app re-checks every \(BackendHelp.pollIntervalSeconds) seconds automatically).")
                .font(.callout)
            CopyableCommand(label: "1 · Start the backend (run in Terminal)",
                            command: state.currentStartCommand)
            CopyableCommand(label: "2 · Launch this app",
                            command: state.provider.launchCommand())
            if state.provider == .saveToken && !state.settings.hasWorkspace {
                Text("Tip: set your SaveToken checkout folder in Settings (sidebar) to get an exact startup command for your machine.")
                    .font(.caption).foregroundColor(.secondary)
            }
            if let ollama = state.ollamaDetected {
                Text(ollama
                     ? "Ollama detected on 127.0.0.1:11434 — you can switch to the Ollama backend above."
                     : "Ollama not detected on 127.0.0.1:11434 (install it from ollama.com if you want that backend).")
                    .font(.caption).foregroundColor(.secondary)
            }
            HStack {
                Button {
                    Task { await state.refresh() }
                } label: {
                    Label("Retry now", systemImage: "arrow.clockwise")
                }
                Text("Automatic retry every \(BackendHelp.pollIntervalSeconds)s is on.")
                    .font(.caption).foregroundColor(.secondary)
            }
        }
        .padding(16)
        .background(RoundedRectangle(cornerRadius: 10).fill(Color.orange.opacity(0.08)))
        .padding(16)
    }
}

// MARK: diagnostics

struct DiagnosticsView: View {
    @EnvironmentObject var state: AppState
    @State private var expanded = true

    var body: some View {
        DisclosureGroup("Diagnostics", isExpanded: $expanded) {
            VStack(alignment: .leading, spacing: 6) {
                diagRow("Endpoint", state.endpointLabel)
                if let h = state.health {
                    diagRow("Server status", h.status ?? "?")
                    diagRow("Active model", h.active_model_id ?? h.model ?? state.activeModel?.id ?? "?")
                    if let gb = h.weights_resident_gb {
                        diagRow("Weights resident", String(format: "%.2f GB (measured)", gb))
                    }
                } else {
                    diagRow("Server status", "no /health response")
                }
                if let m = state.activeModel {
                    if state.provider == .saveToken {
                        diagRow("Default cap", m.default_cap.map { "\($0)" } ?? "?")
                        diagRow("Active cap", m.active_cap.map { "\($0)" } ?? "?")
                        diagRow("Extended cap", m.extended_cap.map { "\($0)" } ?? "?")
                        diagRow("Hard limit", "\(Admission.hardLimit) (architectural)")
                        diagRow("KV cache cost", "12,288 B/token (calibrated M1)")
                        diagRow("Fixed recurrent state", "19,537,920 B (calibrated M1)")
                        diagRow("Estimator safety margin", "2 GiB (calibrated M1)")
                        let prefill = state.estimatedPrefillSeconds()
                        diagRow("Prefill for current prompt",
                                String(format: "~%.1f s (estimated from M1 curve)", prefill))
                    } else {
                        diagRow("Model format", m.isOllamaCloud
                                ? "Ollama cloud / remote (\(m.format ?? "unknown"))"
                                : "Ollama local (\(m.format ?? "unknown"))")
                        diagRow("Context limit", m.context_limit.map { "\($0)" } ?? "managed by Ollama")
                        diagRow("Memory", "managed by Ollama")
                    }
                }
                Text("Values marked (estimated) are predictions, not measurements.")
                    .font(.caption2).foregroundColor(.secondary)
            }
            .font(.caption)
            .padding(.top, 4)
        }
    }

    private func diagRow(_ label: String, _ value: String) -> some View {
        HStack(alignment: .top) {
            Text(label).foregroundColor(.secondary).frame(width: 130, alignment: .leading)
            Text(value).textSelection(.enabled)
        }
    }
}

// MARK: disclaimer (persistent)

struct DisclaimerBar: View {
    var body: some View {
        Text("For research and development only — not for clinical diagnosis or treatment. SaveToken connects only to 127.0.0.1; Ollama models labeled CLOUD may use remote inference. No telemetry or prompt logging by SaveToken.")
            .font(.caption2)
            .foregroundColor(.secondary)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 6)
            .padding(.horizontal, 12)
            .background(Color.gray.opacity(0.08))
    }
}
