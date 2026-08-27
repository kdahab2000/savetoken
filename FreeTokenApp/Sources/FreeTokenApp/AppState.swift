import Foundation
import SwiftUI
import AppKit
import UniformTypeIdentifiers

struct ChatMessage: Identifiable, Equatable {
    let id = UUID()
    let role: String            // "user" | "assistant"
    var content: String
    var stopped: Bool = false
    var warning: String? = nil
    var attachmentNames: [String] = []
    var imageBase64: [String] = []
}

struct Banner: Equatable {
    enum Kind { case error, info }
    let kind: Kind
    let text: String
}

/// First-run / offline help. Commands shown in the UI are built from the
/// user's own settings (see SettingsStore) and never contain
/// maintainer-specific paths. The app never executes shell commands itself.
enum BackendHelp {
    static let host = ServerClient.host
    static let port = BackendProvider.saveToken.port
    static let pollIntervalSeconds = 5
}

enum ConnectionStatus: Equatable {
    case connecting
    case online(activeModel: String)
    case offline(reason: String)
}

enum ResponseLanguagePolicy {
    static let base = "Respond with only the final answer. Do not reveal reasoning, planning, or internal thoughts. Do not output think tags or tokenizer control tokens. Respond in the same language as the user's latest message. If the user writes Arabic, answer in Arabic. Do not translate unless the user explicitly asks for translation. Preserve code, commands, model names, and technical identifiers exactly."

    static func forUserText(_ text: String) -> String {
        let hasArabic = text.unicodeScalars.contains { scalar in
            (0x0600...0x06FF).contains(scalar.value) ||
            (0x0750...0x077F).contains(scalar.value) ||
            (0x08A0...0x08FF).contains(scalar.value)
        }
        if hasArabic {
            return base + " The latest user message contains Arabic. Write the complete answer in Arabic, not English, unless the user explicitly requests a translation."
        }
        return base
    }
}

@MainActor
final class AppState: ObservableObject {
    @Published var provider: BackendProvider
    @Published var status: ConnectionStatus = .connecting
    @Published var models: [ModelInfo] = []
    @Published var health: HealthResponse? = nil
    @Published var messages: [ChatMessage] = []
    @Published var composer: String = ""
    @Published var attachments: [ChatAttachment] = []
    @Published var maxTokens: Int = 512
    @Published var isGenerating = false
    @Published var isSwitching = false
    @Published var streamingText: String = ""
    @Published var streamWarning: String? = nil
    @Published var banner: Banner? = nil
    @Published var lastAdmission: APIErrorDetail? = nil
    @Published var sshAliases: [SSHHost] = []
    @Published var selectedSSHAlias: String?
    @Published var sshCommand: String = SSHClient.identityCommand
    @Published var sshLastResult: SSHCommandResult?
    @Published var sshConfirmation: SSHConfirmation?
    @Published var sshToolsEnabled = false
    @Published var isSSHRunning = false
    @Published var settings: AppSettings
    @Published var ollamaDetected: Bool?

    let speech = SpeechService()

    private(set) var client: ServerClient
    private let settingsStore: SettingsStore
    private let sshClient: SSHClient
    private var selectedModelID: String?
    private var streamTask: Task<Void, Never>?
    private var pollTask: Task<Void, Never>?
    private var pendingModelSSHApproval: CheckedContinuation<Bool, Never>?

    init(client: ServerClient? = nil, settingsStore: SettingsStore = SettingsStore()) {
        let localSSHClient = SSHClient()
        let aliases = localSSHClient.configuredAliases()
        let loaded = settingsStore.load()
        self.settingsStore = settingsStore
        self.settings = loaded
        if let client {
            self.provider = client.provider
            self.client = client
        } else {
            self.provider = loaded.provider
            self.client = ServerClient(
                provider: loaded.provider,
                port: loaded.provider == .saveToken
                    ? loaded.saveTokenPort : loaded.ollamaPort)
        }
        self.sshClient = localSSHClient
        self.sshAliases = aliases
        self.selectedSSHAlias = aliases.first?.alias
        self.selectedModelID = self.provider == .ollama
            ? loaded.selectedOllamaModelID : nil
        self.sshToolsEnabled = loaded.sshToolsEnabled
    }

    var activeModel: ModelInfo? {
        if let selectedModelID {
            return models.first(where: { $0.id == selectedModelID })
        }
        return models.first(where: { $0.isActive })
    }

    var endpointLabel: String { client.endpointLabel }

    var connectionLabel: String {
        switch status {
        case .connecting: return "Connecting…"
        case .online(let m): return "Online — \(m)"
        case .offline(let r): return "Offline — \(r)"
        }
    }
    var isOnline: Bool { if case .online = status { return true } else { return false } }

    func selectProvider(_ newProvider: BackendProvider) {
        guard newProvider != provider else { return }
        guard !isGenerating, !isSwitching else {
            banner = Banner(kind: .error,
                            text: "Stop the current operation before changing providers.")
            return
        }
        provider = newProvider
        client = ServerClient(provider: newProvider)
        settings.provider = newProvider
        if newProvider == .ollama {
            selectedModelID = settings.selectedOllamaModelID
        } else {
            selectedModelID = nil
        }
        settingsStore.save(settings)
        models = []
        health = nil
        banner = nil
        status = .connecting
        Task { await refresh() }
    }

    // MARK: lifecycle

    func start() {
        guard pollTask == nil else { return }
        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                await self?.refresh()
                try? await Task.sleep(nanoseconds: 5_000_000_000)
            }
        }
    }

    func stopPolling() {
        pollTask?.cancel()
        pollTask = nil
    }

    func refresh() async {
        async let ollama = ServerClient.probeOllama(port: settings.ollamaPort)
        do {
            async let h = client.health()
            async let m = client.models()
            let (health, models) = try await (h, m)
            var listed = models.data
            let active: String?
            switch provider {
            case .saveToken:
                active = listed.first(where: { $0.isActive })?.id
                    ?? health.active_model_id ?? health.model
            case .ollama:
                let preferred = selectedModelID ?? settings.selectedOllamaModelID
                active = preferred.flatMap { preferredID in
                    listed.contains(where: { $0.id == preferredID }) ? preferredID : nil
                } ?? listed.first?.id
            }
            selectedModelID = active
            if provider == .ollama, settings.selectedOllamaModelID != active {
                settings.selectedOllamaModelID = active
                settingsStore.save(settings)
            }
            for index in listed.indices {
                listed[index].active = listed[index].id == active
            }
            self.health = health
            self.models = listed
            self.status = .online(activeModel: active ?? "unknown")
        } catch {
            self.status = .offline(reason: "server not reachable at \(endpointLabel)")
            self.health = nil
        }
        self.ollamaDetected = await ollama
    }

    // MARK: settings

    func updateWorkspace(_ path: String) {
        settings.workspacePath = path
        settingsStore.save(settings)
    }

    func setSSHToolsEnabled(_ enabled: Bool) {
        sshToolsEnabled = enabled
        settings.sshToolsEnabled = enabled
        settingsStore.save(settings)
    }

    func setSpeechVoiceIdentifier(_ identifier: String?) {
        settings.speechVoiceIdentifier = identifier
        settingsStore.save(settings)
    }

    func setSpeechRate(_ rate: Double) {
        settings.speechRate = min(max(rate, 0.1), 1.0)
        settingsStore.save(settings)
    }

    func setAutoSpeakReplies(_ enabled: Bool) {
        settings.autoSpeakReplies = enabled
        settingsStore.save(settings)
    }

    func speak(_ message: ChatMessage) {
        if speech.activeMessageID == message.id && speech.isSpeaking {
            speech.stop()
        } else {
            speech.speak(
                text: message.content,
                messageID: message.id,
                voiceIdentifier: settings.speechVoiceIdentifier,
                rate: settings.speechRate)
        }
    }

    func previewSpeech() {
        let selectedLanguage = settings.speechVoiceIdentifier.flatMap { id in
            speech.voices.first(where: { $0.id == id })?.language
        } ?? Locale.current.language.languageCode?.identifier ?? "en"
        let text = selectedLanguage.hasPrefix("ar")
            ? "مرحبًا، هذا اختبار للصوت المحلي في سيف توكن."
            : "Hello, this is SaveToken local speech."
        speech.speak(
            text: text,
            voiceIdentifier: settings.speechVoiceIdentifier,
            rate: settings.speechRate)
    }

    func exportSpeech(_ message: ChatMessage) {
        let panel = NSSavePanel()
        panel.title = "Export spoken response"
        panel.prompt = "Export"
        panel.nameFieldStringValue = "SaveToken-response.wav"
        panel.allowedContentTypes = [.wav]
        panel.canCreateDirectories = true
        guard panel.runModal() == .OK, let url = panel.url else { return }

        Task { [weak self] in
            guard let self else { return }
            do {
                try await self.speech.exportWAV(
                    text: message.content,
                    voiceIdentifier: self.settings.speechVoiceIdentifier,
                    rate: self.settings.speechRate,
                    to: url)
                self.banner = Banner(kind: .info, text: "Exported local speech to \(url.lastPathComponent).")
            } catch {
                self.banner = Banner(kind: .error, text: "Speech export failed: \(error.localizedDescription)")
            }
        }
    }

    /// Startup command for the active provider, built from the user's own
    /// settings — portable on any machine.
    var currentStartCommand: String {
        provider.startCommand(workspacePath: settings.workspacePath)
    }

    // MARK: SSH access

    func refreshSSHAliases() {
        let aliases = sshClient.configuredAliases()
        sshAliases = aliases
        if let selectedSSHAlias, aliases.contains(where: { $0.alias == selectedSSHAlias }) {
            return
        }
        selectedSSHAlias = aliases.first?.alias
    }

    func checkSSHConnection() {
        guard let alias = selectedSSHAlias else {
            banner = Banner(kind: .error, text: "No SSH alias is available in ~/.ssh/config.")
            return
        }
        guard !isSSHRunning else { return }
        isSSHRunning = true
        sshLastResult = nil
        let localSSHClient = sshClient
        Task { [weak self] in
            guard let self else { return }
            do {
                let result = try await localSSHClient.runIdentityCheck(alias: alias)
                self.sshLastResult = result
                self.banner = Banner(
                    kind: result.succeeded ? .info : .error,
                    text: result.succeeded
                        ? "SSH connection to \(alias) succeeded."
                        : "SSH identity check failed on \(alias) (exit \(result.exitCode)).")
            } catch {
                self.banner = Banner(kind: .error, text: "SSH check failed: \(error.localizedDescription)")
            }
            self.isSSHRunning = false
        }
    }

    func requestSSHCommand() {
        guard let alias = selectedSSHAlias else {
            banner = Banner(kind: .error, text: "Select an SSH alias first.")
            return
        }
        guard !isSSHRunning else { return }
        let command = sshCommand.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !command.isEmpty else {
            banner = Banner(kind: .error, text: "Enter a remote command first.")
            return
        }
        sshConfirmation = SSHConfirmation(alias: alias, command: command, source: .manual)
    }

    func confirmSSHCommand() {
        guard let confirmation = sshConfirmation else { return }
        sshConfirmation = nil
        if let continuation = pendingModelSSHApproval {
            pendingModelSSHApproval = nil
            continuation.resume(returning: true)
            return
        }
        executeSSH(alias: confirmation.alias, command: confirmation.command)
    }

    func cancelSSHCommand() {
        sshConfirmation = nil
        if let continuation = pendingModelSSHApproval {
            pendingModelSSHApproval = nil
            continuation.resume(returning: false)
        }
    }

    private func executeSSH(alias: String, command: String) {
        guard !isSSHRunning else { return }
        isSSHRunning = true
        sshLastResult = nil
        let localSSHClient = sshClient
        Task { [weak self] in
            guard let self else { return }
            do {
                let result = try await localSSHClient.run(alias: alias, command: command)
                self.sshLastResult = result
                self.banner = Banner(
                    kind: result.succeeded ? .info : .error,
                    text: result.succeeded
                        ? "SSH command completed on \(alias)."
                        : "SSH command exited with status \(result.exitCode) on \(alias).")
            } catch {
                self.banner = Banner(kind: .error, text: "SSH command failed: \(error.localizedDescription)")
            }
            self.isSSHRunning = false
        }
    }

    private func requestModelSSHApproval(alias: String, command: String) async -> Bool {
        guard sshAliases.contains(where: { $0.alias == alias }) else { return false }
        return await withCheckedContinuation { continuation in
            pendingModelSSHApproval = continuation
            sshConfirmation = SSHConfirmation(alias: alias, command: command, source: .model)
        }
    }

    private func executeApprovedSSH(alias: String, command: String) async -> String {
        guard !isSSHRunning else { return "SSH is already running; try again after it finishes." }
        isSSHRunning = true
        defer { isSSHRunning = false }
        do {
            let result = try await sshClient.run(alias: alias, command: command)
            sshLastResult = result
            return String(result.output.prefix(20_000))
                + (result.output.count > 20_000 ? "\n[output truncated]" : "")
                + "\n[exit \(result.exitCode)]"
        } catch {
            return "SSH error: \(error.localizedDescription)"
        }
    }

    // MARK: model switching (explicit only)

    var switchBlockedReason: String? {
        if isGenerating { return "Stop the current generation before switching models." }
        if isSwitching { return "A switch is already in progress." }
        return nil
    }

    func switchTo(modelID: String) {
        guard !isGenerating, !isSwitching else {
            banner = Banner(kind: .error, text: switchBlockedReason ?? "Switch blocked.")
            return
        }
        if provider == .ollama {
            selectedModelID = modelID
            settings.selectedOllamaModelID = modelID
            settingsStore.save(settings)
            for index in models.indices {
                models[index].active = models[index].id == modelID
            }
            banner = Banner(kind: .info, text: "Selected Ollama model \(modelID).")
            return
        }
        isSwitching = true
        banner = nil
        Task { [weak self] in
            guard let self else { return }
            defer { self.isSwitching = false }
            do {
                let r = try await self.client.switchModel(id: modelID)
                let resident = r.weights_resident_gb.map {
                    String(format: ", %.2f GB resident", $0)
                } ?? ""
                self.banner = Banner(
                    kind: .info,
                    text: "Switched \(r.previous ?? "?") → \(r.model ?? modelID)\(resident).")
                await self.refresh()
            } catch let e as APIError {
                self.banner = Banner(kind: .error,
                                     text: "Switch refused (\(e.detail.code ?? "error")): \(e.detail.message)")
            } catch {
                self.banner = Banner(kind: .error, text: "Switch failed: \(error.localizedDescription)")
            }
        }
    }

    // MARK: generation

    func addAttachments(_ urls: [URL]) {
        for url in urls {
            do {
                let attachment = try AttachmentService.load(url: url)
                attachments.append(attachment)
            } catch {
                banner = Banner(kind: .error, text: "(url.lastPathComponent): (error.localizedDescription)")
            }
        }
    }

    func removeAttachment(_ attachment: ChatAttachment) {
        attachments.removeAll { $0.id == attachment.id }
    }

    private var attachmentText: String {
        attachments.compactMap { attachment in
            guard let content = attachment.content else { return nil }
            return "\n\n--- Attached file: \(attachment.name) ---\n\(content)\n--- End attached file ---"
        }.joined()
    }

    func send() {
        let text = composer.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !isGenerating, isOnline, (!text.isEmpty || !attachments.isEmpty) else { return }
        let pendingAttachments = attachments
        if provider == .saveToken && pendingAttachments.contains(where: { $0.isImage }) {
            banner = Banner(kind: .error, text: "SaveToken MLX accepts text and files, but not images. Select Ollama with a vision model for photos.")
            return
        }
        let messageText = text + attachmentText
        messages.append(ChatMessage(role: "user", content: messageText,
                                     attachmentNames: pendingAttachments.map(\.name),
                                     imageBase64: pendingAttachments.compactMap(\.imageBase64)))
        composer = ""
        attachments.removeAll()
        banner = nil
        lastAdmission = nil
        streamWarning = nil
        streamingText = ""
        isGenerating = true

        if provider == .ollama && sshToolsEnabled && shouldUseSSHTools(for: text) {
            sendWithSSHTools(for: text)
            return
        }

        let payload = [
            ChatMessagePayload(
                role: "system",
                content: ResponseLanguagePolicy.forUserText(messageText)
            )
        ] + messages.map {
            ChatMessagePayload(role: $0.role, content: $0.content,
                               images: $0.imageBase64.isEmpty ? nil : $0.imageBase64)
        }
        let body = ChatRequestBody(messages: payload, max_tokens: maxTokens,
                                   stream: true, allow_maximum_context: nil,
                                   model: provider == .ollama ? activeModel?.id : nil)
        var parser = SSELineParser()
        var accumulatedRaw = ""
        var finishReason: String? = nil
        var sawDone = false

        streamTask = Task { [weak self] in
            guard let self else { return }
            do {
                let lines = self.client.streamChatLines(request: body)
                for try await line in lines {
                    if Task.isCancelled { throw CancellationError() }
                    for event in parser.consume(line: line) {
                        switch event {
                        case .chunk(let chunk):
                            if let warning = chunk.warning { self.streamWarning = warning }
                            if let delta = chunk.choices?.first?.delta?.content,
                               !delta.isEmpty {
                                accumulatedRaw += delta
                                self.streamingText = ResponseTextSanitizer.clean(
                                    accumulatedRaw, finalized: false)
                            }
                            if let fr = chunk.choices?.first?.finish_reason {
                                finishReason = fr
                            }
                        case .done:
                            sawDone = true
                        case .malformed(let raw):
                            self.banner = Banner(
                                kind: .error,
                                text: "Ignored malformed stream data: \(String(raw.prefix(80)))")
                        }
                    }
                }
                for event in parser.flush() {
                    if case .done = event { sawDone = true }
                }
                let visibleText = ResponseTextSanitizer.clean(
                    accumulatedRaw, finalized: true)
                var msg = ChatMessage(role: "assistant", content: visibleText)
                msg.warning = self.streamWarning
                self.appendAssistantMessage(msg)
                if !sawDone {
                    self.banner = Banner(
                        kind: .info,
                        text: "Stream ended without [DONE]\(finishReason.map { " (finish_reason: \($0))" } ?? "").")
                }
            } catch is CancellationError {
                self.commitStopped(ResponseTextSanitizer.clean(
                    accumulatedRaw, finalized: true))
            } catch let e as URLError where e.code == .cancelled {
                self.commitStopped(ResponseTextSanitizer.clean(
                    accumulatedRaw, finalized: true))
            } catch let e as APIError {
                self.lastAdmission = e.detail
                self.banner = Banner(kind: .error, text: e.detail.message)
            } catch {
                self.banner = Banner(kind: .error,
                                     text: "Connection problem: \(error.localizedDescription)")
            }
            self.streamingText = ""
            self.streamWarning = nil
            self.isGenerating = false
            self.streamTask = nil
        }
    }

    /// Tool calls are deliberately opt-in per prompt. Ordinary Ollama chats
    /// keep the low-latency streaming path; only prompts that clearly ask for
    /// a remote/server operation enter the approval-gated SSH path.
    private func shouldUseSSHTools(for text: String) -> Bool {
        let normalized = text.lowercased()
        let indicators = [
            "ssh", "vps", "remote server", "remote host", "hostname",
            "server", "gpu", "docker", "sudo", "systemctl", "journalctl",
            "process", "disk space", "deploy", "service status"
        ]
        return indicators.contains(where: { normalized.contains($0) })
    }

    private func sendWithSSHTools(for userText: String) {
        guard let model = activeModel?.id else {
            banner = Banner(kind: .error, text: "Select an Ollama model before using SSH tools.")
            isGenerating = false
            return
        }

        let visibleHistory = messages.map {
            OllamaMessage(role: $0.role, content: $0.content,
                          images: $0.imageBase64.isEmpty ? nil : $0.imageBase64)
        }
        let system = OllamaMessage(
            role: "system",
            content: ResponseLanguagePolicy.forUserText(userText) + " "
                + "You may call run_ssh only when the user explicitly asks you to inspect or change a server. "
                + "Use only the configured SSH aliases shown by the app. The app will ask the user for approval "
                + "before every command, and you must never ask for passwords or private keys.")
        let tools = [sshToolDefinition()]
        let localClient = client
        let maxTokens = self.maxTokens

        streamTask = Task { [weak self] in
            guard let self else { return }
            do {
                var dialogue = [system] + visibleHistory
                var response = try await localClient.completeOllamaWithTools(
                    model: model, messages: dialogue, tools: tools, maxTokens: maxTokens)
                var round = 0

                while let calls = response.message.tool_calls, !calls.isEmpty, round < 4 {
                    round += 1
                    dialogue.append(OllamaMessage(
                        role: "assistant",
                        content: response.message.content,
                        tool_calls: calls))

                    for call in calls {
                        let function = call.function
                        var toolResult: String
                        if function.name != "run_ssh" {
                            toolResult = "Tool error: unsupported tool \(function.name)."
                        } else if let alias = function.arguments["alias"],
                                  let command = function.arguments["command"],
                                  !alias.isEmpty, !command.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                            self.streamWarning = "SSH approval required for \(alias)…"
                            if await self.requestModelSSHApproval(alias: alias, command: command) {
                                toolResult = await self.executeApprovedSSH(alias: alias, command: command)
                            } else {
                                toolResult = "SSH command denied by the user. Do not retry it unless asked."
                            }
                            self.streamWarning = nil
                        } else {
                            toolResult = "Tool error: run_ssh requires string arguments alias and command."
                        }
                        dialogue.append(OllamaMessage(role: "tool", content: toolResult))
                    }

                    self.streamWarning = "The local model is preparing the final answer…"
                    response = try await localClient.completeOllamaWithTools(
                        model: model, messages: dialogue, tools: tools, maxTokens: maxTokens)
                }

                let finalText: String
                if response.message.tool_calls?.isEmpty == false {
                    finalText = "The model requested too many SSH steps; generation stopped for safety."
                } else {
                    finalText = ResponseTextSanitizer.clean(
                        response.message.content ?? "", finalized: true)
                }
                self.streamingText = finalText
                self.appendAssistantMessage(ChatMessage(role: "assistant", content: finalText))
            } catch let e as APIError {
                self.lastAdmission = e.detail
                self.banner = Banner(kind: .error, text: e.detail.message)
            } catch is CancellationError {
                self.commitStopped(ResponseTextSanitizer.clean(
                    self.streamingText, finalized: true))
            } catch {
                self.banner = Banner(kind: .error,
                                     text: "Ollama tool connection problem: \(error.localizedDescription)")
            }
            self.streamingText = ""
            self.streamWarning = nil
            self.isGenerating = false
            self.streamTask = nil
        }
    }

    private func sshToolDefinition() -> OllamaToolDefinition {
        let configured = sshAliases.map(\.alias).joined(separator: ", ")
        return OllamaToolDefinition(
            type: "function",
            function: OllamaFunctionDefinition(
                name: "run_ssh",
                description: "Run one user-approved command on one configured SSH alias. Allowed aliases: \(configured)",
                parameters: OllamaToolParameters(
                    type: "object",
                    properties: [
                        "alias": OllamaToolProperty(
                            type: "string",
                            description: "Exact alias from the allowed list."),
                        "command": OllamaToolProperty(
                            type: "string",
                            description: "One shell command to run remotely after user approval.")
                    ],
                    required: ["alias", "command"],
                    additionalProperties: false)))
    }

    private func commitStopped(_ partial: String) {
        if partial.isEmpty {
            banner = Banner(kind: .info, text: "Generation stopped before the first token.")
        } else {
            var msg = ChatMessage(role: "assistant", content: partial)
            msg.stopped = true
            messages.append(msg)
        }
    }

    private func appendAssistantMessage(_ message: ChatMessage) {
        messages.append(message)
        if settings.autoSpeakReplies, !message.stopped, !message.content.isEmpty {
            speech.speak(
                text: message.content,
                messageID: message.id,
                voiceIdentifier: settings.speechVoiceIdentifier,
                rate: settings.speechRate)
        }
    }

    func stop() {
        cancelSSHCommand()
        streamTask?.cancel()
    }

    func clear() {
        guard !isGenerating else { return }
        speech.stop()
        messages.removeAll()
        attachments.removeAll()
        banner = nil
        lastAdmission = nil
    }

    // MARK: budget indicator (client-side estimate)

    func currentAssessment() -> BudgetAssessment {
        let active = activeModel
        return Admission.assess(
            promptText: composer,
            maxNewTokens: maxTokens,
            activeCap: active?.active_cap ?? 65536,
            extendedCap: active?.extended_cap ?? 131072)
    }

    func estimatedPrefillSeconds() -> Double {
        let a = currentAssessment()
        let modelID = activeModel?.id ?? ""
        return PrefillEstimator.seconds(
            forTokens: a.estimatedInputTokens,
            curve: PrefillEstimator.curve(forModelID: modelID))
    }
}
