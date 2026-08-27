import SwiftUI
import UniformTypeIdentifiers

struct ChatView: View {
    @EnvironmentObject var state: AppState

    var body: some View {
        VStack(spacing: 0) {
            if !state.isOnline {
                OfflinePanel()
                Divider()
            }
            MessageList()
            Divider()
            BudgetIndicator()
            ComposerBar()
        }
    }
}

// MARK: message history + streaming bubble

struct MessageList: View {
    @EnvironmentObject var state: AppState

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 10) {
                    if state.messages.isEmpty && !state.isGenerating {
                        Text("Start a conversation. The model is a research checkpoint; keep prompts generic and non-clinical.")
                            .foregroundColor(.secondary)
                            .font(.callout)
                            .padding(.top, 24)
                    }
                    ForEach(state.messages) { msg in
                        MessageBubble(message: msg)
                            .id(msg.id)
                    }
                    if state.isGenerating {
                        StreamingBubble()
                            .id("streaming")
                    }
                }
                .padding(12)
            }
            .onChange(of: state.streamingText) { _ in
                proxy.scrollTo("streaming", anchor: .bottom)
            }
            .onChange(of: state.messages.count) { _ in
                if let last = state.messages.last {
                    proxy.scrollTo(last.id, anchor: .bottom)
                }
            }
        }
    }
}

struct MessageBubble: View {
    @EnvironmentObject var state: AppState
    let message: ChatMessage

    var body: some View {
        HStack {
            if message.role == "user" { Spacer(minLength: 60) }
            VStack(alignment: .leading, spacing: 4) {
                Text(message.content)
                    .textSelection(.enabled)
                if !message.attachmentNames.isEmpty {
                    Label(message.attachmentNames.joined(separator: ", "), systemImage: "paperclip")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                HStack(spacing: 6) {
                    if message.stopped {
                        Label("stopped", systemImage: "stop.circle")
                            .font(.caption2).foregroundColor(.orange)
                    }
                    if let warning = message.warning {
                        Text(warning)
                            .font(.caption2).foregroundColor(.orange)
                            .textSelection(.enabled)
                    }
                }
                if message.role == "assistant", !message.content.isEmpty {
                    MessageSpeechActions(speech: state.speech, message: message)
                }
            }
            .padding(10)
            .background(
                RoundedRectangle(cornerRadius: 10)
                    .fill(message.role == "user"
                          ? Color.accentColor.opacity(0.18)
                          : Color.gray.opacity(0.10))
            )
            if message.role == "assistant" { Spacer(minLength: 60) }
        }
    }
}

struct MessageSpeechActions: View {
    @EnvironmentObject var state: AppState
    @ObservedObject var speech: SpeechService
    let message: ChatMessage

    private var isThisMessageSpeaking: Bool {
        speech.isSpeaking && speech.activeMessageID == message.id
    }

    var body: some View {
        HStack(spacing: 8) {
            Button {
                state.speak(message)
            } label: {
                Label(isThisMessageSpeaking ? "Stop speaking" : "Speak",
                      systemImage: isThisMessageSpeaking ? "stop.fill" : "speaker.wave.2.fill")
                    .labelStyle(.iconOnly)
            }
            .buttonStyle(.borderless)
            .help(isThisMessageSpeaking ? "Stop local speech" : "Speak this response locally")

            Button {
                state.exportSpeech(message)
            } label: {
                Label("Export WAV", systemImage: "square.and.arrow.down")
                    .labelStyle(.iconOnly)
            }
            .buttonStyle(.borderless)
            .disabled(speech.isExporting)
            .help("Export this response as a local WAV file")

            if speech.isExporting {
                ProgressView().controlSize(.mini)
            }
            Text("on-device speech")
                .font(.caption2)
                .foregroundColor(.secondary)
        }
    }
}

struct StreamingBubble: View {
    @EnvironmentObject var state: AppState

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                if let warning = state.streamWarning {
                    Text(warning)
                        .font(.caption2).foregroundColor(.orange)
                        .textSelection(.enabled)
                }
                if state.streamingText.isEmpty {
                    HStack(spacing: 6) {
                        ProgressView().controlSize(.small)
                        Text("Prefilling / generating… (long contexts can take minutes)")
                            .font(.caption).foregroundColor(.secondary)
                    }
                } else {
                    Text(state.streamingText + "▍")
                        .textSelection(.enabled)
                }
            }
            .padding(10)
            .background(RoundedRectangle(cornerRadius: 10).fill(Color.gray.opacity(0.10)))
            Spacer(minLength: 60)
        }
    }
}

// MARK: context-budget indicator

struct BudgetIndicator: View {
    @EnvironmentObject var state: AppState

    var body: some View {
        let a = state.currentAssessment()
        let activeCap = state.activeModel?.active_cap ?? 65536
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 6) {
                Image(systemName: gaugeIcon(a.verdict))
                    .foregroundColor(verdictColor(a.verdict))
                Text(budgetLine(a, activeCap: activeCap))
                    .font(.caption)
                    .foregroundColor(verdictColor(a.verdict))
                Spacer()
                Text("hard limit \(Admission.hardLimit) — requests above are always rejected")
                    .font(.caption2).foregroundColor(.secondary)
            }
            if let admission = state.lastAdmission {
                Text(serverAdmissionLine(admission))
                    .font(.caption2).foregroundColor(.red)
                    .textSelection(.enabled)
            }
        }
        .padding(.horizontal, 12)
        .padding(.top, 6)
    }

    private func gaugeIcon(_ v: BudgetVerdict) -> String {
        switch v {
        case .ok: return "checkmark.circle"
        case .overActiveCap, .overExtendedCap: return "exclamationmark.triangle"
        case .overHardLimit: return "xmark.octagon"
        }
    }

    private func verdictColor(_ v: BudgetVerdict) -> Color {
        switch v {
        case .ok: return .green
        case .overActiveCap, .overExtendedCap: return .orange
        case .overHardLimit: return .red
        }
    }

    private func budgetLine(_ a: BudgetAssessment, activeCap: Int) -> String {
        var line = "≈\(a.estimatedInputTokens) prompt tokens (estimated) + \(a.maxNewTokens) max new = \(a.budget) / active cap \(activeCap)"
        switch a.verdict {
        case .ok: break
        case .overActiveCap:
            line += " — server will reject (extended mode off or cap exceeded)"
        case .overExtendedCap:
            line += " — above extended cap; maximum mode requires server-side opt-in"
        case .overHardLimit:
            line += " — exceeds the hard limit"
        }
        return line
    }

    private func serverAdmissionLine(_ d: APIErrorDetail) -> String {
        var parts: [String] = ["Server admission: \(d.code ?? "error")"]
        if let b = d.request_budget { parts.append("budget \(b)") }
        if let c = d.active_cap { parts.append("active cap \(c)") }
        if let l = d.model_context_limit { parts.append("limit \(l)") }
        if let g = d.estimated_peak_memory_gb {
            parts.append(String(format: "est. peak %.2f GB", g))
        }
        return parts.joined(separator: " · ")
    }
}

// MARK: composer

struct ComposerBar: View {
    @EnvironmentObject var state: AppState
    @FocusState private var composerFocused: Bool
    @State private var showingFileImporter = false
    @State private var isDropTargeted = false

    var body: some View {
        VStack(spacing: 6) {
            if !state.attachments.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 6) {
                        ForEach(state.attachments) { attachment in
                            HStack(spacing: 4) {
                                Image(systemName: attachment.isImage ? "photo" : "doc.text")
                                Text(attachment.name).lineLimit(1)
                                Button { state.removeAttachment(attachment) } label: {
                                    Image(systemName: "xmark.circle.fill")
                                }
                                .buttonStyle(.borderless)
                            }
                            .font(.caption)
                            .padding(.horizontal, 8).padding(.vertical, 5)
                            .background(Capsule().fill(Color.gray.opacity(0.14)))
                        }
                    }
                }
            }
            HStack(alignment: .bottom, spacing: 8) {
                ZStack(alignment: .topLeading) {
                    TextEditor(text: $state.composer)
                        .font(.body)
                        .frame(minHeight: 56, maxHeight: 140)
                        .focused($composerFocused)
                        .onSubmit { /* ⏎ inserts newline; Send button sends */ }
                    if state.composer.isEmpty {
                        Text("Message (multiline; ⌘⏎ to send)")
                            .foregroundColor(.secondary.opacity(0.7))
                            .padding(.top, 8).padding(.leading, 5)
                            .allowsHitTesting(false)
                    }
                }
                .background(RoundedRectangle(cornerRadius: 8).fill(Color.gray.opacity(0.06)))
                .onDrop(of: [UTType.fileURL.identifier], isTargeted: $isDropTargeted) { providers in
                    for provider in providers {
                        _ = provider.loadObject(ofClass: URL.self) { url, _ in
                            guard let url else { return }
                            DispatchQueue.main.async { state.addAttachments([url]) }
                        }
                    }
                    return !providers.isEmpty
                }
                .overlay {
                    if isDropTargeted {
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(Color.accentColor, style: StrokeStyle(lineWidth: 2, dash: [6]))
                    }
                }

                VStack(spacing: 6) {
                    Button { showingFileImporter = true } label: {
                        Label("Attach", systemImage: "paperclip")
                    }
                    .help("Attach an image, PDF, or text file")
                    if state.isGenerating {
                        Button {
                            state.stop()
                        } label: {
                            Label("Stop", systemImage: "stop.fill")
                        }
                        .keyboardShortcut(".", modifiers: .command)
                    } else {
                        Button {
                            state.send()
                        } label: {
                            Label("Send", systemImage: "paperplane.fill")
                        }
                        .keyboardShortcut(.return, modifiers: .command)
                        .disabled(state.composer.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && state.attachments.isEmpty
                                  || !state.isOnline)
                    }
                    Button {
                        state.clear()
                    } label: {
                        Label("Clear", systemImage: "trash")
                    }
                    .disabled(state.isGenerating || state.messages.isEmpty)
                }
            }
            HStack {
                Stepper(value: $state.maxTokens, in: 1...262144, step: 128) {
                    Text("max new tokens: \(state.maxTokens)")
                        .font(.caption).monospacedDigit()
                }
                Spacer()
                if !state.isOnline {
                    Text("Offline — start the server: python3 -m freetoken.server")
                        .font(.caption2).foregroundColor(.red)
                }
            }
        }
        .padding(12)
        .fileImporter(isPresented: $showingFileImporter,
                      allowedContentTypes: [.item],
                      allowsMultipleSelection: true) { result in
            if case .success(let urls) = result { state.addAttachments(urls) }
        }
    }
}
