import AVFoundation
import Foundation

struct SpeechVoiceOption: Identifiable, Equatable {
    let id: String
    let name: String
    let language: String

    var label: String {
        let languageName = Locale.current.localizedString(forIdentifier: language)
            ?? language
        return "\(name) — \(languageName)"
    }
}

enum SpeechExportError: LocalizedError {
    case alreadyExporting
    case unsupportedBuffer
    case noAudio

    var errorDescription: String? {
        switch self {
        case .alreadyExporting:
            return "Another speech export is already running."
        case .unsupportedBuffer:
            return "macOS returned an unsupported speech audio buffer."
        case .noAudio:
            return "macOS did not generate any speech audio."
        }
    }
}

private final class SpeechWaveWriter {
    let url: URL
    var file: AVAudioFile?
    var wroteFrames = false
    var finished = false

    init(url: URL) {
        self.url = url
    }

    func append(_ buffer: AVAudioPCMBuffer) throws {
        if file == nil {
            file = try AVAudioFile(
                forWriting: url,
                settings: buffer.format.settings,
                commonFormat: buffer.format.commonFormat,
                interleaved: buffer.format.isInterleaved)
        }
        try file?.write(from: buffer)
        wroteFrames = true
    }
}

/// Local speech playback and WAV export using the voices installed in macOS.
/// Text never leaves this Mac and no microphone permission is required.
final class SpeechService: NSObject, ObservableObject, AVSpeechSynthesizerDelegate {
    @Published private(set) var isSpeaking = false
    @Published private(set) var activeMessageID: UUID?
    @Published private(set) var isExporting = false

    private let synthesizer = AVSpeechSynthesizer()
    private var exportSynthesizer: AVSpeechSynthesizer?

    override init() {
        super.init()
        synthesizer.delegate = self
    }

    var voices: [SpeechVoiceOption] {
        AVSpeechSynthesisVoice.speechVoices()
            .map { SpeechVoiceOption(id: $0.identifier, name: $0.name, language: $0.language) }
            .sorted {
                if $0.language == $1.language { return $0.name < $1.name }
                return $0.language < $1.language
            }
    }

    func speak(text: String, messageID: UUID? = nil,
               voiceIdentifier: String?, rate: Double) {
        let cleaned = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleaned.isEmpty else { return }
        if synthesizer.isSpeaking || synthesizer.isPaused {
            synthesizer.stopSpeaking(at: .immediate)
        }
        let utterance = makeUtterance(
            text: cleaned, voiceIdentifier: voiceIdentifier, rate: rate)
        activeMessageID = messageID
        isSpeaking = true
        synthesizer.speak(utterance)
    }

    func stop() {
        if synthesizer.isSpeaking || synthesizer.isPaused {
            synthesizer.stopSpeaking(at: .immediate)
        }
        isSpeaking = false
        activeMessageID = nil
    }

    func exportWAV(text: String, voiceIdentifier: String?, rate: Double,
                   to url: URL) async throws {
        guard !isExporting else { throw SpeechExportError.alreadyExporting }
        let cleaned = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleaned.isEmpty else { throw SpeechExportError.noAudio }

        isExporting = true
        defer {
            exportSynthesizer = nil
            isExporting = false
        }

        let exportSynthesizer = AVSpeechSynthesizer()
        self.exportSynthesizer = exportSynthesizer
        let utterance = makeUtterance(
            text: cleaned, voiceIdentifier: voiceIdentifier, rate: rate)
        let writer = SpeechWaveWriter(url: url)

        try await withCheckedThrowingContinuation {
            (continuation: CheckedContinuation<Void, Error>) in
            exportSynthesizer.write(utterance) { buffer in
                guard !writer.finished else { return }
                guard let pcmBuffer = buffer as? AVAudioPCMBuffer else {
                    writer.finished = true
                    continuation.resume(throwing: SpeechExportError.unsupportedBuffer)
                    return
                }
                if pcmBuffer.frameLength == 0 {
                    writer.finished = true
                    if writer.wroteFrames {
                        continuation.resume()
                    } else {
                        continuation.resume(throwing: SpeechExportError.noAudio)
                    }
                    return
                }
                do {
                    try writer.append(pcmBuffer)
                } catch {
                    writer.finished = true
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    private func makeUtterance(text: String, voiceIdentifier: String?,
                               rate: Double) -> AVSpeechUtterance {
        let utterance = AVSpeechUtterance(string: text)
        if let voiceIdentifier,
           let voice = AVSpeechSynthesisVoice(identifier: voiceIdentifier) {
            utterance.voice = voice
        }
        utterance.rate = Float(min(max(rate, 0.1), 1.0))
        return utterance
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer,
                           didStart utterance: AVSpeechUtterance) {
        isSpeaking = true
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer,
                           didFinish utterance: AVSpeechUtterance) {
        isSpeaking = false
        activeMessageID = nil
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer,
                           didCancel utterance: AVSpeechUtterance) {
        isSpeaking = false
        activeMessageID = nil
    }
}
