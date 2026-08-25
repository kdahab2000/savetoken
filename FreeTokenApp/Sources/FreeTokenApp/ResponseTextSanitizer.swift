import Foundation

/// Converts raw model text into user-facing chat text.
///
/// Qwen checkpoints can emit a reasoning block and tokenizer control markers
/// as ordinary decoded text. The app keeps the server/API payload untouched,
/// but hides those implementation details from the visible conversation.
enum ResponseTextSanitizer {
    private static let hiddenMarkers = [
        "<|im_end|>",
        "<|endoftext|>",
        "<|end_of_text|>",
        "<|eot_id|>"
    ]

    static func clean(_ raw: String, finalized: Bool = false) -> String {
        var visible = raw

        if let end = visible.range(of: "</think>", options: .caseInsensitive) {
            visible = String(visible[end.upperBound...])
        } else if visible.range(of: "<think>", options: .caseInsensitive) != nil {
            // Keep internal reasoning out of the UI while it is still being
            // generated. Once </think> arrives, the answer becomes visible.
            return ""
        } else if !finalized {
            // This checkpoint can begin a reasoning block without emitting
            // the opening marker in the streamed text. Buffer all text until
            // the stream finishes or </think> confirms the visible answer.
            return ""
        }

        for marker in hiddenMarkers {
            visible = visible.replacingOccurrences(of: marker, with: "")
        }
        return visible.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
