import Foundation
import UniformTypeIdentifiers
import PDFKit

struct ChatAttachment: Identifiable, Equatable {
    let id = UUID()
    let name: String
    let content: String?
    let imageBase64: String?

    var isImage: Bool { imageBase64 != nil }
}

enum AttachmentService {
    static let maxTextCharacters = 200_000

    static func load(url: URL) throws -> ChatAttachment {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        let type = (try? url.resourceValues(forKeys: [.contentTypeKey]).contentType)
            ?? UTType(filenameExtension: url.pathExtension)

        if type?.conforms(to: .image) == true {
            let data = try Data(contentsOf: url)
            return ChatAttachment(name: url.lastPathComponent, content: nil,
                                  imageBase64: data.base64EncodedString())
        }

        if type?.conforms(to: .pdf) == true {
            guard let document = PDFDocument(url: url) else { throw Error.unreadable }
            let text = (0..<document.pageCount).compactMap { document.page(at: $0)?.string }
                .joined(separator: "\n\n")
            return ChatAttachment(name: url.lastPathComponent,
                                  content: clipped(text), imageBase64: nil)
        }

        if type?.conforms(to: .text) == true || type?.conforms(to: .sourceCode) == true {
            let text = try String(contentsOf: url, encoding: .utf8)
            return ChatAttachment(name: url.lastPathComponent,
                                  content: clipped(text), imageBase64: nil)
        }

        throw Error.unsupported
    }

    private static func clipped(_ text: String) -> String {
        guard text.count > maxTextCharacters else { return text }
        return String(text.prefix(maxTextCharacters))
            + "\n\n[Attachment truncated after \(maxTextCharacters) characters.]"
    }

    enum Error: LocalizedError {
        case unreadable, unsupported

        var errorDescription: String? {
            switch self {
            case .unreadable: return "The file could not be read."
            case .unsupported: return "This file type is not supported. Use an image, PDF, or text file."
            }
        }
    }
}
