import Foundation

/// Frames a byte stream into SSE lines while preserving empty separator lines.
/// URLSession.AsyncBytes.lines may omit those empty lines, so the transport
/// uses this small framer before handing lines to SSELineParser.
struct SSELineFramer {
    private var buffer = Data()

    mutating func append(_ byte: UInt8) -> String? {
        if byte == 0x0A { // LF
            var line = buffer
            if line.last == 0x0D { line.removeLast() } // CRLF
            buffer.removeAll(keepingCapacity: true)
            return String(decoding: line, as: UTF8.self)
        }
        buffer.append(byte)
        return nil
    }

    mutating func finish() -> String? {
        guard !buffer.isEmpty else { return nil }
        let line = String(decoding: buffer, as: UTF8.self)
        buffer.removeAll(keepingCapacity: true)
        return line
    }
}

// Incremental Server-Sent-Events parsing, line-based. The transport delivers
// complete lines without terminators; an empty line terminates one SSE event.
// This type is pure and fully unit-tested without any network.

enum SSEEvent: Equatable {
    case chunk(ChatChunk)
    case done
    case malformed(String)
}

struct SSELineParser {
    private var dataBuffer: [String] = []

    /// Feed one complete line; returns any events the line completes.
    mutating func consume(line: String) -> [SSEEvent] {
        if line.hasPrefix("data:") {
            var payload = String(line.dropFirst("data:".count))
            if payload.hasPrefix(" ") { payload.removeFirst() }
            dataBuffer.append(payload)
            return []
        }
        if line.isEmpty {
            guard !dataBuffer.isEmpty else { return [] }
            let payload = dataBuffer.joined(separator: "\n")
            dataBuffer.removeAll()
            return [Self.decode(payload: payload)]
        }
        // "event:", "id:", comments (":"), anything else: ignored per SSE spec.
        return []
    }

    /// Flush a trailing event that was never terminated by an empty line
    /// (e.g. the connection dropped right after the last data line).
    mutating func flush() -> [SSEEvent] {
        guard !dataBuffer.isEmpty else { return [] }
        let payload = dataBuffer.joined(separator: "\n")
        dataBuffer.removeAll()
        return [Self.decode(payload: payload)]
    }

    private static func decode(payload: String) -> SSEEvent {
        if payload == "[DONE]" {
            return .done
        }
        guard let data = payload.data(using: .utf8),
              let chunk = try? JSONDecoder().decode(ChatChunk.self, from: data)
        else {
            return .malformed(payload)
        }
        return .chunk(chunk)
    }
}
