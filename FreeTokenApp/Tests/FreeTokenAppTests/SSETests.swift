import XCTest
@testable import FreeTokenApp

final class SSETests: XCTestCase {
    func testResponseTextSanitizerHidesReasoningAndControlMarkers() {
        let raw = "<think>private reasoning</think>\nHello! 👋<|im_end|>"
        XCTAssertEqual(ResponseTextSanitizer.clean(raw), "Hello! 👋")
    }

    func testResponseTextSanitizerHidesOpenReasoningUntilItCloses() {
        XCTAssertEqual(ResponseTextSanitizer.clean("<think>still thinking"), "")
        XCTAssertEqual(ResponseTextSanitizer.clean("<think>done</think>Answer"), "Answer")
    }

    func testResponseTextSanitizerKeepsOrdinaryText() {
        XCTAssertEqual(
            ResponseTextSanitizer.clean("Hello there", finalized: true),
            "Hello there")
    }

    func testResponseTextSanitizerBuffersUnmarkedStreamUntilFinalized() {
        XCTAssertEqual(ResponseTextSanitizer.clean("planning text"), "")
        XCTAssertEqual(
            ResponseTextSanitizer.clean("planning text", finalized: true),
            "planning text")
    }

    func testLineFramerPreservesEmptySSESeparators() {
        var framer = SSELineFramer()
        let bytes = Array("data: first\n\ndata: [DONE]\n\n".utf8)
        let lines = bytes.compactMap { framer.append($0) }
        XCTAssertEqual(lines, ["data: first", "", "data: [DONE]", ""])
        XCTAssertNil(framer.finish())
    }

    func testLineFramerHandlesCRLFAndUnterminatedLine() {
        var framer = SSELineFramer()
        let bytes = Array("data: ok\r\n\r\ndata: tail".utf8)
        var lines = bytes.compactMap { framer.append($0) }
        if let final = framer.finish() { lines.append(final) }
        XCTAssertEqual(lines, ["data: ok", "", "data: tail"])
        XCTAssertNil(framer.finish())
    }

    private func chunk(content: String? = nil, role: String? = nil,
                       finish: String? = nil, warning: String? = nil) -> String {
        var delta: [String] = []
        if let role { delta.append("\"role\":\"\(role)\"") }
        if let content { delta.append("\"content\":\"\(content)\"") }
        let finishPart = finish.map { ",\"finish_reason\":\"\($0)\"" } ?? ",\"finish_reason\":null"
        let warningPart = warning.map { ",\"warning\":\"\($0)\"" } ?? ""
        return "{\"id\":\"chatcmpl-x\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\(delta.joined(separator: ","))}\(finishPart)}]\(warningPart)}"
    }

    func testSingleEventAcrossChunkedFeeds() {
        var p = SSELineParser()
        let line = "data: " + chunk(content: "Hi")
        // Feed the line in two arbitrary pieces via separate consume calls is
        // not supported by design (transport gives whole lines); instead feed
        // the full line and the terminating blank line separately.
        XCTAssertTrue(p.consume(line: line).isEmpty)
        let events = p.consume(line: "")
        XCTAssertEqual(events.count, 1)
        guard case .chunk(let c) = events[0] else { return XCTFail("expected chunk") }
        XCTAssertEqual(c.choices?.first?.delta?.content, "Hi")
    }

    func testRoleChunkFinishChunkAndDone() {
        var p = SSELineParser()
        var events: [SSEEvent] = []
        events += p.consume(line: "data: " + chunk(content: "", role: "assistant"))
        events += p.consume(line: "")
        events += p.consume(line: "data: " + chunk(content: "Paris"))
        events += p.consume(line: "")
        events += p.consume(line: "data: " + chunk(content: ".", finish: "stop"))
        events += p.consume(line: "")
        events += p.consume(line: "data: [DONE]")
        events += p.consume(line: "")

        XCTAssertEqual(events.count, 4)
        guard case .chunk(let first) = events[0],
              case .chunk(let mid) = events[1],
              case .chunk(let last) = events[2],
              case .done = events[3] else {
            return XCTFail("unexpected event sequence: \(events)")
        }
        XCTAssertEqual(first.choices?.first?.delta?.role, "assistant")
        XCTAssertEqual(mid.choices?.first?.delta?.content, "Paris")
        XCTAssertEqual(last.choices?.first?.finish_reason, "stop")
    }

    func testWarningFieldSurfaces() {
        var p = SSELineParser()
        _ = p.consume(line: "data: " + chunk(content: "", role: "assistant",
                                              warning: "Extended context mode: prefill ~60s"))
        let events = p.consume(line: "")
        guard case .chunk(let c) = events[0] else { return XCTFail() }
        XCTAssertEqual(c.warning, "Extended context mode: prefill ~60s")
    }

    func testMalformedJSONBecomesMalformedNotCrash() {
        var p = SSELineParser()
        _ = p.consume(line: "data: {not json at all")
        let events = p.consume(line: "")
        guard case .malformed(let raw) = events[0] else { return XCTFail() }
        XCTAssertEqual(raw, "{not json at all")
    }

    func testUnknownFieldsIgnoredInChunk() {
        var p = SSELineParser()
        _ = p.consume(line: "data: {\"object\":\"chat.completion.chunk\",\"future_field\":42,\"choices\":[]}")
        let events = p.consume(line: "")
        guard case .chunk(let c) = events[0] else { return XCTFail() }
        XCTAssertEqual(c.object, "chat.completion.chunk")
        XCTAssertEqual(c.choices?.count, 0)
    }

    func testFlushHandlesUnterminatedFinalEvent() {
        var p = SSELineParser()
        _ = p.consume(line: "data: [DONE]")
        let flushed = p.flush()
        XCTAssertEqual(flushed, [.done])
        XCTAssertTrue(p.flush().isEmpty)
    }

    func testBlankLineDispatchesDone() {
        var p = SSELineParser()
        _ = p.consume(line: "data: [DONE]")
        XCTAssertEqual(p.consume(line: ""), [.done])
    }

    func testNonDataFieldsIgnored() {
        var p = SSELineParser()
        XCTAssertTrue(p.consume(line: "event: message").isEmpty)
        XCTAssertTrue(p.consume(line: ": keepalive comment").isEmpty)
        XCTAssertTrue(p.consume(line: "id: 7").isEmpty)
        XCTAssertTrue(p.consume(line: "").isEmpty)  // blank with no data = no event
    }

    func testDoneMarkerMustBeExact() {
        var p = SSELineParser()
        _ = p.consume(line: "data: [DONE] ")  // trailing space -> not the marker
        let events = p.consume(line: "")
        guard case .malformed = events[0] else {
            return XCTFail("expected malformed for near-[DONE] payload")
        }
    }
}
