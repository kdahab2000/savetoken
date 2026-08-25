import Foundation

/// Talks to the freetoken server on loopback only. No other host is ever
/// constructed; no telemetry, no caching of request bodies.
final class ServerClient {
    static let host = "127.0.0.1"

    let provider: BackendProvider
    let baseURL: URL
    private let session: URLSession

    convenience init(port: Int = BackendProvider.saveToken.port) {
        let provider: BackendProvider = port == BackendProvider.ollama.port
            ? .ollama : .saveToken
        self.init(provider: provider, port: port)
    }

    convenience init(provider: BackendProvider) {
        self.init(provider: provider, port: provider.port)
    }

    init(provider: BackendProvider, port: Int) {
        self.provider = provider
        self.baseURL = URL(string: "http://\(Self.host):\(port)")!
        let config = URLSessionConfiguration.ephemeral  // nothing cached on disk
        config.timeoutIntervalForRequest = 30
        config.requestCachePolicy = .reloadIgnoringLocalCacheData
        self.session = URLSession(configuration: config)
    }

    /// Cheap first-run detection: does an Ollama daemon answer on loopback?
    /// Loopback only, 2 s budget, no data sent.
    static func probeOllama(port: Int = BackendProvider.ollama.port) async -> Bool {
        guard let url = URL(string: "http://\(Self.host):\(port)/api/version") else {
            return false
        }
        var req = URLRequest(url: url)
        req.timeoutInterval = 2
        guard let (_, response) = try? await URLSession.shared.data(for: req) else {
            return false
        }
        return (response as? HTTPURLResponse)?.statusCode == 200
    }

    var endpointLabel: String {
        "http://\(Self.host):\(baseURL.port ?? provider.port) (loopback only)"
    }

    // MARK: simple JSON endpoints

    func health() async throws -> HealthResponse {
        switch provider {
        case .saveToken:
            return try await getJSON(path: "health")
        case .ollama:
            let _: OllamaVersionResponse = try await getJSON(path: "api/version")
            return HealthResponse(status: "ok", model: nil,
                                  active_model_id: nil, loaded: true,
                                  weights_resident_gb: nil)
        }
    }

    func models() async throws -> ModelsResponse {
        switch provider {
        case .saveToken:
            return try await getJSON(path: "v1/models")
        case .ollama:
            let tags: OllamaTagsResponse = try await getJSON(path: "api/tags")
            let data = tags.models
                .filter { !ModelInfo.isOllamaEmbeddingID($0.name) }
                .map { tag in
                let details = tag.details
                let context = details?.context_length
                let residentMB = tag.size.map { Int($0 / 1_000_000) }
                return ModelInfo(
                    id: tag.name,
                    revision: tag.modified_at,
                    format: details?.format,
                    dtype: details?.quantization_level ?? details?.parameter_size,
                    context_limit: context,
                    expected_resident_mb: residentMB,
                    research_only: false,
                    active: nil,
                    default_cap: context,
                    extended_cap: context,
                    allow_extended: true,
                    active_cap: context,
                    weights_resident_gb: nil)
                }
            return ModelsResponse(object: "list", data: data)
        }
    }

    func switchModel(id: String) async throws -> SwitchResponse {
        if provider == .ollama {
            return SwitchResponse(switched: true, model: id, previous: nil,
                                  revision: nil, weights_resident_gb: nil)
        }
        struct Body: Encodable { let model: String }
        return try await postJSON(path: "v1/models/switch", body: Body(model: id))
    }

    // MARK: streaming chat

    /// Returns an async sequence of raw SSE lines (without terminators).
    /// Cancelling the consuming task closes the connection, which the server
    /// treats as a generation cancel.
    func streamChatLines(request: ChatRequestBody) -> AsyncThrowingStream<String, Error> {
        switch provider {
        case .saveToken:
            return streamSaveTokenChat(request: request)
        case .ollama:
            return streamOllamaChat(request: request)
        }
    }

    /// Ollama tool calls are intentionally non-streaming here. The app may
    /// need to pause for explicit user approval before executing a requested
    /// local SSH command, then send the tool result back to the model.
    func completeOllamaWithTools(
        model: String,
        messages: [OllamaMessage],
        tools: [OllamaToolDefinition],
        maxTokens: Int
    ) async throws -> OllamaChatResponse {
        guard provider == .ollama else {
            throw APIError.from(
                status: 400,
                body: Data("SSH tools require the Ollama provider".utf8))
        }
        let body = OllamaToolChatRequest(
            model: model,
            messages: messages,
            stream: false,
            think: false,
            tools: tools,
            options: OllamaOptions(num_predict: maxTokens))
        return try await postJSON(path: "api/chat", body: body)
    }

    private func streamSaveTokenChat(request: ChatRequestBody) -> AsyncThrowingStream<String, Error> {
        let url = baseURL.appendingPathComponent("v1/chat/completions")
        let session = self.session
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    var req = URLRequest(url: url)
                    req.httpMethod = "POST"
                    req.setValue("application/json", forHTTPHeaderField: "Content-Type")
                    req.httpBody = try JSONEncoder().encode(request)
                    // Long-context prefill can take many minutes (measured:
                    // ~13 min at 262k); keep the socket open.
                    req.timeoutInterval = 3600
                    let (bytes, response) = try await session.bytes(for: req)
                    if let http = response as? HTTPURLResponse,
                       http.statusCode != 200 {
                        var body = Data()
                        for try await byte in bytes { body.append(byte) }
                        throw APIError.from(status: http.statusCode, body: body)
                    }
                    // Frame bytes ourselves instead of using bytes.lines:
                    // AsyncBytes.lines can omit empty lines, but SSE uses an
                    // empty line to terminate each event.
                    var framer = SSELineFramer()
                    for try await byte in bytes {
                        if let line = framer.append(byte) {
                            continuation.yield(line)
                        }
                    }
                    if let line = framer.finish() {
                        continuation.yield(line)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    // MARK: helpers

    private func getJSON<T: Decodable>(path: String) async throws -> T {
        var req = URLRequest(url: baseURL.appendingPathComponent(path))
        req.httpMethod = "GET"
        return try await run(req)
    }

    private func postJSON<T: Decodable, B: Encodable>(path: String, body: B) async throws -> T {
        var req = URLRequest(url: baseURL.appendingPathComponent(path))
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try JSONEncoder().encode(body)
        return try await run(req)
    }

    private func run<T: Decodable>(_ req: URLRequest) async throws -> T {
        let (data, response) = try await session.data(for: req)
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard (200..<300).contains(status) else {
            throw APIError.from(status: status, body: data)
        }
        do {
            return try JSONDecoder().decode(T.self, from: data)
        } catch {
            throw APIError.from(status: status, body: data)
        }
    }

    /// Ollama's native endpoint streams newline-delimited JSON and supports
    /// think=false. Convert each native message into the same SSE shape used
    /// by the app parser, so the UI has one response path for both providers.
    private func streamOllamaChat(request: ChatRequestBody) -> AsyncThrowingStream<String, Error> {
        let url = baseURL.appendingPathComponent("api/chat")
        let session = self.session
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    guard let model = request.model, !model.isEmpty else {
                        throw APIError.from(
                            status: 400,
                            body: Data("Ollama model is not selected".utf8))
                    }
                    let body = OllamaChatRequest(
                        model: model,
                        messages: request.messages,
                        stream: true,
                        think: false,
                        options: OllamaOptions(num_predict: request.max_tokens))
                    var req = URLRequest(url: url)
                    req.httpMethod = "POST"
                    req.setValue("application/json", forHTTPHeaderField: "Content-Type")
                    req.httpBody = try JSONEncoder().encode(body)
                    req.timeoutInterval = 3600
                    let (bytes, response) = try await session.bytes(for: req)
                    if let http = response as? HTTPURLResponse,
                       http.statusCode != 200 {
                        var errorBody = Data()
                        for try await byte in bytes { errorBody.append(byte) }
                        throw APIError.from(status: http.statusCode, body: errorBody)
                    }

                    let completionID = "ollama-\(UUID().uuidString)"
                    var framer = SSELineFramer()
                    for try await byte in bytes {
                        if let line = framer.append(byte) {
                            for output in try ollamaSSELines(
                                from: line, completionID: completionID,
                                fallbackModel: model) {
                                continuation.yield(output)
                            }
                        }
                    }
                    if let line = framer.finish() {
                        for output in try ollamaSSELines(
                            from: line, completionID: completionID,
                            fallbackModel: model) {
                            continuation.yield(output)
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func ollamaSSELines(
        from line: String,
        completionID: String,
        fallbackModel: String
    ) throws -> [String] {
        guard !line.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return []
        }
        guard let data = line.data(using: .utf8) else { return [] }
        let item = try JSONDecoder().decode(OllamaStreamLine.self, from: data)
        let model = item.model ?? fallbackModel
        if item.done == true {
            let chunk = OutgoingChunk(
                id: completionID,
                object: "chat.completion.chunk",
                created: Int(Date().timeIntervalSince1970),
                model: model,
                choices: [OutgoingChoice(
                    index: 0,
                    delta: OutgoingDelta(role: nil, content: nil),
                    finish_reason: item.done_reason ?? "stop")])
            return [sseLine(chunk), "", "data: [DONE]", ""]
        }
        guard let message = item.message,
              let content = message.content,
              !content.isEmpty else {
            // Ollama may send hidden reasoning in message.reasoning. It is
            // intentionally ignored; only the final content is forwarded.
            return []
        }
        let chunk = OutgoingChunk(
            id: completionID,
            object: "chat.completion.chunk",
            created: Int(Date().timeIntervalSince1970),
            model: model,
            choices: [OutgoingChoice(
                index: 0,
                delta: OutgoingDelta(role: message.role, content: content),
                finish_reason: nil)])
        return [sseLine(chunk), ""]
    }

    private func sseLine(_ chunk: OutgoingChunk) -> String {
        let data = (try? JSONEncoder().encode(chunk)) ?? Data()
        return "data: \(String(decoding: data, as: UTF8.self))"
    }

    private struct OutgoingChunk: Encodable {
        let id: String
        let object: String
        let created: Int
        let model: String
        let choices: [OutgoingChoice]
    }

    private struct OutgoingChoice: Encodable {
        let index: Int
        let delta: OutgoingDelta
        let finish_reason: String?
    }

    private struct OutgoingDelta: Encodable {
        let role: String?
        let content: String?
    }
}
