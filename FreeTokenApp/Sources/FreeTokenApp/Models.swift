import Foundation

enum BackendProvider: String, CaseIterable, Identifiable, Equatable, Codable {
    case saveToken
    case ollama

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .saveToken: return "SaveToken MLX"
        case .ollama: return "Ollama"
        }
    }

    var port: Int {
        switch self {
        case .saveToken: return 8321
        case .ollama: return 11434
        }
    }

    var endpointLabel: String {
        "http://127.0.0.1:\(port) (loopback only)"
    }

    /// Portable startup instructions. When a workspace is configured the
    /// SaveToken command points at it; otherwise a placeholder is shown so no
    /// maintainer-specific path ever appears.
    func startCommand(workspacePath: String? = nil, port: Int? = nil) -> String {
        switch self {
        case .saveToken:
            let trimmed = (workspacePath ?? "")
                .trimmingCharacters(in: .whitespaces)
            let dir = trimmed.isEmpty ? "/path/to/SaveToken" : trimmed
            return "cd \"\(dir)\" && python3 -m freetoken.server --port \(port ?? self.port)"
        case .ollama:
            return "ollama serve"
        }
    }

    func launchCommand() -> String {
        switch self {
        case .saveToken:
            return "open SaveToken.app"
        case .ollama:
            return "open -a Ollama"
        }
    }
}

// Wire types for the freetoken loopback API. Decoding is lenient: every field
// the app does not strictly need is optional, so older/newer server payloads
// never crash the UI.

struct ModelInfo: Decodable, Identifiable, Equatable {
    let id: String
    let revision: String?
    let format: String?
    let dtype: String?
    let context_limit: Int?
    let expected_resident_mb: Int?
    let research_only: Bool?
    var active: Bool?
    let default_cap: Int?
    let extended_cap: Int?
    let allow_extended: Bool?
    let active_cap: Int?
    let weights_resident_gb: Double?

    var isActive: Bool { active ?? false }
    var isOllamaCloud: Bool { Self.isOllamaCloudID(id) }

    static func isOllamaCloudID(_ id: String) -> Bool {
        let normalized = id.lowercased()
        return normalized.hasSuffix(":cloud") || normalized.hasSuffix("-cloud")
    }

    static func isOllamaEmbeddingID(_ id: String) -> Bool {
        let normalized = id.lowercased()
        return normalized.contains("embedding") || normalized.contains("embed-")
    }
}

struct ModelsResponse: Decodable {
    let object: String?
    let data: [ModelInfo]
}

struct HealthResponse: Decodable {
    let status: String?
    let model: String?
    let active_model_id: String?
    let loaded: Bool?
    let weights_resident_gb: Double?
}

struct OllamaVersionResponse: Decodable {
    let version: String?
}

struct OllamaTagsResponse: Decodable {
    let models: [OllamaModelTag]
}

struct OllamaModelTag: Decodable {
    let name: String
    let model: String?
    let modified_at: String?
    let size: Int64?
    let details: OllamaModelDetails?
}

struct OllamaModelDetails: Decodable {
    let format: String?
    let family: String?
    let parameter_size: String?
    let quantization_level: String?
    let context_length: Int?
}

struct ChatMessagePayload: Codable, Equatable {
    let role: String
    let content: String
}

struct ChatRequestBody: Encodable {
    let messages: [ChatMessagePayload]
    let max_tokens: Int
    let stream: Bool
    let allow_maximum_context: Bool?
    let model: String?
}

struct ChunkDelta: Decodable, Equatable {
    let role: String?
    let content: String?
    let reasoning: String?
}

struct ChunkChoice: Decodable, Equatable {
    let index: Int?
    let delta: ChunkDelta?
    let finish_reason: String?
}

struct ChatChunk: Decodable, Equatable {
    let id: String?
    let object: String?
    let created: Int?
    let model: String?
    let choices: [ChunkChoice]?
    let warning: String?
}

struct SwitchResponse: Decodable {
    let switched: Bool?
    let model: String?
    let previous: String?
    let revision: String?
    let weights_resident_gb: Double?
}

struct OllamaChatRequest: Encodable {
    let model: String
    let messages: [ChatMessagePayload]
    let stream: Bool
    let think: Bool
    let options: OllamaOptions
}

struct OllamaToolChatRequest: Encodable {
    let model: String
    let messages: [OllamaMessage]
    let stream: Bool
    let think: Bool
    let tools: [OllamaToolDefinition]
    let options: OllamaOptions
}

struct OllamaMessage: Codable, Equatable {
    let role: String
    let content: String?
    let tool_calls: [OllamaToolCall]?
    let tool_name: String?

    init(role: String, content: String?, tool_calls: [OllamaToolCall]? = nil,
         tool_name: String? = nil) {
        self.role = role
        self.content = content
        self.tool_calls = tool_calls
        self.tool_name = tool_name
    }
}

struct OllamaToolDefinition: Encodable, Equatable {
    let type: String
    let function: OllamaFunctionDefinition
}

struct OllamaFunctionDefinition: Encodable, Equatable {
    let name: String
    let description: String
    let parameters: OllamaToolParameters
}

struct OllamaToolParameters: Encodable, Equatable {
    let type: String
    let properties: [String: OllamaToolProperty]
    let required: [String]
    let additionalProperties: Bool
}

struct OllamaToolProperty: Encodable, Equatable {
    let type: String
    let description: String
}

struct OllamaToolCall: Codable, Equatable {
    let id: String?
    let type: String?
    let function: OllamaFunctionCall
}

struct OllamaFunctionCall: Codable, Equatable {
    let name: String
    let arguments: [String: String]
}

struct OllamaChatResponse: Decodable {
    let model: String?
    let message: OllamaResponseMessage
    let done: Bool?
}

struct OllamaResponseMessage: Decodable {
    let role: String?
    let content: String?
    let reasoning: String?
    let tool_calls: [OllamaToolCall]?
}

struct OllamaOptions: Encodable {
    let num_predict: Int
}

struct OllamaStreamLine: Decodable {
    let model: String?
    let message: OllamaStreamMessage?
    let done: Bool?
    let done_reason: String?
}

struct OllamaStreamMessage: Decodable {
    let role: String?
    let content: String?
    let reasoning: String?
}

// Admission/verification errors carry the backend's admission fields.
struct APIErrorDetail: Decodable {
    let message: String
    let type: String?
    let code: String?
    let request_budget: Int?
    let active_cap: Int?
    let model_context_limit: Int?
    let estimated_peak_memory_gb: Double?
}

struct APIErrorPayload: Decodable {
    let error: APIErrorDetail
}

struct APIError: Error, LocalizedError {
    let status: Int
    let detail: APIErrorDetail

    var errorDescription: String? { detail.message }

    static func from(status: Int, body: Data) -> APIError {
        if let payload = try? JSONDecoder().decode(APIErrorPayload.self, from: body) {
            return APIError(status: status, detail: payload.error)
        }
        let text = String(data: body, encoding: .utf8) ?? "unreadable error body"
        return APIError(status: status, detail: APIErrorDetail(
            message: "HTTP \(status): \(text.prefix(300))",
            type: nil, code: nil, request_budget: nil, active_cap: nil,
            model_context_limit: nil, estimated_peak_memory_gb: nil))
    }
}
