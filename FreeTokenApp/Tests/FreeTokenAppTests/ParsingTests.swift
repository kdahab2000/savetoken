import XCTest
@testable import FreeTokenApp

/// Fixture payloads mirror the exact JSON shapes produced by the freetoken
/// server (captured from /v1/models, /health, admission rejections, and the
/// switch endpoint in M3).
final class ParsingTests: XCTestCase {

    func testResponsePolicyPreservesUserLanguage() {
        XCTAssertTrue(ResponseLanguagePolicy.base.contains("same language"))
        XCTAssertTrue(ResponseLanguagePolicy.base.contains("If the user writes Arabic, answer in Arabic"))
        XCTAssertTrue(ResponseLanguagePolicy.base.contains("Do not translate unless"))
        XCTAssertTrue(ResponseLanguagePolicy.forUserText("اكتب الإجابة بالعربية").contains("complete answer in Arabic"))
    }
    let modelsJSON = """
    {"object":"list","data":[
      {"id":"qwen3.5-healthcare-bf16","object":"model","created":0,"owned_by":"local",
       "revision":"m0-conv-20260824","format":"mlx","dtype":"bfloat16",
       "context_limit":262144,"expected_resident_mb":3764,"research_only":true,
       "active":true,"default_cap":65536,"extended_cap":131072,
       "allow_extended":false,"active_cap":65536,"weights_resident_gb":3.76},
      {"id":"qwen3.5-healthcare-4bit","object":"model","created":0,"owned_by":"local",
       "revision":"m1-quant-20260824","format":"mlx","dtype":"4bit",
       "context_limit":262144,"expected_resident_mb":1177,"research_only":true,
       "active":false}]}
    """

    func testModelsCatalogDecodes() throws {
        let resp = try JSONDecoder().decode(
            ModelsResponse.self, from: Data(modelsJSON.utf8))
        XCTAssertEqual(resp.data.count, 2)
        let bf16 = resp.data[0]
        XCTAssertEqual(bf16.id, "qwen3.5-healthcare-bf16")
        XCTAssertEqual(bf16.revision, "m0-conv-20260824")
        XCTAssertEqual(bf16.format, "mlx")
        XCTAssertEqual(bf16.dtype, "bfloat16")
        XCTAssertEqual(bf16.context_limit, 262144)
        XCTAssertEqual(bf16.default_cap, 65536)
        XCTAssertEqual(bf16.active_cap, 65536)
        XCTAssertEqual(bf16.weights_resident_gb, 3.76)
        XCTAssertTrue(bf16.isActive)
        let q4 = resp.data[1]
        XCTAssertFalse(q4.isActive)
        XCTAssertNil(q4.weights_resident_gb)      // only measured for the active model
        XCTAssertEqual(q4.expected_resident_mb, 1177)
    }

    func testOllamaTagsDecode() throws {
        let json = """
        {"models":[{"name":"qwen3.8:27b-mlx","model":"qwen3.8:27b-mlx",
          "modified_at":"2026-08-15T05:00:00Z","size":18174721847,
          "details":{"format":"safetensors","family":"qwen35",
          "parameter_size":"27B","quantization_level":"nvfp4",
          "context_length":262144}}]}
        """
        let response = try JSONDecoder().decode(
            OllamaTagsResponse.self, from: Data(json.utf8))
        XCTAssertEqual(response.models.count, 1)
        XCTAssertEqual(response.models[0].name, "qwen3.8:27b-mlx")
        XCTAssertEqual(response.models[0].details?.format, "safetensors")
        XCTAssertEqual(response.models[0].details?.context_length, 262144)
    }

    func testOllamaModelClassification() {
        XCTAssertTrue(ModelInfo.isOllamaCloudID("qwen3.5:397b-cloud"))
        XCTAssertFalse(ModelInfo.isOllamaCloudID("qwen3.8:27b-mlx"))
        XCTAssertTrue(ModelInfo.isOllamaEmbeddingID("qwen3-embedding:4b"))
        XCTAssertTrue(ModelInfo.isOllamaEmbeddingID("nomic-embed-text:latest"))
        XCTAssertFalse(ModelInfo.isOllamaEmbeddingID("qwen3-coder:latest"))
    }

    func testHealthDecodes() throws {
        let json = """
        {"status":"ok","model":"qwen3.5-healthcare-bf16",
         "active_model_id":"qwen3.5-healthcare-bf16","loaded":true,
         "weights_resident_gb":3.76}
        """
        let h = try JSONDecoder().decode(HealthResponse.self, from: Data(json.utf8))
        XCTAssertEqual(h.status, "ok")
        XCTAssertEqual(h.active_model_id, "qwen3.5-healthcare-bf16")
        XCTAssertEqual(h.weights_resident_gb, 3.76)
        XCTAssertEqual(h.loaded, true)
    }

    func testAdmissionErrorDecodesWithBackendFields() throws {
        let json = """
        {"error":{"message":"Request budget 70011 exceeds the default cap of 65536. Extended mode (up to 131072) is disabled; start the server with --allow-extended to enable it.",
         "type":"invalid_request_error","code":"default_cap_exceeded",
         "request_budget":70011,"active_cap":65536,
         "model_context_limit":262144,"estimated_peak_memory_gb":6.79}}
        """
        let err = APIError.from(status: 400, body: Data(json.utf8))
        XCTAssertEqual(err.status, 400)
        XCTAssertEqual(err.detail.code, "default_cap_exceeded")
        XCTAssertEqual(err.detail.request_budget, 70011)
        XCTAssertEqual(err.detail.active_cap, 65536)
        XCTAssertEqual(err.detail.model_context_limit, 262144)
        XCTAssertEqual(err.detail.estimated_peak_memory_gb, 6.79)
    }

    func testHardLimitErrorDecodes() throws {
        let json = """
        {"error":{"message":"Request budget (input 11 + max_new_tokens 300000 = 300011) exceeds the model context limit of 262144. This limit is architectural and cannot be raised.",
         "type":"invalid_request_error","code":"context_limit_exceeded",
         "request_budget":300011,"active_cap":65536,
         "model_context_limit":262144,"estimated_peak_memory_gb":9.62}}
        """
        let err = APIError.from(status: 400, body: Data(json.utf8))
        XCTAssertEqual(err.detail.code, "context_limit_exceeded")
        XCTAssertEqual(err.detail.request_budget, 300011)
    }

    func testUnparseableErrorBodyStillProducesMessage() {
        let err = APIError.from(status: 500, body: Data("boom".utf8))
        XCTAssertEqual(err.status, 500)
        XCTAssertTrue(err.detail.message.contains("boom"))
    }

    func testSwitchResponseDecodes() throws {
        let json = """
        {"switched":true,"model":"qwen3.5-healthcare-4bit",
         "previous":"qwen3.5-healthcare-bf16","revision":"m1-quant-20260824",
         "weights_resident_gb":1.18}
        """
        let r = try JSONDecoder().decode(SwitchResponse.self, from: Data(json.utf8))
        XCTAssertEqual(r.switched, true)
        XCTAssertEqual(r.model, "qwen3.5-healthcare-4bit")
        XCTAssertEqual(r.weights_resident_gb, 1.18)
    }
}
