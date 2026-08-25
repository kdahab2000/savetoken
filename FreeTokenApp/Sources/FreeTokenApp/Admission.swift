import Foundation

// Client-side mirror of the server's admission contract, used only for the
// budget indicator before sending. The server's decision is authoritative;
// every value produced here must be displayed as an ESTIMATE in the UI.

enum BudgetVerdict: Equatable {
    case ok                 // within the active cap
    case overActiveCap      // server rejects unless extended mode is enabled
    case overExtendedCap    // requires per-request maximum-mode opt-in
    case overHardLimit      // > 262,144: always rejected
}

struct BudgetAssessment: Equatable {
    let estimatedInputTokens: Int
    let maxNewTokens: Int
    let budget: Int
    let verdict: BudgetVerdict
}

enum Admission {
    /// Architectural ceiling; identical to capacity.MODEL_CONTEXT_LIMIT.
    static let hardLimit = 262144

    /// Rough chars-per-token heuristic. The server counts real tokens after
    /// the chat template; this only drives the pre-send indicator.
    static func estimateTokens(_ text: String) -> Int {
        max(1, Int((Double(text.count) / 3.5).rounded(.up)))
    }

    /// Chat-template overhead measured in M0 (15 tokens for a one-line user
    /// message). Real overhead varies with history length — an estimate.
    static let chatTemplateOverhead = 15

    static func assess(
        promptText: String,
        maxNewTokens: Int,
        activeCap: Int,
        extendedCap: Int
    ) -> BudgetAssessment {
        let est = estimateTokens(promptText) + chatTemplateOverhead
        let budget = est + maxNewTokens
        let verdict: BudgetVerdict
        if budget > hardLimit {
            verdict = .overHardLimit
        } else if budget > extendedCap {
            verdict = .overExtendedCap
        } else if budget > activeCap {
            verdict = .overActiveCap
        } else {
            verdict = .ok
        }
        return BudgetAssessment(
            estimatedInputTokens: est, maxNewTokens: maxNewTokens,
            budget: budget, verdict: verdict)
    }
}

/// Prefill-time estimator interpolating the measured M1 curves. Output is an
/// estimate and must be labeled as one.
enum PrefillEstimator {
    static let bf16Curve: [(tokens: Int, seconds: Double)] = [
        (8192, 4.55), (32768, 23.53), (65536, 59.32),
        (131072, 190.47), (262144, 757.8),
    ]
    static let q4Curve: [(tokens: Int, seconds: Double)] = [
        (8192, 9.62), (32768, 50.83), (65536, 118.3),
        (131072, 330.55), (262144, 804.47),
    ]

    static func curve(forModelID id: String) -> [(tokens: Int, seconds: Double)] {
        id.contains("4bit") ? q4Curve : bf16Curve
    }

    static func seconds(forTokens n: Int,
                        curve: [(tokens: Int, seconds: Double)]) -> Double {
        guard n > 0, let first = curve.first, let last = curve.last else { return 0 }
        if n <= first.tokens { return Double(n) * first.seconds / Double(first.tokens) }
        if n >= last.tokens { return last.seconds }
        for i in 0..<(curve.count - 1) {
            let (x0, y0) = curve[i]
            let (x1, y1) = curve[i + 1]
            if n <= x1 {
                return y0 + (y1 - y0) * Double(n - x0) / Double(x1 - x0)
            }
        }
        return last.seconds
    }
}
