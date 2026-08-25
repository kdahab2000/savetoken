import XCTest
@testable import FreeTokenApp

final class AdmissionTests: XCTestCase {
    func testTokenEstimateFloor() {
        XCTAssertGreaterThanOrEqual(Admission.estimateTokens(""), 1)
        XCTAssertGreaterThanOrEqual(Admission.estimateTokens("a"), 1)
        XCTAssertGreaterThan(Admission.estimateTokens(String(repeating: "x", count: 700)), 100)
    }

    func testBoundaryVerdictsExtendedDisabled() {
        let activeCap = 65536, extendedCap = 131072
        // Build prompt lengths so estimated input = budget - maxNew - overhead.
        func prompt(forBudget budget: Int, maxNew: Int) -> String {
            let est = budget - maxNew - Admission.chatTemplateOverhead
            let chars = max(1, Int(Double(est) * 3.5))
            return String(repeating: "x", count: chars)
        }
        let atCap = Admission.assess(promptText: prompt(forBudget: activeCap, maxNew: 100),
                                     maxNewTokens: 100, activeCap: activeCap,
                                     extendedCap: extendedCap)
        XCTAssertEqual(atCap.budget, activeCap)
        XCTAssertEqual(atCap.verdict, .ok)

        let justOver = Admission.assess(promptText: prompt(forBudget: activeCap + 1, maxNew: 100),
                                        maxNewTokens: 100, activeCap: activeCap,
                                        extendedCap: extendedCap)
        XCTAssertEqual(justOver.budget, activeCap + 1)
        XCTAssertEqual(justOver.verdict, .overActiveCap)

        let overExtended = Admission.assess(promptText: prompt(forBudget: extendedCap + 1, maxNew: 100),
                                            maxNewTokens: 100, activeCap: activeCap,
                                            extendedCap: extendedCap)
        XCTAssertEqual(overExtended.verdict, .overExtendedCap)
    }

    func testHardLimitAlwaysVisible() {
        let a = Admission.assess(promptText: String(repeating: "x", count: 4_000_000),
                                 maxNewTokens: 512, activeCap: 65536, extendedCap: 131072)
        XCTAssertEqual(a.verdict, .overHardLimit)
        XCTAssertGreaterThan(a.budget, Admission.hardLimit)
        XCTAssertEqual(Admission.hardLimit, 262144)
    }

    func testExtendedEnabledRaisesActiveCap() {
        // When the server runs with --allow-extended, active_cap == 131072,
        // so budgets up to the extended cap are .ok on the client side.
        let a = Admission.assess(promptText: String(repeating: "x", count: 350_000),
                                 maxNewTokens: 512, activeCap: 131072, extendedCap: 131072)
        XCTAssertEqual(a.verdict, .ok)
    }

    func testPrefillEstimatorMatchesCalibrationPoints() {
        XCTAssertEqual(PrefillEstimator.seconds(forTokens: 8192, curve: PrefillEstimator.bf16Curve), 4.55, accuracy: 0.001)
        XCTAssertEqual(PrefillEstimator.seconds(forTokens: 262144, curve: PrefillEstimator.bf16Curve), 757.8, accuracy: 0.001)
        XCTAssertEqual(PrefillEstimator.seconds(forTokens: 262144, curve: PrefillEstimator.q4Curve), 804.47, accuracy: 0.001)
    }

    func testPrefillEstimatorMonotonicAndClamped() {
        var prev = -1.0
        for n in [512, 8192, 50000, 131072, 262144] {
            let v = PrefillEstimator.seconds(forTokens: n, curve: PrefillEstimator.bf16Curve)
            XCTAssertGreaterThan(v, prev, "not strictly increasing within calibrated range at \(n)")
            prev = v
        }
        // Beyond the calibrated ceiling the estimate clamps (never extrapolates).
        XCTAssertEqual(PrefillEstimator.seconds(forTokens: 1_000_000, curve: PrefillEstimator.bf16Curve), 757.8, accuracy: 0.001)
        XCTAssertEqual(PrefillEstimator.seconds(forTokens: 0, curve: PrefillEstimator.bf16Curve), 0)
    }

    func testCurveSelectionByModelID() {
        XCTAssertEqual(PrefillEstimator.seconds(forTokens: 8192, curve: PrefillEstimator.curve(forModelID: "qwen3.5-healthcare-4bit")), 9.62, accuracy: 0.001)
        XCTAssertEqual(PrefillEstimator.seconds(forTokens: 8192, curve: PrefillEstimator.curve(forModelID: "qwen3.5-healthcare-bf16")), 4.55, accuracy: 0.001)
    }
}
