import XCTest
@testable import URLGuardCore

/// The iOS UI must make the same conservative claims as the web and Android
/// UIs. These mirror the Python `TestAssessmentLayer` and the Kotlin
/// `RiskAssessmentTest` so the three surfaces cannot drift into saying
/// different things about the same score.
final class RiskAssessmentTests: XCTestCase {

    func testScoreNeverShowsCertainty() {
        // A tree ensemble is never certain, so the UI must never print 0 or
        // 100 for a probability strictly inside (0, 1).
        XCTAssertEqual(RiskAssessment.displayScore(0.0001), 1)
        XCTAssertEqual(RiskAssessment.displayScore(0.9999), 99)
        XCTAssertEqual(RiskAssessment.displayScore(0.5), 50)
        // Exact 0 and 1 are passed through: they mean the model genuinely
        // returned an endpoint, which is a different statement.
        XCTAssertEqual(RiskAssessment.displayScore(0.0), 0)
        XCTAssertEqual(RiskAssessment.displayScore(1.0), 100)
    }

    func testBandBoundariesMatchTheOtherSurfaces() {
        XCTAssertEqual(RiskAssessment.band(for: 0), .low)
        XCTAssertEqual(RiskAssessment.band(for: 30), .low)
        XCTAssertEqual(RiskAssessment.band(for: 31), .suspicious)
        XCTAssertEqual(RiskAssessment.band(for: 70), .suspicious)
        XCTAssertEqual(RiskAssessment.band(for: 71), .high)
        XCTAssertEqual(RiskAssessment.band(for: 100), .high)
    }

    func testConfidenceIsDistanceFromThresholdNotProbability() {
        // At the threshold the model is maximally unsure, whatever the raw
        // probability happens to be.
        XCTAssertEqual(RiskAssessment.decisionMargin(0.49, threshold: 0.49),
                       0.0, accuracy: 1e-9)
        XCTAssertEqual(RiskAssessment.decisionMargin(1.0, threshold: 0.49),
                       1.0, accuracy: 1e-9)
        XCTAssertEqual(RiskAssessment.decisionMargin(0.0, threshold: 0.49),
                       1.0, accuracy: 1e-9)
        // A high probability under a high threshold is NOT high confidence.
        let m = RiskAssessment.decisionMargin(0.80, threshold: 0.95)
        XCTAssertLessThan(m, 0.35)
        XCTAssertEqual(RiskAssessment.marginLabel(m), "Borderline")
    }

    func testSignalsOnlyReportMeasurementsThatArePresent() {
        XCTAssertEqual(RiskAssessment.signals(for: [:]), [])
        let s = RiskAssessment.signals(for: [
            "has_ip": 1, "is_https": 0, "suspicious_word_count": 3,
        ])
        XCTAssertTrue(s.contains { $0.contains("raw IP address") })
        XCTAssertFalse(s.contains { $0.contains("HTTPS") })
        XCTAssertTrue(s.contains { $0.contains("3 matches") })
    }

    func testNoSignalClaimsCertainty() {
        let all: [String: Double] = [
            "is_https": 1, "has_ip": 1, "has_at_symbol": 1, "is_punycode": 1,
            "has_confusable_chars": 1, "has_suspicious_port": 1,
            "is_shortened": 1, "brand_outside_domain": 1,
            "has_double_slash_redirect": 1, "suspicious_word_count": 2,
        ]
        let text = RiskAssessment.signals(for: all).joined(separator: " ").lowercased()
        for banned in ["confirmed", "definitely", "guaranteed", "is malware",
                       "certainly"] {
            XCTAssertFalse(text.contains(banned), "signal claims certainty: \(banned)")
        }
        XCTAssertTrue(RiskAssessment.disclaimer.lowercased()
            .contains("not a guarantee"))
    }

    func testInsufficientBandDoesNotReadAsSafe() {
        let d = RiskAssessment.Band.insufficient.detail.lowercased()
        XCTAssertTrue(d.contains("unresolved"))
        XCTAssertFalse(d.contains(" safe "))
    }
}
