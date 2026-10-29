import Foundation

/// Conservative presentation of a model score, mirroring `webapp/assessment.py`
/// and the Kotlin `RiskAssessment` so the iPhone, the phone and the web UI say
/// the same thing about the same URL.
///
/// The rules that matter:
///   - a score is a RISK BAND, never a verdict;
///   - the UI never displays 0 or 100, because a tree ensemble is never certain;
///   - confidence describes how far the score is from the decision threshold,
///     not a calibrated likelihood;
///   - a signal is only shown when the measurement is actually present.
public enum RiskAssessment {

    public enum Band: String, Sendable {
        case low, suspicious, high, insufficient

        public var label: String {
            switch self {
            case .low: return "LOW RISK"
            case .suspicious: return "SUSPICIOUS"
            case .high: return "HIGH RISK"
            case .insufficient: return "INSUFFICIENT EVIDENCE"
            }
        }

        public var detail: String {
            switch self {
            case .low:
                return "No significant suspicious characteristics were detected."
            case .suspicious:
                return "Some characteristics associated with potentially "
                    + "harmful URLs were detected."
            case .high:
                return "Multiple characteristics associated with potentially "
                    + "harmful URLs were detected. Exercise caution before "
                    + "proceeding."
            case .insufficient:
                return "The available evidence does not support a conclusion "
                    + "either way. Treat this as unresolved rather than safe."
            }
        }
    }

    public static let disclaimer =
        "This assessment is based on machine-learning analysis and available "
        + "security signals. It is not a guarantee that the website is malicious."

    public struct Result: Sendable {
        public let score: Int
        public let band: Band
        public let confidence: Double
        public let confidenceLabel: String
        public let signals: [String]
    }

    /// 0-100, never the endpoints: the model cannot be certain.
    public static func displayScore(_ proba: Double) -> Int {
        let s = Int((proba * 100).rounded())
        if proba > 0.0 && proba < 1.0 { return min(99, max(1, s)) }
        return s
    }

    public static func band(for score: Int) -> Band {
        if score <= 30 { return .low }
        if score <= 70 { return .suspicious }
        return .high
    }

    /// Distance from the decision threshold, 0...1.
    public static func decisionMargin(_ proba: Double,
                                      threshold: Double) -> Double {
        if threshold <= 0.0 || threshold >= 1.0 {
            return abs(proba - 0.5) * 2
        }
        return proba >= threshold
            ? (proba - threshold) / (1.0 - threshold)
            : (threshold - proba) / threshold
    }

    public static func marginLabel(_ margin: Double) -> String {
        if margin >= 0.70 { return "Decisive" }
        if margin >= 0.35 { return "Moderate" }
        return "Borderline"
    }

    /// Plain-language signals for the features actually observed.
    ///
    /// Only properties whose presence is itself the observation are listed --
    /// an IP literal, an '@', punycode. Continuous "unusually long / unusually
    /// many" signals are deliberately NOT included: deciding what counts as
    /// unusual requires the benign-corpus percentiles that live server-side,
    /// and inventing a cutoff here would manufacture evidence.
    public static func signals(for feats: [String: Double]) -> [String] {
        var out: [String] = []
        func on(_ k: String) -> Bool { (feats[k] ?? 0.0) != 0.0 }

        if on("is_https") { out.append("✓ HTTPS enabled") }
        if on("has_ip") {
            out.append("⚠ Host is a raw IP address rather than a domain name")
        }
        if on("has_at_symbol") {
            out.append("⚠ URL contains '@', which can hide the real destination")
        }
        if on("is_punycode") {
            out.append("⚠ Host uses punycode / internationalised characters")
        }
        if on("has_confusable_chars") {
            out.append("⚠ URL mixes scripts that can imitate Latin letters")
        }
        if on("has_suspicious_port") {
            out.append("⚠ URL specifies an uncommon network port")
        }
        if on("is_shortened") {
            out.append("⚠ Host is a known URL shortener, hiding the destination")
        }
        if on("brand_outside_domain") {
            out.append("⚠ A well-known brand name appears outside the "
                + "registered domain")
        }
        if on("has_double_slash_redirect") {
            out.append("⚠ Path contains '//', used in redirect tricks")
        }

        let kw = Int(feats["suspicious_word_count"] ?? 0)
        if kw > 0 {
            out.append("⚠ Suspicious keyword pattern detected (\(kw) matches)")
        }
        return out
    }

    public static func make(proba: Double, threshold: Double,
                            feats: [String: Double]) -> Result {
        let score = displayScore(proba)
        let margin = decisionMargin(proba, threshold: threshold)
        return Result(
            score: score,
            band: band(for: score),
            confidence: min(99.9, max(0.1, margin * 100)),
            confidenceLabel: marginLabel(margin),
            signals: signals(for: feats))
    }
}
