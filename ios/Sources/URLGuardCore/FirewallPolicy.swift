import Foundation

/// The decision the firewall makes about one DNS query.
///
/// Separated from the tunnel plumbing so it can be tested on Linux without a
/// NetworkExtension. Every rule here is one someone could get wrong in a way
/// that either breaks the phone or silently stops protecting it, so each is
/// stated once, in one place, with a test.
public enum FirewallPolicy {

    /// Blocking threshold, deliberately NOT the threshold the training run
    /// selected.
    ///
    /// The training rule targets FPR <= 0.05, which is fine for a screen the
    /// user can dismiss and wrong for a system-wide blocker: at 0.05 roughly
    /// one benign domain in twenty stops resolving and the phone feels broken.
    /// Measured on the hostname model's out-of-fold predictions:
    ///
    ///     0.65 -> FPR 0.0505, recall 0.369
    ///     0.95 -> FPR 0.0033, recall 0.169
    ///
    /// So the default trades away more than half the recall to avoid breaking
    /// roughly 1 domain in 300. The link-scanning layers, which see full URLs
    /// and are far more accurate, are where real coverage comes from. This
    /// matches the Android default exactly.
    public static let defaultThreshold = 0.95

    public enum Decision: Equatable, Sendable {
        /// Forward upstream untouched.
        case allow(reason: String)
        /// Answer NXDOMAIN.
        case block(score: Double)
        /// Not a DNS query we understand -- pass the packet through unchanged.
        case passthrough(reason: String)
    }

    public struct Context: Sendable {
        public let allowlist: Set<String>
        public let threshold: Double
        public let enabled: Bool

        public init(allowlist: Set<String> = [],
                    threshold: Double = FirewallPolicy.defaultThreshold,
                    enabled: Bool = true) {
            self.allowlist = allowlist
            self.threshold = threshold
            self.enabled = enabled
        }
    }

    /// Decide what to do with one parsed DNS payload.
    ///
    /// `score` is the hostname model's probability, supplied by the caller, or
    /// nil when it could not be computed. Nil means ALLOW: a firewall that
    /// blocks when its model fails to load takes the network down, which is a
    /// far worse failure than missing a detection. Fail open, always.
    public static func decide(dns: [UInt8], context: Context,
                              score: (String) -> Double?) -> Decision {
        guard context.enabled else {
            return .passthrough(reason: "the firewall is switched off")
        }
        guard DnsPackets.isStandardQuery(dns) else {
            return .passthrough(reason: "not a standard DNS query")
        }
        let qtype = DnsPackets.questionType(dns)
        // Only forward lookups represent somewhere the user is about to GO.
        guard qtype == DnsPackets.typeA || qtype == DnsPackets.typeAAAA else {
            return .passthrough(reason: "not an address lookup")
        }
        guard let host = DnsPackets.questionName(dns) else {
            return .passthrough(reason: "the queried name could not be read")
        }
        guard !DnsPackets.isNonNavigationalName(host) else {
            return .allow(reason: "reverse, special-use or LAN name")
        }
        if isAllowlisted(host, context.allowlist) {
            return .allow(reason: "you chose to trust this domain")
        }
        guard let p = score(host) else {
            // Fail open.
            return .allow(reason: "the on-device model was unavailable")
        }
        if p >= context.threshold {
            return .block(score: p)
        }
        return .allow(reason: "scored below the blocking threshold")
    }

    /// Allowlisting a domain covers its subdomains: someone who trusts
    /// `example.com` after a warning does not expect to be asked again for
    /// `www.example.com`.
    public static func isAllowlisted(_ host: String,
                                     _ allowlist: Set<String>) -> Bool {
        let h = host.lowercased()
        if allowlist.contains(h) { return true }
        for entry in allowlist where h.hasSuffix(".\(entry)") { return true }
        return false
    }
}
