import XCTest
@testable import URLGuardCore

/// Firewall decisions. The failure modes here are asymmetric: a missed block
/// is one unprotected lookup, a wrong block is a phone that cannot reach a
/// site the user needs. The tests are weighted accordingly.
final class FirewallPolicyTests: XCTestCase {

    private func query(_ name: String, qtype: Int = 1) -> [UInt8] {
        var dns: [UInt8] = [0x12, 0x34, 0x01, 0x00, 0, 1, 0, 0, 0, 0, 0, 0]
        for label in name.split(separator: ".") {
            dns.append(UInt8(label.utf8.count))
            dns += Array(label.utf8)
        }
        dns.append(0)
        dns += [UInt8(qtype >> 8), UInt8(qtype & 0xFF), 0, 1]
        return dns
    }

    func testBlocksOnlyAboveTheThreshold() {
        let ctx = FirewallPolicy.Context(threshold: 0.95)
        let d1 = FirewallPolicy.decide(dns: query("bad.cfd"), context: ctx) { _ in 0.96 }
        XCTAssertEqual(d1, .block(score: 0.96))
        let d2 = FirewallPolicy.decide(dns: query("bad.cfd"), context: ctx) { _ in 0.94 }
        if case .allow = d2 {} else { XCTFail("blocked below the threshold") }
    }

    func testDefaultThresholdIsTheConservativeOneNotTheTrainedOne() {
        // 0.65 was selected by the training rule for FPR <= 0.05. At 0.05,
        // one benign domain in twenty stops resolving and the phone feels
        // broken. The system-wide blocker uses 0.95.
        XCTAssertEqual(FirewallPolicy.defaultThreshold, 0.95)
    }

    func testFailsOpenWhenTheModelIsUnavailable() {
        // The single most important behaviour: no model must never mean no
        // network.
        let d = FirewallPolicy.decide(dns: query("anything.com"),
                                      context: .init()) { _ in nil }
        guard case .allow(let reason) = d else {
            return XCTFail("a missing model blocked the query")
        }
        XCTAssertTrue(reason.contains("unavailable"))
    }

    func testDisabledFirewallTouchesNothing() {
        let d = FirewallPolicy.decide(dns: query("bad.cfd"),
                                      context: .init(enabled: false)) { _ in 1.0 }
        guard case .passthrough = d else {
            return XCTFail("a disabled firewall acted on a query")
        }
    }

    func testOnlyForwardLookupsAreConsidered() {
        let ctx = FirewallPolicy.Context()
        for qtype in [12, 15, 16, 33, 2] {        // PTR, MX, TXT, SRV, NS
            let d = FirewallPolicy.decide(dns: query("x.com", qtype: qtype),
                                          context: ctx) { _ in 1.0 }
            guard case .passthrough = d else {
                return XCTFail("qtype \(qtype) was classified")
            }
        }
        for qtype in [DnsPackets.typeA, DnsPackets.typeAAAA] {
            let d = FirewallPolicy.decide(dns: query("x.com", qtype: qtype),
                                          context: ctx) { _ in 1.0 }
            XCTAssertEqual(d, .block(score: 1.0))
        }
    }

    func testReverseAndLanNamesAreNeverBlockedEvenAtScoreOne() {
        let ctx = FirewallPolicy.Context()
        for name in ["224.166.102.103.in-addr.arpa", "printer.local",
                     "router", "host.internal"] {
            let d = FirewallPolicy.decide(dns: query(name), context: ctx) { _ in 1.0 }
            guard case .allow = d else {
                return XCTFail("\(name) was blocked; it is not a destination")
            }
        }
    }

    func testAllowlistCoversSubdomains() {
        let ctx = FirewallPolicy.Context(allowlist: ["example.com"])
        for host in ["example.com", "www.example.com", "a.b.example.com"] {
            let d = FirewallPolicy.decide(dns: query(host), context: ctx) { _ in 1.0 }
            guard case .allow(let r) = d else {
                return XCTFail("\(host) ignored the allowlist")
            }
            XCTAssertTrue(r.contains("trust"))
        }
        // ...but must not match a domain that merely ends with the same text.
        let d = FirewallPolicy.decide(dns: query("notexample.com"),
                                      context: ctx) { _ in 1.0 }
        XCTAssertEqual(d, .block(score: 1.0))
    }

    func testMalformedQueriesPassThroughUntouched() {
        let ctx = FirewallPolicy.Context()
        for dns in [[UInt8](), [0x12, 0x34], Array(repeating: UInt8(0), count: 12)] {
            let d = FirewallPolicy.decide(dns: dns, context: ctx) { _ in 1.0 }
            if case .block = d { XCTFail("a malformed query was blocked") }
        }
    }

    func testTheHostnameIsTheOnlyThingHandedToTheModel() {
        // The scorer must never be called with anything but a bare hostname:
        // the DNS layer has no URL, and inventing one would feed the model
        // input unlike its training data.
        var seen: [String] = []
        _ = FirewallPolicy.decide(dns: query("Login-Paypal.CFD"),
                                  context: .init()) { h in seen.append(h); return 0.1 }
        XCTAssertEqual(seen, ["login-paypal.cfd"])   // lowercased, no scheme
    }
}
