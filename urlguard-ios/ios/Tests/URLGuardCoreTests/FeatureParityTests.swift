import XCTest
@testable import URLGuardCore

/// The Swift feature port must agree with `src/features.py` exactly.
///
/// This is the same contract, and the same fixture, the Kotlin port is held
/// to. It exists because this project has already shipped three parity bugs
/// between Python and Kotlin -- port extraction on "[::1]:8080", the
/// `_NullParse` path rule, and `urlparse` stripping ";params" -- every one of
/// which produced plausible-looking features that were quietly wrong. A third
/// implementation without the same guard would be a third chance to make that
/// mistake invisibly.
final class FeatureParityTests: XCTestCase {

    struct Fixture: Decodable {
        let feature_names: [String]
        let threshold: Double
        let cases: [Case]
        struct Case: Decodable {
            let url: String
            let features: [Double]
            let probability: Double?
        }
    }

    static var fixture: Fixture!
    static var tld: TldExtract!

    /// Repo root, found from this file's location so the test works wherever
    /// the package is checked out.
    static func repoRoot() -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // URLGuardCoreTests
            .deletingLastPathComponent()   // Tests
            .deletingLastPathComponent()   // ios
            .deletingLastPathComponent()   // repo root
    }

    override class func setUp() {
        super.setUp()
        let root = repoRoot()
        let golden = root.appendingPathComponent(
            "android/app/src/test/resources/golden_vectors.json")
        let psl = root.appendingPathComponent(
            "android/app/src/main/assets/public_suffix_list.txt")
        guard let data = try? Data(contentsOf: golden),
              let parsed = try? JSONDecoder().decode(Fixture.self, from: data),
              let extractor = try? TldExtract.fromSuffixFile(psl) else {
            return
        }
        fixture = parsed
        tld = extractor
    }

    private func requireFixture() throws -> (Fixture, TldExtract) {
        guard let f = Self.fixture, let t = Self.tld else {
            throw XCTSkip("golden_vectors.json or public_suffix_list.txt "
                + "not found; run python src/export_onnx.py")
        }
        return (f, t)
    }

    func testEveryGoldenVectorMatchesPythonExactly() throws {
        let (fixture, tld) = try requireFixture()
        XCTAssertFalse(fixture.cases.isEmpty, "fixture carries no cases")

        var mismatches: [String] = []
        for c in fixture.cases {
            let got = UrlFeatures.vector(c.url, tld: tld,
                                         featureNames: fixture.feature_names)
            XCTAssertEqual(got.count, c.features.count, "feature count")
            for (i, name) in fixture.feature_names.enumerated() {
                let expected = c.features[i]
                let actual = got[i]
                // Tolerance covers float formatting in JSON only; the
                // computation itself is expected to agree exactly.
                if abs(expected - actual) > 1e-9 {
                    mismatches.append(
                        "\(name): expected \(expected), got \(actual) "
                        + "for \(c.url)")
                }
            }
            if mismatches.count > 20 { break }
        }
        XCTAssertTrue(mismatches.isEmpty,
                      "feature drift vs Python:\n" + mismatches.prefix(20)
                        .joined(separator: "\n"))
    }

    func testFeatureNamesAndOrderMatchTheModel() throws {
        let (fixture, tld) = try requireFixture()
        let pairs = UrlFeatures.extract("https://example.com/a?b=1", tld: tld)
        XCTAssertEqual(pairs.map(\.0), fixture.feature_names,
                       "feature names or their order drifted from the model")
    }

    func testPythonLengthSemanticsNotSwiftGraphemeCount() throws {
        let (_, tld) = try requireFixture()
        // "é" as e + combining acute: Python len() == 2, Swift count == 1.
        // Getting this wrong shifts every length, ratio and entropy feature.
        let url = "http://e\u{0301}xample.com/"
        let pairs = UrlFeatures.extract(url, tld: tld)
        let dict = Dictionary(uniqueKeysWithValues: pairs)
        XCTAssertEqual(dict["url_length"], Double(url.unicodeScalars.count))
        XCTAssertNotEqual(dict["url_length"], Double(url.count),
                          "url_length is counting grapheme clusters")
    }

    func testCrc32MatchesZlib() {
        // Values from Python: zlib.crc32(b"...")
        XCTAssertEqual(CRC32.checksum(""), 0)
        XCTAssertEqual(CRC32.checksum("a"), 3904355907)
        XCTAssertEqual(CRC32.checksum("abc"), 891568578)
        XCTAssertEqual(CRC32.checksum("http://example.com"), 3607223980)
    }

    func testNothingResolvesOrConnects() throws {
        let (_, tld) = try requireFixture()
        // Extraction of a host that cannot resolve must still be instant and
        // total: the extractor is pure string work by contract.
        let started = Date()
        for host in ["http://this-host-does-not-exist.invalid/x",
                     "http://10.255.255.1/timeout",
                     "http://[::1]:8080/a"] {
            _ = UrlFeatures.extract(host, tld: tld)
        }
        XCTAssertLessThan(Date().timeIntervalSince(started), 1.0,
                          "extraction appears to be doing I/O")
    }
}

/// Parser edge cases that have actually broken a port before.
final class ParserQuirkTests: XCTestCase {

    private func tld() throws -> TldExtract {
        let psl = FeatureParityTests.repoRoot().appendingPathComponent(
            "android/app/src/main/assets/public_suffix_list.txt")
        guard let t = try? TldExtract.fromSuffixFile(psl) else {
            throw XCTSkip("public suffix list not found")
        }
        return t
    }

    func testPortIsPythonsSecondColonSegment() {
        // Python: netloc.split(":")[1]. For "[::1]:8080" that is "", not
        // "8080". Reproduced on purpose -- the model trained on it.
        let p = UrlFeatures.parseURL("http://[::1]:8080/a")
        XCTAssertEqual(p.netloc, "[::1]:8080")
        let port = p.netloc.contains(":")
            ? String(p.netloc.split(separator: ":",
                                    omittingEmptySubsequences: false)[1])
            : ""
        XCTAssertEqual(port, "")
    }

    func testParamsAreStrippedFromTheLastPathSegment() {
        // urlparse (not urlsplit) removes ";params". A feed URL carrying a
        // literal "&amp;" once made Python report path_length 29 where the
        // Kotlin port said 143.
        let p = UrlFeatures.parseURL("http://a.com/x/y;jsessionid=42")
        XCTAssertEqual(p.path, "/x/y")
        // ...but only for schemes in uses_params.
        let q = UrlFeatures.parseURL("gopher://a.com/x/y;z")
        XCTAssertEqual(q.path, "/x/y;z")
    }

    func testNullParseFallbackGivesEmptyPathNotSlash() {
        // Unbalanced bracket -> CPython raises -> features.py uses _NullParse,
        // whose path is "" rather than "/" when nothing follows the host.
        let p = UrlFeatures.parseURL("http://[/")
        XCTAssertTrue(UrlFeatures.urlparseWouldRaise(p.netloc))
        XCTAssertEqual(p.path, "")
    }

    func testUserinfoIsNotStrippedFromHost() throws {
        let t = try tld()
        // features.py takes netloc.split(":")[0], so host is "user" here,
        // while tldextract still resolves the real registered domain.
        let p = UrlFeatures.parseURL("http://user:pass@evil.com/a")
        let host = String(p.netloc.prefix(while: { $0 != ":" }))
        XCTAssertEqual(host, "user")
        XCTAssertEqual(t.extract("http://user:pass@evil.com/a").registeredDomain,
                       "evil.com")
    }

    func testIPLiteralsExtractAsDomainWithNoSuffix() throws {
        let t = try tld()
        let v4 = t.extract("http://1.2.3.4/a")
        XCTAssertEqual(v4.domain, "1.2.3.4")
        XCTAssertEqual(v4.suffix, "")
        let v6 = t.extract("http://[::1]/a")
        XCTAssertEqual(v6.domain, "[::1]")
        XCTAssertEqual(v6.suffix, "")
    }

    func testCaseIsPreservedButMatchedInsensitively() throws {
        let t = try tld()
        let r = t.extract("http://A.B.COM/")
        XCTAssertEqual(r.subdomain, "A")
        XCTAssertEqual(r.domain, "B")
        XCTAssertEqual(r.suffix, "COM")
    }

    func testPunycodeLabelsDecodeForSuffixMatching() {
        // Without decoding, an IDN suffix never matches the list and every
        // domain-derived feature shifts.
        XCTAssertEqual(Punycode.decodeLabel("xn--fiqs8s"), "中国")
        XCTAssertEqual(Punycode.decodeLabel("xn--80ak6aa92e"), "аррӏе")
        XCTAssertEqual(Punycode.decodeLabel("example"), "example")
        // Invalid punycode falls back to the lowered label, as tldextract does.
        XCTAssertEqual(Punycode.decodeLabel("XN--!!!"), "xn--!!!")
    }

    func testMalformedInputNeverCrashes() throws {
        let t = try tld()
        for u in ["", "   ", "http://", "://", "http://[", "]x[",
                  "http://a@@b", "\u{0000}\u{0001}", "http://.....",
                  "ht!tp://x", "//x", "http://%%%"] {
            _ = UrlFeatures.extract(u, tld: t)
        }
    }
}
