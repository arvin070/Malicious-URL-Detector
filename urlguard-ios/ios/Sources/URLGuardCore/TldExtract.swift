import Foundation

/// Swift port of the exact `tldextract` behaviour `src/features.py` depends on.
///
/// Same reasoning as the Kotlin port in
/// `android/.../TldExtract.kt`: the model's features are defined in terms of
/// tldextract's split into subdomain / domain / suffix, so any library that
/// splits even slightly differently feeds the model numbers it was not trained
/// on -- with no error, just worse predictions.
///
/// Quirks reproduced ON PURPOSE because the model was trained on them:
///   - labels match case-insensitively but are RETURNED with original case,
///     so "A.B.COM" -> subdomain "A", domain "B", suffix "COM";
///   - an IPv4 literal becomes domain "1.2.3.4" with an empty suffix;
///   - a bracketed IPv6 literal becomes domain "[::1]" with an empty suffix.
public struct ExtractResult: Equatable, Sendable {
    public let subdomain: String
    public let domain: String
    public let suffix: String

    /// eTLD+1, the grouping key used throughout the Python pipeline.
    public var registeredDomain: String {
        [domain, suffix].filter { !$0.isEmpty }.joined(separator: ".")
    }
}

private final class TrieNode {
    var matches: [String: TrieNode] = [:]
    var end = false
}

public final class TldExtract {

    private let root: TrieNode

    private init(root: TrieNode) { self.root = root }

    // urllib.parse.scheme_chars
    private static let schemeChars = Set(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-.")

    /// Build from the suffix list dumped out of the live Python extractor
    /// (`android/app/src/main/assets/public_suffix_list.txt`). The same file
    /// is shipped to both platforms so they cannot drift apart.
    public static func fromSuffixLines<S: Sequence>(_ lines: S) -> TldExtract
    where S.Element == String {
        let root = TrieNode()
        for raw in lines {
            let line = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            if line.isEmpty || line.hasPrefix("#") { continue }
            // tldextract's Trie.add_suffix: split on ".", reverse, insert.
            var node = root
            for label in line.split(separator: ".", omittingEmptySubsequences: false)
                .reversed() {
                let key = String(label)
                if let next = node.matches[key] {
                    node = next
                } else {
                    let next = TrieNode()
                    node.matches[key] = next
                    node = next
                }
            }
            node.end = true
        }
        return TldExtract(root: root)
    }

    public static func fromSuffixFile(_ url: URL) throws -> TldExtract {
        let text = try String(contentsOf: url, encoding: .utf8)
        return fromSuffixLines(text.split(separator: "\n",
                                          omittingEmptySubsequences: false)
            .map(String.init))
    }

    // MARK: - netloc extraction

    /// tldextract.remote._schemeless_url
    static func schemelessURL(_ url: String) -> String {
        let chars = Array(url)
        guard let i = indexOf(chars, "//") else { return url }
        if i == 0 { return String(chars[2...]) }
        if i < 2 || chars[i - 1] != ":" { return url }
        for c in chars[0..<(i - 1)] where !schemeChars.contains(c) { return url }
        return String(chars[(i + 2)...])
    }

    private static func indexOf(_ chars: [Character], _ needle: String) -> Int? {
        let n = Array(needle)
        guard chars.count >= n.count else { return nil }
        for i in 0...(chars.count - n.count) {
            if Array(chars[i..<(i + n.count)]) == n { return i }
        }
        return nil
    }

    /// tldextract.remote.lenient_netloc
    static func lenientNetloc(_ url: String) -> String {
        var s = schemelessURL(url)
        s = String(s.prefix(while: { $0 != "/" }))
        s = String(s.prefix(while: { $0 != "?" }))
        s = String(s.prefix(while: { $0 != "#" }))
        // substringAfterLast('@')
        if let at = s.lastIndex(of: "@") {
            s = String(s[s.index(after: at)...])
        }

        if let first = s.first, first == "[" ,
           let close = s.firstIndex(of: "]") {
            return String(s[...close])
        }
        var hostname = String(s.prefix(while: { $0 != ":" }))
        hostname = hostname.trimmingCharacters(in: .whitespacesAndNewlines)
        // rstrip(".。．｡")
        while let last = hostname.last,
              last == "." || last == "。" || last == "．" || last == "｡" {
            hostname.removeLast()
        }
        return hostname
    }

    public static func looksLikeIP(_ s: String) -> Bool {
        guard let first = s.unicodeScalars.first,
              first.value >= 0x30, first.value <= 0x39 else { return false }
        let parts = s.split(separator: ".", omittingEmptySubsequences: false)
        guard parts.count == 4 else { return false }
        for p in parts {
            guard !p.isEmpty, p.count <= 3,
                  p.allSatisfy({ $0.isASCII && $0.isNumber }),
                  let v = Int(p), v >= 0, v <= 255 else { return false }
            // The Python regex has no leading-zero allowance beyond one digit:
            // "01" does not match [0-9]|[1-9][0-9]|...
            if p.count > 1 && p.first == "0" { return false }
        }
        return true
    }

    /// Mirrors `ipaddress.IPv6Address` acceptance closely enough for the
    /// netloc test: hex groups, optional "::" compression, optional trailing
    /// IPv4. Zone ids are rejected, as IPv6Address rejects them.
    public static func looksLikeIPv6(_ s: String) -> Bool {
        if s.isEmpty || s.contains("%") { return false }
        let parts = s.components(separatedBy: "::")
        if parts.count > 2 { return false }

        func groupsOK(_ g: String, allowV4: Bool) -> (Bool, Int) {
            if g.isEmpty { return (true, 0) }
            let items = g.split(separator: ":", omittingEmptySubsequences: false)
            var count = 0
            for (idx, it) in items.enumerated() {
                if allowV4, idx == items.count - 1, it.contains(".") {
                    if !looksLikeIP(String(it)) { return (false, 0) }
                    count += 2
                    continue
                }
                if it.isEmpty || it.count > 4 { return (false, 0) }
                for c in it {
                    let ok = c.isHexDigit && c.isASCII
                    if !ok { return (false, 0) }
                }
                count += 1
            }
            return (true, count)
        }

        if parts.count == 2 {
            let (okA, a) = groupsOK(parts[0], allowV4: false)
            let (okB, b) = groupsOK(parts[1], allowV4: true)
            return okA && okB && (a + b) <= 7
        }
        let (ok, n) = groupsOK(parts[0], allowV4: true)
        return ok && n == 8
    }

    // MARK: - suffix lookup

    /// tldextract.TLDExtract.suffix_index -- index of the first suffix label.
    /// Returns -1 when no suffix matches (tldextract returns None).
    private func suffixIndex(_ labels: [String]) -> Int {
        var node = root
        var suffixIdx = labels.count
        var labelIdx = labels.count

        for label in labels.reversed() {
            let decoded = Punycode.decodeLabel(label)
            if let next = node.matches[decoded] {
                labelIdx -= 1
                node = next
                if node.end { suffixIdx = labelIdx }
                continue
            }
            if node.matches["*"] != nil {
                let isException = node.matches["!\(decoded)"] != nil
                return isException ? labelIdx : labelIdx - 1
            }
            break
        }
        return suffixIdx == labels.count ? -1 : suffixIdx
    }

    /// tldextract.TLDExtract._extract_netloc
    private func extractNetloc(_ netloc: String) -> ExtractResult {
        var dots = netloc
        dots = dots.replacingOccurrences(of: "。", with: ".")
        dots = dots.replacingOccurrences(of: "．", with: ".")
        dots = dots.replacingOccurrences(of: "｡", with: ".")

        if dots.count >= 4, dots.first == "[", dots.last == "]",
           TldExtract.looksLikeIPv6(String(dots.dropFirst().dropLast())) {
            return ExtractResult(subdomain: "", domain: dots, suffix: "")
        }

        let labels = dots.split(separator: ".", omittingEmptySubsequences: false)
            .map(String.init)
        let idx = suffixIndex(labels)

        if idx < 0 {
            if labels.count == 4, TldExtract.looksLikeIP(dots) {
                return ExtractResult(subdomain: "", domain: dots, suffix: "")
            }
            return ExtractResult(
                subdomain: labels.dropLast().joined(separator: "."),
                domain: labels.last ?? "",
                suffix: "")
        }

        let subdomain = idx >= 2
            ? labels[0..<(idx - 1)].joined(separator: ".") : ""
        let domain = idx > 0 ? labels[idx - 1] : ""
        let suffix = labels[idx...].joined(separator: ".")
        return ExtractResult(subdomain: subdomain, domain: domain, suffix: suffix)
    }

    public func extract(_ url: String) -> ExtractResult {
        extractNetloc(TldExtract.lenientNetloc(url))
    }
}
