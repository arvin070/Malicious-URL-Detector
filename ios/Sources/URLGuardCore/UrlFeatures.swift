import Foundation

/// Swift port of `src/features.py` -- all 124 features, in the same order.
///
/// CONTRACT
/// --------
/// For every URL, `extract(url)` must return exactly what Python's
/// `extract_features(url)` returns. `FeatureParityTests` asserts this against
/// the same 625 golden vectors the Kotlin port is held to (600 sampled from
/// the real corpus + 25 hand-picked parser breakers). If you change this file,
/// re-run `python src/export_onnx.py` and keep that test green: a feature that
/// quietly disagrees with training produces a model that is quietly wrong.
///
/// SAFETY: pure string manipulation. Nothing is resolved, fetched or connected
/// to -- the same guarantee the Python extractor makes.
///
/// SWIFT-SPECIFIC PARITY HAZARD
/// ----------------------------
/// `String.count` in Swift counts GRAPHEME CLUSTERS. Python's `len()` counts
/// CODE POINTS. For "é" written as e+U+0301 those differ (1 vs 2), and every
/// length, ratio and entropy feature would shift. So every length in this file
/// goes through `pyLen`, which counts unicode scalars. This is the single
/// easiest way to get a silently-wrong port in Swift and does not arise in
/// Kotlin, whose strings are UTF-16 code units.
///
/// PYTHON QUIRKS REPRODUCED DELIBERATELY
/// -------------------------------------
/// `features.py` computes `host = netloc.split(":")[0]`, which does NOT strip
/// userinfo. So for "http://a@b.com" the host is "a@b.com", and for
/// "http://user:pass@evil.com" the host is literally "user". Those are the
/// values the model was trained on, so `parseURL` reproduces them rather than
/// "fixing" them. tldextract, which IS userinfo-aware, still resolves the real
/// registered domain -- the two disagree by design.
public enum UrlFeatures {

    // MARK: - word lists (mirror src/features.py)

    static let suspiciousWords = [
        "login", "verify", "update", "secure", "account", "banking",
        "confirm", "signin", "webscr", "ebayisapi", "paypal", "free",
        "bonus", "gift", "urgent", "suspend", "password", "wallet",
    ]

    static let shortenerDomains: Set<String> = [
        "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd",
        "buff.ly", "adf.ly", "shorte.st", "rebrand.ly",
    ]

    static let targetBrands = [
        "paypal", "apple", "microsoft", "google", "amazon", "facebook", "netflix",
        "chase", "wellsfargo", "bankofamerica", "ebay", "dropbox", "linkedin",
        "instagram", "whatsapp", "outlook", "office365", "icloud", "steam", "irs",
    ]

    static let commonPorts: Set<String> = ["", "80", "443", "8080", "8443"]

    /// Mirrors ABUSE_PRONE_TLDS in src/features.py. Static domain knowledge,
    /// never fitted from labels -- a per-TLD phishing rate would leak them.
    static let abuseProneTLDs: Set<String> = [
        "tk", "ml", "ga", "cf", "gq",
        "xyz", "top", "work", "click", "link", "buzz", "icu", "cyou", "rest",
        "quest", "monster", "bar", "best", "cam", "surf", "fit", "beauty",
        "sbs", "cfd", "lol", "makeup", "hair", "skin", "mom", "store",
        "zip", "mov",
    ]

    /// Mirrors _EXECUTABLE_EXT in src/features.py.
    static let executableExts: Set<String> = [
        "exe", "apk", "scr", "bat", "cmd", "com", "pif", "msi", "jar", "vbs",
        "js", "ps1", "hta", "dll", "zip", "rar", "7z", "gz", "iso", "img",
        "bin", "sh", "elf",
    ]

    static let vowels: Set<Character> = ["a", "e", "i", "o", "u"]
    static let consonants = Set("bcdfghjklmnpqrstvwxyz")

    /// Schemes for which Python's `urlparse` splits ";params" off the last
    /// path segment (urllib.parse.uses_params). `urlsplit` does NOT, and
    /// features.py calls `urlparse`, so the port has to.
    static let paramSchemes: Set<String> = [
        "", "ftp", "hdl", "prospero", "http", "imap", "https", "shttp",
        "rtsp", "rtspu", "sip", "sips", "mms", "sftp", "tel",
    ]

    public static let charNgramN = 3
    public static let charNgramBuckets = 24
    public static let tokenNgramBuckets = 12

    // MARK: - primitives

    /// Python's `len()`: number of code points, NOT grapheme clusters.
    @inline(__always)
    static func pyLen(_ s: String) -> Int { s.unicodeScalars.count }

    @inline(__always)
    static func ratio(_ n: Int, _ d: Int) -> Double {
        d != 0 ? Double(n) / Double(d) : 0.0
    }

    /// Python's `str.isdigit()` as the Kotlin port reads it: decimal digits.
    @inline(__always)
    static func isDigit(_ s: Unicode.Scalar) -> Bool {
        s.properties.numericType == .decimal
    }

    @inline(__always)
    static func isLetter(_ s: Unicode.Scalar) -> Bool {
        s.properties.isAlphabetic
    }

    static func countScalars(_ s: String,
                             where pred: (Unicode.Scalar) -> Bool) -> Int {
        var n = 0
        for sc in s.unicodeScalars where pred(sc) { n += 1 }
        return n
    }

    static func count(_ s: String, of ch: Unicode.Scalar) -> Int {
        countScalars(s) { $0 == ch }
    }

    /// Python's `shannon_entropy`: base-2 entropy over character frequencies.
    public static func shannonEntropy(_ s: String) -> Double {
        if s.isEmpty { return 0.0 }
        var counts: [Unicode.Scalar: Int] = [:]
        for sc in s.unicodeScalars { counts[sc, default: 0] += 1 }
        let n = Double(pyLen(s))
        var acc = 0.0
        for c in counts.values {
            let p = Double(c) / n
            acc -= p * (log(p) / log(2.0))
        }
        return acc
    }

    /// Python's `has_ip_address`: an anchored IPv4 dotted-quad test.
    public static func hasIPAddress(_ host: String) -> Int {
        let parts = host.split(separator: ".", omittingEmptySubsequences: false)
        guard parts.count == 4 else { return 0 }
        for p in parts {
            guard !p.isEmpty, p.count <= 3,
                  p.unicodeScalars.allSatisfy({ $0.value >= 0x30 && $0.value <= 0x39 })
            else { return 0 }
        }
        return 1
    }

    public static func countSuspiciousWords(_ url: String) -> Int {
        let l = url.lowercased()
        return suspiciousWords.filter { l.contains($0) }.count
    }

    static func maxRun(_ s: String, _ set: Set<Character>) -> Int {
        var best = 0, cur = 0
        for ch in s {
            cur = set.contains(ch) ? cur + 1 : 0
            if cur > best { best = cur }
        }
        return best
    }

    /// Python's `re.split(r"[^a-zA-Z0-9]+", s)` with empties dropped.
    static func tokens(_ s: String) -> [String] {
        var out: [String] = []
        var cur = String.UnicodeScalarView()
        for sc in s.unicodeScalars {
            let isAlnum = (sc.value >= 0x30 && sc.value <= 0x39)
                || (sc.value >= 0x41 && sc.value <= 0x5A)
                || (sc.value >= 0x61 && sc.value <= 0x7A)
            if isAlnum {
                cur.append(sc)
            } else if !cur.isEmpty {
                out.append(String(cur)); cur = String.UnicodeScalarView()
            }
        }
        if !cur.isEmpty { out.append(String(cur)) }
        return out
    }

    /// Python's `_is_confusable`. Returns 1 for any non-ASCII character that
    /// either has no Unicode name (Python's `unicodedata.name` raises) or is a
    /// Cyrillic/Greek/Armenian LETTER -- the scripts that supply Latin
    /// look-alikes for homograph attacks.
    ///
    /// Swift exposes no script property, so the three scripts are matched by
    /// their code blocks. That is exact for these ranges and avoids a
    /// dependency on ICU being present, which it is not on all platforms.
    static func isConfusable(_ s: String) -> Int {
        for sc in s.unicodeScalars {
            let v = sc.value
            if v < 128 { continue }
            // unicodedata.name() raises for these -> Python returns 1.
            if v >= 0x80 && v <= 0x9F { return 1 }
            switch sc.properties.generalCategory {
            case .unassigned, .surrogate, .privateUse, .control:
                return 1
            default:
                break
            }
            guard isLetter(sc) else { continue }
            let cyrillic = (0x0400...0x04FF).contains(Int(v))
                || (0x0500...0x052F).contains(Int(v))
                || (0x2DE0...0x2DFF).contains(Int(v))
                || (0xA640...0xA69F).contains(Int(v))
            let greek = (0x0370...0x03FF).contains(Int(v))
                || (0x1F00...0x1FFF).contains(Int(v))
            let armenian = (0x0530...0x058F).contains(Int(v))
                || (0xFB13...0xFB17).contains(Int(v))
            if cyrillic || greek || armenian { return 1 }
        }
        return 0
    }

    /// Python's `re.sub(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", "", u)`.
    public static func stripScheme(_ u: String) -> String {
        let chars = Array(u)
        guard let first = chars.first, first.isLetter else { return u }
        var i = 1
        while i < chars.count {
            let c = chars[i]
            if c.isLetter || c.isNumber || c == "+" || c == "." || c == "-" {
                i += 1; continue
            }
            break
        }
        if i + 3 <= chars.count, chars[i] == ":", chars[i + 1] == "/",
           chars[i + 2] == "/" {
            return String(chars[(i + 3)...])
        }
        return u
    }

    // MARK: - hashed n-grams

    static func hashedCharNgrams(_ s: String) -> [Double] {
        var counts = [Int](repeating: 0, count: charNgramBuckets)
        // Python slices str by CODE POINT, so iterate scalars.
        let cps = Array(s.unicodeScalars)
        let total = max(cps.count - charNgramN + 1, 0)
        if total == 0 { return [Double](repeating: 0, count: charNgramBuckets) }
        for i in 0..<total {
            var view = String.UnicodeScalarView()
            for j in i..<(i + charNgramN) { view.append(cps[j]) }
            counts[CRC32.bucket(String(view), mod: charNgramBuckets)] += 1
        }
        return counts.map { Double($0) / Double(total) }
    }

    static func hashedTokenBigrams(_ toks: [String]) -> [Double] {
        var counts = [Int](repeating: 0, count: tokenNgramBuckets)
        let total = max(toks.count - 1, 0)
        if total == 0 { return [Double](repeating: 0, count: tokenNgramBuckets) }
        for i in 0..<total {
            counts[CRC32.bucket(toks[i] + "|" + toks[i + 1],
                                mod: tokenNgramBuckets)] += 1
        }
        return counts.map { Double($0) / Double(total) }
    }

    // MARK: - URL parsing

    struct Parsed {
        let scheme: String
        let netloc: String
        let path: String
        let query: String
        let fragment: String
    }

    /// True when CPython's `urlsplit` would raise ValueError on this netloc,
    /// so `features.py` falls back to its `_NullParse`.
    static func urlparseWouldRaise(_ netloc: String) -> Bool {
        let hasOpen = netloc.contains("[")
        let hasClose = netloc.contains("]")
        if hasOpen != hasClose { return true }
        if !hasOpen { return false }
        guard let o = netloc.firstIndex(of: "["),
              let c = netloc.firstIndex(of: "]") , o < c else { return true }
        let bracketed = String(netloc[netloc.index(after: o)..<c])
        if bracketed.hasPrefix("v") { return false }   // IPvFuture, accepted
        return !TldExtract.looksLikeIPv6(bracketed)
    }

    /// Equivalent of Python's `urlparse` for the subset `features.py` uses,
    /// including its `_NullParse` fallback. Python splits fragment first, then
    /// query, then netloc, then path.
    static func parseURL(_ raw: String) -> Parsed {
        var rest = raw
        var scheme = ""
        if let r = rest.range(of: "://") {
            let cand = String(rest[rest.startIndex..<r.lowerBound])
            let ok = !cand.isEmpty && (cand.first?.isLetter ?? false)
                && cand.allSatisfy { $0.isLetter || $0.isNumber
                    || $0 == "+" || $0 == "-" || $0 == "." }
            if ok {
                scheme = cand.lowercased()
                rest = String(rest[r.upperBound...])
            }
        }
        var frag = "", afterFrag = rest
        if let h = rest.firstIndex(of: "#") {
            afterFrag = String(rest[rest.startIndex..<h])
            frag = String(rest[rest.index(after: h)...])
        }
        var query = "", afterQuery = afterFrag
        if let q = afterFrag.firstIndex(of: "?") {
            afterQuery = String(afterFrag[afterFrag.startIndex..<q])
            query = String(afterFrag[afterFrag.index(after: q)...])
        }
        var netloc = afterQuery, path = ""
        if let s = afterQuery.firstIndex(of: "/") {
            netloc = String(afterQuery[afterQuery.startIndex..<s])
            path = String(afterQuery[s...])
        }

        if urlparseWouldRaise(netloc) {
            // _NullParse: `"/" + path if path else ""` -- an empty remainder
            // yields an empty path, not "/". It also does NO ";params"
            // handling, so the split below must be skipped on this branch.
            if path == "/" { path = "" }
            return Parsed(scheme: scheme, netloc: netloc, path: path,
                          query: query, fragment: frag)
        }

        // urlparse() -- unlike urlsplit() -- removes ";params" from the LAST
        // path segment. CPython's _splitparams searches from the final '/'.
        if paramSchemes.contains(scheme), path.contains(";") {
            let chars = Array(path)
            var from = 0
            if let lastSlash = chars.lastIndex(of: "/") { from = lastSlash }
            if let semi = chars[from...].firstIndex(of: ";") {
                path = String(chars[0..<semi])
            }
        }
        return Parsed(scheme: scheme, netloc: netloc, path: path,
                      query: query, fragment: frag)
    }

    // MARK: - scanners replacing simple regexes
    //
    // These replace character-class regexes with direct scalar scans. Besides
    // being faster, it sidesteps NSRegularExpression's UTF-16 ranges, which
    // count a non-BMP character as two -- a quiet way to disagree with Python
    // on exactly the hostile input this model is meant to see.

    static func countPercentEncoded(_ s: String) -> Int {
        let a = Array(s.unicodeScalars)
        var n = 0, i = 0
        func isHex(_ u: Unicode.Scalar) -> Bool {
            (u.value >= 0x30 && u.value <= 0x39)
                || (u.value >= 0x41 && u.value <= 0x46)
                || (u.value >= 0x61 && u.value <= 0x66)
        }
        while i + 2 < a.count {
            if a[i] == "%" && isHex(a[i + 1]) && isHex(a[i + 2]) {
                n += 1; i += 3
            } else { i += 1 }
        }
        return n
    }

    /// `re.findall(r"(//|\.\.|--|__|&&|\?\?)", s)` -- non-overlapping.
    static func countRepeatedSeparators(_ s: String) -> Int {
        let a = Array(s.unicodeScalars)
        let pairs: [(Unicode.Scalar, Unicode.Scalar)] = [
            ("/", "/"), (".", "."), ("-", "-"), ("_", "_"), ("&", "&"), ("?", "?"),
        ]
        var n = 0, i = 0
        while i + 1 < a.count {
            if pairs.contains(where: { $0.0 == a[i] && $0.1 == a[i + 1] }) {
                n += 1; i += 2
            } else { i += 1 }
        }
        return n
    }

    /// `re.search(r"\.[a-zA-Z0-9]{2,5}$", path)`
    static func pathHasFileExt(_ path: String) -> Bool {
        let a = Array(path.unicodeScalars)
        guard a.count >= 3 else { return false }
        var k = a.count - 1
        var run = 0
        func isAlnum(_ u: Unicode.Scalar) -> Bool {
            (u.value >= 0x30 && u.value <= 0x39)
                || (u.value >= 0x41 && u.value <= 0x5A)
                || (u.value >= 0x61 && u.value <= 0x7A)
        }
        while k >= 0 && isAlnum(a[k]) && run < 5 { run += 1; k -= 1 }
        return run >= 2 && run <= 5 && k >= 0 && a[k] == "."
    }

    /// `re.search(r"\b[0-9a-f]{16,}\b", url_lower)`
    static func hasHexLikeSegment(_ s: String) -> Bool {
        let a = Array(s.unicodeScalars)
        func isWord(_ u: Unicode.Scalar) -> Bool {
            (u.value >= 0x30 && u.value <= 0x39)
                || (u.value >= 0x41 && u.value <= 0x5A)
                || (u.value >= 0x61 && u.value <= 0x7A)
                || u == "_"
        }
        func isHexLower(_ u: Unicode.Scalar) -> Bool {
            (u.value >= 0x30 && u.value <= 0x39)
                || (u.value >= 0x61 && u.value <= 0x66)
        }
        var i = 0
        while i < a.count {
            guard isHexLower(a[i]) else { i += 1; continue }
            if i > 0 && isWord(a[i - 1]) {
                while i < a.count && isWord(a[i]) { i += 1 }
                continue
            }
            var j = i
            while j < a.count && isHexLower(a[j]) { j += 1 }
            let runLen = j - i
            let boundary = (j >= a.count) || !isWord(a[j])
            if runLen >= 16 && boundary { return true }
            while j < a.count && isWord(a[j]) { j += 1 }
            i = j
        }
        return false
    }

    /// `re.search(r"[A-Za-z0-9+/]{24,}={0,2}", url)`
    static func hasBase64LikeSegment(_ s: String) -> Bool {
        var run = 0
        for u in s.unicodeScalars {
            let ok = (u.value >= 0x30 && u.value <= 0x39)
                || (u.value >= 0x41 && u.value <= 0x5A)
                || (u.value >= 0x61 && u.value <= 0x7A)
                || u == "+" || u == "/"
            if ok { run += 1; if run >= 24 { return true } } else { run = 0 }
        }
        return false
    }

    /// Mirrors _EXECUTABLE_EXT: `\.(ext|...)$` on the path, case-insensitive.
    static func pathHasExecutableExt(_ path: String) -> Bool {
        guard let dot = path.lastIndex(of: ".") else { return false }
        let ext = String(path[path.index(after: dot)...]).lowercased()
        return executableExts.contains(ext)
    }

    static func isSpecialChar(_ u: Unicode.Scalar) -> Bool {
        // re: [^a-zA-Z0-9./:-]
        let v = u.value
        if (v >= 0x30 && v <= 0x39) || (v >= 0x41 && v <= 0x5A)
            || (v >= 0x61 && v <= 0x7A) { return false }
        return !(u == "." || u == "/" || u == ":" || u == "-")
    }

    static func isQuerySpecialChar(_ u: Unicode.Scalar) -> Bool {
        // re: [^a-zA-Z0-9=&_.\-]
        let v = u.value
        if (v >= 0x30 && v <= 0x39) || (v >= 0x41 && v <= 0x5A)
            || (v >= 0x61 && v <= 0x7A) { return false }
        return !(u == "=" || u == "&" || u == "_" || u == "." || u == "-")
    }

    // MARK: - main

    /// Returns the 124 features keyed by name, in the same insertion order as
    /// `features.py`. Callers index by name (see `UrlClassifier`) so order can
    /// never silently diverge from the trained model.
    public static func extract(_ rawURL: String,
                               tld: TldExtract) -> [(String, Double)] {
        let url = rawURL.trimmingCharacters(in: .whitespacesAndNewlines)
        let parsed = parseURL(url.contains("://") ? url : "http://\(url)")
        let netloc = parsed.netloc
        let host = String(netloc.prefix(while: { $0 != ":" }))
        // Python: `netloc.split(":")[1] if ":" in netloc else ""`.
        // For "[::1]:8080" that is the empty segment between the first two
        // colons, NOT "8080" -- reproduce it rather than "fixing" it.
        let port: String = netloc.contains(":")
            ? String(netloc.split(separator: ":",
                                  omittingEmptySubsequences: false)[1])
            : ""
        let ext = tld.extract(url)
        let registeredDomain = ext.registeredDomain

        let path = parsed.path
        let query = parsed.query
        let fragment = parsed.fragment
        let urlL = url.lowercased()
        let hostL = host.lowercased()
        let domain = ext.domain
        let subdomain = ext.subdomain
        let suffix = ext.suffix

        var f: [(String, Double)] = []
        f.reserveCapacity(192)
        func put(_ k: String, _ v: Double) { f.append((k, v)) }
        func put(_ k: String, _ v: Int) { f.append((k, Double(v))) }
        func put(_ k: String, _ v: Bool) { f.append((k, v ? 1.0 : 0.0)) }

        let urlLen = pyLen(url)
        let hostLen = pyLen(host)
        let pathLen = pyLen(path)
        let queryLen = pyLen(query)
        let numDigits = countScalars(url, where: isDigit)
        let numSpecial = countScalars(url, where: isSpecialChar)

        // ---- ORIGINAL 21 ----
        put("url_length", urlLen)
        put("host_length", hostLen)
        put("path_length", pathLen)
        put("num_dots", count(url, of: "."))
        put("num_hyphens", count(url, of: "-"))
        put("num_underscores", count(url, of: "_"))
        put("num_slashes", count(url, of: "/"))
        put("num_digits", numDigits)
        put("num_special_chars", numSpecial)
        put("num_subdomains", subdomain.isEmpty ? 0
            : subdomain.split(separator: ".",
                              omittingEmptySubsequences: false).count)
        put("has_ip", hasIPAddress(host))
        put("has_at_symbol", url.contains("@"))
        put("has_double_slash_redirect", path.contains("//"))
        put("is_https", parsed.scheme == "https")
        put("is_shortened", shortenerDomains.contains(registeredDomain))
        put("suspicious_word_count", countSuspiciousWords(url))
        put("entropy", shannonEntropy(host))
        put("query_length", queryLen)
        put("num_params", query.isEmpty ? 0 : count(query, of: "="))
        put("tld_length", pyLen(suffix))
        put("digit_ratio_in_host",
            ratio(countScalars(host, where: isDigit), hostLen))

        // ---- component lengths ----
        put("fragment_length", pyLen(fragment))
        put("domain_length", pyLen(domain))
        put("subdomain_length", pyLen(subdomain))
        put("netloc_length", pyLen(netloc))

        // ---- per-component separator / special-char counts ----
        let pathNumDigits = countScalars(path, where: isDigit)
        put("host_num_dots", count(host, of: "."))
        put("host_num_hyphens", count(host, of: "-"))
        put("host_num_digits", countScalars(host, where: isDigit))
        put("path_num_dots", count(path, of: "."))
        put("path_num_hyphens", count(path, of: "-"))
        put("path_num_digits", pathNumDigits)
        put("domain_num_hyphens", count(domain, of: "-"))
        put("query_num_special", countScalars(query, where: isQuerySpecialChar))

        // ---- query / path structure ----
        put("path_depth", path.split(separator: "/").filter { !$0.isEmpty }.count)
        put("num_query_params", query.isEmpty ? 0
            : query.split(separator: "&", omittingEmptySubsequences: false)
                .filter { !$0.isEmpty }.count)
        let maxParam = query.split(separator: "&", omittingEmptySubsequences: false)
            .filter { $0.contains("=") }
            .map { seg -> Int in
                guard let eq = seg.firstIndex(of: "=") else { return 0 }
                return pyLen(String(seg[seg.index(after: eq)...]))
            }.max() ?? 0
        put("max_param_value_length", maxParam)
        put("has_fragment", !fragment.isEmpty)
        put("path_has_file_ext", pathHasFileExt(path))

        // ---- encoding / obfuscation ----
        put("num_percent_encoded", countPercentEncoded(url))
        put("num_percent_signs", count(url, of: "%"))
        let noScheme = stripScheme(url)
        put("repeated_separators", countRepeatedSeparators(noScheme))
        put("num_equals", count(url, of: "="))
        put("num_ampersands", count(url, of: "&"))
        put("num_question_marks", count(url, of: "?"))
        put("has_hex_like_segment", hasHexLikeSegment(urlL))
        put("has_base64_like_segment", hasBase64LikeSegment(url))

        // ---- port ----
        put("has_port", !port.isEmpty)
        put("has_suspicious_port", !port.isEmpty && !commonPorts.contains(port))

        // ---- internationalisation / homograph ----
        put("is_punycode", hostL.contains("xn--"))
        put("has_non_ascii", url.unicodeScalars.contains { $0.value > 127 })
        put("has_confusable_chars", isConfusable(url))

        // ---- entropy, per component ----
        put("host_entropy", shannonEntropy(host))
        put("path_entropy", shannonEntropy(path))
        put("query_entropy", shannonEntropy(query))
        put("domain_entropy", shannonEntropy(domain))
        put("url_entropy", shannonEntropy(url))

        // ---- character-class ratios ----
        put("digit_ratio_in_url", ratio(numDigits, urlLen))
        put("letter_ratio_in_url",
            ratio(countScalars(url, where: isLetter), urlLen))
        put("letter_ratio_in_host",
            ratio(countScalars(host, where: isLetter), hostLen))
        put("digit_ratio_in_path", ratio(pathNumDigits, pathLen))
        put("special_ratio_in_url", ratio(numSpecial, urlLen))
        put("path_ratio_of_url", ratio(pathLen, urlLen))
        put("query_ratio_of_url", ratio(queryLen, urlLen))
        put("host_ratio_of_url", ratio(hostLen, urlLen))

        // ---- domain / subdomain structure ----
        let subParts = subdomain.split(separator: ".").map(String.init)
            .filter { !$0.isEmpty }
        put("subdomain_depth", subParts.count)
        put("longest_subdomain_length", subParts.map { pyLen($0) }.max() ?? 0)
        put("has_www", (subParts.first?.lowercased() ?? "") == "www")
        put("tld_in_subdomain", !suffix.isEmpty && subdomain.contains(suffix))
        put("tld_in_path",
            !suffix.isEmpty && path.lowercased().contains(".\(suffix)"))
        put("domain_is_digits_only",
            !domain.isEmpty && domain.unicodeScalars.allSatisfy(isDigit))

        // ---- brand-abuse signals ----
        let brandInURL = targetBrands.filter { urlL.contains($0) }
        put("brand_in_url", brandInURL.count)
        put("brand_in_subdomain",
            targetBrands.contains { subdomain.lowercased().contains($0) })
        put("brand_in_path",
            targetBrands.contains { path.lowercased().contains($0) })
        put("brand_outside_domain",
            !brandInURL.isEmpty
            && !targetBrands.contains { domain.lowercased().contains($0) })
        put("suspicious_words_in_host",
            suspiciousWords.filter { hostL.contains($0) }.count)
        put("suspicious_words_in_path",
            suspiciousWords.filter { path.lowercased().contains($0) }.count)

        // ---- token statistics ----
        let urlTokens = tokens(urlL)
        let hostTokens = tokens(hostL)
        put("num_tokens", urlTokens.count)
        put("mean_token_length",
            ratio(urlTokens.reduce(0) { $0 + pyLen($1) }, urlTokens.count))
        put("max_token_length", urlTokens.map { pyLen($0) }.max() ?? 0)
        put("num_host_tokens", hostTokens.count)
        put("max_host_token_length", hostTokens.map { pyLen($0) }.max() ?? 0)

        let dom = domain.lowercased()
        put("domain_vowel_ratio",
            ratio(dom.filter { vowels.contains($0) }.count, pyLen(dom)))
        put("domain_max_consonant_run", maxRun(dom, consonants))
        let domScalars = Array(dom.unicodeScalars)
        var switches = 0
        if domScalars.count > 1 {
            for i in 0..<(domScalars.count - 1)
            where isDigit(domScalars[i]) != isDigit(domScalars[i + 1]) {
                switches += 1
            }
        }
        put("domain_digit_letter_switches", switches)
        var repeatCounts: [Unicode.Scalar: Int] = [:]
        for sc in urlL.unicodeScalars { repeatCounts[sc, default: 0] += 1 }
        put("url_char_repeat_max", repeatCounts.values.max() ?? 0)
        put("host_hyphen_groups", hostL.contains("-")
            ? hostL.split(separator: "-", omittingEmptySubsequences: false)
                .filter { !$0.isEmpty }.count - 1
            : 0)

        // ---- TLD and payload indicators ----
        let lastSuffixLabel = suffix.lowercased()
            .split(separator: ".").last.map(String.init) ?? ""
        put("tld_abuse_prone", abuseProneTLDs.contains(lastSuffixLabel))
        put("path_has_executable_ext", pathHasExecutableExt(path))

        // ---- hashed n-grams ----
        let ngSrc = stripScheme(urlL)
        for (i, v) in hashedCharNgrams(ngSrc).enumerated() {
            put(String(format: "cng3_%02d", i), v)
        }
        for (i, v) in hashedTokenBigrams(tokens(ngSrc)).enumerated() {
            put(String(format: "tng2_%02d", i), v)
        }

        return f
    }

    /// Feature values in the order the model expects, given its name list.
    public static func vector(_ rawURL: String, tld: TldExtract,
                              featureNames: [String]) -> [Double] {
        let pairs = extract(rawURL, tld: tld)
        var byName: [String: Double] = [:]
        byName.reserveCapacity(pairs.count)
        for (k, v) in pairs { byName[k] = v }
        return featureNames.map { byName[$0] ?? 0.0 }
    }
}
