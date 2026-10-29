import Foundation

/// Punycode (RFC 3492) decoding for IDN labels.
///
/// WHY THIS IS HAND-WRITTEN
/// ------------------------
/// `tldextract` decodes every `xn--` label before matching it against the
/// public suffix list, because the list stores suffixes in Unicode form
/// ("中国", not "xn--fiqs8s"). Skip the decode and every internationalised
/// domain gets the wrong suffix, hence the wrong `domain`, `subdomain`,
/// `tld_length` and `domain_entropy` -- silently.
///
/// The Kotlin port gets this from `java.net.IDN`. Foundation has no equivalent
/// that works on Linux, so the algorithm is implemented directly. It is fully
/// specified and deterministic, and the golden-vector test covers the
/// punycode cases in the fixture.
enum Punycode {

    private static let base = 36
    private static let tmin = 1
    private static let tmax = 26
    private static let skew = 38
    private static let damp = 700
    private static let initialBias = 72
    private static let initialN = 128

    private static func digit(_ c: Unicode.Scalar) -> Int? {
        switch c.value {
        case 0x30...0x39: return Int(c.value) - 0x30 + 26   // '0'-'9' -> 26..35
        case 0x41...0x5A: return Int(c.value) - 0x41        // 'A'-'Z' -> 0..25
        case 0x61...0x7A: return Int(c.value) - 0x61        // 'a'-'z' -> 0..25
        default: return nil
        }
    }

    private static func adapt(_ delta: Int, _ numPoints: Int,
                              _ firstTime: Bool) -> Int {
        var delta = firstTime ? delta / damp : delta / 2
        delta += delta / numPoints
        var k = 0
        while delta > ((base - tmin) * tmax) / 2 {
            delta /= (base - tmin)
            k += base
        }
        return k + (((base - tmin + 1) * delta) / (delta + skew))
    }

    /// Decode one label. Returns nil when the input is not valid punycode, so
    /// the caller can fall back to the original label -- which is what
    /// `tldextract` does when `IDN.toUnicode` raises.
    static func decode(_ input: String) -> String? {
        var output: [Unicode.Scalar] = []
        var n = initialN
        var i = 0
        var bias = initialBias

        let scalars = Array(input.unicodeScalars)
        // Everything before the last delimiter is literal ASCII.
        var start = 0
        if let delim = scalars.lastIndex(of: "-") {
            for s in scalars[0..<delim] {
                guard s.isASCII else { return nil }
                output.append(s)
            }
            start = delim + 1
        }

        var idx = start
        while idx < scalars.count {
            let oldi = i
            var w = 1
            var k = base
            while true {
                guard idx < scalars.count else { return nil }
                guard let d = digit(scalars[idx]) else { return nil }
                idx += 1
                // Overflow guards, as required by the RFC.
                guard d <= (Int.max - i) / w else { return nil }
                i += d * w
                let t: Int
                if k <= bias { t = tmin }
                else if k >= bias + tmax { t = tmax }
                else { t = k - bias }
                if d < t { break }
                guard w <= Int.max / (base - t) else { return nil }
                w *= (base - t)
                k += base
            }
            let outLen = output.count + 1
            bias = adapt(i - oldi, outLen, oldi == 0)
            guard i / outLen <= Int.max - n else { return nil }
            n += i / outLen
            i %= outLen
            guard let scalar = Unicode.Scalar(UInt32(n)) else { return nil }
            output.insert(scalar, at: i)
            i += 1
        }

        var s = ""
        s.unicodeScalars.append(contentsOf: output)
        return s
    }

    /// `tldextract._decode_punycode`: lowercase, then decode only `xn--`
    /// labels, falling back to the lowered label when decoding fails.
    static func decodeLabel(_ label: String) -> String {
        let lowered = label.lowercased()
        guard lowered.hasPrefix("xn--") else { return lowered }
        let body = String(lowered.dropFirst(4))
        return decode(body) ?? lowered
    }
}
