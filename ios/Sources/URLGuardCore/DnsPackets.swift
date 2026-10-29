import Foundation

/// Minimal IPv4/UDP/DNS packet handling for the on-device firewall.
///
/// Swift port of `android/.../DnsPackets.kt`, kept byte-for-byte compatible so
/// both platforms drop and allow the same queries. Only what the guard needs:
/// read the queried name out of a DNS request, and build either an NXDOMAIN
/// refusal or a reply wrapped back into IPv4+UDP with correct checksums.
///
/// Deliberately narrow: IPv4 + UDP only. Anything else (IPv6, TCP, fragments)
/// is reported as unparseable and passed through untouched, because silently
/// mangling traffic the guard does not understand would be far worse than not
/// filtering it.
///
/// This file is pure byte manipulation with no Apple dependency, so it is
/// COMPILED AND TESTED ON LINUX. A firewall that corrupts packets is a bricked
/// phone, and "it looked right" is not a standard worth shipping against.
public enum DnsPackets {

    public static let ipv4Version = 4
    public static let protoUDP = 17
    public static let typeA = 1
    public static let typeAAAA = 28

    public struct Ipv4Udp: Sendable {
        public let srcAddr: [UInt8]
        public let dstAddr: [UInt8]
        public let srcPort: Int
        public let dstPort: Int
        public let payload: [UInt8]
        public let ihl: Int
    }

    /// Returns nil when the datagram is not a well-formed IPv4 UDP packet.
    public static func parseIpv4Udp(_ packet: [UInt8],
                                    length: Int? = nil) -> Ipv4Udp? {
        let len = length ?? packet.count
        guard len >= 28, packet.count >= len else { return nil }
        guard Int(packet[0] >> 4) == ipv4Version else { return nil }
        let ihl = Int(packet[0] & 0x0F) * 4
        guard ihl >= 20, len >= ihl + 8 else { return nil }
        guard Int(packet[9]) == protoUDP else { return nil }

        // Fragmented packets cannot be reassembled here; leave them alone.
        let flagsFrag = (Int(packet[6]) << 8) | Int(packet[7])
        if (flagsFrag & 0x2000) != 0 || (flagsFrag & 0x1FFF) != 0 { return nil }

        let src = Array(packet[12..<16])
        let dst = Array(packet[16..<20])
        let sPort = (Int(packet[ihl]) << 8) | Int(packet[ihl + 1])
        let dPort = (Int(packet[ihl + 2]) << 8) | Int(packet[ihl + 3])
        let udpLen = (Int(packet[ihl + 4]) << 8) | Int(packet[ihl + 5])
        guard udpLen >= 8 else { return nil }
        let payloadLen = min(udpLen - 8, len - ihl - 8)
        guard payloadLen >= 0 else { return nil }
        let payload = Array(packet[(ihl + 8)..<(ihl + 8 + payloadLen)])
        return Ipv4Udp(srcAddr: src, dstAddr: dst, srcPort: sPort,
                       dstPort: dPort, payload: payload, ihl: ihl)
    }

    /// Queried name from a DNS message, lowercased and without the trailing
    /// dot. Returns nil if the message has no question or is malformed.
    public static func questionName(_ dns: [UInt8]) -> String? {
        guard dns.count >= 13 else { return nil }
        let qdCount = (Int(dns[4]) << 8) | Int(dns[5])
        guard qdCount >= 1 else { return nil }

        var name = ""
        var i = 12
        var guardCount = 0
        while i < dns.count {
            let len = Int(dns[i])
            if len == 0 { break }
            // Compression pointers are not legal in a question section.
            if (len & 0xC0) != 0 { return nil }
            i += 1
            guard i + len <= dns.count else { return nil }
            if !name.isEmpty { name.append(".") }
            for k in 0..<len {
                name.append(Character(Unicode.Scalar(dns[i + k])))
            }
            i += len
            guardCount += 1
            if guardCount > 127 { return nil }
        }
        let lowered = name.lowercased()
        return lowered.isEmpty ? nil : lowered
    }

    /// QTYPE of the first question, or -1 if it cannot be read.
    ///
    /// Needed because only forward lookups represent somewhere the user is
    /// about to GO. A PTR query is a reverse lookup of an address that was
    /// already resolved -- blocking one protects nobody and breaks reverse
    /// lookups.
    public static func questionType(_ dns: [UInt8]) -> Int {
        guard let end = questionEnd(dns), end >= 4, end <= dns.count else {
            return -1
        }
        let t = end - 4
        return (Int(dns[t]) << 8) | Int(dns[t + 1])
    }

    /// Names that must never be classified, whatever they score.
    ///
    /// `*.in-addr.arpa` / `*.ip6.arpa` are reverse lookups; the model sees a
    /// long digit-heavy label and scores it like a DGA domain. Observed on a
    /// real Android device: `224.166.102.103.in-addr.arpa` scored 0.97 and was
    /// blocked, which is pure false-positive noise.
    ///
    /// The rest are special-use names (RFC 6761/8375) resolved locally, which
    /// are not web destinations at all.
    public static func isNonNavigationalName(_ host: String) -> Bool {
        var h = host.lowercased()
        while h.hasSuffix(".") { h.removeLast() }
        if h.isEmpty { return true }
        if h.hasSuffix(".arpa") || h == "arpa" { return true }
        for suffix in specialUseSuffixes where h == suffix || h.hasSuffix(".\(suffix)") {
            return true
        }
        // A single label ("router", "nas") is a LAN name, never a public site.
        if !h.contains(".") { return true }
        return false
    }

    static let specialUseSuffixes = [
        "local", "localhost", "home.arpa", "internal", "intranet",
        "lan", "private", "corp", "home", "test", "invalid", "onion",
    ]

    /// True if this looks like a standard query we should inspect.
    public static func isStandardQuery(_ dns: [UInt8]) -> Bool {
        guard dns.count >= 12 else { return false }
        let flags = (Int(dns[2]) << 8) | Int(dns[3])
        let isResponse = (flags & 0x8000) != 0
        let opcode = (flags >> 11) & 0xF
        return !isResponse && opcode == 0
    }

    /// Copy of the request header answering NXDOMAIN (RCODE 3), question
    /// echoed back.
    public static func buildNxDomain(_ request: [UInt8]) -> [UInt8] {
        guard let qEnd = questionEnd(request), qEnd <= request.count else {
            return []
        }
        var out = Array(request[0..<qEnd])
        // QR=1, RD copied from request, RA=1, RCODE=3
        let rd = Int(request[2]) & 0x01
        out[2] = UInt8(0x80 | rd)
        out[3] = UInt8(0x80 | 0x03)
        out[6] = 0; out[7] = 0          // ANCOUNT
        out[8] = 0; out[9] = 0          // NSCOUNT
        out[10] = 0; out[11] = 0        // ARCOUNT
        return out
    }

    static func questionEnd(_ dns: [UInt8]) -> Int? {
        guard dns.count >= 13 else { return nil }
        var i = 12
        while i < dns.count {
            let len = Int(dns[i])
            if len == 0 { i += 1; break }
            if (len & 0xC0) != 0 { return nil }
            i += len + 1
        }
        return i + 4 <= dns.count ? i + 4 : nil
    }

    /// Wrap a DNS payload in IPv4+UDP, swapping the request's endpoints so the
    /// reply goes back to the asking socket.
    public static func buildResponsePacket(request: Ipv4Udp,
                                           dnsPayload: [UInt8]) -> [UInt8] {
        let total = 20 + 8 + dnsPayload.count
        var p = [UInt8](repeating: 0, count: total)

        p[0] = 0x45                                  // IPv4, IHL=5
        p[1] = 0                                     // DSCP/ECN
        writeShort(&p, 2, total)
        writeShort(&p, 4, 0)                         // identification
        writeShort(&p, 6, 0x4000)                    // don't fragment
        p[8] = 64                                    // TTL
        p[9] = UInt8(protoUDP)
        writeShort(&p, 10, 0)                        // checksum placeholder
        for (i, b) in request.dstAddr.enumerated() { p[12 + i] = b }  // src
        for (i, b) in request.srcAddr.enumerated() { p[16 + i] = b }  // dst

        let udpLen = 8 + dnsPayload.count
        writeShort(&p, 20, request.dstPort)
        writeShort(&p, 22, request.srcPort)
        writeShort(&p, 24, udpLen)
        writeShort(&p, 26, 0)                        // UDP checksum placeholder
        for (i, b) in dnsPayload.enumerated() { p[28 + i] = b }

        writeShort(&p, 10, checksum(p, from: 0, len: 20))
        writeShort(&p, 26, udpChecksum(p, payloadLen: dnsPayload.count))
        return p
    }

    static func writeShort(_ b: inout [UInt8], _ off: Int, _ v: Int) {
        b[off] = UInt8((v >> 8) & 0xFF)
        b[off + 1] = UInt8(v & 0xFF)
    }

    static func checksum(_ data: [UInt8], from: Int, len: Int) -> Int {
        var sum: UInt64 = 0
        var i = from
        let end = from + len
        while i + 1 < end {
            sum += UInt64((Int(data[i]) << 8) | Int(data[i + 1]))
            i += 2
        }
        if i < end { sum += UInt64(Int(data[i]) << 8) }
        while (sum >> 16) != 0 { sum = (sum & 0xFFFF) + (sum >> 16) }
        return Int(~sum & 0xFFFF)
    }

    /// UDP checksum over the IPv4 pseudo-header + UDP header + payload.
    static func udpChecksum(_ pkt: [UInt8], payloadLen: Int) -> Int {
        let udpLen = 8 + payloadLen
        var sum: UInt64 = 0
        var i = 12
        while i < 20 {                               // src + dst addresses
            sum += UInt64((Int(pkt[i]) << 8) | Int(pkt[i + 1]))
            i += 2
        }
        sum += UInt64(protoUDP)
        sum += UInt64(udpLen)
        i = 20
        let end = 20 + udpLen
        while i + 1 < end {
            sum += UInt64((Int(pkt[i]) << 8) | Int(pkt[i + 1]))
            i += 2
        }
        if i < end { sum += UInt64(Int(pkt[i]) << 8) }
        while (sum >> 16) != 0 { sum = (sum & 0xFFFF) + (sum >> 16) }
        let c = Int(~sum & 0xFFFF)
        // 0 means "no checksum" in UDP, so the all-ones form is sent instead.
        return c == 0 ? 0xFFFF : c
    }
}
