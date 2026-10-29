package com.aics.urlguard

import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Minimal IPv4/UDP/DNS packet handling for the VPN layer.
 *
 * Only what the guard needs: read the queried name out of a DNS request, and
 * build either an NXDOMAIN refusal or a copy of an upstream reply, wrapped back
 * into IPv4+UDP with correct checksums.
 *
 * Deliberately narrow: IPv4 + UDP only. Anything else (IPv6, TCP, fragments) is
 * reported as unparseable and passed through untouched by the service, because
 * silently mangling traffic the guard does not understand would be far worse
 * than not filtering it.
 */
object DnsPackets {

    const val IPV4 = 4
    const val PROTO_UDP = 17

    data class Ipv4Udp(
        val srcAddr: ByteArray, val dstAddr: ByteArray,
        val srcPort: Int, val dstPort: Int,
        val payload: ByteArray,
        val ihl: Int,
    )

    /** Returns null when the datagram is not a well-formed IPv4 UDP packet. */
    fun parseIpv4Udp(packet: ByteArray, length: Int): Ipv4Udp? {
        if (length < 28) return null
        val version = (packet[0].toInt() ushr 4) and 0xF
        if (version != IPV4) return null
        val ihl = (packet[0].toInt() and 0xF) * 4
        if (ihl < 20 || length < ihl + 8) return null
        if ((packet[9].toInt() and 0xFF) != PROTO_UDP) return null

        // Fragmented packets cannot be reassembled here; leave them alone.
        val flagsFrag = ((packet[6].toInt() and 0xFF) shl 8) or (packet[7].toInt() and 0xFF)
        val moreFragments = (flagsFrag and 0x2000) != 0
        val fragOffset = flagsFrag and 0x1FFF
        if (moreFragments || fragOffset != 0) return null

        val src = packet.copyOfRange(12, 16)
        val dst = packet.copyOfRange(16, 20)
        val sPort = ((packet[ihl].toInt() and 0xFF) shl 8) or (packet[ihl + 1].toInt() and 0xFF)
        val dPort = ((packet[ihl + 2].toInt() and 0xFF) shl 8) or (packet[ihl + 3].toInt() and 0xFF)
        val udpLen = ((packet[ihl + 4].toInt() and 0xFF) shl 8) or (packet[ihl + 5].toInt() and 0xFF)
        if (udpLen < 8) return null
        val payloadLen = minOf(udpLen - 8, length - ihl - 8)
        if (payloadLen < 0) return null
        val payload = packet.copyOfRange(ihl + 8, ihl + 8 + payloadLen)
        return Ipv4Udp(src, dst, sPort, dPort, payload, ihl)
    }

    /**
     * Queried name from a DNS message, lowercased and without the trailing dot.
     * Returns null if the message has no question or is malformed.
     */
    fun questionName(dns: ByteArray): String? {
        if (dns.size < 13) return null
        val qdCount = ((dns[4].toInt() and 0xFF) shl 8) or (dns[5].toInt() and 0xFF)
        if (qdCount < 1) return null

        val sb = StringBuilder()
        var i = 12
        var guard = 0
        while (i < dns.size) {
            val len = dns[i].toInt() and 0xFF
            if (len == 0) break
            // Compression pointers are not legal in a question section.
            if ((len and 0xC0) != 0) return null
            i++
            if (i + len > dns.size) return null
            if (sb.isNotEmpty()) sb.append('.')
            for (k in 0 until len) {
                sb.append((dns[i + k].toInt() and 0xFF).toChar())
            }
            i += len
            if (++guard > 127) return null
        }
        val name = sb.toString().lowercase()
        return name.ifEmpty { null }
    }

    const val TYPE_A = 1
    const val TYPE_AAAA = 28

    /**
     * QTYPE of the first question, or -1 if it cannot be read.
     *
     * Needed because only forward lookups represent somewhere the user is about
     * to GO. A PTR query is a reverse lookup of an IP that was already
     * resolved -- blocking one protects nobody and breaks reverse lookups.
     */
    fun questionType(dns: ByteArray): Int {
        val end = questionEnd(dns) ?: return -1
        if (end < 4 || end > dns.size) return -1
        val t = end - 4
        return ((dns[t].toInt() and 0xFF) shl 8) or (dns[t + 1].toInt() and 0xFF)
    }

    /**
     * Names that must never be classified, whatever they score.
     *
     * `*.in-addr.arpa` / `*.ip6.arpa` are reverse lookups; the model sees a long
     * digit-heavy label and scores it like a DGA domain. Observed on a real
     * device: `224.166.102.103.in-addr.arpa` scored 0.97 and was blocked, which
     * is pure false-positive noise.
     *
     * The rest are special-use names (RFC 6761/8375) that are resolved locally
     * and are not web destinations at all.
     */
    fun isNonNavigationalName(host: String): Boolean {
        val h = host.trimEnd('.').lowercase()
        if (h.isEmpty()) return true
        if (h.endsWith(".arpa") || h == "arpa") return true
        for (suffix in SPECIAL_USE_SUFFIXES) {
            if (h == suffix || h.endsWith(".$suffix")) return true
        }
        // A single label ("router", "nas") is a LAN name, never a public site.
        if (!h.contains('.')) return true
        return false
    }

    private val SPECIAL_USE_SUFFIXES = listOf(
        "local", "localhost", "home.arpa", "internal", "intranet",
        "lan", "private", "corp", "home", "test", "invalid", "onion",
    )

    /** True if this looks like a standard query we should inspect. */
    fun isStandardQuery(dns: ByteArray): Boolean {
        if (dns.size < 12) return false
        val flags = ((dns[2].toInt() and 0xFF) shl 8) or (dns[3].toInt() and 0xFF)
        val isResponse = (flags and 0x8000) != 0
        val opcode = (flags ushr 11) and 0xF
        return !isResponse && opcode == 0
    }

    /** Copy of the request header answering NXDOMAIN (RCODE 3), question echoed. */
    fun buildNxDomain(request: ByteArray): ByteArray {
        val qEnd = questionEnd(request) ?: return ByteArray(0)
        val out = request.copyOfRange(0, qEnd)
        // QR=1, RD copied from request, RA=1, RCODE=3
        val rd = (request[2].toInt() and 0x01)
        out[2] = (0x80 or (rd shl 0)).toByte()
        out[3] = (0x80 or 0x03).toByte()
        out[6] = 0; out[7] = 0          // ANCOUNT
        out[8] = 0; out[9] = 0          // NSCOUNT
        out[10] = 0; out[11] = 0        // ARCOUNT
        return out
    }

    private fun questionEnd(dns: ByteArray): Int? {
        if (dns.size < 13) return null
        var i = 12
        while (i < dns.size) {
            val len = dns[i].toInt() and 0xFF
            if (len == 0) { i++; break }
            if ((len and 0xC0) != 0) return null
            i += len + 1
        }
        return if (i + 4 <= dns.size) i + 4 else null
    }

    /**
     * Wrap a DNS payload in IPv4+UDP, swapping the request's endpoints so the
     * reply goes back to the asking socket.
     */
    fun buildResponsePacket(request: Ipv4Udp, dnsPayload: ByteArray): ByteArray {
        val total = 20 + 8 + dnsPayload.size
        val buf = ByteBuffer.allocate(total).order(ByteOrder.BIG_ENDIAN)

        buf.put((0x45).toByte())                    // IPv4, IHL=5
        buf.put(0)                                  // DSCP/ECN
        buf.putShort(total.toShort())
        buf.putShort(0)                             // identification
        buf.putShort(0x4000.toShort())              // don't fragment
        buf.put(64)                                 // TTL
        buf.put(PROTO_UDP.toByte())
        buf.putShort(0)                             // checksum placeholder
        buf.put(request.dstAddr)                    // src = who was asked
        buf.put(request.srcAddr)                    // dst = who asked

        val udpLen = 8 + dnsPayload.size
        buf.putShort(request.dstPort.toShort())
        buf.putShort(request.srcPort.toShort())
        buf.putShort(udpLen.toShort())
        buf.putShort(0)                             // UDP checksum placeholder
        buf.put(dnsPayload)

        val pkt = buf.array()
        writeShort(pkt, 10, checksum(pkt, 0, 20))
        writeShort(pkt, 26, udpChecksum(pkt, dnsPayload.size))
        return pkt
    }

    private fun writeShort(b: ByteArray, off: Int, v: Int) {
        b[off] = ((v ushr 8) and 0xFF).toByte()
        b[off + 1] = (v and 0xFF).toByte()
    }

    private fun checksum(data: ByteArray, from: Int, len: Int): Int {
        var sum = 0L
        var i = from
        val end = from + len
        while (i + 1 < end) {
            sum += ((data[i].toInt() and 0xFF) shl 8) or (data[i + 1].toInt() and 0xFF)
            i += 2
        }
        if (i < end) sum += (data[i].toInt() and 0xFF) shl 8
        while ((sum shr 16) != 0L) sum = (sum and 0xFFFF) + (sum shr 16)
        return (sum.inv() and 0xFFFF).toInt()
    }

    /** UDP checksum over the IPv4 pseudo-header + UDP header + payload. */
    private fun udpChecksum(pkt: ByteArray, payloadLen: Int): Int {
        val udpLen = 8 + payloadLen
        var sum = 0L
        for (i in 12 until 20 step 2) {             // src + dst addresses
            sum += ((pkt[i].toInt() and 0xFF) shl 8) or (pkt[i + 1].toInt() and 0xFF)
        }
        sum += PROTO_UDP.toLong()
        sum += udpLen.toLong()
        var i = 20
        val end = 20 + udpLen
        while (i + 1 < end) {
            sum += ((pkt[i].toInt() and 0xFF) shl 8) or (pkt[i + 1].toInt() and 0xFF)
            i += 2
        }
        if (i < end) sum += (pkt[i].toInt() and 0xFF) shl 8
        while ((sum shr 16) != 0L) sum = (sum and 0xFFFF) + (sum shr 16)
        val c = (sum.inv() and 0xFFFF).toInt()
        // 0 means "no checksum" in UDP, so the all-ones form is sent instead.
        return if (c == 0) 0xFFFF else c
    }
}
