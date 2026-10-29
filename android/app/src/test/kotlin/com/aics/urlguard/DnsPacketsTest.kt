package com.aics.urlguard

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test
import java.nio.ByteBuffer

/**
 * The DNS layer rewrites raw IP packets. A checksum bug here does not show up
 * as a crash -- it shows up as "the user's internet is broken", which is the
 * worst possible failure for a security tool. So the packet maths is tested
 * directly.
 */
class DnsPacketsTest {

    /** Builds a realistic IPv4/UDP DNS query for `name`. */
    private fun query(name: String, txId: Int = 0x1234): ByteArray {
        val labels = name.split(".").filter { it.isNotEmpty() }
        val dnsLen = 12 + labels.sumOf { it.length + 1 } + 1 + 4
        val dns = ByteBuffer.allocate(dnsLen)
        dns.putShort(txId.toShort())
        dns.putShort(0x0100.toShort())     // standard query, RD
        dns.putShort(1.toShort()); dns.putShort(0); dns.putShort(0); dns.putShort(0)
        for (l in labels) { dns.put(l.length.toByte()); dns.put(l.toByteArray()) }
        dns.put(0.toByte())
        dns.putShort(1.toShort()); dns.putShort(1.toShort())   // A, IN
        val payload = dns.array()

        val total = 20 + 8 + payload.size
        val p = ByteBuffer.allocate(total)
        p.put(0x45.toByte()); p.put(0.toByte()); p.putShort(total.toShort())
        p.putShort(0); p.putShort(0x4000.toShort()); p.put(64.toByte()); p.put(17.toByte())
        p.putShort(0)
        p.put(byteArrayOf(10, 111, 222.toByte(), 1))      // src
        p.put(byteArrayOf(10, 111, 222.toByte(), 2))      // dst
        p.putShort(40000.toShort()); p.putShort(53.toShort())
        p.putShort((8 + payload.size).toShort()); p.putShort(0.toShort())
        p.put(payload)
        return p.array()
    }

    @Test
    fun `parses a well formed query`() {
        val pkt = query("secure-paypal-login.tk")
        val parsed = DnsPackets.parseIpv4Udp(pkt, pkt.size)
        assertNotNull(parsed)
        assertEquals(53, parsed!!.dstPort)
        assertEquals(40000, parsed.srcPort)
        assertEquals("secure-paypal-login.tk", DnsPackets.questionName(parsed.payload))
        assertTrue(DnsPackets.isStandardQuery(parsed.payload))
    }

    @Test
    fun `question name is lowercased and dot joined`() {
        val pkt = query("A.Very.Deep.Example.CO.UK")
        val parsed = DnsPackets.parseIpv4Udp(pkt, pkt.size)!!
        assertEquals("a.very.deep.example.co.uk", DnsPackets.questionName(parsed.payload))
    }

    @Test
    fun `rejects non udp, short, and fragmented packets`() {
        val pkt = query("example.com")
        assertNull(DnsPackets.parseIpv4Udp(pkt, 10), "short packet accepted")

        val tcp = pkt.copyOf(); tcp[9] = 6
        assertNull(DnsPackets.parseIpv4Udp(tcp, tcp.size), "TCP accepted")

        val frag = pkt.copyOf(); frag[6] = 0x20; frag[7] = 0
        assertNull(DnsPackets.parseIpv4Udp(frag, frag.size), "fragment accepted")

        val v6 = pkt.copyOf(); v6[0] = 0x65
        assertNull(DnsPackets.parseIpv4Udp(v6, v6.size), "non-IPv4 accepted")
    }

    @Test
    fun `nxdomain reply echoes the question and sets rcode 3`() {
        val pkt = query("bad.example", txId = 0xBEEF)
        val parsed = DnsPackets.parseIpv4Udp(pkt, pkt.size)!!
        val reply = DnsPackets.buildNxDomain(parsed.payload)

        assertTrue(reply.isNotEmpty())
        // transaction id preserved -- the resolver matches on it
        assertEquals(0xBE, reply[0].toInt() and 0xFF)
        assertEquals(0xEF, reply[1].toInt() and 0xFF)
        assertTrue((reply[2].toInt() and 0x80) != 0, "QR bit not set")
        assertEquals(3, reply[3].toInt() and 0x0F, "RCODE is not NXDOMAIN")
        assertEquals(1, ((reply[4].toInt() and 0xFF) shl 8) or (reply[5].toInt() and 0xFF))
        assertEquals(0, ((reply[6].toInt() and 0xFF) shl 8) or (reply[7].toInt() and 0xFF))
        assertEquals("bad.example", DnsPackets.questionName(reply))
    }

    @Test
    fun `response packet has valid checksums and swapped endpoints`() {
        val pkt = query("example.com")
        val parsed = DnsPackets.parseIpv4Udp(pkt, pkt.size)!!
        val reply = DnsPackets.buildNxDomain(parsed.payload)
        val out = DnsPackets.buildResponsePacket(parsed, reply)

        // A correct IPv4 header checksums to zero when re-summed including it.
        assertEquals(0, ones(out, 0, 20), "bad IPv4 header checksum")

        // Endpoints swapped so the reply reaches the asking socket.
        val back = DnsPackets.parseIpv4Udp(out, out.size)!!
        assertEquals(parsed.dstPort, back.srcPort)
        assertEquals(parsed.srcPort, back.dstPort)
        assertArrayEquals(parsed.dstAddr, back.srcAddr)
        assertArrayEquals(parsed.srcAddr, back.dstAddr)

        // UDP checksum validates over the pseudo-header.
        assertEquals(0, udpOnes(out), "bad UDP checksum")
        assertEquals("example.com", DnsPackets.questionName(back.payload))
    }

    @Test
    fun `malformed dns payloads are rejected, not guessed`() {
        assertNull(DnsPackets.questionName(ByteArray(4)))
        assertNull(DnsPackets.questionName(ByteArray(20)))       // qdcount 0
        // compression pointer in the question section is illegal
        val bad = ByteArray(24)
        bad[5] = 1; bad[12] = 0xC0.toByte()
        assertNull(DnsPackets.questionName(bad))
        // a label running past the end of the buffer
        val over = ByteArray(16); over[5] = 1; over[12] = 100
        assertNull(DnsPackets.questionName(over))
    }

    // --- helpers: one's-complement sums, which must be 0 for a valid packet ---
    private fun ones(b: ByteArray, from: Int, len: Int): Int {
        var sum = 0L
        var i = from
        while (i + 1 < from + len) {
            sum += ((b[i].toInt() and 0xFF) shl 8) or (b[i + 1].toInt() and 0xFF); i += 2
        }
        while ((sum shr 16) != 0L) sum = (sum and 0xFFFF) + (sum shr 16)
        return (sum.inv() and 0xFFFF).toInt()
    }

    private fun udpOnes(pkt: ByteArray): Int {
        val udpLen = ((pkt[24].toInt() and 0xFF) shl 8) or (pkt[25].toInt() and 0xFF)
        var sum = 0L
        for (i in 12 until 20 step 2) {
            sum += ((pkt[i].toInt() and 0xFF) shl 8) or (pkt[i + 1].toInt() and 0xFF)
        }
        sum += 17L; sum += udpLen.toLong()
        var i = 20
        while (i + 1 < 20 + udpLen) {
            sum += ((pkt[i].toInt() and 0xFF) shl 8) or (pkt[i + 1].toInt() and 0xFF); i += 2
        }
        if (i < 20 + udpLen) sum += (pkt[i].toInt() and 0xFF) shl 8
        while ((sum shr 16) != 0L) sum = (sum and 0xFFFF) + (sum shr 16)
        return (sum.inv() and 0xFFFF).toInt()
    }
}

/**
 * Regression tests for the reverse-DNS false positives found on a real device:
 * `224.166.102.103.in-addr.arpa` scored 0.97 and was blocked, which protects
 * nobody and breaks reverse lookups.
 */
class DnsScopeTest {

    @Test
    fun `reverse dns names are never classified`() {
        assertTrue(DnsPackets.isNonNavigationalName("224.166.102.103.in-addr.arpa"))
        assertTrue(DnsPackets.isNonNavigationalName("78.182.250.142.in-addr.arpa"))
        assertTrue(DnsPackets.isNonNavigationalName("1.0.0.0.ip6.arpa"))
        assertTrue(DnsPackets.isNonNavigationalName("in-addr.arpa"))
    }

    @Test
    fun `special use and lan names are never classified`() {
        for (h in listOf("printer.local", "localhost", "nas.lan", "router",
                         "db.internal", "wiki.corp", "foo.invalid", "x.onion",
                         "myhost.home.arpa", "")) {
            assertTrue(DnsPackets.isNonNavigationalName(h), "not skipped: '$h'")
        }
    }

    @Test
    fun `real sites are still classified`() {
        for (h in listOf("google.com", "secure-paypal-login.tk",
                         "en.wikipedia.org", "a.b.example.co.uk")) {
            assertFalse(DnsPackets.isNonNavigationalName(h), "wrongly skipped: '$h'")
        }
    }

    @Test
    fun `question type is read correctly`() {
        // A query for example.com, QTYPE=1 (A), QCLASS=1
        val dns = byteArrayOf(
            0x12, 0x34, 0x01, 0x00, 0x00, 0x01, 0, 0, 0, 0, 0, 0,
            7, 'e'.code.toByte(), 'x'.code.toByte(), 'a'.code.toByte(),
            'm'.code.toByte(), 'p'.code.toByte(), 'l'.code.toByte(), 'e'.code.toByte(),
            3, 'c'.code.toByte(), 'o'.code.toByte(), 'm'.code.toByte(),
            0, 0, 1, 0, 1
        )
        assertEquals(DnsPackets.TYPE_A, DnsPackets.questionType(dns))

        val ptr = dns.copyOf()
        ptr[ptr.size - 3] = 12          // PTR
        assertEquals(12, DnsPackets.questionType(ptr))

        val aaaa = dns.copyOf()
        aaaa[aaaa.size - 3] = 28        // AAAA
        assertEquals(DnsPackets.TYPE_AAAA, DnsPackets.questionType(aaaa))

        assertEquals(-1, DnsPackets.questionType(ByteArray(8)))
    }
}
