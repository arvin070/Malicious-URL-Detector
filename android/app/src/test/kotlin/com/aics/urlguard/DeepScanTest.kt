package com.aics.urlguard

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

/**
 * The fast-first decision rule.
 *
 * These cover the part that can be tested without a device or a server: WHEN
 * a second opinion is sought. The network behaviour itself is exercised
 * against the real backend by the Python suite.
 */
class DeepScanTest {

    private val threshold = 0.49

    @Test
    fun `confident verdicts on either side never call the server`() {
        // Far below: clearly benign to the on-device model.
        assertFalse(DeepScan.isBorderline(0.02, threshold))
        assertFalse(DeepScan.isBorderline(0.10, threshold))
        // Far above: clearly malicious.
        assertFalse(DeepScan.isBorderline(0.95, threshold))
        assertFalse(DeepScan.isBorderline(0.99, threshold))
    }

    @Test
    fun `scores near the threshold are borderline from both directions`() {
        assertTrue(DeepScan.isBorderline(threshold, threshold))
        assertTrue(DeepScan.isBorderline(threshold - 0.01, threshold))
        assertTrue(DeepScan.isBorderline(threshold + 0.01, threshold))
        // The band is symmetric: a near-miss below the threshold deserves the
        // same second look as a near-hit above it. Being just under the line
        // is exactly the case a URL-only model gets wrong.
        assertEquals(
            DeepScan.isBorderline(threshold - DeepScan.BORDERLINE_MARGIN, threshold),
            DeepScan.isBorderline(threshold + DeepScan.BORDERLINE_MARGIN, threshold)
        )
    }

    @Test
    fun `the band edges are inclusive and nothing beyond them qualifies`() {
        assertTrue(DeepScan.isBorderline(threshold + DeepScan.BORDERLINE_MARGIN, threshold))
        assertTrue(DeepScan.isBorderline(threshold - DeepScan.BORDERLINE_MARGIN, threshold))
        assertFalse(DeepScan.isBorderline(threshold + DeepScan.BORDERLINE_MARGIN + 1e-6, threshold))
        assertFalse(DeepScan.isBorderline(threshold - DeepScan.BORDERLINE_MARGIN - 1e-6, threshold))
    }

    @Test
    fun `the wait is capped at something a user will tolerate`() {
        // A link tap that stalls for seconds feels broken, and the on-device
        // answer is already in hand.
        assertTrue(DeepScan.TIMEOUT_MS <= 3000L, "deep scan deadline too long")
        assertTrue(DeepScan.TIMEOUT_MS >= 1000L,
            "deep scan deadline too short to be useful")
    }

    @Test
    fun `the borderline band does not swallow the whole score range`() {
        // If the margin were wide enough to cover most scores, every tap
        // would wait on the network and the fast path would not exist.
        assertTrue(DeepScan.BORDERLINE_MARGIN > 0.0)
        assertTrue(DeepScan.BORDERLINE_MARGIN <= 0.25,
            "borderline band is too wide to be a fast path")
    }

    @Test
    fun `host extraction never throws on malformed input`() {
        for (u in listOf("", "not a url", "http://", "http://[", "://x",
                "https://ok.example/path")) {
            DeepScan.hostOf(u)   // must not throw
        }
        assertEquals("ok.example", DeepScan.hostOf("https://ok.example/path"))
    }
}
