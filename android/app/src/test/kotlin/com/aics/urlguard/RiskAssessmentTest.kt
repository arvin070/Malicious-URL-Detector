package com.aics.urlguard

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

/**
 * The Android UI must make the same conservative claims as the web UI.
 *
 * These mirror the Python `TestAssessmentLayer` tests so the two surfaces
 * cannot drift into saying different things about the same score.
 */
class RiskAssessmentTest {

    @Test
    fun `bands match the specified ranges`() {
        assertEquals(RiskAssessment.Band.LOW, RiskAssessment.bandFor(0))
        assertEquals(RiskAssessment.Band.LOW, RiskAssessment.bandFor(30))
        assertEquals(RiskAssessment.Band.SUSPICIOUS, RiskAssessment.bandFor(31))
        assertEquals(RiskAssessment.Band.SUSPICIOUS, RiskAssessment.bandFor(70))
        assertEquals(RiskAssessment.Band.HIGH, RiskAssessment.bandFor(71))
        assertEquals(RiskAssessment.Band.HIGH, RiskAssessment.bandFor(100))
    }

    @Test
    fun `score never reads as certainty`() {
        for (p in listOf(0.00001, 0.0001, 0.5, 0.9999, 0.999999)) {
            val s = RiskAssessment.displayScore(p)
            assertTrue(s in 1..99, "displayScore($p) = $s implies certainty")
        }
    }

    @Test
    fun `confidence is low near the threshold and high far from it`() {
        val near = RiskAssessment.decisionMargin(0.69, 0.70)
        val far = RiskAssessment.decisionMargin(0.999, 0.70)
        val lowEnd = RiskAssessment.decisionMargin(0.01, 0.70)
        assertTrue(near < 0.2, "near-threshold margin too high: $near")
        assertTrue(far > 0.9, "decisive margin too low: $far")
        assertTrue(lowEnd > 0.9, "a very low score should be a confident LOW RISK")
        assertEquals("Borderline", RiskAssessment.marginLabel(near))
        assertEquals("Decisive", RiskAssessment.marginLabel(far))
    }

    @Test
    fun `band wording never claims certainty`() {
        val banned = listOf("definitely", "confirmed", "guarantee", "100%",
                            "is malicious", "is safe", "proven")
        for (b in RiskAssessment.Band.values()) {
            val text = b.description.lowercase()
            for (word in banned) {
                assertFalse(text.contains(word), "${b.name} overstates: ${b.description}")
            }
        }
        assertTrue(RiskAssessment.DISCLAIMER.lowercase().contains("not a guarantee"))
    }

    @Test
    fun `low risk is not described as safe`() {
        val low = RiskAssessment.Band.LOW.description.lowercase()
        assertFalse(low.contains("safe"))
        assertTrue(low.contains("no significant suspicious characteristics"))
    }

    @Test
    fun `signals are only emitted for features actually present`() {
        val none = RiskAssessment.signalsFor(emptyMap())
        assertTrue(none.isEmpty(), "signals invented from no data: $none")

        val https = RiskAssessment.signalsFor(mapOf("is_https" to 1.0))
        assertEquals(listOf("✓ HTTPS enabled"), https)

        val ip = RiskAssessment.signalsFor(mapOf("has_ip" to 1.0, "is_https" to 0.0))
        assertEquals(1, ip.size)
        assertTrue(ip[0].contains("raw IP address"))
    }

    @Test
    fun `signals never use proof language`() {
        val all = RiskAssessment.signalsFor(mapOf(
            "is_https" to 1.0, "has_ip" to 1.0, "has_at_symbol" to 1.0,
            "is_punycode" to 1.0, "has_confusable_chars" to 1.0,
            "has_suspicious_port" to 1.0, "is_shortened" to 1.0,
            "brand_outside_domain" to 1.0, "has_double_slash_redirect" to 1.0,
            "suspicious_word_count" to 3.0,
        ))
        assertEquals(10, all.size)
        for (s in all) {
            val l = s.lowercase()
            for (w in listOf("proof", "confirmed", "definitely", "guaranteed")) {
                assertFalse(l.contains(w), "signal overstates: $s")
            }
        }
    }

    @Test
    fun `no continuous threshold is invented on device`() {
        /* "Unusually long" requires the benign-corpus percentiles that live
           server-side. The Android layer must not guess a cutoff. */
        val huge = RiskAssessment.signalsFor(mapOf(
            "url_length" to 5000.0, "num_hyphens" to 40.0, "num_dots" to 30.0,
        ))
        assertTrue(huge.isEmpty(), "device invented a length/count threshold: $huge")
    }
}
