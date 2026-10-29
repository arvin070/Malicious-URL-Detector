package com.aics.urlguard

import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Conservative presentation of a model score, mirroring `webapp/assessment.py`
 * so the phone and the web UI say the same thing about the same URL.
 *
 * The rules that matter:
 *   - a score is a RISK BAND, never a verdict;
 *   - the UI never displays 0 or 100, because a tree ensemble is never certain;
 *   - confidence describes how far the score is from the decision threshold,
 *     not a calibrated likelihood;
 *   - a signal is only shown when the measurement is actually present.
 */
object RiskAssessment {

    enum class Band(val label: String, val description: String) {
        LOW("LOW RISK",
            "No significant suspicious characteristics were detected."),
        SUSPICIOUS("SUSPICIOUS",
            "Some characteristics associated with potentially harmful URLs " +
                "were detected."),
        HIGH("HIGH RISK",
            "Multiple characteristics associated with potentially harmful URLs " +
                "were detected. Exercise caution before proceeding."),
    }

    const val DISCLAIMER =
        "This assessment is based on machine-learning analysis and available " +
            "security signals. It is not a guarantee that the website is malicious."

    data class Result(
        val score: Int,
        val band: Band,
        val confidence: Double,
        val confidenceLabel: String,
        val signals: List<String>,
    )

    /** 0-100, never the endpoints: the model cannot be certain. */
    fun displayScore(proba: Double): Int {
        val s = (proba * 100).roundToInt()
        return if (proba > 0.0 && proba < 1.0) min(99, max(1, s)) else s
    }

    fun bandFor(score: Int): Band = when {
        score <= 30 -> Band.LOW
        score <= 70 -> Band.SUSPICIOUS
        else -> Band.HIGH
    }

    /** Distance from the decision threshold, 0..1. See the web layer for why. */
    fun decisionMargin(proba: Double, threshold: Double): Double {
        if (threshold <= 0.0 || threshold >= 1.0) return kotlin.math.abs(proba - 0.5) * 2
        return if (proba >= threshold) (proba - threshold) / (1.0 - threshold)
        else (threshold - proba) / threshold
    }

    fun marginLabel(margin: Double): String = when {
        margin >= 0.70 -> "Decisive"
        margin >= 0.35 -> "Moderate"
        else -> "Borderline"
    }

    /**
     * Plain-language signals for the features actually observed.
     *
     * Only properties whose presence is itself the observation are listed --
     * an IP literal, an '@', punycode. Continuous "unusually long / unusually
     * many" signals are deliberately NOT included: deciding what counts as
     * unusual requires the benign-corpus percentiles that live server-side,
     * and inventing a cutoff here would manufacture evidence.
     */
    fun signalsFor(feats: Map<String, Double>): List<String> {
        val out = ArrayList<String>()
        fun on(k: String) = (feats[k] ?: 0.0) != 0.0

        if (on("is_https")) out.add("✓ HTTPS enabled")
        if (on("has_ip")) out.add("⚠ Host is a raw IP address rather than a domain name")
        if (on("has_at_symbol")) out.add("⚠ URL contains '@', which can hide the real destination")
        if (on("is_punycode")) out.add("⚠ Host uses punycode / internationalised characters")
        if (on("has_confusable_chars")) out.add("⚠ URL mixes scripts that can imitate Latin letters")
        if (on("has_suspicious_port")) out.add("⚠ URL specifies an uncommon network port")
        if (on("is_shortened")) out.add("⚠ Host is a known URL shortener, hiding the destination")
        if (on("brand_outside_domain")) out.add("⚠ A well-known brand name appears outside the registered domain")
        if (on("has_double_slash_redirect")) out.add("⚠ Path contains '//', used in redirect tricks")

        val kw = (feats["suspicious_word_count"] ?: 0.0).toInt()
        if (kw > 0) out.add("⚠ Suspicious keyword pattern detected ($kw matches)")

        return out
    }

    fun of(url: String, verdict: UrlClassifier.Verdict,
           feats: Map<String, Double>): Result {
        val score = displayScore(verdict.score)
        val margin = decisionMargin(verdict.score, verdict.threshold)
        return Result(
            score = score,
            band = bandFor(score),
            confidence = (margin * 100).coerceIn(0.1, 99.9),
            confidenceLabel = marginLabel(margin),
            signals = signalsFor(feats),
        )
    }
}
