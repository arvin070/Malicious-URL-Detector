package com.aics.urlguard

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.BeforeAll
import org.junit.jupiter.api.Test
import java.io.File
import kotlin.math.abs

/**
 * End-to-end check: URL -> Kotlin features -> bundled ONNX model -> probability,
 * compared against the probability Python produced for the same URL.
 *
 * FeatureParityTest proves the features match. This proves the whole serving
 * path matches, which is the claim that actually matters: the phone returns the
 * same verdict as the validated desktop model.
 */
class InferenceParityTest {

    companion object {
        private lateinit var clf: UrlClassifier
        private lateinit var cases: List<FeatureParityTest.Companion.Case>

        private val goldenDir: File by lazy { FeatureParityTest.goldenDir }

        private val assets: File by lazy {
            listOf(File("src/main/assets"), File("app/src/main/assets"),
                   File("android/app/src/main/assets"))
                .first { it.isDirectory }
        }

        @BeforeAll
        @JvmStatic
        fun setUp() {
            val meta = File(assets, "model_meta.json").readText()
            val names = Regex("\"feature_names\":\\s*\\[(.*?)]", RegexOption.DOT_MATCHES_ALL)
                .find(meta)!!.groupValues[1]
                .split(",").map { it.trim().trim('"') }.filter { it.isNotEmpty() }
            val thr = Regex("\"threshold\":\\s*([0-9.eE+-]+)").find(meta)!!
                .groupValues[1].toDouble()

            clf = UrlClassifier.create(
                modelBytes = File(assets, "url_model.onnx").readBytes(),
                suffixListLines = File(assets, "public_suffix_list.txt").readLines().asSequence(),
                featureNames = names,
                threshold = thr,
                modelName = "XGBoost",
            )
            cases = FeatureParityTest.Companion.parseGoldenPublic(
                File(goldenDir, "golden_vectors.json").readText()
            )
        }
    }

    @Test
    fun `scores match python on every golden url`() {
        var worst = 0.0
        var worstUrl = ""
        var flips = 0
        for (c in cases) {
            val got = clf.classify(c.url)
            val d = abs(got.score - c.proba)
            if (d > worst) { worst = d; worstUrl = c.url }
            if ((got.score >= clf.threshold) != (c.proba >= clf.threshold)) flips++
        }
        assertTrue(worst < 1e-5,
            "max score delta $worst on '${worstUrl.take(90)}' exceeds 1e-5")
        assertEquals(0, flips, "$flips verdicts differ from Python")
    }

    @Test
    fun `probabilities stay within the unit interval`() {
        val hostile = listOf("", "   ", "http://", "h".repeat(4000), "://",
                             "http://[", "https://[]]/", "http://%%%/")
        for (u in hostile + cases.take(50).map { it.url }) {
            val s = clf.classify(u).score
            assertTrue(s in 0.0..1.0, "score out of range for '${u.take(40)}': $s")
            assertTrue(s.isFinite(), "non-finite score for '${u.take(40)}'")
        }
    }

    @Test
    fun `scoring is deterministic`() {
        val u = "http://secure-paypal-login.tk/verify/9231"
        val first = clf.classify(u).score
        repeat(20) { assertEquals(first, clf.classify(u).score, 0.0) }
    }

    @Test
    fun `obvious phish outscores obvious benign`() {
        val phish = clf.classify("http://secure-paypal-login.tk/verify/9231")
        val benign = clf.classify("https://en.wikipedia.org/wiki/Phishing")
        assertTrue(phish.score > benign.score,
            "phish ${phish.score} !> benign ${benign.score}")
    }
}
