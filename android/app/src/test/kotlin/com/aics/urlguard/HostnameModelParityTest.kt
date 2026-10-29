package com.aics.urlguard

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.BeforeAll
import org.junit.jupiter.api.Test
import java.io.File
import kotlin.math.abs

/**
 * Parity for the DNS layer's hostname-only model.
 *
 * Kept separate from [InferenceParityTest] because it is a DIFFERENT model with
 * a different threshold, and because its fixture is built from bare hostnames
 * -- what a DNS query actually carries. Testing it on full URLs would exercise
 * input it never sees in production.
 *
 * This layer can block name resolution device-wide, so a drift here does not
 * merely mispredict: it takes domains off the air for the user.
 */
class HostnameModelParityTest {

    companion object {
        private lateinit var clf: UrlClassifier
        private lateinit var cases: List<FeatureParityTest.Companion.Case>

        private val assets: File by lazy {
            listOf(File("src/main/assets/hostname"),
                   File("app/src/main/assets/hostname"),
                   File("android/app/src/main/assets/hostname"))
                .first { it.isDirectory }
        }
        private val golden: File by lazy {
            listOf(File("src/test/resources/hostname"),
                   File("app/src/test/resources/hostname"),
                   File("android/app/src/test/resources/hostname"))
                .first { it.isDirectory }
        }
        private val psl: File by lazy {
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
                suffixListLines = File(psl, "public_suffix_list.txt").readLines().asSequence(),
                featureNames = names,
                threshold = thr,
                modelName = "XGBoost-hostname",
            )
            cases = FeatureParityTest.parseGoldenPublic(
                File(golden, "golden_vectors.json").readText()
            )
        }
    }

    @Test
    fun `serves at the conservative dns threshold`() {
        // 0.95, NOT the 0.65 the training rule picked. At 0.65 the measured FPR
        // is 0.0505 -- roughly 1 benign domain in 20 would stop resolving.
        assertEquals(0.95, clf.threshold, 1e-9,
            "DNS threshold changed; re-check the FPR this implies")
    }

    @Test
    fun `features match python on every golden hostname`() {
        val tld = TldExtract.fromSuffixLines(
            File(psl, "public_suffix_list.txt").readLines().asSequence()
        )
        var bad = 0
        val detail = StringBuilder()
        for (c in cases) {
            val got = UrlFeatures.extract(c.url, tld)
            val names = got.keys.toList()
            for (i in names.indices) {
                val e = c.features[i]
                val a = got[names[i]]!!
                if (abs(e - a) > 1e-9 * maxOf(1.0, abs(e))) {
                    bad++
                    if (detail.length < 2000) {
                        detail.append("\n${c.url}: ${names[i]} python=$e kotlin=$a")
                    }
                }
            }
        }
        assertEquals(0, bad, "hostname feature drift:$detail")
    }

    @Test
    fun `scores match python on every golden hostname`() {
        var worst = 0.0
        var worstHost = ""
        var flips = 0
        for (c in cases) {
            val v = clf.classify(c.url)
            val d = abs(v.score - c.proba)
            if (d > worst) { worst = d; worstHost = c.url }
            if ((v.score >= clf.threshold) != (c.proba >= clf.threshold)) flips++
        }
        assertTrue(worst < 1e-5, "max delta $worst on '$worstHost'")
        assertEquals(0, flips, "$flips DNS verdicts differ from Python")
    }

    @Test
    fun `common hostnames are not blocked at the serving threshold`() {
        // Not an accuracy claim -- a smoke check that the severe threshold does
        // what it is for. These are everyday hosts; blocking any of them would
        // be an immediately visible outage for the user.
        val everyday = listOf(
            "google.com", "www.google.com", "wikipedia.org", "en.wikipedia.org",
            "github.com", "amazon.in", "youtube.com", "gmail.com",
            "stackoverflow.com", "microsoft.com", "apple.com", "cloudflare.com",
            "whatsapp.com", "instagram.com", "netflix.com", "paypal.com",
        )
        val blocked = everyday.filter { clf.classify(it).malicious }
        assertTrue(blocked.isEmpty(),
            "DNS layer would block everyday hostnames: $blocked")
    }

    @Test
    fun `scoring a hostname never throws and stays in range`() {
        val hostile = listOf("", " ", ".", "..", "a", "xn--80ak6aa92e.com",
                             "1.2.3.4", "[::1]", "h".repeat(300), "-.-")
        for (h in hostile) {
            val s = clf.classify(h).score
            assertTrue(s in 0.0..1.0 && s.isFinite(), "bad score for '$h': $s")
        }
    }
}
