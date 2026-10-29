package com.aics.urlguard

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.BeforeAll
import java.io.File
import kotlin.math.abs

/**
 * Proves the Kotlin feature port agrees with `src/features.py`, URL by URL and
 * feature by feature.
 *
 * This is the test that makes the Android app trustworthy. The model is served
 * exactly as validated ONLY IF it is fed exactly the numbers it was trained on;
 * a port that is 99% right is a model that is quietly wrong, with no error
 * anywhere to notice. So every one of the 122 features is compared on every
 * golden URL, and any drift fails the build.
 *
 * Fixtures come from `python src/export_onnx.py`, which writes
 * `assets/golden_vectors.json` (600 URLs sampled from the real corpus + 25
 * hand-picked parser breakers).
 */
class FeatureParityTest {

    companion object {
        private lateinit var golden: Golden
        private lateinit var tld: TldExtract

        private val assetsDir: File by lazy {
            val candidates = listOf(
                File("src/main/assets"),
                File("app/src/main/assets"),
                File("android/app/src/main/assets"),
            )
            candidates.firstOrNull { it.isDirectory }
                ?: error("assets dir not found from ${File(".").absolutePath}")
        }

        /** Test fixtures live outside src/main/assets so they never ship. */
        val goldenDir: File by lazy {
            listOf(File("src/test/resources"), File("app/src/test/resources"),
                   File("android/app/src/test/resources"))
                .firstOrNull { it.isDirectory }
                ?: error("test resources not found from ${File(".").absolutePath}")
        }

        data class Case(val url: String, val features: List<Double>, val proba: Double)
        data class Golden(val featureNames: List<String>, val threshold: Double,
                          val cases: List<Case>)

        @BeforeAll
        @JvmStatic
        fun load() {
            tld = TldExtract.fromSuffixLines(
                File(assetsDir, "public_suffix_list.txt").readLines().asSequence()
            )
            golden = parseGolden(File(goldenDir, "golden_vectors.json").readText())
        }

        /** Minimal JSON reader -- avoids pulling a parser into unit tests. */
        fun parseGoldenPublic(text: String): List<Case> = parseGolden(text).cases

        private fun parseGolden(text: String): Golden {
            val names = Regex("\"feature_names\":\\[(.*?)]").find(text)!!
                .groupValues[1].split(",").map { it.trim().trim('"') }
            val threshold = Regex("\"threshold\":([0-9.eE+-]+)").find(text)!!
                .groupValues[1].toDouble()
            val casesBlob = text.substringAfter("\"cases\":[").dropLast(2)

            val cases = ArrayList<Case>()
            var i = 0
            while (i < casesBlob.length) {
                val start = casesBlob.indexOf("{\"url\":", i)
                if (start < 0) break
                val urlStart = start + "{\"url\":".length
                val url = readJsonString(casesBlob, urlStart)
                val featStart = casesBlob.indexOf("\"features\":[", urlStart) + "\"features\":[".length
                val featEnd = casesBlob.indexOf(']', featStart)
                val feats = casesBlob.substring(featStart, featEnd)
                    .split(",").map { it.toDouble() }
                val pStart = casesBlob.indexOf("\"proba\":", featEnd) + "\"proba\":".length
                val pEnd = casesBlob.indexOf('}', pStart)
                val proba = casesBlob.substring(pStart, pEnd).toDouble()
                cases.add(Case(url, feats, proba))
                i = pEnd + 1
            }
            return Golden(names, threshold, cases)
        }

        private fun readJsonString(s: String, quoteIdx: Int): String {
            require(s[quoteIdx] == '"')
            val sb = StringBuilder()
            var i = quoteIdx + 1
            while (i < s.length) {
                val c = s[i]
                when {
                    c == '"' -> return sb.toString()
                    c == '\\' -> {
                        when (val e = s[i + 1]) {
                            'n' -> { sb.append('\n'); i += 2 }
                            't' -> { sb.append('\t'); i += 2 }
                            'r' -> { sb.append('\r'); i += 2 }
                            'b' -> { sb.append('\b'); i += 2 }
                            'f' -> { sb.append(''); i += 2 }
                            '/' -> { sb.append('/'); i += 2 }
                            '"' -> { sb.append('"'); i += 2 }
                            '\\' -> { sb.append('\\'); i += 2 }
                            'u' -> { sb.append(s.substring(i + 2, i + 6).toInt(16).toChar()); i += 6 }
                            else -> { sb.append(e); i += 2 }
                        }
                    }
                    else -> { sb.append(c); i++ }
                }
            }
            error("unterminated JSON string")
        }
    }

    @Test
    fun `fixture loaded and covers the full feature set`() {
        assertEquals(124, golden.featureNames.size, "expected 124 features")
        assertTrue(golden.cases.size >= 500,
            "golden fixture too small: ${golden.cases.size}")
    }

    @Test
    fun `every feature matches python on every golden url`() {
        val failures = StringBuilder()
        var compared = 0
        var badUrls = 0

        for (case in golden.cases) {
            val got = UrlFeatures.extract(case.url, tld)
            var urlBad = false

            if (got.size != golden.featureNames.size) {
                failures.append("\n${case.url.take(80)}: produced ${got.size} features, " +
                    "expected ${golden.featureNames.size}")
                badUrls++
                continue
            }

            // Order must match too: the ONNX model indexes by position.
            if (got.keys.toList() != golden.featureNames) {
                val mismatch = got.keys.toList().zip(golden.featureNames)
                    .withIndex().firstOrNull { it.value.first != it.value.second }
                failures.append("\n${case.url.take(60)}: feature ORDER differs at " +
                    "index ${mismatch?.index}: ${mismatch?.value}")
                badUrls++
                continue
            }

            for ((idx, name) in golden.featureNames.withIndex()) {
                val expected = case.features[idx]
                val actual = got[name]!!
                compared++
                val tol = 1e-9 * maxOf(1.0, abs(expected))
                if (abs(expected - actual) > tol) {
                    if (failures.length < 6000) {
                        failures.append("\n${case.url.take(70)}\n    $name: " +
                            "python=$expected kotlin=$actual")
                    }
                    urlBad = true
                }
            }
            if (urlBad) badUrls++
        }

        assertTrue(failures.isEmpty(),
            "feature parity FAILED on $badUrls/${golden.cases.size} URLs " +
                "($compared comparisons):$failures")
    }

    @Test
    fun `tldextract port matches python registered domains`() {
        // Spot-checks of the quirks the port reproduces on purpose.
        assertEquals("example.co.uk",
            tld.extract("http://login.secure.example.co.uk/x").registeredDomain)
        assertEquals("example.com", tld.extract("https://a.b.example.com").registeredDomain)
        assertEquals("b.com", tld.extract("http://a@b.com").registeredDomain)
        assertEquals("evil.com",
            tld.extract("http://user:pass@evil.com/login").registeredDomain)
        // case is preserved, as in tldextract
        val up = tld.extract("http://A.B.COM/Path")
        assertEquals("A", up.subdomain)
        assertEquals("B", up.domain)
        assertEquals("COM", up.suffix)
        // IP literals: domain carries the literal, suffix is empty
        assertEquals("1.2.3.4", tld.extract("http://1.2.3.4/a").domain)
        assertEquals("", tld.extract("http://1.2.3.4/a").suffix)
    }

    @Test
    fun `ngram block is scheme invariant and sums to one`() {
        val a = UrlFeatures.extract("http://secure-paypal-login.tk/verify/9231", tld)
        val b = UrlFeatures.extract("secure-paypal-login.tk/verify/9231", tld)
        val c = UrlFeatures.extract("HTTPS://secure-paypal-login.tk/verify/9231", tld)
        for (k in a.keys.filter { it.startsWith("cng3_") || it.startsWith("tng2_") }) {
            assertEquals(a[k]!!, b[k]!!, 1e-12, k)
            assertEquals(a[k]!!, c[k]!!, 1e-12, k)
        }
        val sum = a.filterKeys { it.startsWith("cng3_") }.values.sum()
        assertEquals(1.0, sum, 1e-9)
    }

    @Test
    fun `extraction never throws on hostile input`() {
        val hostile = listOf(
            "", "   ", "a.b", "http://", "ftp://x.com/y", "http://[",
            "https://[]]/", "http://%%%/", " .com", "http://a@@b.com",
            "http://[::1]:8080/x", "h".repeat(5000), "://", "//", "?",
            "http://😀.com/😀", "x".repeat(1) + "￿",
        )
        for (u in hostile) {
            val f = UrlFeatures.extract(u, tld)
            assertEquals(124, f.size, "wrong feature count for ${u.take(20)}")
            for ((k, v) in f) {
                assertTrue(v.isFinite(), "$k not finite for ${u.take(20)}: $v")
            }
        }
    }
}
