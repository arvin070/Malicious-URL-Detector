package com.aics.urlguard

import java.util.zip.CRC32
import kotlin.math.ln

/**
 * Kotlin port of `src/features.py` -- all 122 features, in the same order.
 *
 * CONTRACT
 * --------
 * For every URL, `extract(url)` must return exactly what Python's
 * `extract_features(url)` returns. `FeatureParityTest` asserts this against 625
 * golden vectors (600 sampled from the real corpus + 25 hand-picked parser
 * breakers) and fails the build on any drift. If you change this file, you must
 * re-run `python src/export_onnx.py` and keep that test green -- a feature that
 * quietly disagrees with training produces a model that is quietly wrong.
 *
 * SAFETY: pure string manipulation. Nothing is resolved, fetched or connected
 * to -- the same guarantee the Python extractor makes, enforced here by
 * `FeatureNetworkIsolationTest`.
 *
 * PYTHON QUIRKS REPRODUCED DELIBERATELY
 * -------------------------------------
 * `features.py` computes `host = netloc.split(":")[0]`, which does NOT strip
 * userinfo. So for "http://a@b.com" the host is "a@b.com", and for
 * "http://user:pass@evil.com" the host is literally "user". Those are the
 * values the model was trained on, so `parseUrl` reproduces them rather than
 * "fixing" them. tldextract, which IS userinfo-aware, still resolves the real
 * registered domain -- the two disagree by design.
 */
object UrlFeatures {

    private val SUSPICIOUS_WORDS = listOf(
        "login", "verify", "update", "secure", "account", "banking",
        "confirm", "signin", "webscr", "ebayisapi", "paypal", "free",
        "bonus", "gift", "urgent", "suspend", "password", "wallet",
    )

    private val SHORTENER_DOMAINS = setOf(
        "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd",
        "buff.ly", "adf.ly", "shorte.st", "rebrand.ly",
    )

    private val TARGET_BRANDS = listOf(
        "paypal", "apple", "microsoft", "google", "amazon", "facebook", "netflix",
        "chase", "wellsfargo", "bankofamerica", "ebay", "dropbox", "linkedin",
        "instagram", "whatsapp", "outlook", "office365", "icloud", "steam", "irs",
    )

    private val COMMON_PORTS = setOf("", "80", "443", "8080", "8443")

    /** Mirrors ABUSE_PRONE_TLDS in src/features.py. Static domain knowledge,
     *  never fitted from labels -- a per-TLD phishing rate would leak them. */
    private val ABUSE_PRONE_TLDS = setOf(
        "tk", "ml", "ga", "cf", "gq",
        "xyz", "top", "work", "click", "link", "buzz", "icu", "cyou", "rest",
        "quest", "monster", "bar", "best", "cam", "surf", "fit", "beauty",
        "sbs", "cfd", "lol", "makeup", "hair", "skin", "mom", "store",
        "zip", "mov",
    )

    /** Mirrors _EXECUTABLE_EXT in src/features.py. */
    private val EXECUTABLE_EXT = Regex(
        "\\.(exe|apk|scr|bat|cmd|com|pif|msi|jar|vbs|js|ps1|hta|dll|" +
            "zip|rar|7z|gz|iso|img|bin|sh|elf)$",
        RegexOption.IGNORE_CASE)

    private val VOWELS = "aeiou".toSet()
    private val CONSONANTS = "bcdfghjklmnpqrstvwxyz".toSet()

    private val HEX_RE = Regex("%[0-9a-fA-F]{2}")
    private val TOKEN_SPLIT = Regex("[^a-zA-Z0-9]+")
    private val SCHEME_RE = Regex("^[a-zA-Z][a-zA-Z0-9+.\\-]*://")

    /**
     * Schemes for which Python's `urlparse` splits ";params" off the last path
     * segment (urllib.parse.uses_params). `urlsplit` does NOT do this, and
     * features.py calls `urlparse`, so the port has to.
     */
    private val PARAM_SCHEMES = setOf(
        "", "ftp", "hdl", "prospero", "http", "imap", "https", "shttp",
        "rtsp", "rtspu", "sip", "sips", "mms", "sftp", "tel")
    private val SPECIAL_RE = Regex("[^a-zA-Z0-9./:-]")
    private val QUERY_SPECIAL_RE = Regex("[^a-zA-Z0-9=&_.\\-]")
    private val REPEATED_SEP_RE = Regex("(//|\\.\\.|--|__|&&|\\?\\?)")
    private val FILE_EXT_RE = Regex("\\.[a-zA-Z0-9]{2,5}$")
    private val HEX_SEG_RE = Regex("\\b[0-9a-f]{16,}\\b")
    private val B64_SEG_RE = Regex("[A-Za-z0-9+/]{24,}={0,2}")

    const val CHAR_NGRAM_N = 3
    const val CHAR_NGRAM_BUCKETS = 24
    const val TOKEN_NGRAM_BUCKETS = 12

    // ------------------------------------------------------------ helpers

    private fun ratio(n: Number, d: Number): Double {
        val dd = d.toDouble()
        return if (dd != 0.0) n.toDouble() / dd else 0.0
    }

    /** Python's `shannon_entropy`: base-2 entropy over character frequencies. */
    fun shannonEntropy(s: String): Double {
        if (s.isEmpty()) return 0.0
        val counts = HashMap<Char, Int>()
        for (c in s) counts[c] = (counts[c] ?: 0) + 1
        val n = s.length.toDouble()
        var acc = 0.0
        for (c in counts.values) {
            val p = c / n
            acc -= p * (ln(p) / LN2)
        }
        return acc
    }

    private val LN2 = ln(2.0)

    private val IPV4_SIMPLE = Regex("^\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}$")

    /** Python's `has_ip_address`: `re.match` is a PREFIX match, but the pattern
     *  is anchored with `$`, so this is a full match. */
    fun hasIpAddress(host: String): Int =
        if (IPV4_SIMPLE.matches(host)) 1 else 0

    fun countSuspiciousWords(url: String): Int {
        val l = url.lowercase()
        return SUSPICIOUS_WORDS.count { l.contains(it) }
    }

    private fun maxRun(s: String, charset: Set<Char>): Int {
        var best = 0
        var cur = 0
        for (ch in s) {
            cur = if (ch in charset) cur + 1 else 0
            if (cur > best) best = cur
        }
        return best
    }

    private fun tokens(s: String): List<String> =
        s.split(TOKEN_SPLIT).filter { it.isNotEmpty() }

    /**
     * Python's `_is_confusable`. Returns 1 for any non-ASCII character that
     * either has no Unicode name (Python's `unicodedata.name` raises) or is a
     * Cyrillic/Greek/Armenian LETTER -- the scripts that supply Latin
     * look-alikes for homograph attacks.
     */
    internal fun isConfusable(s: String): Int {
        var i = 0
        while (i < s.length) {
            val cp = s.codePointAt(i)
            i += Character.charCount(cp)
            if (cp < 128) continue

            // unicodedata.name() raises for these -> Python returns 1.
            if (cp in 0x80..0x9F) return 1
            when (Character.getType(cp).toByte()) {
                Character.UNASSIGNED,
                Character.SURROGATE,
                Character.PRIVATE_USE,
                Character.CONTROL -> return 1
            }

            val script = try {
                Character.UnicodeScript.of(cp)
            } catch (_: IllegalArgumentException) {
                return 1
            }
            if (Character.isLetter(cp) && (
                    script == Character.UnicodeScript.CYRILLIC ||
                        script == Character.UnicodeScript.GREEK ||
                        script == Character.UnicodeScript.ARMENIAN
                    )
            ) return 1
        }
        return 0
    }

    fun stripScheme(u: String): String = SCHEME_RE.replace(u, "")

    /**
     * CRC32 bucket. Python uses `zlib.crc32`; Java's `java.util.zip.CRC32` is
     * the same CRC-32/ISO-HDLC, so the two agree bit for bit. Python's own
     * `hash()` is salted per process and would differ between runs, which is
     * why neither side uses it.
     */
    private fun bucket(s: String, mod: Int): Int {
        val crc = CRC32()
        crc.update(s.toByteArray(Charsets.UTF_8))
        return (crc.value % mod).toInt()
    }

    private fun hashedCharNgrams(s: String): DoubleArray {
        val counts = IntArray(CHAR_NGRAM_BUCKETS)
        // Python slices by CODE UNIT on str (code points). Kotlin strings are
        // UTF-16, so iterate code points to keep the windows identical for
        // non-BMP input.
        val cps = s.codePoints().toArray()
        val total = maxOf(cps.size - CHAR_NGRAM_N + 1, 0)
        for (i in 0 until total) {
            val gram = String(cps, i, CHAR_NGRAM_N)
            counts[bucket(gram, CHAR_NGRAM_BUCKETS)]++
        }
        val out = DoubleArray(CHAR_NGRAM_BUCKETS)
        if (total == 0) return out
        for (i in counts.indices) out[i] = counts[i].toDouble() / total
        return out
    }

    private fun hashedTokenBigrams(toks: List<String>): DoubleArray {
        val counts = IntArray(TOKEN_NGRAM_BUCKETS)
        val total = maxOf(toks.size - 1, 0)
        for (i in 0 until total) {
            counts[bucket(toks[i] + "|" + toks[i + 1], TOKEN_NGRAM_BUCKETS)]++
        }
        val out = DoubleArray(TOKEN_NGRAM_BUCKETS)
        if (total == 0) return out
        for (i in counts.indices) out[i] = counts[i].toDouble() / total
        return out
    }

    // ------------------------------------------------------- URL parsing

    private data class Parsed(
        val scheme: String,
        val netloc: String,
        val path: String,
        val query: String,
        val fragment: String,
    )

    /**
     * True when CPython's `urlsplit` would raise ValueError on this netloc, so
     * `features.py` falls back to its `_NullParse`.
     *
     * CPython raises when brackets are unbalanced, or when the bracketed host
     * is not a valid IPv6 literal (a bracketed IPv4 is rejected too). Getting
     * this wrong only shows up on a handful of hostile URLs -- which is exactly
     * why the golden fixture carries them.
     */
    private fun urlparseWouldRaise(netloc: String): Boolean {
        val hasOpen = netloc.contains('[')
        val hasClose = netloc.contains(']')
        if (hasOpen != hasClose) return true
        if (!hasOpen) return false
        val bracketed = netloc.substringAfter('[').substringBefore(']')
        if (bracketed.startsWith("v")) return false   // IPvFuture, accepted
        return !TldExtract.looksLikeIpv6(bracketed)
    }

    /**
     * Equivalent of Python's `urlparse` for the subset `features.py` uses,
     * including its `_NullParse` fallback. Python splits fragment first, then
     * query, then netloc, then path.
     *
     * The fallback differs in exactly one observable way: `_NullParse` builds
     * the path as `"/" + rest if rest else ""`, so a URL ending right after the
     * host's slash gets path "" where `urlparse` gives "/".
     */
    private fun parseUrl(raw: String): Parsed {
        var rest = raw
        var scheme = ""
        val schemeSep = rest.indexOf("://")
        if (schemeSep >= 0) {
            val cand = rest.substring(0, schemeSep)
            if (cand.isNotEmpty() && cand[0].isLetter() &&
                cand.all { it.isLetterOrDigit() || it == '+' || it == '-' || it == '.' }
            ) {
                scheme = cand.lowercase()
                rest = rest.substring(schemeSep + 3)
            }
        }
        val frag: String
        val afterFrag: String
        val hashIdx = rest.indexOf('#')
        if (hashIdx >= 0) {
            afterFrag = rest.substring(0, hashIdx); frag = rest.substring(hashIdx + 1)
        } else {
            afterFrag = rest; frag = ""
        }
        val query: String
        val afterQuery: String
        val qIdx = afterFrag.indexOf('?')
        if (qIdx >= 0) {
            afterQuery = afterFrag.substring(0, qIdx); query = afterFrag.substring(qIdx + 1)
        } else {
            afterQuery = afterFrag; query = ""
        }
        val slashIdx = afterQuery.indexOf('/')
        val netloc: String
        var path: String
        if (slashIdx >= 0) {
            netloc = afterQuery.substring(0, slashIdx)
            path = afterQuery.substring(slashIdx)
        } else {
            netloc = afterQuery
            path = ""
        }
        if (urlparseWouldRaise(netloc)) {
            // _NullParse: `"/" + path if path else ""` -- an empty remainder
            // yields an empty path, not "/". It also does NO ";params"
            // handling, so the split below must be skipped on this branch.
            if (path == "/") path = ""
            return Parsed(scheme, netloc, path, query, frag)
        }

        // urlparse() -- unlike urlsplit() -- removes ";params" from the LAST
        // path segment. Missed on the first pass, and it only shows up on URLs
        // that contain a semicolon: a feed URL carrying a literal "&amp;"
        // made Python report path_length 29 where this port said 143.
        // CPython's _splitparams searches from the final '/'.
        if (scheme in PARAM_SCHEMES && path.contains(';')) {
            val from = path.lastIndexOf('/').let { if (it >= 0) it else 0 }
            val i = path.indexOf(';', from)
            if (i >= 0) path = path.substring(0, i)
        }
        return Parsed(scheme, netloc, path, query, frag)
    }

    // ------------------------------------------------------------ main

    /**
     * Returns the 122 features keyed by name, in the same insertion order as
     * `features.py`. Callers should still index by name (see [UrlClassifier])
     * so that order can never silently diverge from the trained model.
     */
    fun extract(rawUrl: String, tld: TldExtract): LinkedHashMap<String, Double> {
        val url = rawUrl.trim()
        val parsed = parseUrl(if (url.contains("://")) url else "http://$url")
        val netloc = parsed.netloc
        val host = netloc.substringBefore(':')
        // Python: `netloc.split(":")[1] if ":" in netloc else ""`.
        // For "[::1]:8080" that is the empty segment between the first two
        // colons, NOT "8080" -- reproduce it rather than "fixing" it.
        val port = if (netloc.contains(':')) netloc.split(":")[1] else ""
        val ext = tld.extract(url)
        val registeredDomain = ext.registeredDomain

        val path = parsed.path
        val query = parsed.query
        val fragment = parsed.fragment
        val urlL = url.lowercase()
        val hostL = host.lowercase()
        val domain = ext.domain
        val subdomain = ext.subdomain
        val suffix = ext.suffix

        val f = LinkedHashMap<String, Double>(192)
        fun put(k: String, v: Number) { f[k] = v.toDouble() }
        fun put(k: String, v: Boolean) { f[k] = if (v) 1.0 else 0.0 }

        val numDigits = url.count { it.isDigit() }
        val numSpecial = SPECIAL_RE.findAll(url).count()

        // ---- ORIGINAL 21 ----
        put("url_length", url.length)
        put("host_length", host.length)
        put("path_length", path.length)
        put("num_dots", url.count { it == '.' })
        put("num_hyphens", url.count { it == '-' })
        put("num_underscores", url.count { it == '_' })
        put("num_slashes", url.count { it == '/' })
        put("num_digits", numDigits)
        put("num_special_chars", numSpecial)
        put("num_subdomains", if (subdomain.isNotEmpty()) subdomain.split(".").size else 0)
        put("has_ip", hasIpAddress(host))
        put("has_at_symbol", url.contains("@"))
        put("has_double_slash_redirect", path.contains("//"))
        put("is_https", parsed.scheme == "https")
        put("is_shortened", registeredDomain in SHORTENER_DOMAINS)
        put("suspicious_word_count", countSuspiciousWords(url))
        put("entropy", shannonEntropy(host))
        put("query_length", query.length)
        put("num_params", if (query.isNotEmpty()) query.count { it == '=' } else 0)
        put("tld_length", suffix.length)
        put("digit_ratio_in_host",
            if (host.isNotEmpty()) host.count { it.isDigit() }.toDouble() / host.length else 0.0)

        // ---- component lengths ----
        put("fragment_length", fragment.length)
        put("domain_length", domain.length)
        put("subdomain_length", subdomain.length)
        put("netloc_length", netloc.length)

        // ---- per-component separator / special-char counts ----
        val pathNumDigits = path.count { it.isDigit() }
        put("host_num_dots", host.count { it == '.' })
        put("host_num_hyphens", host.count { it == '-' })
        put("host_num_digits", host.count { it.isDigit() })
        put("path_num_dots", path.count { it == '.' })
        put("path_num_hyphens", path.count { it == '-' })
        put("path_num_digits", pathNumDigits)
        put("domain_num_hyphens", domain.count { it == '-' })
        put("query_num_special", QUERY_SPECIAL_RE.findAll(query).count())

        // ---- query / path structure ----
        put("path_depth", path.split("/").count { it.isNotEmpty() })
        put("num_query_params", if (query.isNotEmpty()) query.split("&").count { it.isNotEmpty() } else 0)
        put("max_param_value_length",
            query.split("&").filter { it.contains("=") }
                .maxOfOrNull { it.substringAfter("=").length } ?: 0)
        put("has_fragment", fragment.isNotEmpty())
        put("path_has_file_ext", FILE_EXT_RE.containsMatchIn(path))

        // ---- encoding / obfuscation ----
        put("num_percent_encoded", HEX_RE.findAll(url).count())
        put("num_percent_signs", url.count { it == '%' })
        val noScheme = stripScheme(url)
        put("repeated_separators", REPEATED_SEP_RE.findAll(noScheme).count())
        put("num_equals", url.count { it == '=' })
        put("num_ampersands", url.count { it == '&' })
        put("num_question_marks", url.count { it == '?' })
        put("has_hex_like_segment", HEX_SEG_RE.containsMatchIn(urlL))
        put("has_base64_like_segment", B64_SEG_RE.containsMatchIn(url))

        // ---- port ----
        put("has_port", port.isNotEmpty())
        put("has_suspicious_port", port.isNotEmpty() && port !in COMMON_PORTS)

        // ---- internationalisation / homograph ----
        put("is_punycode", hostL.contains("xn--"))
        put("has_non_ascii", url.any { it.code > 127 })
        put("has_confusable_chars", isConfusable(url))

        // ---- entropy, per component ----
        put("host_entropy", shannonEntropy(host))
        put("path_entropy", shannonEntropy(path))
        put("query_entropy", shannonEntropy(query))
        put("domain_entropy", shannonEntropy(domain))
        put("url_entropy", shannonEntropy(url))

        // ---- character-class ratios ----
        val lettersUrl = url.count { it.isLetter() }
        put("digit_ratio_in_url", ratio(numDigits, url.length))
        put("letter_ratio_in_url", ratio(lettersUrl, url.length))
        put("letter_ratio_in_host", ratio(host.count { it.isLetter() }, host.length))
        put("digit_ratio_in_path", ratio(pathNumDigits, path.length))
        put("special_ratio_in_url", ratio(numSpecial, url.length))
        put("path_ratio_of_url", ratio(path.length, url.length))
        put("query_ratio_of_url", ratio(query.length, url.length))
        put("host_ratio_of_url", ratio(host.length, url.length))

        // ---- domain / subdomain structure ----
        val subParts = subdomain.split(".").filter { it.isNotEmpty() }
        put("subdomain_depth", subParts.size)
        put("longest_subdomain_length", subParts.maxOfOrNull { it.length } ?: 0)
        put("has_www", subParts.isNotEmpty() && subParts[0].lowercase() == "www")
        put("tld_in_subdomain", suffix.isNotEmpty() && subdomain.contains(suffix))
        put("tld_in_path", suffix.isNotEmpty() && path.lowercase().contains(".$suffix"))
        put("domain_is_digits_only",
            domain.isNotEmpty() && domain.all { it.isDigit() })

        // ---- brand-abuse signals ----
        val brandInUrl = TARGET_BRANDS.filter { urlL.contains(it) }
        put("brand_in_url", brandInUrl.size)
        put("brand_in_subdomain", TARGET_BRANDS.any { subdomain.lowercase().contains(it) })
        put("brand_in_path", TARGET_BRANDS.any { path.lowercase().contains(it) })
        put("brand_outside_domain",
            brandInUrl.isNotEmpty() && !TARGET_BRANDS.any { domain.lowercase().contains(it) })
        put("suspicious_words_in_host", SUSPICIOUS_WORDS.count { hostL.contains(it) })
        put("suspicious_words_in_path", SUSPICIOUS_WORDS.count { path.lowercase().contains(it) })

        // ---- token statistics ----
        val urlTokens = tokens(urlL)
        val hostTokens = tokens(hostL)
        put("num_tokens", urlTokens.size)
        put("mean_token_length", ratio(urlTokens.sumOf { it.length }, urlTokens.size))
        put("max_token_length", urlTokens.maxOfOrNull { it.length } ?: 0)
        put("num_host_tokens", hostTokens.size)
        put("max_host_token_length", hostTokens.maxOfOrNull { it.length } ?: 0)

        val dom = domain.lowercase()
        put("domain_vowel_ratio", ratio(dom.count { it in VOWELS }, dom.length))
        put("domain_max_consonant_run", maxRun(dom, CONSONANTS))
        put("domain_digit_letter_switches",
            (0 until maxOf(dom.length - 1, 0)).count { dom[it].isDigit() != dom[it + 1].isDigit() })
        put("url_char_repeat_max",
            if (urlL.isEmpty()) 0 else urlL.groupingBy { it }.eachCount().values.max())
        put("host_hyphen_groups",
            if (hostL.contains("-")) hostL.split("-").count { it.isNotEmpty() } - 1 else 0)

        // ---- TLD and payload indicators ----
        put("tld_abuse_prone",
            suffix.lowercase().substringAfterLast('.') in ABUSE_PRONE_TLDS)
        put("path_has_executable_ext", EXECUTABLE_EXT.containsMatchIn(path))

        // ---- hashed n-grams ----
        val ngSrc = stripScheme(urlL)
        hashedCharNgrams(ngSrc).forEachIndexed { i, v ->
            f["cng3_%02d".format(i)] = v
        }
        hashedTokenBigrams(tokens(ngSrc)).forEachIndexed { i, v ->
            f["tng2_%02d".format(i)] = v
        }

        return f
    }
}
