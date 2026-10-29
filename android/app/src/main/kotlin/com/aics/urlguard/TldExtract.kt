package com.aics.urlguard

import java.net.IDN

/**
 * Kotlin port of the exact `tldextract` behaviour that `src/features.py` relies on.
 *
 * WHY A PORT AND NOT A LIBRARY
 * ----------------------------
 * The model's features (`num_subdomains`, `tld_length`, `domain_entropy`, the
 * brand-abuse signals, ...) are all defined in terms of tldextract's split of a
 * URL into subdomain / domain / suffix. Any library that splits even slightly
 * differently would feed the model different numbers from the ones it was
 * trained on -- silently, with no error, just worse predictions.
 *
 * So this is a deliberate line-by-line port of tldextract's `lenient_netloc`,
 * `_extract_netloc` and `Trie.suffix_index`, and the suffix list it reads is
 * dumped straight out of the live Python extractor (see
 * `android/app/src/main/assets/public_suffix_list.txt`) rather than fetched
 * separately. `FeatureParityTest` asserts the result against golden vectors
 * produced by the Python code.
 *
 * Quirks that are reproduced ON PURPOSE, because the model was trained on them:
 *   - labels are matched case-insensitively but RETURNED with original case,
 *     so "A.B.COM" -> subdomain "A", domain "B", suffix "COM";
 *   - an IPv4 literal becomes domain="1.2.3.4" with an empty suffix;
 *   - a bracketed IPv6 literal becomes domain="[::1]" with an empty suffix.
 */
data class ExtractResult(
    val subdomain: String,
    val domain: String,
    val suffix: String,
) {
    /** eTLD+1, the grouping key used throughout the Python pipeline. */
    val registeredDomain: String
        get() = listOf(domain, suffix).filter { it.isNotEmpty() }.joinToString(".")
}

private class TrieNode {
    val matches = HashMap<String, TrieNode>()
    var end = false
}

class TldExtract private constructor(private val root: TrieNode) {

    companion object {
        private val IPV4_RE = Regex(
            "^(?:(?:[0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\\.)" +
                "{3}(?:[0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])$"
        )

        // urllib.parse.scheme_chars
        private val SCHEME_CHARS: Set<Char> =
            ("abcdefghijklmnopqrstuvwxyz" +
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ" +
                "0123456789" +
                "+-.").toSet()

        fun fromSuffixLines(lines: Sequence<String>): TldExtract {
            val root = TrieNode()
            for (raw in lines) {
                val line = raw.trim()
                if (line.isEmpty() || line.startsWith("#")) continue
                // tldextract's Trie.add_suffix: split on ".", reverse, insert.
                var node = root
                for (label in line.split(".").asReversed()) {
                    node = node.matches.getOrPut(label) { TrieNode() }
                }
                node.end = true
            }
            return TldExtract(root)
        }

        /** tldextract.remote._schemeless_url */
        internal fun schemelessUrl(url: String): String {
            val i = url.indexOf("//")
            if (i == 0) return url.substring(2)
            if (i < 2 || url[i - 1] != ':' ||
                url.substring(0, i - 1).any { it !in SCHEME_CHARS }
            ) return url
            return url.substring(i + 2)
        }

        /** tldextract.remote.lenient_netloc */
        internal fun lenientNetloc(url: String): String {
            val afterUserinfo = schemelessUrl(url)
                .substringBefore('/')
                .substringBefore('?')
                .substringBefore('#')
                .substringAfterLast('@')

            if (afterUserinfo.isNotEmpty() && afterUserinfo[0] == '[') {
                val close = afterUserinfo.indexOf(']')
                if (close >= 0) return afterUserinfo.substring(0, close + 1)
            }
            val hostname = afterUserinfo.substringBefore(':').trim()
            // rstrip(".。．｡")
            return hostname.trimEnd('.', '。', '．', '｡')
        }

        fun looksLikeIp(s: String): Boolean {
            if (s.isEmpty() || !s[0].isDigit()) return false
            return IPV4_RE.matches(s)
        }

        fun looksLikeIpv6(s: String): Boolean {
            if (s.isEmpty()) return false
            // Mirrors ipaddress.IPv6Address acceptance closely enough for the
            // netloc test: hex groups, optional "::" compression, optional
            // trailing IPv4. Rejects zone ids, which IPv6Address also rejects.
            if (s.contains('%')) return false
            val parts = s.split("::")
            if (parts.size > 2) return false
            fun groupsOk(g: String, allowV4: Boolean): Pair<Boolean, Int> {
                if (g.isEmpty()) return true to 0
                val items = g.split(":")
                var count = 0
                for ((idx, it) in items.withIndex()) {
                    if (allowV4 && idx == items.size - 1 && it.contains('.')) {
                        if (!looksLikeIp(it)) return false to 0
                        count += 2
                        continue
                    }
                    if (it.isEmpty() || it.length > 4) return false to 0
                    if (!it.all { c -> c.isDigit() || c in 'a'..'f' || c in 'A'..'F' }) {
                        return false to 0
                    }
                    count += 1
                }
                return true to count
            }
            return if (parts.size == 2) {
                val (okA, a) = groupsOk(parts[0], false)
                val (okB, b) = groupsOk(parts[1], true)
                okA && okB && (a + b) <= 7
            } else {
                val (ok, n) = groupsOk(parts[0], true)
                ok && n == 8
            }
        }

        /** tldextract._decode_punycode */
        internal fun decodePunycode(label: String): String {
            val lowered = label.lowercase()
            if (!lowered.startsWith("xn--")) return lowered
            return try {
                IDN.toUnicode(lowered, IDN.ALLOW_UNASSIGNED)
            } catch (_: IllegalArgumentException) {
                lowered
            }
        }
    }

    /**
     * tldextract.TLDExtract.suffix_index -- index of the first suffix label.
     * Returns -1 when no suffix matches (tldextract returns None).
     */
    private fun suffixIndex(labels: List<String>): Int {
        var node = root
        var suffixIdx = labels.size
        var labelIdx = labels.size

        for (label in labels.asReversed()) {
            val decoded = decodePunycode(label)
            val next = node.matches[decoded]
            if (next != null) {
                labelIdx -= 1
                node = next
                if (node.end) suffixIdx = labelIdx
                continue
            }
            val wildcard = node.matches["*"]
            if (wildcard != null) {
                val isException = node.matches.containsKey("!$decoded")
                return if (isException) labelIdx else labelIdx - 1
            }
            break
        }
        return if (suffixIdx == labels.size) -1 else suffixIdx
    }

    /** tldextract.TLDExtract._extract_netloc */
    private fun extractNetloc(netloc: String): ExtractResult {
        val dots = netloc
            .replace('。', '.')
            .replace('．', '.')
            .replace('｡', '.')

        if (dots.length >= 4 && dots[0] == '[' && dots[dots.length - 1] == ']' &&
            looksLikeIpv6(dots.substring(1, dots.length - 1))
        ) {
            return ExtractResult("", dots, "")
        }

        val labels = dots.split(".")
        val idx = suffixIndex(labels)

        if (idx < 0) {
            if (labels.size == 4 && looksLikeIp(dots)) {
                return ExtractResult("", dots, "")
            }
            return ExtractResult(
                subdomain = labels.dropLast(1).joinToString("."),
                domain = labels.last(),
                suffix = "",
            )
        }

        val subdomain = if (idx >= 2) labels.subList(0, idx - 1).joinToString(".") else ""
        val domain = if (idx > 0) labels[idx - 1] else ""
        val suffix = labels.subList(idx, labels.size).joinToString(".")
        return ExtractResult(subdomain, domain, suffix)
    }

    fun extract(url: String): ExtractResult = extractNetloc(lenientNetloc(url))
}
