package com.aics.urlguard

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.*
import org.json.JSONObject
import kotlin.concurrent.thread

/**
 * Manual scanner and result screen.
 *
 * Scoring is LOCAL first: the bundled ONNX model answers immediately and works
 * with no network and no account. When the user is signed in, the same URL is
 * also sent to the backend so the scan joins their history and carries the
 * authoritative model version. If that call fails the local result still
 * stands, and the screen says the sync did not happen rather than silently
 * dropping it.
 */
class ScanActivity : Activity() {

    companion object { const val EXTRA_URL = "url" }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_scan)
        ApiClient.configure(this)

        val input = findViewById<EditText>(R.id.scan_input)
        intent.getStringExtra(EXTRA_URL)?.let { input.setText(it) }

        findViewById<Button>(R.id.scan_go).setOnClickListener { run(input.text.toString().trim()) }
        findViewById<Button>(R.id.scan_clear).setOnClickListener {
            input.setText(""); findViewById<View>(R.id.result_block).visibility = View.GONE
        }
        if (!intent.getStringExtra(EXTRA_URL).isNullOrBlank()) run(input.text.toString().trim())
    }

    private fun run(url: String) {
        if (url.isBlank()) return
        val hostnameScope = findViewById<CheckBox>(R.id.scope_hostname).isChecked
        findViewById<ProgressBar>(R.id.scan_progress).visibility = View.VISIBLE
        findViewById<View>(R.id.result_block).visibility = View.GONE

        thread {
            val started = System.nanoTime()
            val local = runCatching {
                val clf = if (hostnameScope) Guard.hostname(this) else Guard.full(this)
                val subject = if (hostnameScope) hostnameOf(url) else url
                val v = clf.classify(subject)
                Triple(v, RiskAssessment.of(subject, v, v.features), subject)
            }
            val ms = (System.nanoTime() - started) / 1_000_000.0

            val wantWeb = findViewById<CheckBox>(R.id.scope_website).isChecked
            var remote: JSONObject? = null
            var syncNote = ""
            if (SessionStore(this).isSignedIn || wantWeb) {
                val r = runCatching {
                    ApiClient.scan(this, url,
                        if (hostnameScope) "hostname" else "url",
                        analyzeWebsite = wantWeb && !hostnameScope)
                }
                remote = r.getOrNull()
                if (r.isFailure) {
                    syncNote = if (wantWeb)
                        "Website analysis unavailable (offline or server error). " +
                        "The on-device URL assessment below still stands."
                    else "Not synced to your account (offline or server error)."
                }
            } else {
                syncNote = "Not signed in — this scan is not saved to an account."
            }

            runOnUiThread {
                findViewById<ProgressBar>(R.id.scan_progress).visibility = View.GONE
                local.onFailure {
                    Toast.makeText(this, "On-device model unavailable: ${it.message}",
                        Toast.LENGTH_LONG).show()
                }
                local.onSuccess { (v, a, subject) ->
                    render(subject, v, a, ms, remote, syncNote, hostnameScope)
                    ScanHistory(this).add(ScanHistory.Entry(
                        subject, v.score, a.band == RiskAssessment.Band.HIGH,
                        if (hostnameScope) "manual_hostname" else "manual",
                        System.currentTimeMillis()))
                }
            }
        }
    }

    private fun hostnameOf(u: String): String {
        var s = u.trim().replace(Regex("^[a-zA-Z][a-zA-Z0-9+.\\-]*://"), "")
        s = s.substringBefore('/').substringBefore('?').substringBefore('#')
        return s.substringAfterLast('@').substringBefore(':').lowercase()
    }

    private fun render(url: String, v: UrlClassifier.Verdict,
                       a: RiskAssessment.Result, ms: Double,
                       remote: JSONObject?, syncNote: String, hostnameScope: Boolean) {
        findViewById<View>(R.id.result_block).visibility = View.VISIBLE

        // Detection breakdown comes from the backend when it ran; otherwise the
        // rows say Not available rather than being filled in from the local
        // model, which has no page evidence at all.
        val bd = remote?.optJSONObject("assessment")?.optJSONArray("breakdown")
        findViewById<TextView>(R.id.r_breakdown).text = when {
            bd != null -> (0 until bd.length()).joinToString("\n") { i ->
                val o = bd.getJSONObject(i)
                val v2 = o.optString("value")
                val sc = if (o.has("score") && !o.isNull("score"))
                    "  (${o.optInt("score")})" else ""
                "  ${o.optString("label").padEnd(22)} $v2$sc"
            }
            hostnameScope ->
                "  Hostname-only assessment; webpage content was unavailable.\n" +
                "  URL pattern            " + a.band.label + "\n" +
                "  Website structure      Not available\n" +
                "  Credential form        Not available\n" +
                "  Form destination       Not available\n" +
                "  Brand/domain match     Not available\n" +
                "  DNS/reputation         Not available"
            else ->
                "  URL pattern            " + a.band.label + "\n" +
                "  Website structure      Not available (page not analysed)\n" +
                "  Credential form        Not available\n" +
                "  Form destination       Not available\n" +
                "  Brand/domain match     Not available\n" +
                "  DNS/reputation         Not available"
        }

        findViewById<TextView>(R.id.r_band).text = a.band.label
        findViewById<TextView>(R.id.r_band).setTextColor(
            when (a.band) {
                RiskAssessment.Band.LOW -> 0xFF1D8F5C.toInt()
                RiskAssessment.Band.SUSPICIOUS -> 0xFFA9741A.toInt()
                RiskAssessment.Band.HIGH -> 0xFFC23B2A.toInt()
            })
        findViewById<TextView>(R.id.r_desc).text = a.band.description
        findViewById<TextView>(R.id.r_url).text = url
        findViewById<TextView>(R.id.r_scores).text =
            "Risk score ${a.score}/100     Model confidence " +
            "%.1f".format(a.confidence) + "% (${a.confidenceLabel})"

        val modelVersion = remote?.optString("model_version")
            ?.takeIf { it.isNotBlank() && it != "null" } ?: "on-device bundle"
        findViewById<TextView>(R.id.r_meta).text = buildString {
            append("Detection time ${"%.1f".format(ms)} ms on this device\n")
            append("Model: $modelVersion\n")
            append("Scope: ${if (hostnameScope) "Hostname only" else "Full URL"}")
            val wm = remote?.optString("web_model_version")
            if (!wm.isNullOrBlank() && wm != "null") append("\nWebsite model: $wm")
            if (syncNote.isNotBlank()) append("\n$syncNote")
        }

        findViewById<TextView>(R.id.r_signals).text =
            if (a.signals.isEmpty()) "No security signals were raised for this URL."
            else a.signals.joinToString("\n")

        val f = v.features
        fun g(k: String) = (f[k] ?: 0.0)
        findViewById<TextView>(R.id.r_chars).text = listOf(
            "URL length" to g("url_length").toInt().toString(),
            "HTTPS" to if (g("is_https") != 0.0) "Enabled" else "Not enabled",
            "Hostname length" to g("host_length").toInt().toString(),
            "Subdomains" to g("subdomain_depth").toInt().toString(),
            "Path depth" to g("path_depth").toInt().toString(),
            "Query parameters" to g("num_query_params").toInt().toString(),
            "Special characters" to g("num_special_chars").toInt().toString(),
            "Hostname entropy" to "%.2f".format(g("host_entropy")),
            "IP address used" to if (g("has_ip") != 0.0) "Yes" else "No",
            "Punycode / IDN" to if (g("is_punycode") != 0.0) "Yes" else "No",
            "Abuse-prone TLD" to if (g("tld_abuse_prone") != 0.0) "Yes" else "No",
        ).joinToString("\n") { "${it.first.padEnd(22)} ${it.second}" }

        findViewById<TextView>(R.id.r_dns).text =
            if (hostnameScope)
                "Hostname-only assessment; webpage content was unavailable.\n" +
                "DNS resolution, IP address, domain age and reputation: Not available."
            else
                "DNS resolution: Not available\nIP address: Not available\n" +
                "Domain age: Not available\nReputation: Not available\n\n" +
                "URLGuard analyses the URL text only and never contacts the " +
                "address, so this information is not collected."

        findViewById<TextView>(R.id.r_action).text = when (a.band) {
            RiskAssessment.Band.HIGH ->
                "Recommended action: do not open this link."
            RiskAssessment.Band.SUSPICIOUS ->
                "Recommended action: proceed only if you are expecting this link."
            RiskAssessment.Band.LOW ->
                "Recommended action: no specific caution indicated by this assessment."
        }
        findViewById<TextView>(R.id.r_disclaimer).text = RiskAssessment.DISCLAIMER
    }
}
