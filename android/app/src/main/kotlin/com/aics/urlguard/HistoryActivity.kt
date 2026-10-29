package com.aics.urlguard

import android.app.Activity
import android.os.Bundle
import android.text.format.DateUtils
import android.view.View
import android.widget.*
import kotlin.concurrent.thread

/**
 * Scan history.
 *
 * Signed in: the account's server-side history (only ever that account's).
 * Signed out: the device-local log, clearly labelled as local-only.
 */
class HistoryActivity : Activity() {

    companion object { const val EXTRA_THREATS_ONLY = "threats_only" }

    private var threatsOnly = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_history)
        ApiClient.configure(this)
        threatsOnly = intent.getBooleanExtra(EXTRA_THREATS_ONLY, false)
        title = if (threatsOnly) "Recent threats" else "Scan history"
        findViewById<TextView>(R.id.h_title).text = title

        findViewById<EditText>(R.id.h_search).addTextChangedListener(
            object : android.text.TextWatcher {
                override fun afterTextChanged(s: android.text.Editable?) = load()
                override fun beforeTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}
                override fun onTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}
            })
        findViewById<Button>(R.id.h_clear).setOnClickListener {
            ScanHistory(this).clear(); load()
        }
        load()
    }

    private fun load() {
        val q = findViewById<EditText>(R.id.h_search).text.toString().trim()
        val list = findViewById<TextView>(R.id.h_list)
        val note = findViewById<TextView>(R.id.h_note)

        if (!SessionStore(this).isSignedIn) {
            note.text = "Device-local history. Sign in to keep history across devices."
            val items = ScanHistory(this).all()
                .filter { !threatsOnly || it.blocked }
                .filter { q.isBlank() || it.url.contains(q, true) }
            list.text = if (items.isEmpty()) "Not enough data yet."
            else items.joinToString("\n") {
                val mark = if (it.blocked) "⚠" else "✓"
                "$mark  ${"%.2f".format(it.score)}  ${it.url.take(58)}\n" +
                "    ${it.source} · ${DateUtils.getRelativeTimeSpanString(it.at)}"
            }
            return
        }

        note.text = "Your account history. Only you can see these scans."
        findViewById<ProgressBar>(R.id.h_progress).visibility = View.VISIBLE
        thread {
            val r = runCatching {
                ApiClient.history(this, limit = 50,
                    q = q.ifBlank { null },
                    band = if (threatsOnly) "high" else null)
            }
            runOnUiThread {
                findViewById<ProgressBar>(R.id.h_progress).visibility = View.GONE
                r.onSuccess { arr ->
                    if (arr.length() == 0) { list.text = "Not enough data yet."; return@onSuccess }
                    val sb = StringBuilder()
                    for (i in 0 until arr.length()) {
                        val o = arr.getJSONObject(i)
                        val mark = when (o.optString("risk_key")) {
                            "high" -> "⚠"; "suspicious" -> "!"; else -> "✓" }
                        sb.append("$mark  ${o.optInt("risk_score")}/100  ")
                        sb.append(o.optString("domain").ifBlank { o.optString("url") }.take(50))
                        sb.append("\n    ${o.optString("risk_band")} · ")
                        sb.append(o.optString("model_version").ifBlank { "model n/a" })
                        sb.append(" · ${o.optString("source")}\n")
                    }
                    list.text = sb.toString()
                }.onFailure {
                    list.text = "Could not load account history (offline). " +
                        "Device-local scans are still recorded."
                }
            }
        }
    }
}
