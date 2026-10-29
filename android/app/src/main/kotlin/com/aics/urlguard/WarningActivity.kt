package com.aics.urlguard

import android.app.Activity
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.TextView

/**
 * The interstitial the user sees instead of a flagged page.
 *
 * Design choices that matter more than they look:
 *  - "Go back" is the prominent, default action; proceeding is deliberately the
 *    quieter one. The model's measured FPR is not zero, so proceeding must stay
 *    POSSIBLE -- a blocker that cannot be overridden gets uninstalled -- but it
 *    should not be the path of least resistance.
 *  - The score and the signals behind it are shown, not hidden. A warning the
 *    user cannot interrogate trains them to click through warnings.
 */
class WarningActivity : Activity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_warning)

        val url = intent.getStringExtra(EXTRA_URL).orEmpty()
        val proba = intent.getDoubleExtra(EXTRA_SCORE, 0.0)
        val threshold = intent.getDoubleExtra(EXTRA_THRESHOLD, 0.5)
        val signals = intent.getStringExtra(EXTRA_SIGNALS).orEmpty()
        val detectedAt = intent.getLongExtra(EXTRA_TIME, System.currentTimeMillis())

        val score = RiskAssessment.displayScore(proba)
        val margin = RiskAssessment.decisionMargin(proba, threshold)
        val band = RiskAssessment.bandFor(score)

        findViewById<TextView>(R.id.warn_url).text = url
        findViewById<TextView>(R.id.warn_band).text = band.label
        findViewById<TextView>(R.id.warn_score).text = buildString {
            append("Risk score $score/100")
            append("     Detection confidence ")
            append("%.1f".format((margin * 100).coerceIn(0.1, 99.9)))
            append("% (")
            append(RiskAssessment.marginLabel(margin))
            append(")")
        }
        findViewById<TextView>(R.id.warn_time).text =
            "Detected " + java.text.SimpleDateFormat("d MMM yyyy, HH:mm:ss",
                java.util.Locale.getDefault()).format(java.util.Date(detectedAt))
        findViewById<TextView>(R.id.warn_signals).text =
            if (signals.isBlank()) "No individual signals were recorded for this URL."
            else signals

        val signalsView = findViewById<View>(R.id.warn_signals)
        findViewById<Button>(R.id.warn_details).setOnClickListener {
            signalsView.visibility =
                if (signalsView.visibility == View.VISIBLE) View.GONE else View.VISIBLE
        }

        findViewById<Button>(R.id.warn_back).setOnClickListener { finish() }

        findViewById<Button>(R.id.warn_proceed).setOnClickListener {
            val host = android.net.Uri.parse(url).host
            if (!host.isNullOrEmpty()) GuardPrefs(this).allow(host)
            BrowserHandoff.open(this, url)
            finish()
        }
    }

    companion object {
        const val EXTRA_URL = "url"
        const val EXTRA_SCORE = "score"
        const val EXTRA_THRESHOLD = "threshold"
        const val EXTRA_SIGNALS = "signals"
        const val EXTRA_TIME = "time"
    }
}
