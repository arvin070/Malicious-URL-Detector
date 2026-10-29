package com.aics.urlguard

import android.app.Activity
import android.content.ComponentName
import android.content.Intent
import android.content.pm.ActivityInfo
import android.content.pm.PackageManager
import android.util.Log
import android.net.Uri
import android.os.Bundle
import android.widget.Toast

/**
 * The accurate layer: every http/https link the user TAPS is routed here first.
 *
 * This is where the full 122-feature model runs, because a tapped link carries
 * the whole URL -- path, query and all. That is the model validated at
 * ROC-AUC 0.958 +/- 0.011 under domain-aware cross-validation.
 *
 * It is also where the phishing actually arrives: links in SMS, WhatsApp,
 * Telegram, Gmail. Typing a URL by hand into Chrome does not pass through here
 * (no Android app can intercept that without an AccessibilityService) -- the
 * DNS layer is the fallback for those, at much lower accuracy.
 *
 * The activity has no UI of its own: it classifies and immediately either hands
 * off to a real browser or shows [WarningActivity].
 */
class LinkInterceptorActivity : Activity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val url = intent?.dataString
        if (url.isNullOrBlank()) { finish(); return }

        val prefs = GuardPrefs(this)
        val host = Uri.parse(url).host ?: ""

        if (!prefs.linkGuardEnabled || (host.isNotEmpty() && prefs.isAllowed(host))) {
            openInBrowser(url); finish(); return
        }

        val verdict = try {
            Guard.full(this).classify(url)
        } catch (t: Throwable) {
            // Fail OPEN: if the model cannot load, never strand the user on a
            // link they asked to open.
            Toast.makeText(this, "URL Guard unavailable", Toast.LENGTH_SHORT).show()
            openInBrowser(url); finish(); return
        }

        // FAST PATH. The on-device model has already answered, with no
        // network call, and for a confident answer that is the whole
        // decision -- the user tapped a link and is waiting.
        if (!GuardPrefs(this).deepScanEnabled ||
            !DeepScan.isBorderline(verdict.score, verdict.threshold)) {
            decide(url, verdict, null)
            return
        }

        // BORDERLINE. Only here is a second opinion worth the wait, and the
        // wait is capped. The network call cannot run on the main thread, so
        // the decision is deferred to the callback; every failure path inside
        // DeepScan returns the local verdict unchanged.
        Thread {
            val outcome = try {
                DeepScan.analyse(this, url, verdict)
            } catch (t: Throwable) {
                null
            }
            runOnUiThread { decide(url, verdict, outcome) }
        }.apply { isDaemon = true }.start()
    }

    /**
     * Apply whichever verdict we ended up with. [outcome] is null when no
     * deep scan ran or it failed, in which case the on-device verdict stands.
     */
    private fun decide(url: String, local: UrlClassifier.Verdict,
                       outcome: DeepScan.Outcome?) {
        if (isFinishing || isDestroyed) return

        val score = outcome?.score ?: local.score
        val blocked = outcome?.malicious ?: local.malicious
        val source = if (outcome?.used == true) "link+server" else "link"

        ScanHistory(this).add(
            ScanHistory.Entry(url, score, blocked, source,
                System.currentTimeMillis())
        )

        if (blocked) {
            val signals = RiskAssessment.signalsFor(local.features).toMutableList()
            // The server's account of what it did, when it contributed one.
            outcome?.takeIf { it.used }?.let { signals.add(0, it.note) }
            startActivity(
                Intent(this, WarningActivity::class.java)
                    .putExtra(WarningActivity.EXTRA_URL, url)
                    .putExtra(WarningActivity.EXTRA_SCORE, score)
                    .putExtra(WarningActivity.EXTRA_THRESHOLD, local.threshold)
                    .putExtra(WarningActivity.EXTRA_TIME, System.currentTimeMillis())
                    .putExtra(WarningActivity.EXTRA_SIGNALS,
                        if (signals.isEmpty()) ""
                        else signals.joinToString("\n"))
            )
        } else {
            openInBrowser(url)
        }
        finish()
    }

    private fun openInBrowser(url: String) {
        BrowserHandoff.open(this, url)
    }
}

/**
 * Hands a URL to a real browser without bouncing back into this app.
 *
 * Launching a plain ACTION_VIEW would re-enter [LinkInterceptorActivity] and
 * loop forever, so the guard's own component is excluded from the candidates
 * and an explicit browser is chosen instead.
 */
object BrowserHandoff {

    private const val TAG = "BrowserHandoff"

    /**
     * Browsers are resolved with CATEGORY_BROWSABLE explicitly set.
     *
     * Without it the query returns almost nothing once this app holds the
     * BROWSER role: the bare VIEW intent resolves through the preferred-activity
     * machinery, which points at us, and the real browsers never appear. Adding
     * the category makes it an ordinary filter match again. (Observed on an API
     * 34 emulator: Chrome has the right https/BROWSABLE filter, but a
     * category-less query listed only this app.)
     */
    fun open(ctx: Activity, url: String) {
        val uri = Uri.parse(url)
        val launch = Intent(Intent.ACTION_VIEW, uri)
            .addCategory(Intent.CATEGORY_BROWSABLE)
            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)

        val chosen = resolveBrowser(ctx, uri)
        if (chosen == null) {
            Log.w(TAG, "no browser found for $uri")
            Toast.makeText(ctx, "No other browser installed", Toast.LENGTH_LONG).show()
            return
        }
        launch.component = ComponentName(chosen.packageName, chosen.name)
        try {
            ctx.startActivity(launch)
        } catch (t: Throwable) {
            Log.w(TAG, "explicit launch failed, falling back to chooser", t)
            // Last resort: let the user pick. Keeps a safe link openable even if
            // the component we resolved has since gone away.
            try {
                ctx.startActivity(
                    Intent.createChooser(
                        Intent(Intent.ACTION_VIEW, uri)
                            .addCategory(Intent.CATEGORY_BROWSABLE),
                        "Open with"
                    ).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                )
            } catch (t2: Throwable) {
                Toast.makeText(ctx, "Could not open browser", Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun resolveBrowser(ctx: Activity, uri: Uri): ActivityInfo? {
        val self = ctx.packageName
        val pm = ctx.packageManager

        // A generic https probe, not the real URL: a site with a verified app
        // link would otherwise resolve to that app instead of a browser.
        val probe = Intent(Intent.ACTION_VIEW, Uri.parse("https://example.com"))
            .addCategory(Intent.CATEGORY_BROWSABLE)

        val candidates = (
            pm.queryIntentActivities(probe, PackageManager.MATCH_ALL) +
                pm.queryIntentActivities(
                    Intent(Intent.ACTION_VIEW, uri)
                        .addCategory(Intent.CATEGORY_BROWSABLE),
                    PackageManager.MATCH_ALL
                )
            )
            .map { it.activityInfo }
            .filter { it.packageName != self && it.exported && it.enabled }
            .distinctBy { it.packageName to it.name }

        Log.i(TAG, "browser candidates: " +
            candidates.joinToString { "${it.packageName}/${it.name}" }.ifEmpty { "none" })

        return PREFERRED_BROWSERS.firstNotNullOfOrNull { pkg ->
            candidates.firstOrNull { it.packageName == pkg }
        } ?: candidates.firstOrNull()
    }

    /** Tried in order; anything else installed is used as a fallback. */
    private val PREFERRED_BROWSERS = listOf(
        "com.android.chrome",
        "org.mozilla.firefox",
        "com.brave.browser",
        "com.microsoft.emmx",
        "com.opera.browser",
        "com.duckduckgo.mobile.android",
        "com.sec.android.app.sbrowser",
    )

    /** Whether any browser other than us can handle http links. */
    fun hasOtherBrowser(ctx: Activity): Boolean = resolveBrowser(
        ctx, Uri.parse("https://example.com")) != null
}
