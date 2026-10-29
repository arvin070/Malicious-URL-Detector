package com.aics.urlguard

import android.content.Context
import org.json.JSONObject
import java.util.concurrent.Callable
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

/**
 * Second-stage analysis for links the on-device model cannot settle.
 *
 * WHY THE DEVICE DOES NOT DECIDE EVERYTHING
 * -----------------------------------------
 * The ONNX model on the phone sees the URL text and nothing else. That is
 * enough for the clear cases and it is instant, which matters: the user tapped
 * a link and is waiting. Most taps resolve on-device with no network call at
 * all.
 *
 * The cases it cannot settle are the ones near its own threshold. For those,
 * and only those, the BACKEND can retrieve the page and run the webpage,
 * JavaScript, redirect and fusion layers. The phone never fetches the page
 * itself: pulling attacker-chosen content into the process that is supposed to
 * be protecting the user would defeat the point.
 *
 * EVERY FAILURE PATH OPENS THE LINK
 * ---------------------------------
 * No network, no account, slow server, malformed response, timeout: all of
 * them fall back to the on-device verdict. A security feature that strands
 * the user on a link they asked for, because a server was slow, gets switched
 * off -- and then protects nobody.
 */
object DeepScan {

    /**
     * Band around the on-device threshold where a second opinion is worth
     * waiting for. Outside it the local model is already confident and the
     * network round trip buys nothing.
     */
    const val BORDERLINE_MARGIN = 0.20

    /**
     * Hard cap on the wait. A link tap that stalls longer than this feels
     * broken, and the on-device verdict is already available -- so the
     * deadline is a user-experience limit, not a network timeout to tune up.
     */
    const val TIMEOUT_MS = 2500L

    private val pool = Executors.newCachedThreadPool { r ->
        Thread(r, "urlguard-deepscan").apply { isDaemon = true }
    }

    data class Outcome(
        val used: Boolean,
        val score: Double,
        val malicious: Boolean,
        val band: String?,
        /** Why the deep scan did or did not contribute. Shown, not hidden. */
        val note: String,
    )

    /** True when the local score is close enough to its threshold to be unsure. */
    fun isBorderline(score: Double, threshold: Double): Boolean =
        kotlin.math.abs(score - threshold) <= BORDERLINE_MARGIN

    /**
     * Ask the backend for a full-layer assessment. Blocking; call off the main
     * thread. Returns the local verdict unchanged on any failure.
     */
    fun analyse(ctx: Context, url: String, local: UrlClassifier.Verdict): Outcome {
        val prefs = GuardPrefs(ctx)
        if (!prefs.deepScanEnabled) {
            return Outcome(false, local.score, local.malicious, null,
                "Deep scan is off; the on-device URL model decided this.")
        }
        if (!isBorderline(local.score, local.threshold)) {
            return Outcome(false, local.score, local.malicious, null,
                "The on-device model was confident, so no server call was made.")
        }

        val task = Callable {
            ApiClient.scan(ctx, url, "url", analyzeWebsite = prefs.deepScanFetchesPage)
        }
        val json: JSONObject = try {
            pool.submit(task).get(TIMEOUT_MS, TimeUnit.MILLISECONDS)
        } catch (t: Throwable) {
            return Outcome(false, local.score, local.malicious, null,
                "The server could not be reached in time; the on-device " +
                    "result stands.")
        }

        val assessment = json.optJSONObject("assessment")
            ?: return Outcome(false, local.score, local.malicious, null,
                "The server response could not be read; the on-device " +
                    "result stands.")

        val risk = assessment.optInt("risk_score", -1)
        if (risk < 0) {
            return Outcome(false, local.score, local.malicious, null,
                "The server returned no score; the on-device result stands.")
        }
        val bandKey = assessment.optJSONObject("risk_band")?.optString("key")
        val bandLabel = assessment.optJSONObject("risk_band")?.optString("label")

        // "insufficient" means the server could not settle it either. That is
        // not a reason to block, and it is not a reason to pretend the link is
        // fine: the local verdict stands and the wording says why.
        if (bandKey == "insufficient") {
            return Outcome(true, risk / 100.0, local.malicious, bandKey,
                "The server analysed the page and still could not reach a " +
                    "conclusion. Treat this link as unresolved.")
        }

        val blocked = bandKey == "high"
        val layers = assessment.optJSONObject("layers")
        val pageRead = layers?.optJSONObject("website")?.optBoolean("available") == true
        return Outcome(true, risk / 100.0, blocked, bandKey,
            if (pageRead)
                "The server retrieved and analysed the page ($bandLabel). " +
                    "No script was run and nothing was submitted."
            else
                "The server re-analysed the URL ($bandLabel). The page " +
                    "itself could not be retrieved.")
    }

    /**
     * Host for display, never used for a decision.
     *
     * Uses java.net.URI rather than android.net.Uri deliberately: this is a
     * pure string operation, and the framework class is a stub under plain
     * JVM unit tests, so depending on it would make this untestable off a
     * device for no benefit.
     */
    fun hostOf(url: String): String =
        try { java.net.URI(url).host ?: "" } catch (t: Throwable) { "" }
}
