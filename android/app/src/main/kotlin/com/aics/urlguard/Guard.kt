package com.aics.urlguard

import android.content.Context
import android.content.SharedPreferences
import org.json.JSONArray
import org.json.JSONObject

/**
 * Lazily-built, process-wide holder for the two classifiers.
 *
 * There are two models on purpose, and they are NOT interchangeable:
 *
 *  - [full]     scores a complete URL (path, query and all). Used by the link
 *               handler, where the whole URL is available. This is the model
 *               validated at ROC-AUC 0.958 +/- 0.011 under domain-aware CV.
 *
 *  - [hostname] scores a bare hostname. Used by the DNS layer, because a DNS
 *               query carries nothing else -- HTTPS encrypts the path. It was
 *               retrained on hostnames alone and is MUCH weaker
 *               (ROC-AUC 0.764 +/- 0.021). It is served at a deliberately
 *               severe threshold so it almost never blocks anything legitimate;
 *               see [GuardPrefs.dnsThreshold].
 *
 * Mixing them up would silently serve a model input it was never trained on,
 * so each is loaded from its own asset directory with its own metadata.
 */
object Guard {

    @Volatile private var fullClf: UrlClassifier? = null
    @Volatile private var hostClf: UrlClassifier? = null

    fun full(ctx: Context): UrlClassifier =
        fullClf ?: synchronized(this) {
            fullClf ?: load(ctx, "").also { fullClf = it }
        }

    fun hostname(ctx: Context): UrlClassifier =
        hostClf ?: synchronized(this) {
            hostClf ?: load(ctx, "hostname/").also { hostClf = it }
        }

    private fun load(ctx: Context, prefix: String): UrlClassifier {
        val am = ctx.applicationContext.assets
        val meta = JSONObject(am.open("${prefix}model_meta.json").use {
            it.readBytes().toString(Charsets.UTF_8)
        })
        val namesJson = meta.getJSONArray("feature_names")
        val names = (0 until namesJson.length()).map { namesJson.getString(it) }
        val modelBytes = am.open("${prefix}url_model.onnx").use { it.readBytes() }
        // The suffix list is shared: both models were trained with the same one.
        val psl = am.open("public_suffix_list.txt").use {
            it.readBytes().toString(Charsets.UTF_8)
        }.lineSequence()

        return UrlClassifier.create(
            modelBytes = modelBytes,
            suffixListLines = psl,
            featureNames = names,
            threshold = meta.getDouble("threshold"),
            modelName = meta.optString("model", "model"),
        )
    }
}

/** Settings, kept deliberately small. */
class GuardPrefs(ctx: Context) {
    private val sp: SharedPreferences =
        ctx.applicationContext.getSharedPreferences("urlguard", Context.MODE_PRIVATE)

    var linkGuardEnabled: Boolean
        get() = sp.getBoolean("link_guard", true)
        set(v) = sp.edit().putBoolean("link_guard", v).apply()

    var dnsGuardEnabled: Boolean
        get() = sp.getBoolean("dns_guard", false)
        set(v) = sp.edit().putBoolean("dns_guard", v).apply()

    /**
     * Blocking threshold for the DNS layer.
     *
     * Default 0.95, NOT the 0.65 the training run selected. The training rule
     * targets FPR <= 0.05, which is fine for a screen the user can dismiss but
     * not for a system-wide blocker: at 0.05 roughly one benign domain in
     * twenty stops resolving and the phone feels broken. Measured on the
     * hostname model's out-of-fold predictions:
     *
     *      0.65 -> FPR 0.0505, recall 0.369
     *      0.95 -> FPR 0.0033, recall 0.169
     *
     * So the default trades away more than half the recall to avoid breaking
     * roughly 1 domain in 300. The link-handler layer, which sees full URLs and
     * is far more accurate, is where real coverage comes from.
     */
    var dnsThreshold: Float
        get() = sp.getFloat("dns_threshold", 0.95f)
        set(v) = sp.edit().putFloat("dns_threshold", v).apply()

    /**
     * Whether a borderline link may be sent to the backend for a second
     * opinion. Off by default: it transmits the URL the user tapped to the
     * server, and that is the user's decision to make, not a default.
     */
    var deepScanEnabled: Boolean
        get() = sp.getBoolean("deep_scan", false)
        set(v) = sp.edit().putBoolean("deep_scan", v).apply()

    /**
     * Whether a deep scan may also ask the backend to RETRIEVE the page.
     * Separate from [deepScanEnabled] because it is a bigger step: it causes
     * a request to the suspicious site, albeit from the server rather than
     * the phone. Off by default for the same reason.
     */
    var deepScanFetchesPage: Boolean
        get() = sp.getBoolean("deep_scan_fetch", false)
        set(v) = sp.edit().putBoolean("deep_scan_fetch", v).apply()

    /** Hostnames the user chose to trust after seeing a warning. */
    fun allowlist(): MutableSet<String> =
        sp.getStringSet("allowlist", emptySet())!!.toMutableSet()

    fun allow(host: String) {
        val s = allowlist(); s.add(host.lowercase())
        sp.edit().putStringSet("allowlist", s).apply()
    }

    fun isAllowed(host: String): Boolean = host.lowercase() in allowlist()

    fun clearAllowlist() = sp.edit().remove("allowlist").apply()
}

/** Ring buffer of recent verdicts, shown on the main screen. */
class ScanHistory(ctx: Context) {
    private val sp = ctx.applicationContext
        .getSharedPreferences("urlguard_history", Context.MODE_PRIVATE)

    data class Entry(val url: String, val score: Double, val blocked: Boolean,
                     val source: String, val at: Long)

    fun add(e: Entry) {
        val arr = JSONArray(sp.getString("items", "[]"))
        val next = JSONArray()
        next.put(JSONObject().apply {
            put("url", e.url); put("score", e.score); put("blocked", e.blocked)
            put("source", e.source); put("at", e.at)
        })
        for (i in 0 until minOf(arr.length(), MAX - 1)) next.put(arr.get(i))
        sp.edit().putString("items", next.toString()).apply()
    }

    fun all(): List<Entry> {
        val arr = JSONArray(sp.getString("items", "[]"))
        return (0 until arr.length()).map { i ->
            val o = arr.getJSONObject(i)
            Entry(o.getString("url"), o.getDouble("score"), o.getBoolean("blocked"),
                  o.optString("source", "link"), o.optLong("at"))
        }
    }

    fun clear() = sp.edit().remove("items").apply()

    companion object { const val MAX = 100 }
}
