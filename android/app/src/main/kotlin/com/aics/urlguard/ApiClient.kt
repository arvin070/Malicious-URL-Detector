package com.aics.urlguard

import android.content.Context
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedReader
import java.net.HttpURLConnection
import java.net.URL
import java.nio.charset.StandardCharsets

/**
 * Talks to the URLGuard backend for authentication and history.
 *
 * WHAT DOES **NOT** GO THROUGH HERE
 * ---------------------------------
 * Link interception and the DNS layer never call this class. They score
 * locally with the bundled ONNX model, so protection keeps working with no
 * network, no account, and without a single intercepted URL leaving the
 * device. The backend is for sign-in and for syncing the history of scans the
 * user ran deliberately.
 *
 * NO SECRETS ARE SHIPPED
 * ----------------------
 * There is no API key in this class and none in the APK. The only credential
 * that exists at runtime is the user's own token, obtained by signing in and
 * held in app-private storage.
 *
 * Note on storage: [SessionStore] uses app-private SharedPreferences, which on
 * a non-rooted device is readable only by this app. EncryptedSharedPreferences
 * would add at-rest encryption and is the production choice; it is not used
 * here because its artifact is still alpha.
 */
object ApiClient {

    private const val TAG = "ApiClient"
    private const val TIMEOUT_MS = 12_000

    class ApiException(val status: Int, val code: String?, message: String)
        : Exception(message)

    data class Result(val json: JSONObject)

    @Volatile var baseUrl: String = DEFAULT_BASE
    const val DEFAULT_BASE = "http://10.0.2.2:5000"   // emulator -> host machine

    fun configure(ctx: Context) {
        baseUrl = SessionStore(ctx).baseUrl
    }

    // ------------------------------------------------------------ transport
    private fun request(
        ctx: Context, method: String, path: String,
        body: JSONObject? = null, authed: Boolean = true, allowRetry: Boolean = true,
    ): JSONObject {
        val store = SessionStore(ctx)
        val conn = (URL(store.baseUrl + path).openConnection() as HttpURLConnection).apply {
            requestMethod = method
            connectTimeout = TIMEOUT_MS
            readTimeout = TIMEOUT_MS
            setRequestProperty("Accept", "application/json")
            if (body != null) {
                doOutput = true
                setRequestProperty("Content-Type", "application/json")
            }
            if (authed) store.accessToken?.let {
                setRequestProperty("Authorization", "Bearer $it")
            }
        }
        try {
            body?.let {
                conn.outputStream.use { os ->
                    os.write(it.toString().toByteArray(StandardCharsets.UTF_8))
                }
            }
            val status = conn.responseCode
            val stream = if (status in 200..299) conn.inputStream else conn.errorStream
            val text = stream?.bufferedReader()?.use(BufferedReader::readText).orEmpty()
            val json = if (text.isBlank()) JSONObject() else JSONObject(text)

            if (status == 401 && authed && allowRetry && store.refreshToken != null) {
                val code = json.optString("code")
                if (code in listOf("token_expired", "token_invalid", "auth_required")) {
                    if (refresh(ctx)) {
                        return request(ctx, method, path, body, authed, allowRetry = false)
                    }
                }
            }
            if (status !in 200..299) {
                throw ApiException(status, json.optString("code", null),
                    json.optString("error", "Request failed ($status)"))
            }
            return json
        } finally {
            conn.disconnect()
        }
    }

    private fun refresh(ctx: Context): Boolean {
        val store = SessionStore(ctx)
        val rt = store.refreshToken ?: return false
        return try {
            val d = request(ctx, "POST", "/api/auth/refresh",
                JSONObject().put("refresh_token", rt), authed = false)
            store.save(d)
            true
        } catch (e: Exception) {
            Log.i(TAG, "refresh failed; signing out locally")
            store.clear()
            false
        }
    }

    // ----------------------------------------------------------------- auth
    fun login(ctx: Context, email: String, password: String): JSONObject {
        val d = request(ctx, "POST", "/api/auth/login",
            JSONObject().put("email", email).put("password", password),
            authed = false)
        SessionStore(ctx).save(d)
        return d
    }

    fun register(ctx: Context, name: String, email: String,
                 password: String, confirm: String): JSONObject {
        val d = request(ctx, "POST", "/api/auth/register",
            JSONObject().put("name", name).put("email", email)
                .put("password", password).put("confirm_password", confirm),
            authed = false)
        SessionStore(ctx).save(d)
        return d
    }

    fun forgot(ctx: Context, email: String): JSONObject =
        request(ctx, "POST", "/api/auth/forgot",
            JSONObject().put("email", email), authed = false)

    fun logout(ctx: Context) {
        val store = SessionStore(ctx)
        runCatching {
            request(ctx, "POST", "/api/auth/logout",
                JSONObject().put("refresh_token", store.refreshToken ?: ""),
                authed = false)
        }
        store.clear()
    }

    fun me(ctx: Context): JSONObject = request(ctx, "GET", "/api/auth/me")

    fun updateProfile(ctx: Context, name: String?, prefs: JSONObject?): JSONObject {
        val body = JSONObject()
        name?.let { body.put("name", it) }
        prefs?.let { body.put("preferences", it) }
        return request(ctx, "PATCH", "/api/auth/profile", body)
    }

    // ----------------------------------------------------------- scans
    /**
     * Server-side scan; records history against the signed-in account.
     *
     * `analyzeWebsite` asks the BACKEND to retrieve and analyse the page. The
     * app never downloads or renders an arbitrary page itself: doing that on
     * the device would mean pulling attacker-controlled content into the
     * phone's process, which is exactly what the interception layer exists to
     * avoid. The backend fetch is GET-only, runs no JavaScript and is
     * SSRF-guarded.
     */
    fun scan(ctx: Context, url: String, scope: String = "url",
             analyzeWebsite: Boolean = false): JSONObject =
        request(ctx, "POST", "/api/scan",
            JSONObject().put("url", url).put("scope", scope)
                .put("analyze_website", analyzeWebsite))

    fun history(ctx: Context, limit: Int = 25, offset: Int = 0,
                q: String? = null, band: String? = null): JSONArray {
        val sb = StringBuilder("/api/history?limit=$limit&offset=$offset")
        if (!q.isNullOrBlank()) sb.append("&q=").append(java.net.URLEncoder.encode(q, "UTF-8"))
        if (!band.isNullOrBlank()) sb.append("&band=").append(band)
        return request(ctx, "GET", sb.toString()).optJSONArray("items") ?: JSONArray()
    }

    fun stats(ctx: Context): JSONObject = request(ctx, "GET", "/api/stats")

    fun meta(ctx: Context): JSONObject =
        request(ctx, "GET", "/api/meta", authed = false)
}

/**
 * Session storage. Holds the user's own tokens and nothing else.
 *
 * The access token is short-lived; the refresh token is what survives a
 * restart. Both live in app-private storage and are cleared on sign-out.
 */
class SessionStore(ctx: Context) {
    private val sp = ctx.applicationContext
        .getSharedPreferences("urlguard_session", Context.MODE_PRIVATE)

    var baseUrl: String
        get() = sp.getString("base_url", ApiClient.DEFAULT_BASE)!!
        set(v) = sp.edit().putString("base_url", v.trimEnd('/')).apply()

    val accessToken: String? get() = sp.getString("access", null)
    val refreshToken: String? get() = sp.getString("refresh", null)
    val isSignedIn: Boolean get() = refreshToken != null

    val userName: String? get() = sp.getString("name", null)
    val userEmail: String? get() = sp.getString("email", null)
    val createdAt: Double get() = sp.getFloat("created_at", 0f).toDouble()

    fun save(d: org.json.JSONObject) {
        val e = sp.edit()
        d.optString("access_token", null)?.let { e.putString("access", it) }
        d.optString("refresh_token", null)?.let { e.putString("refresh", it) }
        d.optJSONObject("user")?.let { u ->
            e.putString("name", u.optString("name"))
            e.putString("email", u.optString("email"))
            e.putFloat("created_at", u.optDouble("created_at", 0.0).toFloat())
            e.putString("preferences", u.optJSONObject("preferences")?.toString() ?: "{}")
        }
        e.apply()
    }

    fun preferences(): org.json.JSONObject =
        runCatching { org.json.JSONObject(sp.getString("preferences", "{}")!!) }
            .getOrDefault(org.json.JSONObject())

    fun clear() = sp.edit().remove("access").remove("refresh")
        .remove("name").remove("email").remove("created_at")
        .remove("preferences").apply()
}
