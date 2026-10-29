package com.aics.urlguard

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.text.format.DateUtils
import android.view.View
import android.widget.*
import org.json.JSONObject
import kotlin.concurrent.thread

/** Account details, security preferences and sign-out. */
class ProfileActivity : Activity() {

    private lateinit var store: SessionStore
    private lateinit var prefs: GuardPrefs

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_profile)
        ApiClient.configure(this)
        store = SessionStore(this)
        prefs = GuardPrefs(this)

        findViewById<TextView>(R.id.p_account).text =
            if (store.isSignedIn) buildString {
                append("Name       ${store.userName ?: "Not available"}\n")
                append("Email      ${store.userEmail ?: "Not available"}\n")
                val c = store.createdAt
                append("Created    " + if (c > 0)
                    DateUtils.formatDateTime(this@ProfileActivity, (c * 1000).toLong(),
                        DateUtils.FORMAT_SHOW_DATE or DateUtils.FORMAT_SHOW_YEAR)
                    else "Not available")
            } else "Not signed in.\n\nProtection still runs locally on this device."

        findViewById<Switch>(R.id.p_warn).apply {
            isChecked = prefs.linkGuardEnabled
            setOnCheckedChangeListener { _, v -> prefs.linkGuardEnabled = v; syncPrefs() }
        }
        findViewById<Switch>(R.id.p_dns).apply {
            isChecked = prefs.dnsGuardEnabled
            setOnCheckedChangeListener { _, v -> prefs.dnsGuardEnabled = v; syncPrefs() }
        }
        findViewById<Button>(R.id.p_clear_trusted).setOnClickListener {
            prefs.clearAllowlist()
            Toast.makeText(this, "Trusted sites cleared", Toast.LENGTH_SHORT).show()
        }
        findViewById<Button>(R.id.p_signout).apply {
            text = if (store.isSignedIn) "Sign out" else "Sign in"
            setOnClickListener {
                if (store.isSignedIn) {
                    thread {
                        ApiClient.logout(this@ProfileActivity)
                        runOnUiThread { goLogin() }
                    }
                } else goLogin()
            }
        }
        findViewById<TextView>(R.id.p_privacy).text =
            "What is stored, and why\n\n" +
            "• Scans you run while signed in are saved to your account: the URL, " +
            "the assessment, the risk score, the model version and the time. " +
            "That is what makes history and analytics possible.\n\n" +
            "• Intercepted links and DNS lookups are scored ON THIS DEVICE and " +
            "are not sent anywhere.\n\n" +
            "• The URL is never fetched. No DNS resolution, IP, domain age or " +
            "reputation data is collected, which is why those fields read " +
            "\"Not available\".\n\n" +
            "• Your password is stored by the server only as an Argon2id hash."

        loadModelInfo()
    }

    private fun goLogin() {
        startActivity(Intent(this, LoginActivity::class.java)
            .addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_NEW_TASK))
        finish()
    }

    /** Preferences live on the account when signed in, so the web UI agrees. */
    private fun syncPrefs() {
        if (!store.isSignedIn) return
        thread {
            runCatching {
                ApiClient.updateProfile(this, null, JSONObject()
                    .put("warn_on_suspicious", prefs.linkGuardEnabled)
                    .put("dns_layer_enabled", prefs.dnsGuardEnabled))
            }
        }
    }

    private fun loadModelInfo() {
        val tv = findViewById<TextView>(R.id.p_model)
        thread {
            val r = runCatching { ApiClient.meta(this) }
            runOnUiThread {
                tv.text = r.fold(
                    onSuccess = { m ->
                        "Detection model  ${m.optString("model_version", "Not available")}\n" +
                        "Dataset          ${m.optString("dataset_version", "Not available")}\n" +
                        "Features         ${m.optInt("n_features")}\n" +
                        "Threshold        ${Math.round(m.optDouble("threshold", 0.0) * 100)}/100"
                    },
                    onFailure = {
                        "Server model info unavailable (offline).\n" +
                        "The on-device model bundled with this app is still in use."
                    })
            }
        }
    }
}
