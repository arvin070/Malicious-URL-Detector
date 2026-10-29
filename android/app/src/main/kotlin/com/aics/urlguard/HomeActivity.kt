package com.aics.urlguard

import android.app.Activity
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.net.VpnService
import android.os.Build
import android.os.Bundle
import android.widget.*
import kotlin.concurrent.thread

/** Landing screen: who you are, what is on, and the quick actions. */
class HomeActivity : Activity() {

    private lateinit var prefs: GuardPrefs

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_home)
        ApiClient.configure(this)
        prefs = GuardPrefs(this)
        Notifier.ensureChannels(this)
        if (Build.VERSION.SDK_INT >= 33) {
            requestPermissions(arrayOf(android.Manifest.permission.POST_NOTIFICATIONS), 7)
        }

        findViewById<Button>(R.id.action_scan).setOnClickListener {
            startActivity(Intent(this, ScanActivity::class.java))
        }
        findViewById<Button>(R.id.action_clipboard).setOnClickListener { scanClipboard() }
        findViewById<Button>(R.id.action_history).setOnClickListener {
            startActivity(Intent(this, HistoryActivity::class.java)
                .putExtra(HistoryActivity.EXTRA_THREATS_ONLY, false))
        }
        findViewById<Button>(R.id.action_threats).setOnClickListener {
            startActivity(Intent(this, HistoryActivity::class.java)
                .putExtra(HistoryActivity.EXTRA_THREATS_ONLY, true))
        }
        findViewById<Button>(R.id.action_profile).setOnClickListener {
            startActivity(Intent(this, ProfileActivity::class.java))
        }

        findViewById<Switch>(R.id.link_switch).apply {
            isChecked = prefs.linkGuardEnabled
            setOnCheckedChangeListener { _, v -> prefs.linkGuardEnabled = v }
        }
        findViewById<Switch>(R.id.dns_switch).apply {
            isChecked = prefs.dnsGuardEnabled && GuardVpnService.isRunning(this@HomeActivity)
            setOnCheckedChangeListener { _, v ->
                prefs.dnsGuardEnabled = v
                if (v) {
                    val consent = VpnService.prepare(this@HomeActivity)
                    if (consent != null) startActivityForResult(consent, 101)
                    else GuardVpnService.start(this@HomeActivity)
                } else GuardVpnService.stop(this@HomeActivity)
            }
        }
    }

    override fun onResume() {
        super.onResume()
        val store = SessionStore(this)
        findViewById<TextView>(R.id.welcome).text =
            if (store.isSignedIn) "Welcome back, ${store.userName ?: store.userEmail}"
            else "Welcome"
        findViewById<TextView>(R.id.account_line).text =
            if (store.isSignedIn) store.userEmail
            else "Not signed in — protection still runs locally, but history is not synced."
        findViewById<Switch>(R.id.dns_switch).isChecked =
            prefs.dnsGuardEnabled && GuardVpnService.isRunning(this)
        loadSummary()
    }

    @Deprecated("single consent flow")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == 101) {
            if (resultCode == RESULT_OK) GuardVpnService.start(this)
            else {
                prefs.dnsGuardEnabled = false
                findViewById<Switch>(R.id.dns_switch).isChecked = false
            }
        }
    }

    /** Local counts always; server counts only when signed in. */
    private fun loadSummary() {
        val local = ScanHistory(this).all()
        val high = local.count { it.blocked }
        findViewById<TextView>(R.id.summary).text =
            if (local.isEmpty()) "No links checked on this device yet."
            else "$high of ${local.size} links checked on this device were flagged."

        if (!SessionStore(this).isSignedIn) {
            findViewById<TextView>(R.id.server_summary).text = ""
            return
        }
        thread {
            val r = runCatching { ApiClient.stats(this) }
            runOnUiThread {
                findViewById<TextView>(R.id.server_summary).text = r.fold(
                    onSuccess = { s ->
                        if (!s.optBoolean("sufficient_data"))
                            "No scans recorded on your account yet."
                        else "Account: ${s.optInt("total")} scans · " +
                             "${s.optInt("high")} high risk · " +
                             "${s.optInt("suspicious")} suspicious"
                    },
                    onFailure = { "Account summary unavailable (offline)." })
            }
        }
    }

    private fun scanClipboard() {
        val cm = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        val text = cm.primaryClip?.getItemAt(0)?.coerceToText(this)?.toString()?.trim()
        if (text.isNullOrBlank()) {
            Toast.makeText(this, "Clipboard is empty", Toast.LENGTH_SHORT).show(); return
        }
        startActivity(Intent(this, ScanActivity::class.java)
            .putExtra(ScanActivity.EXTRA_URL, text))
    }
}
