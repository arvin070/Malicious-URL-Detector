package com.aics.urlguard

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.text.InputType
import android.view.View
import android.widget.*
import kotlin.concurrent.thread

/**
 * Sign-in, registration and password reset.
 *
 * Signing in is OPTIONAL. Link interception and the DNS layer are local and
 * work without an account; an account only adds synced history and shared
 * settings. The screen says so rather than gating protection behind a login.
 */
class LoginActivity : Activity() {

    private enum class Mode { LOGIN, REGISTER, FORGOT }
    private var mode = Mode.LOGIN

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_login)
        ApiClient.configure(this)

        findViewById<EditText>(R.id.server_url).setText(SessionStore(this).baseUrl)

        findViewById<Button>(R.id.pw_toggle).setOnClickListener {
            val f = findViewById<EditText>(R.id.password)
            val hidden = f.inputType and InputType.TYPE_TEXT_VARIATION_PASSWORD != 0
            f.inputType = InputType.TYPE_CLASS_TEXT or
                (if (hidden) InputType.TYPE_TEXT_VARIATION_VISIBLE_PASSWORD
                 else InputType.TYPE_TEXT_VARIATION_PASSWORD)
            f.setSelection(f.text.length)
            (it as Button).text = if (hidden) "Hide" else "Show"
        }

        findViewById<Button>(R.id.primary).setOnClickListener { submit() }
        findViewById<Button>(R.id.switch_register).setOnClickListener { setMode(Mode.REGISTER) }
        findViewById<Button>(R.id.switch_login).setOnClickListener { setMode(Mode.LOGIN) }
        findViewById<Button>(R.id.switch_forgot).setOnClickListener { setMode(Mode.FORGOT) }
        findViewById<Button>(R.id.skip).setOnClickListener {
            startActivity(Intent(this, HomeActivity::class.java)); finish()
        }
        setMode(Mode.LOGIN)
    }

    private fun setMode(m: Mode) {
        mode = m
        findViewById<View>(R.id.name_row).visibility =
            if (m == Mode.REGISTER) View.VISIBLE else View.GONE
        findViewById<View>(R.id.password_row).visibility =
            if (m == Mode.FORGOT) View.GONE else View.VISIBLE
        findViewById<View>(R.id.confirm_row).visibility =
            if (m == Mode.REGISTER) View.VISIBLE else View.GONE
        findViewById<Button>(R.id.primary).text = when (m) {
            Mode.LOGIN -> "Sign in"; Mode.REGISTER -> "Create account"
            Mode.FORGOT -> "Request reset"
        }
        findViewById<View>(R.id.switch_register).visibility =
            if (m == Mode.LOGIN) View.VISIBLE else View.GONE
        findViewById<View>(R.id.switch_forgot).visibility =
            if (m == Mode.LOGIN) View.VISIBLE else View.GONE
        findViewById<View>(R.id.switch_login).visibility =
            if (m == Mode.LOGIN) View.GONE else View.VISIBLE
        msg("")
    }

    private fun msg(t: String) {
        findViewById<TextView>(R.id.message).apply {
            text = t; visibility = if (t.isBlank()) View.GONE else View.VISIBLE
        }
    }

    private fun submit() {
        val base = findViewById<EditText>(R.id.server_url).text.toString().trim()
        if (base.isNotEmpty()) SessionStore(this).baseUrl = base
        val email = findViewById<EditText>(R.id.email).text.toString().trim()
        val pass = findViewById<EditText>(R.id.password).text.toString()
        val name = findViewById<EditText>(R.id.name).text.toString().trim()
        val confirm = findViewById<EditText>(R.id.confirm).text.toString()

        if (mode == Mode.REGISTER && pass != confirm) {
            msg("Passwords do not match."); return
        }
        val btn = findViewById<Button>(R.id.primary)
        btn.isEnabled = false
        findViewById<ProgressBar>(R.id.progress).visibility = View.VISIBLE

        thread {
            val outcome = runCatching {
                when (mode) {
                    Mode.LOGIN -> ApiClient.login(this, email, pass)
                    Mode.REGISTER -> ApiClient.register(this, name, email, pass, confirm)
                    Mode.FORGOT -> ApiClient.forgot(this, email)
                }
            }
            runOnUiThread {
                btn.isEnabled = true
                findViewById<ProgressBar>(R.id.progress).visibility = View.GONE
                outcome.onSuccess { d ->
                    if (mode == Mode.FORGOT) {
                        msg(d.optString("message") + " " + d.optString("delivery"))
                        setMode(Mode.LOGIN)
                    } else {
                        startActivity(Intent(this, HomeActivity::class.java))
                        finish()
                    }
                }.onFailure { e ->
                    msg(when (e) {
                        is ApiClient.ApiException -> e.message ?: "Sign-in failed."
                        else -> "Could not reach the server. Check the address, " +
                                "or continue without an account."
                    })
                }
            }
        }
    }
}
