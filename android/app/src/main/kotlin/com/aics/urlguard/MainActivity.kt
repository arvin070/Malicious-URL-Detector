package com.aics.urlguard

import android.app.Activity
import android.content.Intent
import android.os.Bundle

/**
 * Launcher router.
 *
 * Deliberately does no work of its own: it decides whether the user lands on
 * sign-in or on the home screen and gets out of the way. Protection does not
 * depend on either -- the interception and DNS layers run from their own
 * components whether or not this activity is ever opened.
 */
class MainActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        ApiClient.configure(this)
        val next = if (SessionStore(this).isSignedIn) HomeActivity::class.java
                   else LoginActivity::class.java
        startActivity(Intent(this, next))
        finish()
    }
}
