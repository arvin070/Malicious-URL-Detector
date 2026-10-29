package com.aics.urlguard

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.net.VpnService

/**
 * Restarts the DNS layer after a reboot, but ONLY if the user had it on.
 *
 * VpnService.prepare() returning null means consent is still granted; if the
 * user has since revoked it, this does nothing rather than popping a dialog out
 * of nowhere at boot.
 */
class BootReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != Intent.ACTION_BOOT_COMPLETED) return
        if (!GuardPrefs(context).dnsGuardEnabled) return
        if (VpnService.prepare(context) != null) return   // consent revoked
        GuardVpnService.start(context)
    }
}
