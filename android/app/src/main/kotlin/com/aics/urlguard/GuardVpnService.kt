package com.aics.urlguard

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.net.VpnService
import android.os.Build
import android.os.ParcelFileDescriptor
import android.util.Log
import java.io.FileInputStream
import java.io.FileOutputStream
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.InetAddress
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.ThreadPoolExecutor
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

/**
 * The background layer: a local VPN that inspects DNS queries only.
 *
 * WHAT IT CAN AND CANNOT SEE
 * --------------------------
 * HTTPS encrypts the path and query, so a DNS query reveals the HOSTNAME and
 * nothing else. This service therefore scores hostnames with the weaker
 * hostname-only model, at a severe threshold (see [GuardPrefs.dnsThreshold]).
 * It is a coarse net that runs everywhere; [LinkInterceptorActivity] is the
 * accurate layer, and it only fires on links the user taps.
 *
 * SCOPE
 * -----
 * Only the virtual DNS address is routed into the tunnel
 * ([VIRTUAL_DNS] / 32). Web traffic, app traffic and everything else keep their
 * normal path and are never read -- the guard sees DNS questions, not content.
 *
 * FAIL OPEN
 * ---------
 * Any error forwarding, parsing or classifying results in the query being
 * passed through untouched. A security tool that breaks the user's connectivity
 * when it malfunctions gets uninstalled, and then protects nothing.
 */
class GuardVpnService : VpnService() {

    private var tun: ParcelFileDescriptor? = null
    private val running = AtomicBoolean(false)
    private var worker: Thread? = null
    private var pool: ExecutorService? = null

    /**
     * Writes to the tun fd are serialised. The reader hands each query to the
     * pool, so several replies can be ready at once, and interleaving two
     * packets into the fd would corrupt both.
     */
    private val writeLock = Any()

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            stopGuard()
            return START_NOT_STICKY
        }
        if (running.get()) return START_STICKY
        startForeground(NOTIF_ID, buildNotification())
        startGuard()
        // START_STICKY so Android restarts the guard after a low-memory kill;
        // a protection service that silently stays dead is worse than none.
        return START_STICKY
    }

    private fun startGuard() {
        val fd = try {
            Builder()
                .setSession("URL Guard")
                .addAddress(VIRTUAL_CLIENT, 32)
                .addDnsServer(VIRTUAL_DNS)
                // Route ONLY our virtual DNS address. Everything else keeps its
                // normal path, so this cannot break or observe other traffic.
                .addRoute(VIRTUAL_DNS, 32)
                .setBlocking(true)
                .also { b ->
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                        b.setMetered(false)
                    }
                }
                .establish()
        } catch (t: Throwable) {
            Log.e(TAG, "could not establish VPN", t)
            stopSelf()
            return
        }
        if (fd == null) { stopSelf(); return }
        tun = fd
        running.set(true)
        // Each query waits on an upstream round trip (up to UPSTREAM_TIMEOUT_MS).
        // Handling them on the reader thread would mean one slow or dead
        // resolver stalls EVERY DNS lookup on the device, which users
        // experience as "the internet stopped working". A small pool keeps
        // queries independent; the bounded queue plus CallerRunsPolicy means a
        // flood degrades to the old serial behaviour instead of growing without
        // limit.
        pool = ThreadPoolExecutor(
            2, 8, 30L, TimeUnit.SECONDS, LinkedBlockingQueue(128),
            ThreadPoolExecutor.CallerRunsPolicy()
        )
        worker = Thread({ pump(fd) }, "urlguard-dns").apply { isDaemon = true; start() }
    }

    private fun pump(fd: ParcelFileDescriptor) {
        val input = FileInputStream(fd.fileDescriptor)
        val output = FileOutputStream(fd.fileDescriptor)
        val buf = ByteArray(32 * 1024)
        val prefs = GuardPrefs(this)
        val history = ScanHistory(this)

        while (running.get()) {
            val n = try {
                input.read(buf)
            } catch (t: Throwable) {
                if (running.get()) Log.w(TAG, "tun read failed", t)
                break
            }
            if (n <= 0) continue

            val pkt = DnsPackets.parseIpv4Udp(buf, n)
            if (pkt == null || pkt.dstPort != 53) continue

            // buf is reused by the next read, so the packet must be handed off
            // as its own copy before the reader loops.
            pool?.execute {
                try {
                    handleQuery(pkt, output, prefs, history)
                } catch (t: Throwable) {
                    Log.w(TAG, "query handling failed, passing through", t)
                    runCatching { forwardAndReply(pkt, output) }
                }
            }
        }
    }

    private fun handleQuery(
        pkt: DnsPackets.Ipv4Udp,
        output: FileOutputStream,
        prefs: GuardPrefs,
        history: ScanHistory,
    ) {
        val dns = pkt.payload
        if (!DnsPackets.isStandardQuery(dns)) { forwardAndReply(pkt, output); return }

        // Only forward lookups (A / AAAA) describe somewhere the user is about
        // to go. PTR, SRV, TXT and friends are plumbing; classifying them
        // produced real false positives on device (reverse-DNS names scoring
        // 0.97+ purely for being digit-heavy).
        val qtype = DnsPackets.questionType(dns)
        if (qtype != DnsPackets.TYPE_A && qtype != DnsPackets.TYPE_AAAA) {
            forwardAndReply(pkt, output); return
        }

        val host = DnsPackets.questionName(dns)
        if (host == null ||
            DnsPackets.isNonNavigationalName(host) ||
            prefs.isAllowed(host)
        ) { forwardAndReply(pkt, output); return }

        val score = try {
            Guard.hostname(this).score(Guard.hostname(this).featureVector(host))
        } catch (t: Throwable) {
            Log.w(TAG, "scoring failed, allowing", t)
            forwardAndReply(pkt, output); return
        }

        if (score >= prefs.dnsThreshold) {
            val reply = DnsPackets.buildNxDomain(dns)
            if (reply.isEmpty()) { forwardAndReply(pkt, output); return }
            writePacket(output, DnsPackets.buildResponsePacket(pkt, reply))
            history.add(ScanHistory.Entry(host, score, true, "dns",
                System.currentTimeMillis()))
            Notifier.blockedHost(this, host, score)
        } else {
            forwardAndReply(pkt, output)
        }
    }

    /** Relay the query to the real resolver through a protected socket. */
    private fun forwardAndReply(pkt: DnsPackets.Ipv4Udp, output: FileOutputStream) {
        DatagramSocket().use { sock ->
            // protect() keeps this socket OUT of our own tunnel; without it the
            // forwarded query would loop straight back into the VPN.
            if (!protect(sock)) return
            sock.soTimeout = UPSTREAM_TIMEOUT_MS
            val upstream = InetAddress.getByName(UPSTREAM_DNS)
            sock.send(DatagramPacket(pkt.payload, pkt.payload.size, upstream, 53))
            val back = ByteArray(4096)
            val resp = DatagramPacket(back, back.size)
            sock.receive(resp)
            writePacket(
                output,
                DnsPackets.buildResponsePacket(pkt, back.copyOfRange(0, resp.length))
            )
        }
    }

    private fun writePacket(output: FileOutputStream, bytes: ByteArray) {
        synchronized(writeLock) { output.write(bytes) }
    }

    private fun stopGuard() {
        running.set(false)
        pool?.shutdownNow()
        pool = null
        runCatching { tun?.close() }
        tun = null
        worker?.interrupt()
        worker = null
        stopForegroundCompat()
        stopSelf()
    }

    @Suppress("DEPRECATION")
    private fun stopForegroundCompat() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            stopForeground(STOP_FOREGROUND_REMOVE)
        } else {
            stopForeground(true)
        }
    }

    override fun onRevoke() {
        // The user switched to another VPN, or revoked permission.
        stopGuard()
        super.onRevoke()
    }

    override fun onDestroy() {
        running.set(false)
        pool?.shutdownNow()
        runCatching { tun?.close() }
        super.onDestroy()
    }

    private fun buildNotification(): Notification {
        Notifier.ensureChannels(this)
        val open = PendingIntent.getActivity(
            this, 0, Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )
        val stop = PendingIntent.getService(
            this, 1, Intent(this, GuardVpnService::class.java).setAction(ACTION_STOP),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )
        return Notification.Builder(this, Notifier.CH_STATUS)
            .setContentTitle("URL Guard is on")
            .setContentText("Checking DNS lookups on this device")
            .setSmallIcon(android.R.drawable.ic_lock_idle_lock)
            .setOngoing(true)
            .setContentIntent(open)
            .addAction(
                Notification.Action.Builder(null as android.graphics.drawable.Icon?,
                    "Turn off", stop).build()
            )
            .build()
    }

    companion object {
        private const val TAG = "GuardVpn"
        const val ACTION_STOP = "com.aics.urlguard.STOP_VPN"
        const val NOTIF_ID = 42

        // Link-local-ish private addresses used only inside the tunnel.
        const val VIRTUAL_CLIENT = "10.111.222.1"
        const val VIRTUAL_DNS = "10.111.222.2"
        const val UPSTREAM_DNS = "1.1.1.1"
        const val UPSTREAM_TIMEOUT_MS = 4000

        fun start(ctx: Context) {
            val i = Intent(ctx, GuardVpnService::class.java)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                ctx.startForegroundService(i)
            } else {
                ctx.startService(i)
            }
        }

        fun stop(ctx: Context) {
            ctx.startService(
                Intent(ctx, GuardVpnService::class.java).setAction(ACTION_STOP)
            )
        }

        fun isRunning(ctx: Context): Boolean {
            val am = ctx.getSystemService(Context.ACTIVITY_SERVICE)
                as android.app.ActivityManager
            @Suppress("DEPRECATION")
            return am.getRunningServices(Int.MAX_VALUE).any {
                it.service.className == GuardVpnService::class.java.name
            }
        }
    }
}

/** Notification channels and the "we blocked something" alert. */
object Notifier {
    const val CH_STATUS = "guard_status"
    const val CH_ALERT = "guard_alert"

    fun ensureChannels(ctx: Context) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        val nm = ctx.getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(
            NotificationChannel(CH_STATUS, "Protection status",
                NotificationManager.IMPORTANCE_LOW)
        )
        nm.createNotificationChannel(
            NotificationChannel(CH_ALERT, "Blocked lookups",
                NotificationManager.IMPORTANCE_HIGH)
        )
    }

    fun blockedHost(ctx: Context, host: String, score: Double) {
        ensureChannels(ctx)
        val n = Notification.Builder(ctx, CH_ALERT)
            .setContentTitle("Blocked a suspicious lookup")
            .setContentText("$host  (score ${"%.2f".format(score)})")
            .setSmallIcon(android.R.drawable.stat_sys_warning)
            .setAutoCancel(true)
            .setContentIntent(
                PendingIntent.getActivity(ctx, 2,
                    Intent(ctx, MainActivity::class.java),
                    PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)
            )
            .build()
        ctx.getSystemService(NotificationManager::class.java)
            .notify(host.hashCode(), n)
    }
}
