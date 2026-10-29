package com.vox.evensuite

import android.Manifest
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.media.projection.MediaProjectionManager
import android.graphics.drawable.Icon
import android.os.Build
import android.os.IBinder
import android.util.Log
import com.vox.evensuite.ble.BleManager

/**
 * Keeps the app alive with the screen off: microphone while hands-free
 * listening is on, connected device while the glasses are linked. The
 * notification's Stop button turns listening off ("so it doesn't go forever").
 *
 * The glasses link counts as wanted from the moment the user connects until
 * they disconnect - not only while it is up. That is what keeps Android from
 * freezing or killing the app while the link is down and being rebuilt (it
 * used to stop with the link, and then nothing could reconnect). If the
 * process is killed anyway, the system restarts this service (sticky) and it
 * dials the saved pair again by itself.
 */
class SuiteService : Service() {

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent == null) {
            // The system restarted us after the process died: take the glasses link up again.
            Log.i(TAG, "restarted by the system")
            if (BleManager.restore(this)) {
                goForeground()
                return START_STICKY
            }
            stopSelfCompat()
            return START_NOT_STICKY
        }
        when (intent.action) {
            ACTION_STOP_LISTENING -> {
                onStopListening?.invoke()
                return if (keepLink) START_STICKY else START_NOT_STICKY
            }
            ACTION_DISCONNECT -> {
                // "Disconnect" in the notification: the user is done with the glasses for now.
                BleManager.init(this) // (the process may have been restarted just for this)
                BleManager.disconnect()
                if (!listening && !mirroring) {
                    stopSelfCompat()
                    return START_NOT_STICKY
                }
            }
            ACTION_LINK -> {
                if (!keepLink && BleManager.hasSavedLink(this)) BleManager.restore(this) // e.g. at boot
            }
            ACTION_UPDATE -> {
                listening = intent.getBooleanExtra("listening", false)
                glasses = intent.getBooleanExtra("glasses", false)
                mic = intent.getBooleanExtra("mic", false)
                text = intent.getStringExtra("text") ?: ""
            }
            ACTION_STOP_MIRROR -> stopMirrorNow()
            ACTION_START_MIRROR -> {
                // Android 14: the service must be in the foreground as
                // "mediaProjection" before the projection is created.
                mirroring = true
                if (!goForeground()) {
                    mirroring = false
                    onMirrorStopped?.invoke("Android did not allow screen capture to start")
                    return START_NOT_STICKY
                }
                startMirror(intent)
                return START_NOT_STICKY
            }
        }
        if (!listening && !keepLink && !mirroring) {
            stopSelfCompat()
            return START_NOT_STICKY
        }
        goForeground()
        return if (keepLink) START_STICKY else START_NOT_STICKY
    }

    private fun goForeground(): Boolean {
        val notification = build()
        val types = serviceTypes()
        if (tryForeground(notification, types)) return true
        // Android refuses the microphone type when started from the background:
        // keep the glasses link (and screen sharing) without it.
        val noMic = types and ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE.inv()
        if (noMic != types && noMic != 0 && tryForeground(notification, noMic)) return true
        if (!mirroring) stopSelfCompat()
        return false
    }

    private fun tryForeground(notification: Notification, types: Int): Boolean = try {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(NOTIFICATION_ID, notification, types)
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
        true
    } catch (e: Exception) {
        Log.e(TAG, "startForeground($types) failed: $e")
        false
    }

    private fun startMirror(intent: Intent) {
        val code = intent.getIntExtra("code", 0)
        val data: Intent? = if (Build.VERSION.SDK_INT >= 33) {
            intent.getParcelableExtra("data", Intent::class.java)
        } else {
            @Suppress("DEPRECATION")
            intent.getParcelableExtra("data")
        }
        val mpm = getSystemService(MediaProjectionManager::class.java)
        val projection = try {
            if (data == null) null else mpm.getMediaProjection(code, data)
        } catch (e: Exception) {
            Log.e(TAG, "getMediaProjection: $e")
            null
        }
        if (projection == null) {
            mirroring = false
            onMirrorStopped?.invoke("Screen sharing was not allowed")
            refreshAfterMirror()
            return
        }
        mirror?.release()
        val m = ScreenMirror(
            this, projection,
            onFrame = { frame -> onMirrorFrame?.invoke(frame) },
            onStopped = {
                mirror = null
                mirroring = false
                onMirrorStopped?.invoke(null)
                refreshAfterMirror()
            },
        )
        applyConfig(m, pendingConfig)
        mirror = m
        try {
            m.start()
        } catch (e: Exception) {
            Log.e(TAG, "mirror start: $e")
            m.release()
            mirror = null
            mirroring = false
            onMirrorStopped?.invoke("Could not capture the screen: ${e.message}")
            refreshAfterMirror()
        }
    }

    private fun stopMirrorNow() {
        val m = mirror
        mirror = null
        mirroring = false
        m?.release()
        onMirrorStopped?.invoke(null)
        refreshAfterMirror()
    }

    private fun refreshAfterMirror() {
        if (!listening && !keepLink) stopSelfCompat() else goForeground()
    }

    private fun serviceTypes(): Int {
        var types = 0
        // Hands-free, or ready for a TouchBar hold while the phone is in a pocket.
        if ((listening || (mic && glasses)) && granted(Manifest.permission.RECORD_AUDIO)) {
            types = types or ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE
        }
        val btOk = Build.VERSION.SDK_INT < Build.VERSION_CODES.S || granted(Manifest.permission.BLUETOOTH_CONNECT)
        if (keepLink && btOk) {
            types = types or ServiceInfo.FOREGROUND_SERVICE_TYPE_CONNECTED_DEVICE
        }
        if (mirroring) types = types or ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION
        if (types == 0) types = ServiceInfo.FOREGROUND_SERVICE_TYPE_CONNECTED_DEVICE
        return types
    }

    private fun granted(permission: String) =
        checkSelfPermission(permission) == PackageManager.PERMISSION_GRANTED

    private fun stopSelfCompat() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            stopForeground(STOP_FOREGROUND_REMOVE)
        } else {
            @Suppress("DEPRECATION")
            stopForeground(true)
        }
        stopSelf()
    }

    private fun build(): Notification {
        val nm = getSystemService(NotificationManager::class.java)
        if (nm.getNotificationChannel(CHANNEL_ID) == null) {
            nm.createNotificationChannel(
                NotificationChannel(CHANNEL_ID, "Listening and glasses", NotificationManager.IMPORTANCE_LOW).apply {
                    description = "Shown while Even Suite listens for \"Claude\" or stays linked to your glasses"
                    setShowBadge(false)
                }
            )
        }
        val open = PendingIntent.getActivity(
            this, 0,
            Intent(this, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )
        val title = when {
            mirroring -> "Showing your screen on the glasses"
            listening -> "Listening for “Claude”"
            else -> "Glasses connected"
        }
        val idle = when {
            !BleManager.appAttached -> "Kept linked to your G1. Open Even Suite to use it."
            keepLink -> "Even Suite is linked to your G1"
            else -> "Even Suite"
        }
        val builder = Notification.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_notification)
            .setContentTitle(title)
            .setContentText(text.ifEmpty { idle })
            .setContentIntent(open)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setCategory(Notification.CATEGORY_SERVICE)
        if (listening) {
            val stop = PendingIntent.getService(
                this, 1,
                Intent(this, SuiteService::class.java).setAction(ACTION_STOP_LISTENING),
                PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
            )
            builder.addAction(Notification.Action.Builder(null as Icon?, "Stop listening", stop).build())
        }
        if (keepLink) {
            val off = PendingIntent.getService(
                this, 3,
                Intent(this, SuiteService::class.java).setAction(ACTION_DISCONNECT),
                PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
            )
            builder.addAction(Notification.Action.Builder(null as Icon?, "Disconnect glasses", off).build())
        }
        if (mirroring) {
            val stop = PendingIntent.getService(
                this, 2,
                Intent(this, SuiteService::class.java).setAction(ACTION_STOP_MIRROR),
                PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
            )
            builder.addAction(Notification.Action.Builder(null as Icon?, "Stop sharing", stop).build())
        }
        return builder.build()
    }

    companion object {
        private const val TAG = "SuiteService"
        private const val CHANNEL_ID = "suite"
        private const val NOTIFICATION_ID = 7
        const val ACTION_UPDATE = "com.vox.evensuite.UPDATE"
        const val ACTION_LINK = "com.vox.evensuite.LINK"
        const val ACTION_DISCONNECT = "com.vox.evensuite.DISCONNECT"
        const val ACTION_STOP_LISTENING = "com.vox.evensuite.STOP_LISTENING"
        const val ACTION_START_MIRROR = "com.vox.evensuite.START_MIRROR"
        const val ACTION_STOP_MIRROR = "com.vox.evensuite.STOP_MIRROR"

        private var listening = false
        private var glasses = false

        /** The glasses link is wanted (by the app, or saved natively) - keeps this service up. */
        private val keepLink: Boolean get() = glasses || BleManager.isWanted
        private var mic = false
        private var text = ""
        @Volatile private var mirroring = false
        private var mirror: ScreenMirror? = null
        private var pendingConfig: Map<*, *>? = null

        /** Set by MainActivity: frames and the end of screen sharing go to Flutter. */
        var onMirrorFrame: ((Map<String, Any>) -> Unit)? = null
        var onMirrorStopped: ((String?) -> Unit)? = null

        fun startMirror(context: Context, code: Int, data: Intent, config: Map<*, *>?): String? {
            pendingConfig = config
            return try {
                context.startForegroundService(
                    Intent(context, SuiteService::class.java)
                        .setAction(ACTION_START_MIRROR)
                        .putExtra("code", code)
                        .putExtra("data", data)
                )
                null
            } catch (e: Exception) {
                e.message ?: e.toString()
            }
        }

        fun stopMirror(context: Context) {
            if (!mirroring && mirror == null) return
            try {
                context.startService(Intent(context, SuiteService::class.java).setAction(ACTION_STOP_MIRROR))
            } catch (e: Exception) {
                mirror?.release()
                mirror = null
                mirroring = false
            }
        }

        fun mirrorStats(): Map<String, Any>? = mirror?.takeStats()

        fun configureMirror(config: Map<*, *>?) {
            pendingConfig = config
            mirror?.let { applyConfig(it, config) }
        }

        private fun applyConfig(m: ScreenMirror, config: Map<*, *>?) {
            (config?.get("region") as? String)?.let { m.region = it }
            (config?.get("strip") as? Int)?.let { m.strip = it }
            (config?.get("intervalMs") as? Int)?.let { m.intervalMs = it.toLong().coerceIn(200L, 5000L) }
            (config?.get("cx") as? Int)?.let { m.cx = it }
            (config?.get("cy") as? Int)?.let { m.cy = it }
            (config?.get("cw") as? Int)?.let { m.cw = it }
            (config?.get("ch") as? Int)?.let { m.ch = it }
            (config?.get("fit") as? String)?.let { m.fit = it }
            (config?.get("rows") as? Int)?.let { m.rows = it }
            m.refresh()
        }

        /** Set by MainActivity; tells Flutter to turn hands-free off. */
        var onStopListening: (() -> Unit)? = null

        /** The glasses link was asked for, or given up: the service follows (see the class comment). */
        fun linkChanged(context: Context) {
            try {
                val intent = Intent(context, SuiteService::class.java).setAction(ACTION_LINK)
                if (BleManager.isWanted) {
                    context.startForegroundService(intent)
                } else if (!listening && !mirroring) {
                    glasses = false
                    context.startService(intent) // nothing keeps it: it stops itself
                }
            } catch (e: Exception) {
                Log.e(TAG, "linkChanged: $e")
            }
        }

        /** Returns null on success, or why Android refused. */
        fun update(context: Context, listening: Boolean, glasses: Boolean, text: String, mic: Boolean = false): String? {
            val intent = Intent(context, SuiteService::class.java)
                .setAction(ACTION_UPDATE)
                .putExtra("listening", listening)
                .putExtra("glasses", glasses)
                .putExtra("mic", mic)
                .putExtra("text", text)
            return try {
                if (!listening && !glasses && !mirroring && !BleManager.isWanted) {
                    // Nothing to keep alive: just let a running service stop itself.
                    if (Companion.listening || Companion.glasses) context.startService(intent)
                    Companion.listening = false
                    Companion.glasses = false
                } else {
                    context.startForegroundService(intent)
                }
                null
            } catch (e: Exception) {
                Log.e(TAG, "update: $e")
                e.message ?: e.toString()
            }
        }
    }
}
