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
import android.graphics.drawable.Icon
import android.os.Build
import android.os.IBinder
import android.util.Log

/**
 * Keeps the app alive with the screen off: microphone while hands-free
 * listening is on, connected device while the glasses are linked. The
 * notification's Stop button turns listening off ("so it doesn't go forever").
 */
class SuiteService : Service() {

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_STOP_LISTENING -> {
                onStopListening?.invoke()
                return START_NOT_STICKY
            }
            ACTION_UPDATE -> {
                listening = intent.getBooleanExtra("listening", false)
                glasses = intent.getBooleanExtra("glasses", false)
                text = intent.getStringExtra("text") ?: ""
            }
        }
        if (!listening && !glasses) {
            stopSelfCompat()
            return START_NOT_STICKY
        }
        try {
            val notification = build()
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                startForeground(NOTIFICATION_ID, notification, serviceTypes())
            } else {
                startForeground(NOTIFICATION_ID, notification)
            }
        } catch (e: Exception) {
            // Android refuses some types when started from the background.
            Log.e(TAG, "startForeground failed: $e")
            stopSelfCompat()
        }
        return START_NOT_STICKY
    }

    private fun serviceTypes(): Int {
        var types = 0
        if (listening && granted(Manifest.permission.RECORD_AUDIO)) {
            types = types or ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE
        }
        val btOk = Build.VERSION.SDK_INT < Build.VERSION_CODES.S || granted(Manifest.permission.BLUETOOTH_CONNECT)
        if (glasses && btOk) {
            types = types or ServiceInfo.FOREGROUND_SERVICE_TYPE_CONNECTED_DEVICE
        }
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
            listening -> "Listening for “Claude”"
            else -> "Glasses connected"
        }
        val builder = Notification.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_notification)
            .setContentTitle(title)
            .setContentText(text.ifEmpty { if (glasses) "Even Suite is linked to your G1" else "Even Suite" })
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
        return builder.build()
    }

    companion object {
        private const val TAG = "SuiteService"
        private const val CHANNEL_ID = "suite"
        private const val NOTIFICATION_ID = 7
        const val ACTION_UPDATE = "com.vox.evensuite.UPDATE"
        const val ACTION_STOP_LISTENING = "com.vox.evensuite.STOP_LISTENING"

        private var listening = false
        private var glasses = false
        private var text = ""

        /** Set by MainActivity; tells Flutter to turn hands-free off. */
        var onStopListening: (() -> Unit)? = null

        /** Returns null on success, or why Android refused. */
        fun update(context: Context, listening: Boolean, glasses: Boolean, text: String): String? {
            val intent = Intent(context, SuiteService::class.java)
                .setAction(ACTION_UPDATE)
                .putExtra("listening", listening)
                .putExtra("glasses", glasses)
                .putExtra("text", text)
            return try {
                if (!listening && !glasses) {
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
