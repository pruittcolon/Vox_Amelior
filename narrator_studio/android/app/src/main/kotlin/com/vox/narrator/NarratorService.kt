package com.vox.narrator

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.support.v4.media.MediaMetadataCompat
import android.support.v4.media.session.MediaSessionCompat
import android.support.v4.media.session.PlaybackStateCompat
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import androidx.media.app.NotificationCompat.MediaStyle

/**
 * Keeps reading (and exporting) alive with the screen off, and gives the
 * notification, lock screen and headset buttons play / pause / skip.
 */
class NarratorService : Service() {
    companion object {
        private const val CHANNEL = "narrator"
        private const val ID = 41

        fun show(context: Context, kind: String, title: String, text: String, playing: Boolean) {
            val intent = Intent(context, NarratorService::class.java)
                .putExtra("kind", kind)
                .putExtra("title", title)
                .putExtra("text", text)
                .putExtra("playing", playing)
            ContextCompat.startForegroundService(context, intent)
        }

        fun stop(context: Context) {
            context.stopService(Intent(context, NarratorService::class.java))
        }
    }

    private var session: MediaSessionCompat? = null
    private var wakeLock: PowerManager.WakeLock? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        val manager = getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(
            NotificationChannel(CHANNEL, "Reading and exporting", NotificationManager.IMPORTANCE_LOW),
        )
        session = MediaSessionCompat(this, "NarratorStudio").apply {
            setCallback(object : MediaSessionCompat.Callback() {
                override fun onPlay() = Bridge.emit("media", "play")
                override fun onPause() = Bridge.emit("media", "pause")
                override fun onSkipToNext() = Bridge.emit("media", "next")
                override fun onSkipToPrevious() = Bridge.emit("media", "previous")
                override fun onStop() = Bridge.emit("media", "stop")
            })
            isActive = true
        }
        val power = getSystemService(Context.POWER_SERVICE) as PowerManager
        wakeLock = power.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "narrator:work").apply {
            setReferenceCounted(false)
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        intent?.getStringExtra("action")?.let { action ->
            Bridge.emit("media", action)
            return START_NOT_STICKY
        }
        val kind = intent?.getStringExtra("kind") ?: "play"
        val title = intent?.getStringExtra("title") ?: "Narrator Studio"
        val text = intent?.getStringExtra("text") ?: ""
        val playing = intent?.getBooleanExtra("playing", true) ?: true

        val notification = build(kind, title, text, playing)
        val type = if (kind == "export") {
            ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC
        } else {
            ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PLAYBACK
        }
        if (Build.VERSION.SDK_INT >= 29) startForeground(ID, notification, type) else startForeground(ID, notification)

        if (playing || kind == "export") {
            wakeLock?.acquire(4 * 60 * 60 * 1000L)
        } else {
            wakeLock?.takeIf { it.isHeld }?.release()
        }
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        wakeLock?.takeIf { it.isHeld }?.release()
        session?.run {
            isActive = false
            release()
        }
        super.onDestroy()
    }

    private fun action(name: String, code: Int): PendingIntent =
        PendingIntent.getService(
            this, code,
            Intent(this, NarratorService::class.java).putExtra("action", name),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )

    private fun build(kind: String, title: String, text: String, playing: Boolean): Notification {
        val open = PendingIntent.getActivity(
            this, 0,
            Intent(this, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP),
            PendingIntent.FLAG_IMMUTABLE,
        )
        val builder = NotificationCompat.Builder(this, CHANNEL)
            .setSmallIcon(R.drawable.ic_notification)
            .setContentTitle(title)
            .setContentText(text)
            .setContentIntent(open)
            .setOnlyAlertOnce(true)
            .setOngoing(playing || kind == "export")
            .setVisibility(NotificationCompat.VISIBILITY_PUBLIC)

        if (kind == "export") {
            return builder.setProgress(0, 0, true).build()
        }

        val s = session!!
        s.setMetadata(
            MediaMetadataCompat.Builder()
                .putString(MediaMetadataCompat.METADATA_KEY_TITLE, title)
                .putString(MediaMetadataCompat.METADATA_KEY_ARTIST, text)
                .build(),
        )
        s.setPlaybackState(
            PlaybackStateCompat.Builder()
                .setActions(
                    PlaybackStateCompat.ACTION_PLAY or PlaybackStateCompat.ACTION_PAUSE or
                        PlaybackStateCompat.ACTION_PLAY_PAUSE or PlaybackStateCompat.ACTION_SKIP_TO_NEXT or
                        PlaybackStateCompat.ACTION_SKIP_TO_PREVIOUS or PlaybackStateCompat.ACTION_STOP,
                )
                .setState(
                    if (playing) PlaybackStateCompat.STATE_PLAYING else PlaybackStateCompat.STATE_PAUSED,
                    PlaybackStateCompat.PLAYBACK_POSITION_UNKNOWN, 1f,
                )
                .build(),
        )
        return builder
            .addAction(android.R.drawable.ic_media_previous, "Back", action("previous", 1))
            .addAction(
                if (playing) android.R.drawable.ic_media_pause else android.R.drawable.ic_media_play,
                if (playing) "Pause" else "Play",
                action(if (playing) "pause" else "play", 2),
            )
            .addAction(android.R.drawable.ic_media_next, "Next", action("next", 3))
            .setStyle(MediaStyle().setMediaSession(s.sessionToken).setShowActionsInCompactView(0, 1, 2))
            .build()
    }
}
