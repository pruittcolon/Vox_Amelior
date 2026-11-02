package com.vox.evensuite

import android.app.Notification
import android.content.pm.ApplicationInfo
import android.os.Build
import android.service.notification.NotificationListenerService
import android.service.notification.StatusBarNotification

/**
 * Sees the phone's notifications (after the user grants "Notification
 * access") so the app can show them on the glasses. Only ordinary posted
 * notifications are passed on: no ongoing ones (music, downloads), group
 * summaries or the app's own.
 */
class NotifyListener : NotificationListenerService() {

    override fun onNotificationPosted(sbn: StatusBarNotification) {
        val sink = onPosted ?: return
        if (sbn.packageName == packageName) return
        val n = sbn.notification ?: return
        if (n.flags and Notification.FLAG_ONGOING_EVENT != 0) return
        if (n.flags and Notification.FLAG_GROUP_SUMMARY != 0) return
        val extras = n.extras ?: return
        val title = extras.getCharSequence(Notification.EXTRA_TITLE)?.toString() ?: ""
        val text = (extras.getCharSequence(Notification.EXTRA_BIG_TEXT)
            ?: extras.getCharSequence(Notification.EXTRA_TEXT))?.toString() ?: ""
        if (title.isBlank() && text.isBlank()) return
        sink(
            mapOf(
                "key" to sbn.key,
                "package" to sbn.packageName,
                "app" to appName(sbn, extras),
                "title" to title,
                "text" to text,
                "time" to sbn.postTime,
            )
        )
    }

    private fun appName(sbn: StatusBarNotification, extras: android.os.Bundle): String {
        val info: ApplicationInfo? = try {
            if (Build.VERSION.SDK_INT >= 33) {
                extras.getParcelable("android.appInfo", ApplicationInfo::class.java)
            } else {
                @Suppress("DEPRECATION")
                extras.getParcelable("android.appInfo")
            }
        } catch (_: Exception) {
            null
        } ?: try {
            packageManager.getApplicationInfo(sbn.packageName, 0)
        } catch (_: Exception) {
            null
        }
        return info?.let { packageManager.getApplicationLabel(it).toString() } ?: sbn.packageName
    }

    companion object {
        /** Set by MainActivity while the app is running. */
        @Volatile
        var onPosted: ((Map<String, Any>) -> Unit)? = null
    }
}
