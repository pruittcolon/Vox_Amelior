package com.vox.evensuite

import android.Manifest
import android.app.AlarmManager
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.media.AudioManager
import android.net.Uri
import android.os.Build
import android.os.SystemClock
import android.provider.AlarmClock
import android.provider.ContactsContract
import android.provider.Settings
import android.view.KeyEvent
import org.json.JSONArray
import org.json.JSONObject

/**
 * Small things Claude (or a quick voice command) can do on the phone:
 * reminders and timers (our own alarms and notifications, so they work in
 * the background and survive a reboot), the Clock app's alarms, the
 * flashlight, music and volume, opening apps, calls, texts, maps and search.
 * Every action answers {ok, say} or {ok: false, error}.
 */
object PhoneActions {
    /** A reminder went off while the app is running (it shows it on the glasses). */
    @Volatile var onReminder: ((String, String) -> Unit)? = null

    private const val PREFS = "phone_reminders"
    private const val CHANNEL = "reminders"

    fun run(ctx: Context, a: Map<*, *>): Map<String, Any?> = try {
        when (a["do"] as? String ?: "") {
            "remind" -> remind(ctx, a["id"] as? String ?: "${System.currentTimeMillis()}",
                (a["at"] as Number).toLong(), a["text"] as? String ?: "Reminder")
            "cancelReminder" -> {
                cancel(ctx, a["id"] as? String ?: "")
                ok("Reminder cancelled")
            }
            "reminders" -> mapOf("ok" to true, "list" to list(ctx))
            "alarm" -> startOrFail(ctx, Intent(AlarmClock.ACTION_SET_ALARM)
                .putExtra(AlarmClock.EXTRA_HOUR, (a["hour"] as Number).toInt())
                .putExtra(AlarmClock.EXTRA_MINUTES, (a["minute"] as? Number)?.toInt() ?: 0)
                .putExtra(AlarmClock.EXTRA_MESSAGE, a["label"] as? String ?: "")
                .putExtra(AlarmClock.EXTRA_SKIP_UI, true), "Alarm set")
            "torch" -> torch(ctx, a["on"] as? Boolean ?: true)
            "media" -> media(ctx, a["key"] as? String ?: "play")
            "volume" -> volume(ctx, a["dir"] as? String ?: "up")
            "open" -> open(ctx, a["app"] as? String ?: "")
            "call" -> call(ctx, a["to"] as? String ?: "")
            "text" -> text(ctx, a["to"] as? String ?: "", a["body"] as? String ?: "")
            "navigate" -> {
                val place = a["place"] as? String ?: ""
                val nav = Intent(Intent.ACTION_VIEW, Uri.parse("google.navigation:q=" + Uri.encode(place)))
                if (nav.resolveActivity(ctx.packageManager) != null) startOrFail(ctx, nav, "Directions to $place")
                else startOrFail(ctx, Intent(Intent.ACTION_VIEW, Uri.parse("geo:0,0?q=" + Uri.encode(place))), "Directions to $place")
            }
            "search" -> {
                val q = a["query"] as? String ?: ""
                startOrFail(ctx, Intent(Intent.ACTION_VIEW, Uri.parse("https://www.google.com/search?q=" + Uri.encode(q))), "Searching")
            }
            "status" -> mapOf(
                "ok" to true,
                "overlay" to Settings.canDrawOverlays(ctx),
                "exactAlarms" to exactAllowed(ctx),
                "contacts" to granted(ctx, Manifest.permission.READ_CONTACTS),
                "calls" to granted(ctx, Manifest.permission.CALL_PHONE),
            )
            "openOverlaySettings" -> startOrFail(ctx, Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                Uri.parse("package:${ctx.packageName}")), "")
            "openAlarmSettings" -> if (Build.VERSION.SDK_INT >= 31) startOrFail(ctx,
                Intent(Settings.ACTION_REQUEST_SCHEDULE_EXACT_ALARM, Uri.parse("package:${ctx.packageName}")), "") else ok("")
            else -> fail("I can't do that on the phone yet")
        }
    } catch (e: Exception) {
        fail("${e.message ?: e}")
    }

    private fun ok(say: String) = mapOf("ok" to true, "say" to say)
    private fun fail(why: String) = mapOf("ok" to false, "error" to why)
    private fun granted(ctx: Context, p: String) = ctx.checkSelfPermission(p) == PackageManager.PERMISSION_GRANTED

    /** Opening another app from the background needs "Display over other apps" (Android 10+). */
    private fun startOrFail(ctx: Context, intent: Intent, say: String): Map<String, Any?> {
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        if (intent.resolveActivity(ctx.packageManager) == null) return fail("No app on this phone can do that")
        ctx.startActivity(intent)
        return if (Settings.canDrawOverlays(ctx) || MainActivity.inForeground) ok(say)
        else mapOf("ok" to true, "say" to say, "note" to "background")
    }

    // ----------------------------------------------------------- reminders

    private fun exactAllowed(ctx: Context): Boolean {
        if (Build.VERSION.SDK_INT < 31) return true
        return (ctx.getSystemService(Context.ALARM_SERVICE) as AlarmManager).canScheduleExactAlarms()
    }

    private fun store(ctx: Context) = ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

    private fun load(ctx: Context): JSONArray = try {
        JSONArray(store(ctx).getString("list", "[]"))
    } catch (_: Exception) {
        JSONArray()
    }

    private fun save(ctx: Context, a: JSONArray) = store(ctx).edit().putString("list", a.toString()).apply()

    fun list(ctx: Context): List<Map<String, Any>> {
        val a = load(ctx)
        return (0 until a.length()).map {
            val o = a.getJSONObject(it)
            mapOf("id" to o.getString("id"), "at" to o.getLong("at"), "text" to o.getString("text"))
        }
    }

    private fun remind(ctx: Context, id: String, at: Long, text: String): Map<String, Any?> {
        val a = load(ctx)
        val keep = JSONArray()
        for (i in 0 until a.length()) if (a.getJSONObject(i).getString("id") != id) keep.put(a.getJSONObject(i))
        keep.put(JSONObject().put("id", id).put("at", at).put("text", text))
        save(ctx, keep)
        schedule(ctx, id, at, text)
        return mapOf("ok" to true, "say" to "", "exact" to exactAllowed(ctx))
    }

    private fun pending(ctx: Context, id: String, text: String): PendingIntent {
        val i = Intent(ctx, ReminderReceiver::class.java).putExtra("id", id).putExtra("text", text)
        return PendingIntent.getBroadcast(ctx, id.hashCode(), i,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)
    }

    fun schedule(ctx: Context, id: String, at: Long, text: String) {
        val am = ctx.getSystemService(Context.ALARM_SERVICE) as AlarmManager
        val pi = pending(ctx, id, text)
        val when_ = maxOf(at, System.currentTimeMillis() + 1000)
        if (exactAllowed(ctx)) am.setExactAndAllowWhileIdle(AlarmManager.RTC_WAKEUP, when_, pi)
        else am.setAndAllowWhileIdle(AlarmManager.RTC_WAKEUP, when_, pi)
    }

    fun cancel(ctx: Context, id: String) {
        val a = load(ctx)
        val keep = JSONArray()
        var text = ""
        for (i in 0 until a.length()) {
            val o = a.getJSONObject(i)
            if (o.getString("id") == id) text = o.getString("text") else keep.put(o)
        }
        save(ctx, keep)
        (ctx.getSystemService(Context.ALARM_SERVICE) as AlarmManager).cancel(pending(ctx, id, text))
    }

    /** After a reboot: set them all again. */
    fun rescheduleAll(ctx: Context) {
        for (r in list(ctx)) schedule(ctx, r["id"] as String, r["at"] as Long, r["text"] as String)
    }

    fun fire(ctx: Context, id: String, text: String) {
        val a = load(ctx)
        val keep = JSONArray()
        for (i in 0 until a.length()) if (a.getJSONObject(i).getString("id") != id) keep.put(a.getJSONObject(i))
        save(ctx, keep)
        val nm = ctx.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        if (Build.VERSION.SDK_INT >= 26 && nm.getNotificationChannel(CHANNEL) == null) {
            nm.createNotificationChannel(NotificationChannel(CHANNEL, "Reminders", NotificationManager.IMPORTANCE_HIGH))
        }
        val open = PendingIntent.getActivity(ctx, 3000, Intent(ctx, MainActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)
        val n = android.app.Notification.Builder(ctx, CHANNEL)
            .setSmallIcon(android.R.drawable.ic_popup_reminder)
            .setContentTitle("Reminder")
            .setContentText(text)
            .setCategory(android.app.Notification.CATEGORY_REMINDER)
            .setAutoCancel(true)
            .setContentIntent(open)
            .build()
        nm.notify(4000 + (id.hashCode() and 0xfff), n)
        onReminder?.invoke(id, text)
    }

    // ----------------------------------------------------------- the rest

    private fun torch(ctx: Context, on: Boolean): Map<String, Any?> {
        val cm = ctx.getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val id = cm.cameraIdList.firstOrNull {
            cm.getCameraCharacteristics(it).get(CameraCharacteristics.FLASH_INFO_AVAILABLE) == true
        } ?: return fail("No flashlight")
        cm.setTorchMode(id, on)
        return ok(if (on) "Flashlight on" else "Flashlight off")
    }

    private fun media(ctx: Context, key: String): Map<String, Any?> {
        val code = when (key) {
            "pause" -> KeyEvent.KEYCODE_MEDIA_PAUSE
            "next" -> KeyEvent.KEYCODE_MEDIA_NEXT
            "previous" -> KeyEvent.KEYCODE_MEDIA_PREVIOUS
            "toggle" -> KeyEvent.KEYCODE_MEDIA_PLAY_PAUSE
            else -> KeyEvent.KEYCODE_MEDIA_PLAY
        }
        val am = ctx.getSystemService(Context.AUDIO_SERVICE) as AudioManager
        val t = SystemClock.uptimeMillis()
        am.dispatchMediaKeyEvent(KeyEvent(t, t, KeyEvent.ACTION_DOWN, code, 0))
        am.dispatchMediaKeyEvent(KeyEvent(t, t, KeyEvent.ACTION_UP, code, 0))
        return ok(when (key) {
            "pause" -> "Paused"
            "next" -> "Next track"
            "previous" -> "Previous track"
            else -> "Playing"
        })
    }

    private fun volume(ctx: Context, dir: String): Map<String, Any?> {
        val am = ctx.getSystemService(Context.AUDIO_SERVICE) as AudioManager
        val adj = when (dir) {
            "down" -> AudioManager.ADJUST_LOWER
            "mute" -> AudioManager.ADJUST_TOGGLE_MUTE
            else -> AudioManager.ADJUST_RAISE
        }
        repeat(if (dir == "mute") 1 else 2) { am.adjustStreamVolume(AudioManager.STREAM_MUSIC, adj, 0) }
        val pct = am.getStreamVolume(AudioManager.STREAM_MUSIC) * 100 / maxOf(1, am.getStreamMaxVolume(AudioManager.STREAM_MUSIC))
        return ok(if (dir == "mute") "Sound toggled" else "Volume $pct%")
    }

    private fun open(ctx: Context, name: String): Map<String, Any?> {
        val pm = ctx.packageManager
        val main = Intent(Intent.ACTION_MAIN).addCategory(Intent.CATEGORY_LAUNCHER)
        val apps = pm.queryIntentActivities(main, 0)
        val want = name.lowercase().trim()
        val best = apps.minByOrNull {
            val label = it.loadLabel(pm).toString().lowercase()
            when {
                label == want -> 0
                label.startsWith(want) -> 1
                label.contains(want) -> 2
                want.contains(label) -> 3
                else -> 9
            }
        } ?: return fail("No apps found")
        val label = best.loadLabel(pm).toString()
        val l = label.lowercase()
        if (!(l == want || l.startsWith(want) || l.contains(want) || want.contains(l))) return fail("No app called $name")
        val launch = pm.getLaunchIntentForPackage(best.activityInfo.packageName) ?: return fail("Can't open $label")
        return startOrFail(ctx, launch, "Opening $label")
    }

    /** A number as given, or the first number of a contact with that name. */
    private fun number(ctx: Context, to: String): Pair<String, String>? {
        val digits = to.filter { it.isDigit() || it == '+' }
        if (digits.length >= 3) return digits to to
        if (!granted(ctx, Manifest.permission.READ_CONTACTS)) return null
        ctx.contentResolver.query(
            ContactsContract.CommonDataKinds.Phone.CONTENT_URI,
            arrayOf(ContactsContract.CommonDataKinds.Phone.NUMBER, ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME),
            "${ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME} LIKE ?", arrayOf("%$to%"), null,
        )?.use { c -> if (c.moveToFirst()) return c.getString(0) to c.getString(1) }
        return null
    }

    private fun call(ctx: Context, to: String): Map<String, Any?> {
        val (num, who) = number(ctx, to) ?: return fail(
            if (granted(ctx, Manifest.permission.READ_CONTACTS)) "No contact called $to" else "Allow contacts to call by name")
        val direct = granted(ctx, Manifest.permission.CALL_PHONE)
        val i = Intent(if (direct) Intent.ACTION_CALL else Intent.ACTION_DIAL, Uri.parse("tel:" + Uri.encode(num)))
        return startOrFail(ctx, i, if (direct) "Calling $who" else "Dialing $who")
    }

    private fun text(ctx: Context, to: String, body: String): Map<String, Any?> {
        val (num, who) = number(ctx, to) ?: return fail(
            if (granted(ctx, Manifest.permission.READ_CONTACTS)) "No contact called $to" else "Allow contacts to text by name")
        val i = Intent(Intent.ACTION_SENDTO, Uri.parse("smsto:" + Uri.encode(num))).putExtra("sms_body", body)
        return startOrFail(ctx, i, "Message to $who ready - tap send")
    }
}

/** A reminder's time came. */
class ReminderReceiver : BroadcastReceiver() {
    override fun onReceive(ctx: Context, intent: Intent) {
        PhoneActions.fire(ctx, intent.getStringExtra("id") ?: "", intent.getStringExtra("text") ?: "Reminder")
    }
}

/** Reminders survive a restart of the phone, and so does the glasses link (if it was wanted). */
class BootReceiver : BroadcastReceiver() {
    override fun onReceive(ctx: Context, intent: Intent) {
        if (intent.action != Intent.ACTION_BOOT_COMPLETED) return
        PhoneActions.rescheduleAll(ctx)
        if (com.vox.evensuite.ble.BleManager.hasSavedLink(ctx)) {
            try {
                ctx.startForegroundService(
                    Intent(ctx, SuiteService::class.java).setAction(SuiteService.ACTION_LINK)
                )
            } catch (e: Exception) {
                android.util.Log.e("BootReceiver", "glasses link: $e")
            }
        }
    }
}
