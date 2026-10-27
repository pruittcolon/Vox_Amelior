package com.vox.evensuite

import android.Manifest
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.location.Location
import android.location.LocationManager
import android.content.pm.PackageManager
import android.media.projection.MediaProjectionManager
import android.net.Uri
import android.os.Build
import android.provider.Settings
import android.view.KeyEvent
import com.vox.evensuite.ble.BleManager
import com.vox.evensuite.speech.GlassesMic
import com.vox.evensuite.speech.SpeechEngine
import com.vox.evensuite.speech.TtsEngine
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel

class MainActivity : FlutterActivity() {

    private lateinit var bleChannel: MethodChannel
    private lateinit var systemChannel: MethodChannel
    private val bleEvents = Sink()
    private val speechEvents = Sink()
    private val screenEvents = Sink()
    private val ttsEvents = Sink()
    private val notifyEvents = Sink()
    private var tts: TtsEngine? = null
    private var launchTab: String? = null
    private var pendingScreen: MethodChannel.Result? = null
    private var pendingScreenConfig: Map<*, *>? = null
    private var speech: SpeechEngine? = null

    private var pendingPermissions: MethodChannel.Result? = null
    private var pendingAsked: List<String> = emptyList()
    private var volumeKeys = false

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        val messenger = flutterEngine.dartExecutor.binaryMessenger
        BleManager.init(this)

        bleChannel = MethodChannel(messenger, "method.bluetooth")
        bleChannel.setMethodCallHandler(::onBleCall)
        EventChannel(messenger, "eventBleReceive").setStreamHandler(bleEvents)
        BleManager.listener = object : BleManager.Listener {
            override fun onPairFound(info: Map<String, String>) {
                bleChannel.invokeMethod("foundPairedGlasses", info)
            }

            override fun onStatus(method: String, info: Map<String, Any>) {
                bleChannel.invokeMethod(method, info)
            }

            override fun onReceive(lr: String, data: ByteArray) {
                bleEvents.send(mapOf("lr" to lr, "data" to data, "type" to "Receive"))
            }
        }

        val engine = SpeechEngine(this) { event -> speechEvents.send(event) }
        speech = engine
        MethodChannel(messenger, "suite/speech").setMethodCallHandler { call, result ->
            when (call.method) {
                "available" -> result.success(engine.isAvailable())
                "start" -> {
                    engine.start(call.arguments as? Map<*, *>)
                    result.success(null)
                }
                "stop" -> {
                    engine.stop()
                    result.success(null)
                }
                "finish" -> {
                    engine.finish()
                    result.success(null)
                }
                else -> result.notImplemented()
            }
        }
        EventChannel(messenger, "suite/speechEvents").setStreamHandler(speechEvents)

        val voice = TtsEngine(this) { event -> ttsEvents.send(event) }
        tts = voice
        MethodChannel(messenger, "suite/tts").setMethodCallHandler { call, result ->
            val a = call.arguments as? Map<*, *>
            when (call.method) {
                "speak" -> {
                    voice.speak(a?.get("text") as? String ?: "", a?.get("id") as? String ?: "",
                        a?.get("flush") as? Boolean ?: false)
                    result.success(null)
                }
                "stop" -> {
                    voice.stop()
                    result.success(null)
                }
                "configure" -> {
                    voice.configure(a)
                    result.success(null)
                }
                "voices" -> result.success(voice.voices())
                "engines" -> result.success(voice.engines())
                else -> result.notImplemented()
            }
        }
        EventChannel(messenger, "suite/ttsEvents").setStreamHandler(ttsEvents)

        EventChannel(messenger, "suite/notifications").setStreamHandler(notifyEvents)
        NotifyListener.onPosted = { m -> runOnUiThread { notifyEvents.send(m) } }
        launchTab = intent?.getStringExtra(EXTRA_TAB)

        MethodChannel(messenger, "suite/screen").setMethodCallHandler(::onScreenCall)
        EventChannel(messenger, "suite/screenFrames").setStreamHandler(screenEvents)
        SuiteService.onMirrorFrame = { frame -> runOnUiThread { screenEvents.send(frame) } }
        SuiteService.onMirrorStopped = { reason ->
            runOnUiThread { screenEvents.send(mapOf("stopped" to true, "reason" to (reason ?: ""))) }
        }

        systemChannel = MethodChannel(messenger, "suite/system")
        systemChannel.setMethodCallHandler(::onSystemCall)
        SuiteService.onStopListening = {
            runOnUiThread { systemChannel.invokeMethod("stopListening", null) }
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        val tab = intent.getStringExtra(EXTRA_TAB) ?: return
        if (::systemChannel.isInitialized) systemChannel.invokeMethod("openTab", tab)
    }

    override fun onDestroy() {
        if (isFinishing) {
            // Swiped away: nothing is left to drive the mic or the glasses.
            speech?.dispose()
            tts?.dispose()
            NotifyListener.onPosted = null
            BleManager.listener = null
            BleManager.disconnect()
            SuiteService.update(this, listening = false, glasses = false, text = "")
            SuiteService.onStopListening = null
            SuiteService.stopMirror(this)
            SuiteService.onMirrorFrame = null
            SuiteService.onMirrorStopped = null
        }
        super.onDestroy()
    }

    // ------------------------------------------------------------ bluetooth

    private fun onBleCall(call: MethodCall, result: MethodChannel.Result) {
        val args = call.arguments as? Map<*, *>
        when (call.method) {
            "startScan" -> {
                val error = BleManager.startScan()
                if (error == null) result.success("Scanning") else result.error("Bluetooth", error, null)
            }
            "stopScan" -> {
                BleManager.stopScan()
                result.success(null)
            }
            "connectToGlasses" -> {
                val channel = (args?.get("channel") as? String)
                    ?: (args?.get("deviceName") as? String)?.removePrefix("Pair_")
                    ?: ""
                if (channel.isEmpty()) {
                    result.error("InvalidArguments", "No glasses channel given", null)
                    return
                }
                BleManager.prefer2M = args?.get("phy2m") as? Boolean ?: true
                val error = BleManager.connect(channel, args?.get("saved") as? Map<*, *>)
                if (error == null) result.success("Connecting") else result.error("Bluetooth", error, null)
            }
            "disconnectFromGlasses" -> {
                BleManager.disconnect()
                result.success(null)
            }
            "send" -> {
                val data = args?.get("data") as? ByteArray
                if (data != null && data.isNotEmpty()) BleManager.send(data, args?.get("lr") as? String)
                result.success(null)
            }
            "sendBatch" -> {
                val packets = (args?.get("packets") as? List<*>)?.filterIsInstance<ByteArray>() ?: emptyList()
                val gap = (args?.get("gapMs") as? Number)?.toLong() ?: 0L
                BleManager.sendBatch(packets, args?.get("lr") as? String, gap.coerceIn(0L, 50L)) { sent ->
                    result.success(sent)
                }
            }
            "status" -> result.success(BleManager.status())
            // Kept for compatibility with the demo's Dart code; the app listens with the phone mic.
            "startEvenAI", "stopEvenAI" -> result.success(null)
            else -> result.notImplemented()
        }
    }

    // ---------------------------------------------------------------- screen

    private fun onScreenCall(call: MethodCall, result: MethodChannel.Result) {
        val args = call.arguments as? Map<*, *>
        when (call.method) {
            "start" -> {
                // Android shows its own "start recording or casting?" prompt.
                pendingScreen?.success("busy")
                pendingScreen = result
                pendingScreenConfig = args
                val mpm = getSystemService(MediaProjectionManager::class.java)
                try {
                    startActivityForResult(mpm.createScreenCaptureIntent(), SCREEN_REQUEST)
                } catch (e: Exception) {
                    pendingScreen = null
                    result.success("error: ${e.message}")
                }
            }
            "stop" -> {
                SuiteService.stopMirror(this)
                result.success(null)
            }
            "configure" -> {
                SuiteService.configureMirror(args)
                result.success(null)
            }
            else -> result.notImplemented()
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != SCREEN_REQUEST) return
        val result = pendingScreen ?: return
        pendingScreen = null
        if (resultCode != RESULT_OK || data == null) {
            result.success("denied")
            return
        }
        val error = SuiteService.startMirror(this, resultCode, data, pendingScreenConfig)
        result.success(error?.let { "error: $it" } ?: "ok")
    }

    // --------------------------------------------------------------- system

    private val prefs by lazy { getSharedPreferences("even_suite", Context.MODE_PRIVATE) }

    private fun onSystemCall(call: MethodCall, result: MethodChannel.Result) {
        val args = call.arguments as? Map<*, *>
        when (call.method) {
            "prefsGetAll" -> result.success(HashMap(prefs.all))
            "prefsSet" -> {
                val key = args?.get("key") as? String
                if (key == null) {
                    result.error("InvalidArguments", "key missing", null)
                    return
                }
                val edit = prefs.edit()
                when (val value = args?.get("value")) {
                    null -> edit.remove(key)
                    is Boolean -> edit.putBoolean(key, value)
                    is Int -> edit.putLong(key, value.toLong())
                    is Long -> edit.putLong(key, value)
                    is Double -> edit.putFloat(key, value.toFloat())
                    else -> edit.putString(key, value.toString())
                }
                edit.apply()
                result.success(null)
            }
            "service" -> {
                val error = SuiteService.update(
                    this,
                    listening = args?.get("listening") as? Boolean ?: false,
                    glasses = args?.get("glasses") as? Boolean ?: false,
                    text = args?.get("text") as? String ?: "",
                )
                result.success(error)
            }
            "permissions" -> askPermissions(
                (args?.get("kinds") as? List<*>)?.filterIsInstance<String>() ?: emptyList(), result
            )
            "termuxInstalled" -> result.success(termuxInstalled())
            "runTermux" -> result.success(runTermux(args?.get("command") as? String ?: ""))
            "openAppSettings" -> {
                startActivity(
                    Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.parse("package:$packageName"))
                        .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                )
                result.success(null)
            }
            "openTermux" -> {
                val launch = packageManager.getLaunchIntentForPackage(TERMUX)
                if (launch != null) startActivity(launch.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
                result.success(launch != null)
            }
            "nativeStats" -> {
                val m = HashMap<String, Any>()
                m["ble"] = BleManager.stats()
                m["mic"] = GlassesMic.takeStats()
                speech?.let { m["speech"] = it.takeStats() }
                SuiteService.mirrorStats()?.let { m["screen"] = it }
                result.success(m)
            }
            "volumeKeys" -> {
                volumeKeys = args?.get("enabled") as? Boolean ?: false
                result.success(null)
            }
            "notify" -> {
                postNotification(
                    (args?.get("id") as? Number)?.toInt() ?: 1,
                    args?.get("title") as? String ?: "",
                    args?.get("text") as? String ?: "",
                    args?.get("tab") as? String ?: "",
                )
                result.success(null)
            }
            "notificationAccess" -> result.success(hasNotificationAccess())
            "openNotificationAccess" -> {
                try {
                    startActivity(Intent(Settings.ACTION_NOTIFICATION_LISTENER_SETTINGS).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
                } catch (_: Exception) {
                    startActivity(
                        Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.parse("package:$packageName"))
                            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                    )
                }
                result.success(null)
            }
            "location" -> result.success(lastLocation())
            "launchTab" -> {
                result.success(launchTab)
                launchTab = null
            }
            else -> result.notImplemented()
        }
    }

    // --------------------------------------------------------- notifications

    /** "Claude finished ..." on the phone; tapping it opens [tab] in the app. */
    private fun postNotification(id: Int, title: String, text: String, tab: String) {
        if (Build.VERSION.SDK_INT >= 33 &&
            checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
        ) return
        val nm = getSystemService(NotificationManager::class.java)
        if (nm.getNotificationChannel(DONE_CHANNEL) == null) {
            nm.createNotificationChannel(
                NotificationChannel(DONE_CHANNEL, "Claude finished", NotificationManager.IMPORTANCE_HIGH).apply {
                    description = "When Claude Code finishes a job in Termux"
                }
            )
        }
        val open = PendingIntent.getActivity(
            this, 100 + (id % 1000),
            Intent(this, MainActivity::class.java)
                .addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP)
                .putExtra(EXTRA_TAB, tab),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )
        val n = Notification.Builder(this, DONE_CHANNEL)
            .setSmallIcon(R.drawable.ic_notification)
            .setContentTitle(title)
            .setContentText(text)
            .setStyle(Notification.BigTextStyle().bigText(text))
            .setContentIntent(open)
            .setAutoCancel(true)
            .setCategory(Notification.CATEGORY_STATUS)
            .build()
        try {
            nm.notify(DONE_BASE + (id % 1000), n)
        } catch (_: SecurityException) {
        }
    }

    private fun hasNotificationAccess(): Boolean {
        val enabled = Settings.Secure.getString(contentResolver, "enabled_notification_listeners") ?: return false
        val me = ComponentName(this, NotifyListener::class.java)
        return enabled.split(":").any { ComponentName.unflattenFromString(it) == me }
    }

    /** Last known position for the weather on the look-up screen (no GPS fix is requested). */
    private fun lastLocation(): Map<String, Double>? {
        val fine = checkSelfPermission(Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED
        val coarse = checkSelfPermission(Manifest.permission.ACCESS_COARSE_LOCATION) == PackageManager.PERMISSION_GRANTED
        if (!fine && !coarse) return null
        val lm = getSystemService(LocationManager::class.java) ?: return null
        var best: Location? = null
        for (provider in listOf(LocationManager.NETWORK_PROVIDER, LocationManager.GPS_PROVIDER, LocationManager.PASSIVE_PROVIDER)) {
            val l = try {
                lm.getLastKnownLocation(provider)
            } catch (_: Exception) {
                null
            } ?: continue
            if (best == null || l.time > best.time) best = l
        }
        return best?.let { mapOf("lat" to it.latitude, "lon" to it.longitude) }
    }

    private fun permissionsFor(kind: String): List<String> = when (kind) {
        "bluetooth" -> if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            listOf(Manifest.permission.BLUETOOTH_SCAN, Manifest.permission.BLUETOOTH_CONNECT, Manifest.permission.ACCESS_FINE_LOCATION)
        } else {
            listOf(Manifest.permission.ACCESS_FINE_LOCATION)
        }
        "microphone" -> listOf(Manifest.permission.RECORD_AUDIO)
        "notifications" -> if (Build.VERSION.SDK_INT >= 33) listOf(Manifest.permission.POST_NOTIFICATIONS) else emptyList()
        "termux" -> listOf(TERMUX_PERMISSION)
        "location" -> listOf(Manifest.permission.ACCESS_COARSE_LOCATION, Manifest.permission.ACCESS_FINE_LOCATION)
        else -> emptyList()
    }

    private fun askPermissions(kinds: List<String>, result: MethodChannel.Result) {
        val wanted = kinds.flatMap(::permissionsFor).distinct()
        val missing = wanted.filter { checkSelfPermission(it) != PackageManager.PERMISSION_GRANTED }
        if (missing.isEmpty()) {
            result.success(kinds.associateWith { true })
            return
        }
        pendingPermissions?.success(kinds.associateWith { false })
        pendingPermissions = result
        pendingAsked = kinds
        requestPermissions(missing.toTypedArray(), PERMISSION_REQUEST)
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode != PERMISSION_REQUEST) return
        val result = pendingPermissions ?: return
        pendingPermissions = null
        result.success(pendingAsked.associateWith { kind ->
            permissionsFor(kind).all { checkSelfPermission(it) == PackageManager.PERMISSION_GRANTED }
        })
    }

    // --------------------------------------------------------------- termux

    private fun termuxInstalled() = try {
        packageManager.getPackageInfo(TERMUX, 0)
        true
    } catch (_: PackageManager.NameNotFoundException) {
        false
    }

    /** Runs [command] with Termux's bash in the background. Returns "ok" or a reason. */
    private fun runTermux(command: String): String {
        if (command.isBlank()) return "No command set"
        if (!termuxInstalled()) return "not-installed"
        if (checkSelfPermission(TERMUX_PERMISSION) != PackageManager.PERMISSION_GRANTED) return "no-permission"
        val intent = Intent("com.termux.RUN_COMMAND").apply {
            setClassName(TERMUX, "com.termux.app.RunCommandService")
            putExtra("com.termux.RUN_COMMAND_PATH", "/data/data/com.termux/files/usr/bin/bash")
            putExtra("com.termux.RUN_COMMAND_ARGUMENTS", arrayOf("-lc", command))
            putExtra("com.termux.RUN_COMMAND_WORKDIR", "/data/data/com.termux/files/home")
            putExtra("com.termux.RUN_COMMAND_BACKGROUND", true)
            putExtra("com.termux.RUN_COMMAND_SESSION_ACTION", "0")
        }
        return try {
            startForegroundService(intent)
            "ok"
        } catch (e: Exception) {
            e.message ?: e.toString()
        }
    }

    // ---------------------------------------------------------- volume keys

    override fun dispatchKeyEvent(event: KeyEvent): Boolean {
        if (volumeKeys && (event.keyCode == KeyEvent.KEYCODE_VOLUME_UP || event.keyCode == KeyEvent.KEYCODE_VOLUME_DOWN)) {
            if (event.action == KeyEvent.ACTION_DOWN && event.repeatCount == 0) {
                systemChannel.invokeMethod("volumeKey", if (event.keyCode == KeyEvent.KEYCODE_VOLUME_UP) "up" else "down")
            }
            return true
        }
        return super.dispatchKeyEvent(event)
    }

    /** Event channel sink that tolerates Flutter not listening yet. */
    private class Sink : EventChannel.StreamHandler {
        private var sink: EventChannel.EventSink? = null

        override fun onListen(arguments: Any?, events: EventChannel.EventSink?) {
            sink = events
        }

        override fun onCancel(arguments: Any?) {
            sink = null
        }

        fun send(event: Any) {
            sink?.success(event)
        }
    }

    companion object {
        private const val PERMISSION_REQUEST = 41
        private const val SCREEN_REQUEST = 42
        private const val TERMUX = "com.termux"
        private const val TERMUX_PERMISSION = "com.termux.permission.RUN_COMMAND"
        private const val EXTRA_TAB = "tab"
        private const val DONE_CHANNEL = "claude_done"
        private const val DONE_BASE = 2000
    }
}
