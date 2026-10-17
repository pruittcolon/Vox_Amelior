package com.vox.narrator

import android.Manifest
import android.content.ContentValues
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.Handler
import android.os.Looper
import android.provider.MediaStore
import android.provider.OpenableColumns
import android.view.WindowManager
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import java.io.File
import java.util.concurrent.Executors

/** Sends events from native code (notification buttons, audio focus) to Dart. */
object Bridge {
    private val main = Handler(Looper.getMainLooper())
    var channel: MethodChannel? = null

    fun emit(method: String, args: Any? = null) {
        main.post { channel?.invokeMethod(method, args) }
    }
}

class MainActivity : FlutterActivity() {
    private val main = Handler(Looper.getMainLooper())
    private val io = Executors.newSingleThreadExecutor()
    private val player by lazy { PcmPlayer(applicationContext) }
    private val recorder = MicRecorder()
    private var pendingShare: Map<String, Any?>? = null
    private val permissionWaiters = mutableMapOf<Int, MethodChannel.Result>()
    private var nextPermissionCode = 7100

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        pendingShare = readShare(intent)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        val share = readShare(intent) ?: return
        Bridge.emit("shared", share)
    }

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        val channel = MethodChannel(flutterEngine.dartExecutor.binaryMessenger, "narrator/native")
        Bridge.channel = channel
        channel.setMethodCallHandler { call, result ->
            try {
                handle(call, result)
            } catch (e: Exception) {
                result.error("native", e.message ?: e.toString(), null)
            }
        }
    }

    override fun onDestroy() {
        if (isFinishing) {
            player.stop()
            NarratorService.stop(this)
        }
        super.onDestroy()
    }

    private fun background(result: MethodChannel.Result, work: () -> Any?) {
        io.execute {
            try {
                val value = work()
                main.post { result.success(value) }
            } catch (e: Exception) {
                main.post { result.error("native", e.message ?: e.toString(), null) }
            }
        }
    }

    private fun handle(call: MethodCall, result: MethodChannel.Result) {
        when (call.method) {
            // ---- Speech output ------------------------------------------------
            "player.open" -> {
                player.open(call.argument<Int>("sampleRate")!!)
                result.success(null)
            }
            "player.write" -> player.write(call.argument<ByteArray>("pcm")!!) { frames ->
                main.post { result.success(frames) }
            }
            "player.position" -> result.success(player.position())
            "player.pause" -> { player.pause(); result.success(null) }
            "player.resume" -> { player.resume(); result.success(null) }
            "player.stop" -> { player.stop(); result.success(null) }

            // ---- Background service (notification, lock screen, headset) -------
            "service.start", "service.update" -> {
                NarratorService.show(
                    this,
                    call.argument<String>("kind") ?: "play",
                    call.argument<String>("title") ?: "Narrator Studio",
                    call.argument<String>("text") ?: "",
                    call.argument<Boolean>("playing") ?: true,
                )
                result.success(null)
            }
            "service.stop" -> { NarratorService.stop(this); result.success(null) }

            // ---- Audio files ----------------------------------------------------
            "encode.m4a" -> background(result) {
                M4aEncoder.encode(
                    File(call.argument<String>("wav")!!),
                    File(call.argument<String>("out")!!),
                    call.argument<Int>("bitrate") ?: 96000,
                ) { p -> Bridge.emit("encodeProgress", p) }
                null
            }
            "decode.audio" -> background(result) {
                AudioDecoder.decode(
                    call.argument<String>("path")!!,
                    call.argument<Double>("maxSeconds") ?: 30.0,
                )
            }

            // ---- Microphone (voice cloning) -------------------------------------
            "record.start" -> {
                if (!hasPermission(Manifest.permission.RECORD_AUDIO)) {
                    result.error("permission", "Microphone permission is needed to record a voice sample.", null)
                } else {
                    recorder.start(call.argument<Int>("sampleRate") ?: 24000)
                    result.success(null)
                }
            }
            "record.level" -> result.success(recorder.level)
            "record.stop" -> background(result) { recorder.stop() }

            // ---- Permissions ----------------------------------------------------
            "perm.request" -> {
                val name = when (call.argument<String>("name")) {
                    "mic" -> Manifest.permission.RECORD_AUDIO
                    "notifications" ->
                        if (Build.VERSION.SDK_INT >= 33) Manifest.permission.POST_NOTIFICATIONS else null
                    else -> null
                }
                if (name == null || hasPermission(name)) {
                    result.success(true)
                } else {
                    val code = nextPermissionCode++
                    permissionWaiters[code] = result
                    requestPermissions(arrayOf(name), code)
                }
            }

            // ---- Files ----------------------------------------------------------
            "files.saveToDownloads" -> background(result) {
                saveToDownloads(
                    File(call.argument<String>("path")!!),
                    call.argument<String>("name")!!,
                    call.argument<String>("mime") ?: "application/octet-stream",
                )
            }
            "files.share" -> {
                val file = File(call.argument<String>("path")!!)
                val uri = FileProvider.getUriForFile(this, "$packageName.files", file)
                val send = Intent(Intent.ACTION_SEND).apply {
                    type = call.argument<String>("mime") ?: "application/octet-stream"
                    putExtra(Intent.EXTRA_STREAM, uri)
                    putExtra(Intent.EXTRA_SUBJECT, call.argument<String>("subject") ?: file.name)
                    addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                }
                startActivity(Intent.createChooser(send, "Share"))
                result.success(null)
            }
            "files.shared" -> {
                result.success(pendingShare)
                pendingShare = null
            }

            // ---- App ------------------------------------------------------------
            "app.keepScreenOn" -> {
                if (call.argument<Boolean>("on") == true) {
                    window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                } else {
                    window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                }
                result.success(null)
            }
            "app.openClaude" -> {
                val launch = packageManager.getLaunchIntentForPackage("com.anthropic.claude")
                if (launch != null) {
                    startActivity(launch)
                    result.success("app")
                } else {
                    startActivity(Intent(Intent.ACTION_VIEW, Uri.parse("https://claude.ai/new")))
                    result.success("web")
                }
            }
            "app.openUrl" -> {
                startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(call.argument<String>("url")!!)))
                result.success(null)
            }
            else -> result.notImplemented()
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        val waiter = permissionWaiters.remove(requestCode) ?: return
        waiter.success(grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED)
    }

    private fun hasPermission(name: String) =
        ContextCompat.checkSelfPermission(this, name) == PackageManager.PERMISSION_GRANTED

    /** Text or a file shared / opened from another app, copied into our cache. */
    private fun readShare(intent: Intent?): Map<String, Any?>? {
        if (intent == null) return null
        return try {
            when (intent.action) {
                Intent.ACTION_SEND -> {
                    @Suppress("DEPRECATION")
                    val stream = intent.getParcelableExtra<Uri>(Intent.EXTRA_STREAM)
                    if (stream != null) {
                        copyIn(stream, intent.type)
                    } else {
                        val text = intent.getStringExtra(Intent.EXTRA_TEXT) ?: return null
                        mapOf(
                            "text" to text,
                            "title" to intent.getStringExtra(Intent.EXTRA_SUBJECT),
                        )
                    }
                }
                Intent.ACTION_VIEW -> intent.data?.let { copyIn(it, intent.type) }
                else -> null
            }
        } catch (e: Exception) {
            mapOf("error" to (e.message ?: e.toString()))
        }
    }

    private fun copyIn(uri: Uri, mime: String?): Map<String, Any?> {
        var name = "shared"
        contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
            if (c.moveToFirst() && !c.isNull(0)) name = c.getString(0)
        }
        val type = mime ?: contentResolver.getType(uri)
        if (!name.contains('.')) {
            name += when {
                type == "application/pdf" -> ".pdf"
                type == "text/markdown" -> ".md"
                else -> ".txt"
            }
        }
        val dir = File(cacheDir, "shared").apply { mkdirs() }
        val safe = name.replace(Regex("[^A-Za-z0-9._ -]"), "_")
        val out = File(dir, "${System.currentTimeMillis()}-$safe")
        contentResolver.openInputStream(uri)!!.use { input -> out.outputStream().use { input.copyTo(it) } }
        return mapOf("path" to out.absolutePath, "name" to name, "mime" to type)
    }

    private fun saveToDownloads(file: File, name: String, mime: String): String {
        if (Build.VERSION.SDK_INT >= 29) {
            val values = ContentValues().apply {
                put(MediaStore.Downloads.DISPLAY_NAME, name)
                put(MediaStore.Downloads.MIME_TYPE, mime)
                put(MediaStore.Downloads.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS + "/Narrator Studio")
                put(MediaStore.Downloads.IS_PENDING, 1)
            }
            val uri = contentResolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
                ?: throw IllegalStateException("Android refused to create the file in Downloads")
            contentResolver.openOutputStream(uri)!!.use { out -> file.inputStream().use { it.copyTo(out) } }
            values.clear()
            values.put(MediaStore.Downloads.IS_PENDING, 0)
            contentResolver.update(uri, values, null, null)
            return "Downloads/Narrator Studio/$name"
        }
        if (!hasPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE)) {
            throw IllegalStateException("Storage permission is off. Use Share instead.")
        }
        @Suppress("DEPRECATION")
        val dir = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), "Narrator Studio")
        dir.mkdirs()
        val out = File(dir, name)
        file.copyTo(out, overwrite = true)
        return out.absolutePath
    }
}
