package com.vox.evensuite.speech

import android.content.Context
import android.content.Intent
import android.media.AudioFormat
import android.media.AudioManager
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.util.Log

/**
 * Android's built-in speech recognizer (no extra SDK, no API key).
 *
 * In continuous mode every recognition session is restarted as soon as it
 * ends, which is how the app keeps listening for the wake word. All calls
 * must happen on the main thread (SpeechRecognizer requires it).
 */
class SpeechEngine(
    private val context: Context,
    private val emit: (Map<String, Any?>) -> Unit,
) : RecognitionListener {

    private val main = Handler(Looper.getMainLooper())
    private val audio = context.getSystemService(AudioManager::class.java)

    private var recognizer: SpeechRecognizer? = null
    private var usingOnDevice = false
    private var active = false
    private var continuous = false
    private var language: String? = null
    private var preferOffline = true
    private var quiet = true
    private var engine = "auto"
    private var audioSource = "phone" // or "glasses" (Android 13+)
    private var failures = 0
    private var muted = false
    private var lastLevelAt = 0L
    private var finishing = false
    private var usingGlasses = false
    private var silenceMs = 1500L

    // Diagnostics
    private var statSessions = 0
    private val statErrors = mutableMapOf<String, Int>()

    fun takeStats(): Map<String, Any> {
        val m = mapOf("sessions" to statSessions, "errors" to HashMap(statErrors), "onDevice" to usingOnDevice,
            "glasses" to usingGlasses)
        statSessions = 0
        statErrors.clear()
        return m
    }

    fun isAvailable(): Boolean = SpeechRecognizer.isRecognitionAvailable(context) ||
        (Build.VERSION.SDK_INT >= 31 && SpeechRecognizer.isOnDeviceRecognitionAvailable(context))

    fun start(args: Map<*, *>?) {
        continuous = args?.get("continuous") as? Boolean ?: false
        language = (args?.get("language") as? String)?.takeIf { it.isNotBlank() }
        preferOffline = args?.get("preferOffline") as? Boolean ?: true
        quiet = args?.get("quiet") as? Boolean ?: true
        audioSource = args?.get("audioSource") as? String ?: "phone"
        silenceMs = ((args?.get("silenceMs") as? Number)?.toLong() ?: 1500L).coerceIn(500L, 5000L)
        val newEngine = args?.get("engine") as? String ?: "auto"
        if (newEngine != engine) {
            destroyRecognizer()
            engine = newEngine
        }
        failures = 0
        active = true
        finishing = false
        main.removeCallbacksAndMessages(null)
        emit(mapOf("type" to "state", "listening" to true))
        startSession()
    }

    /** Stop listening for good. */
    fun stop() {
        active = false
        finishing = false
        main.removeCallbacksAndMessages(null)
        try {
            recognizer?.cancel()
        } catch (_: Exception) {
        }
        GlassesMic.closeStreamNow()
        unmute()
        emit(mapOf("type" to "state", "listening" to false))
    }

    /** End the current utterance now and deliver its final text (push-to-talk release). */
    fun finish() {
        if (!active) return
        finishing = true
        if (usingGlasses) GlassesMic.endStream()
        try {
            recognizer?.stopListening()
        } catch (_: Exception) {
        }
    }

    fun dispose() {
        stop()
        destroyRecognizer()
    }

    // ------------------------------------------------------------ sessions

    private fun startSession() {
        if (!active) return
        val r = recognizer ?: createRecognizer() ?: run {
            active = false
            emit(mapOf("type" to "error", "code" to -1, "message" to "Speech recognition is not available on this phone. Install or enable the Google app.", "fatal" to true))
            emit(mapOf("type" to "state", "listening" to false))
            return
        }
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
            putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 3)
            putExtra(RecognizerIntent.EXTRA_CALLING_PACKAGE, context.packageName)
            if (preferOffline) putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, true)
            language?.let { putExtra(RecognizerIntent.EXTRA_LANGUAGE, it) }
            // How long a pause ends what you are saying (the app's "send after a pause").
            putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_COMPLETE_SILENCE_LENGTH_MILLIS, silenceMs)
            putExtra(RecognizerIntent.EXTRA_SPEECH_INPUT_POSSIBLY_COMPLETE_SILENCE_LENGTH_MILLIS, (silenceMs - 400L).coerceAtLeast(400L))
        }
        // Glasses microphone: decoded G1 audio through a pipe. A recognizer that
        // does not support this simply uses the phone's microphone instead.
        usingGlasses = false
        if (audioSource == "glasses" && Build.VERSION.SDK_INT >= 33) {
            val pipe = GlassesMic.openStream()
            if (pipe != null) {
                usingGlasses = true
                intent.putExtra(RecognizerIntent.EXTRA_AUDIO_SOURCE, pipe)
                intent.putExtra(RecognizerIntent.EXTRA_AUDIO_SOURCE_CHANNEL_COUNT, 1)
                intent.putExtra(RecognizerIntent.EXTRA_AUDIO_SOURCE_ENCODING, AudioFormat.ENCODING_PCM_16BIT)
                intent.putExtra(RecognizerIntent.EXTRA_AUDIO_SOURCE_SAMPLING_RATE, 16000)
            }
        }
        emit(mapOf("type" to "source", "glasses" to usingGlasses))
        // The Google recognizer plays a chime at every start. While waiting for
        // the wake word that would chime every few seconds, so mute briefly.
        statSessions++
        if (quiet && continuous) mute()
        try {
            r.startListening(intent)
        } catch (e: Exception) {
            Log.e(TAG, "startListening: $e")
            destroyRecognizer()
            retryLater()
        }
    }

    private fun createRecognizer(): SpeechRecognizer? {
        val onDeviceOk = Build.VERSION.SDK_INT >= 31 && SpeechRecognizer.isOnDeviceRecognitionAvailable(context)
        val useOnDevice = when (engine) {
            "ondevice" -> onDeviceOk
            "default" -> false
            else -> onDeviceOk && Build.VERSION.SDK_INT >= 33
        }
        val r = try {
            if (useOnDevice && Build.VERSION.SDK_INT >= 31) {
                SpeechRecognizer.createOnDeviceSpeechRecognizer(context)
            } else if (SpeechRecognizer.isRecognitionAvailable(context)) {
                SpeechRecognizer.createSpeechRecognizer(context)
            } else {
                null
            }
        } catch (e: Exception) {
            Log.e(TAG, "create recognizer: $e")
            null
        } ?: return null
        usingOnDevice = useOnDevice
        r.setRecognitionListener(this)
        recognizer = r
        emit(mapOf("type" to "engine", "onDevice" to useOnDevice))
        return r
    }

    private fun destroyRecognizer() {
        try {
            recognizer?.destroy()
        } catch (_: Exception) {
        }
        recognizer = null
    }

    private fun next(delayMs: Long) {
        if (!active) return
        if (!continuous) {
            active = false
            unmute()
            emit(mapOf("type" to "state", "listening" to false))
            return
        }
        main.postDelayed({ startSession() }, delayMs)
    }

    private fun retryLater() {
        failures++
        val delay = minOf(10_000L, 300L * (1L shl minOf(failures, 5)))
        next(delay)
    }

    private fun mute() {
        if (muted) return
        muted = true
        for (stream in STREAMS) {
            try {
                audio.adjustStreamVolume(stream, AudioManager.ADJUST_MUTE, 0)
            } catch (_: Exception) {
                // Do Not Disturb can forbid touching some streams.
            }
        }
    }

    private fun unmute() {
        if (!muted) return
        muted = false
        for (stream in STREAMS) {
            try {
                audio.adjustStreamVolume(stream, AudioManager.ADJUST_UNMUTE, 0)
            } catch (_: Exception) {
            }
        }
    }

    // ---------------------------------------------------- RecognitionListener

    override fun onReadyForSpeech(params: Bundle?) {
        // The start chime has played by now.
        main.postDelayed({ unmute() }, 350)
        emit(mapOf("type" to "ready"))
    }

    override fun onBeginningOfSpeech() {
        emit(mapOf("type" to "begin"))
    }

    override fun onRmsChanged(rmsdB: Float) {
        val now = SystemClock.uptimeMillis()
        if (now - lastLevelAt < 120) return
        lastLevelAt = now
        emit(mapOf("type" to "level", "value" to rmsdB.toDouble()))
    }

    override fun onBufferReceived(buffer: ByteArray?) {}

    override fun onEndOfSpeech() {
        emit(mapOf("type" to "endOfSpeech"))
    }

    override fun onError(error: Int) {
        statErrors["$error"] = (statErrors["$error"] ?: 0) + 1
        unmute()
        if (usingGlasses) GlassesMic.closeStreamNow()
        val wasFinishing = finishing
        finishing = false
        when (error) {
            SpeechRecognizer.ERROR_NO_MATCH, SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> {
                failures = 0
                emit(mapOf("type" to "end", "empty" to true, "finishing" to wasFinishing))
                next(120)
            }
            SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> {
                active = false
                emit(mapOf("type" to "error", "code" to error, "message" to "Microphone permission is off for Even Suite.", "fatal" to true))
                emit(mapOf("type" to "state", "listening" to false))
            }
            SpeechRecognizer.ERROR_RECOGNIZER_BUSY, SpeechRecognizer.ERROR_CLIENT -> {
                destroyRecognizer()
                emit(mapOf("type" to "end", "empty" to true, "finishing" to wasFinishing))
                retryLater()
            }
            else -> {
                // 12/13: language not supported/unavailable on-device -> use the normal recognizer.
                if ((error == 12 || error == 13) && usingOnDevice) {
                    engine = "default"
                    destroyRecognizer()
                }
                emit(mapOf("type" to "error", "code" to error, "message" to describe(error), "fatal" to false))
                emit(mapOf("type" to "end", "empty" to true, "finishing" to wasFinishing))
                retryLater()
            }
        }
    }

    override fun onResults(results: Bundle?) {
        unmute()
        if (usingGlasses) GlassesMic.closeStreamNow()
        failures = 0
        val wasFinishing = finishing
        finishing = false
        val list = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION) ?: arrayListOf()
        emit(mapOf("type" to "final", "text" to (list.firstOrNull() ?: ""), "alts" to list, "finishing" to wasFinishing))
        emit(mapOf("type" to "end", "empty" to list.isEmpty(), "finishing" to wasFinishing))
        next(100)
    }

    override fun onPartialResults(partialResults: Bundle?) {
        val list = partialResults?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION) ?: return
        val text = list.firstOrNull() ?: return
        if (text.isNotBlank()) emit(mapOf("type" to "partial", "text" to text))
    }

    override fun onEvent(eventType: Int, params: Bundle?) {}

    private fun describe(error: Int) = when (error) {
        SpeechRecognizer.ERROR_NETWORK, SpeechRecognizer.ERROR_NETWORK_TIMEOUT -> "No network for speech recognition"
        SpeechRecognizer.ERROR_AUDIO -> "Microphone busy or unavailable"
        SpeechRecognizer.ERROR_SERVER -> "Speech service error"
        12, 13 -> "Language not available offline"
        else -> "Speech error $error"
    }

    companion object {
        private const val TAG = "SpeechEngine"
        private val STREAMS = intArrayOf(
            AudioManager.STREAM_MUSIC,
            AudioManager.STREAM_NOTIFICATION,
            AudioManager.STREAM_SYSTEM,
        )
    }
}
