package com.vox.evensuite.speech

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFocusRequest
import android.media.AudioManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.speech.tts.Voice
import android.util.Log

/**
 * The phone's own text-to-speech (no extra SDK): reads Claude's answers
 * aloud, sentence by sentence as they arrive. Other audio ducks while it
 * talks. Events: start / done / error per utterance id.
 */
class TtsEngine(
    private val context: Context,
    private val emit: (Map<String, Any?>) -> Unit,
) {
    private val main = Handler(Looper.getMainLooper())
    private val audio = context.getSystemService(AudioManager::class.java)
    private val attributes = AudioAttributes.Builder()
        .setUsage(AudioAttributes.USAGE_ASSISTANT)
        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
        .build()

    private var tts: TextToSpeech? = null
    private var ready = false
    private var engine = ""
    private var voiceName = ""
    private var rate = 1f
    private var pitch = 1f
    private val pending = mutableListOf<Triple<String, String, Boolean>>()
    private val active = mutableSetOf<String>()
    private var focus: AudioFocusRequest? = null

    fun configure(args: Map<*, *>?) {
        val newEngine = args?.get("engine") as? String ?: ""
        voiceName = args?.get("voice") as? String ?: ""
        rate = ((args?.get("rate") as? Number)?.toFloat() ?: 1f).coerceIn(0.3f, 3f)
        pitch = ((args?.get("pitch") as? Number)?.toFloat() ?: 1f).coerceIn(0.3f, 3f)
        if (newEngine != engine) {
            engine = newEngine
            shutdown()
        }
        if (ready) apply()
    }

    fun speak(text: String, id: String, flush: Boolean) {
        val t = tts ?: create()
        if (!ready) {
            pending.add(Triple(text, id, flush))
            return
        }
        requestFocus()
        active.add(id)
        val params = Bundle()
        val r = t.speak(text, if (flush) TextToSpeech.QUEUE_FLUSH else TextToSpeech.QUEUE_ADD, params, id)
        if (r != TextToSpeech.SUCCESS) {
            active.remove(id)
            emit(mapOf("type" to "error", "id" to id, "message" to "speak failed ($r)"))
            if (active.isEmpty()) abandonFocus()
        }
    }

    fun stop() {
        pending.clear()
        active.clear()
        try {
            tts?.stop()
        } catch (_: Exception) {
        }
        abandonFocus()
    }

    fun voices(): List<Map<String, Any>> {
        val t = tts ?: return emptyList()
        if (!ready) return emptyList()
        val list = try {
            t.voices ?: emptySet()
        } catch (_: Exception) {
            emptySet<Voice>()
        }
        return list
            .filter { !it.features.contains(TextToSpeech.Engine.KEY_FEATURE_NOT_INSTALLED) }
            .sortedWith(compareBy<Voice>({ it.locale.toLanguageTag() }, { -it.quality }, { it.name }))
            .map {
                mapOf(
                    "name" to it.name,
                    "locale" to it.locale.toLanguageTag(),
                    "quality" to it.quality,
                    "network" to it.isNetworkConnectionRequired,
                )
            }
    }

    fun engines(): List<Map<String, String>> {
        val t = tts ?: create()
        return try {
            t.engines.map { mapOf("name" to it.name, "label" to it.label) }
        } catch (_: Exception) {
            emptyList()
        }
    }

    fun dispose() {
        stop()
        shutdown()
    }

    private fun create(): TextToSpeech {
        ready = false
        val listener = TextToSpeech.OnInitListener { status ->
            main.post {
                ready = status == TextToSpeech.SUCCESS
                if (!ready) {
                    emit(mapOf("type" to "error", "id" to "", "message" to "Text-to-speech did not start ($status)"))
                    for ((_, id, _) in pending) emit(mapOf("type" to "error", "id" to id, "message" to "no voice"))
                    pending.clear()
                    return@post
                }
                apply()
                emit(mapOf("type" to "ready"))
                val queued = pending.toList()
                pending.clear()
                for ((text, id, flush) in queued) speak(text, id, flush)
            }
        }
        val t = if (engine.isEmpty()) TextToSpeech(context, listener) else TextToSpeech(context, listener, engine)
        t.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(utteranceId: String?) {
                main.post { emit(mapOf("type" to "start", "id" to (utteranceId ?: ""))) }
            }

            override fun onDone(utteranceId: String?) {
                main.post { finished(utteranceId, "done", null) }
            }

            @Deprecated("Deprecated in Java")
            override fun onError(utteranceId: String?) {
                main.post { finished(utteranceId, "error", "error") }
            }

            override fun onError(utteranceId: String?, errorCode: Int) {
                main.post { finished(utteranceId, "error", "error $errorCode") }
            }

            override fun onStop(utteranceId: String?, interrupted: Boolean) {
                main.post { finished(utteranceId, "done", null) }
            }
        })
        tts = t
        return t
    }

    private fun finished(id: String?, type: String, message: String?) {
        val key = id ?: ""
        active.remove(key)
        if (active.isEmpty()) abandonFocus()
        emit(mapOf("type" to type, "id" to key, "message" to (message ?: "")))
    }

    private fun apply() {
        val t = tts ?: return
        try {
            t.setAudioAttributes(attributes)
            t.setSpeechRate(rate)
            t.setPitch(pitch)
            if (voiceName.isNotEmpty()) {
                t.voices?.firstOrNull { it.name == voiceName }?.let { t.voice = it }
            }
        } catch (e: Exception) {
            Log.e(TAG, "apply: $e")
        }
    }

    private fun shutdown() {
        ready = false
        try {
            tts?.shutdown()
        } catch (_: Exception) {
        }
        tts = null
    }

    private fun requestFocus() {
        if (focus != null) return
        val req = AudioFocusRequest.Builder(AudioManager.AUDIOFOCUS_GAIN_TRANSIENT_MAY_DUCK)
            .setAudioAttributes(attributes)
            .build()
        try {
            audio.requestAudioFocus(req)
            focus = req
        } catch (_: Exception) {
        }
    }

    private fun abandonFocus() {
        val f = focus ?: return
        focus = null
        try {
            audio.abandonAudioFocusRequest(f)
        } catch (_: Exception) {
        }
    }

    companion object {
        private const val TAG = "TtsEngine"
    }
}
