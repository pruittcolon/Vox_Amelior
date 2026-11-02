package com.vox.evensuite.speech

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Handler
import android.os.Looper
import android.util.Log

/**
 * Raw 16 kHz mono PCM (16-bit little endian) for the app's own speech
 * engine (Parakeet): from the phone's microphone in 100 ms chunks, or from
 * the glasses' microphone (decoded from LC3). Chunks are handed over on the
 * main thread, as Flutter's event channels need.
 */
class PcmRecorder(private val context: Context, private val send: (ByteArray) -> Unit) {
    private val main = Handler(Looper.getMainLooper())
    @Volatile private var running = false
    private var thread: Thread? = null
    private var record: AudioRecord? = null

    /** Returns where the sound comes from ("phone", "glasses") or why it can't ("no-permission", "error"). */
    fun start(source: String): String {
        stop()
        if (source == "glasses" && GlassesMic.supported) {
            running = true
            GlassesMic.openPcm { pcm -> post(pcm) }
            return "glasses"
        }
        if (context.checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            return "no-permission"
        }
        val min = AudioRecord.getMinBufferSize(RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        val rec = try {
            AudioRecord(
                MediaRecorder.AudioSource.VOICE_RECOGNITION, RATE, AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT, maxOf(min, CHUNK * 4),
            )
        } catch (e: Exception) {
            Log.e(TAG, "AudioRecord: $e")
            return "error"
        }
        if (rec.state != AudioRecord.STATE_INITIALIZED) {
            rec.release()
            return "error"
        }
        record = rec
        running = true
        rec.startRecording()
        thread = Thread {
            val buf = ByteArray(CHUNK)
            while (running) {
                val n = rec.read(buf, 0, buf.size)
                if (n > 0) post(buf.copyOf(n)) else if (n < 0) break
            }
        }.apply {
            name = "PcmRecorder"
            start()
        }
        return "phone"
    }

    private fun post(pcm: ByteArray) {
        if (running) main.post { if (running) send(pcm) }
    }

    fun stop() {
        running = false
        GlassesMic.closePcm()
        val t = thread
        thread = null
        try {
            t?.join(300)
        } catch (_: InterruptedException) {
        }
        record?.let {
            try {
                it.stop()
            } catch (_: Exception) {
            }
            it.release()
        }
        record = null
    }

    companion object {
        private const val TAG = "PcmRecorder"
        private const val RATE = 16000
        private const val CHUNK = 3200 // 100 ms of 16-bit samples
    }
}
