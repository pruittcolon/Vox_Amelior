package com.vox.narrator

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import java.io.ByteArrayOutputStream
import kotlin.math.sqrt

/** Records a short voice sample (16-bit mono) for cloning. */
class MicRecorder {
    @Volatile private var record: AudioRecord? = null
    @Volatile private var thread: Thread? = null
    private val data = ByteArrayOutputStream()
    private var rate = 24000

    /** Loudness of the last buffer, 0..1, for the level meter. */
    @Volatile var level = 0.0
        private set

    @SuppressLint("MissingPermission")
    fun start(sampleRate: Int) {
        stopInternal()
        rate = sampleRate
        data.reset()
        val min = AudioRecord.getMinBufferSize(sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        val r = AudioRecord(
            MediaRecorder.AudioSource.MIC, sampleRate, AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT, maxOf(min, sampleRate),
        )
        record = r
        r.startRecording()
        thread = Thread {
            val buf = ByteArray(4096)
            while (record === r) {
                val n = r.read(buf, 0, buf.size)
                if (n <= 0) continue
                synchronized(data) { data.write(buf, 0, n) }
                var sum = 0.0
                var i = 0
                while (i + 1 < n) {
                    val s = ((buf[i].toInt() and 0xFF) or (buf[i + 1].toInt() shl 8)).toShort() / 32768.0
                    sum += s * s
                    i += 2
                }
                level = minOf(1.0, sqrt(sum / (n / 2)) * 4)
            }
        }.apply { start() }
    }

    fun stop(): Map<String, Any> {
        stopInternal()
        val bytes = synchronized(data) { data.toByteArray() }
        return mapOf("sampleRate" to rate, "pcm" to bytes)
    }

    private fun stopInternal() {
        val r = record ?: return
        record = null
        try { r.stop() } catch (_: IllegalStateException) {}
        thread?.join(500)
        thread = null
        r.release()
        level = 0.0
    }
}
