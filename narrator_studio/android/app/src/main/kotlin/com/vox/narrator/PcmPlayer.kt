package com.vox.narrator

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFocusRequest
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import java.util.concurrent.Executors

/**
 * Streams 16-bit mono PCM from Dart to the speaker.
 *
 * Writes are blocking on a writer thread, so Dart's `await write()` returns
 * once the audio is in the track's buffer; that is the back-pressure that
 * keeps generation at most a buffer ahead of what is heard.
 */
class PcmPlayer(private val context: Context) {
    private val writer = Executors.newSingleThreadExecutor()
    private val audio = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
    @Volatile private var track: AudioTrack? = null
    @Volatile private var written = 0L
    private var focus: AudioFocusRequest? = null

    private val focusListener = AudioManager.OnAudioFocusChangeListener { change ->
        when (change) {
            AudioManager.AUDIOFOCUS_LOSS, AudioManager.AUDIOFOCUS_LOSS_TRANSIENT -> Bridge.emit("media", "pause")
            AudioManager.AUDIOFOCUS_LOSS_TRANSIENT_CAN_DUCK -> track?.setVolume(0.3f)
            AudioManager.AUDIOFOCUS_GAIN -> track?.setVolume(1f)
        }
    }

    @Synchronized
    fun open(sampleRate: Int) {
        stop()
        val attributes = AudioAttributes.Builder()
            .setUsage(AudioAttributes.USAGE_MEDIA)
            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
            .build()
        val min = AudioTrack.getMinBufferSize(
            sampleRate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT,
        )
        // About half a second of buffer: smooth, but pause/seek still feel instant.
        val bytes = maxOf(min, sampleRate)
        val t = AudioTrack.Builder()
            .setAudioAttributes(attributes)
            .setAudioFormat(
                AudioFormat.Builder()
                    .setSampleRate(sampleRate)
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build(),
            )
            .setTransferMode(AudioTrack.MODE_STREAM)
            .setBufferSizeInBytes(bytes)
            .build()
        val request = AudioFocusRequest.Builder(AudioManager.AUDIOFOCUS_GAIN)
            .setAudioAttributes(attributes)
            .setOnAudioFocusChangeListener(focusListener)
            .build()
        audio.requestAudioFocus(request)
        focus = request
        written = 0
        track = t
        t.play()
    }

    /** Queues [pcm]; [done] gets the total number of frames written so far. */
    fun write(pcm: ByteArray, done: (Long) -> Unit) {
        val t = track
        if (t == null) {
            done(written)
            return
        }
        writer.execute {
            var offset = 0
            while (offset < pcm.size && track === t) {
                val n = try {
                    t.write(pcm, offset, pcm.size - offset, AudioTrack.WRITE_BLOCKING)
                } catch (e: IllegalStateException) {
                    -1
                }
                if (n <= 0) break
                offset += n
            }
            if (track === t) written += offset / 2
            done(written)
        }
    }

    /** Frames actually played since [open]. */
    fun position(): Long {
        val t = track ?: return 0
        return try {
            t.playbackHeadPosition.toLong() and 0xFFFFFFFFL
        } catch (e: IllegalStateException) {
            0
        }
    }

    fun pause() {
        try { track?.pause() } catch (_: IllegalStateException) {}
    }

    fun resume() {
        try { track?.play() } catch (_: IllegalStateException) {}
    }

    @Synchronized
    fun stop() {
        val t = track ?: return
        track = null
        try {
            t.pause()
            t.flush()
        } catch (_: IllegalStateException) {}
        // Release after any blocked write has returned.
        writer.execute { t.release() }
        focus?.let { audio.abandonAudioFocusRequest(it) }
        focus = null
    }
}
