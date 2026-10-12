package com.vox.evensuite.speech

import android.os.ParcelFileDescriptor
import android.util.Log
import com.vox.evensuite.lc3.Lc3
import java.io.IOException
import java.io.OutputStream
import java.util.concurrent.Executors

/**
 * Streams the glasses' microphone (LC3 packets over Bluetooth) into Android's
 * speech recognizer as 16 kHz PCM through a pipe (Android 13+
 * RecognizerIntent.EXTRA_AUDIO_SOURCE). Decoding and writing happen on one
 * worker thread so Bluetooth callbacks never block.
 */
object GlassesMic {
    private const val TAG = "GlassesMic"
    private val worker = Executors.newSingleThreadExecutor()
    private var handle = 0L // only touched on the worker thread
    @Volatile private var out: OutputStream? = null
    private var readSide: ParcelFileDescriptor? = null
    @Volatile private var streaming = false

    val supported get() = Lc3.loaded

    /** A fresh pipe for one recognition session; returns the end the recognizer reads. */
    @Synchronized
    fun openStream(): ParcelFileDescriptor? {
        if (!Lc3.loaded) return null
        closeStreamNow()
        val pipe = try {
            ParcelFileDescriptor.createPipe()
        } catch (e: IOException) {
            Log.e(TAG, "createPipe: $e")
            return null
        }
        readSide = pipe[0]
        out = ParcelFileDescriptor.AutoCloseOutputStream(pipe[1])
        streaming = true
        worker.execute { if (handle == 0L) handle = Lc3.open() }
        return pipe[0]
    }

    /** One 0xF1 packet's audio (200 bytes of LC3). */
    fun onPacket(lc3: ByteArray) {
        if (!streaming) return
        val o = out ?: return
        worker.execute {
            if (handle == 0L) return@execute
            try {
                o.write(Lc3.decode(handle, lc3))
            } catch (e: IOException) {
                // That session's pipe is closed (the recognizer is done with it).
            }
        }
    }

    /** End of speech: the recognizer sees end-of-file and finishes. */
    @Synchronized
    fun endStream() {
        streaming = false
        val o = out
        out = null
        // Queued after any packets still being decoded, so none are lost.
        worker.execute { closeQuietly(o) }
    }

    /** Session over: release both pipe ends and the decoder state. */
    @Synchronized
    fun closeStreamNow() {
        streaming = false
        val o = out
        val r = readSide
        out = null
        readSide = null
        worker.execute {
            closeQuietly(o)
            try {
                r?.close()
            } catch (_: IOException) {
            }
            if (handle != 0L) {
                Lc3.close(handle)
                handle = 0L
            }
        }
    }

    private fun closeQuietly(o: OutputStream?) {
        try {
            o?.close()
        } catch (_: IOException) {
        }
    }
}
