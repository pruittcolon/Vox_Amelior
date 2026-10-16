package com.vox.narrator

import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Decodes any audio file Android can play (mp3, m4a, wav, ogg...) to mono float PCM. */
object AudioDecoder {
    fun decode(path: String, maxSeconds: Double): Map<String, Any> {
        val extractor = MediaExtractor()
        extractor.setDataSource(path)
        var trackIndex = -1
        for (i in 0 until extractor.trackCount) {
            val mime = extractor.getTrackFormat(i).getString(MediaFormat.KEY_MIME) ?: continue
            if (mime.startsWith("audio/")) {
                trackIndex = i
                break
            }
        }
        require(trackIndex >= 0) { "No audio track in this file" }
        extractor.selectTrack(trackIndex)
        val format = extractor.getTrackFormat(trackIndex)
        var sampleRate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
        var channels = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
        val codec = MediaCodec.createDecoderByType(format.getString(MediaFormat.KEY_MIME)!!)
        codec.configure(format, null, null, 0)
        codec.start()

        val out = ByteArrayOutputStream()
        val info = MediaCodec.BufferInfo()
        var inputDone = false
        var outputDone = false
        var pcmFloat = false
        var frames = 0L
        try {
            while (!outputDone) {
                if (!inputDone) {
                    val index = codec.dequeueInputBuffer(10_000)
                    if (index >= 0) {
                        val buffer = codec.getInputBuffer(index)!!
                        val n = extractor.readSampleData(buffer, 0)
                        if (n < 0) {
                            codec.queueInputBuffer(index, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            inputDone = true
                        } else {
                            codec.queueInputBuffer(index, 0, n, extractor.sampleTime, 0)
                            extractor.advance()
                        }
                    }
                }
                val outIndex = codec.dequeueOutputBuffer(info, 10_000)
                if (outIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                    val f = codec.outputFormat
                    sampleRate = f.getInteger(MediaFormat.KEY_SAMPLE_RATE)
                    channels = f.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
                    pcmFloat = f.containsKey(MediaFormat.KEY_PCM_ENCODING) &&
                        f.getInteger(MediaFormat.KEY_PCM_ENCODING) == android.media.AudioFormat.ENCODING_PCM_FLOAT
                } else if (outIndex >= 0) {
                    val buffer = codec.getOutputBuffer(outIndex)!!
                    buffer.position(info.offset)
                    buffer.limit(info.offset + info.size)
                    frames += mixDown(buffer.slice().order(ByteOrder.LITTLE_ENDIAN), channels, pcmFloat, out)
                    codec.releaseOutputBuffer(outIndex, false)
                    if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) outputDone = true
                    if (frames >= maxSeconds * sampleRate) outputDone = true
                }
            }
        } finally {
            codec.stop()
            codec.release()
            extractor.release()
        }
        return mapOf("sampleRate" to sampleRate, "pcm" to out.toByteArray())
    }

    /** Averages the channels into float32 little-endian samples; returns frames added. */
    private fun mixDown(buffer: ByteBuffer, channels: Int, pcmFloat: Boolean, out: ByteArrayOutputStream): Int {
        val bytesPerSample = if (pcmFloat) 4 else 2
        val frames = buffer.remaining() / (bytesPerSample * channels)
        val result = ByteBuffer.allocate(frames * 4).order(ByteOrder.LITTLE_ENDIAN)
        for (i in 0 until frames) {
            var sum = 0f
            for (c in 0 until channels) {
                sum += if (pcmFloat) buffer.float else buffer.short / 32768f
            }
            result.putFloat(sum / channels)
        }
        out.write(result.array(), 0, frames * 4)
        return frames
    }
}
