package com.vox.narrator

import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMuxer
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Turns a 16-bit mono WAV into an AAC .m4a (what podcast hosts accept). */
object M4aEncoder {
    fun encode(wav: File, out: File, bitrate: Int, progress: (Double) -> Unit) {
        RandomAccessFile(wav, "r").use { raf ->
            val (sampleRate, channels, dataStart, dataSize) = readWavHeader(raf)
            val format = MediaFormat.createAudioFormat(MediaFormat.MIMETYPE_AUDIO_AAC, sampleRate, channels).apply {
                setInteger(MediaFormat.KEY_AAC_PROFILE, MediaCodecInfo.CodecProfileLevel.AACObjectLC)
                setInteger(MediaFormat.KEY_BIT_RATE, bitrate)
                setInteger(MediaFormat.KEY_MAX_INPUT_SIZE, 16384)
            }
            val codec = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_AUDIO_AAC)
            codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
            codec.start()
            out.delete()
            val muxer = MediaMuxer(out.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
            var track = -1
            var muxing = false
            val info = MediaCodec.BufferInfo()
            val frameBytes = 2 * channels
            var readBytes = 0L
            var inputDone = false
            var outputDone = false
            val chunk = ByteArray(8192)
            raf.seek(dataStart)
            var lastReport = 0.0
            try {
                while (!outputDone) {
                    if (!inputDone) {
                        val index = codec.dequeueInputBuffer(10_000)
                        if (index >= 0) {
                            val buffer = codec.getInputBuffer(index)!!
                            buffer.clear()
                            val want = minOf(buffer.capacity().toLong(), chunk.size.toLong(), dataSize - readBytes).toInt()
                            val n = if (want > 0) raf.read(chunk, 0, want - want % frameBytes) else -1
                            val pts = readBytes / frameBytes * 1_000_000L / sampleRate
                            if (n <= 0) {
                                codec.queueInputBuffer(index, 0, 0, pts, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                                inputDone = true
                            } else {
                                buffer.put(chunk, 0, n)
                                codec.queueInputBuffer(index, 0, n, pts, 0)
                                readBytes += n
                                val p = readBytes.toDouble() / dataSize
                                if (p - lastReport >= 0.02) {
                                    lastReport = p
                                    progress(p)
                                }
                            }
                        }
                    }
                    val outIndex = codec.dequeueOutputBuffer(info, 10_000)
                    when {
                        outIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                            track = muxer.addTrack(codec.outputFormat)
                            muxer.start()
                            muxing = true
                        }
                        outIndex >= 0 -> {
                            val encoded = codec.getOutputBuffer(outIndex)!!
                            if (info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG != 0) info.size = 0
                            if (info.size > 0 && muxing) {
                                encoded.position(info.offset)
                                encoded.limit(info.offset + info.size)
                                muxer.writeSampleData(track, encoded, info)
                            }
                            codec.releaseOutputBuffer(outIndex, false)
                            if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) outputDone = true
                        }
                    }
                }
            } finally {
                codec.stop()
                codec.release()
                if (muxing) muxer.stop()
                muxer.release()
            }
            progress(1.0)
        }
    }

    private data class WavInfo(val sampleRate: Int, val channels: Int, val dataStart: Long, val dataSize: Long)

    private fun readWavHeader(raf: RandomAccessFile): WavInfo {
        val head = ByteArray(12)
        raf.readFully(head)
        require(String(head, 0, 4) == "RIFF" && String(head, 8, 4) == "WAVE") { "Not a WAV file" }
        var sampleRate = 24000
        var channels = 1
        while (raf.filePointer < raf.length()) {
            val chunkHead = ByteArray(8)
            raf.readFully(chunkHead)
            val id = String(chunkHead, 0, 4)
            val size = ByteBuffer.wrap(chunkHead, 4, 4).order(ByteOrder.LITTLE_ENDIAN).int.toLong() and 0xFFFFFFFFL
            if (id == "fmt ") {
                val fmt = ByteArray(size.toInt())
                raf.readFully(fmt)
                val b = ByteBuffer.wrap(fmt).order(ByteOrder.LITTLE_ENDIAN)
                channels = b.getShort(2).toInt()
                sampleRate = b.getInt(4)
            } else if (id == "data") {
                val start = raf.filePointer
                val available = raf.length() - start
                return WavInfo(sampleRate, channels, start, minOf(size, available))
            } else {
                raf.seek(raf.filePointer + size + (size and 1))
            }
        }
        throw IllegalArgumentException("WAV file has no data")
    }
}
