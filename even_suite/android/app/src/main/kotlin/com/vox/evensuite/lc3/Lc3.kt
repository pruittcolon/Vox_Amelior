package com.vox.evensuite.lc3

/** LC3 decoder for the G1 microphone (native, from the Even Realities demo's liblc3). */
object Lc3 {
    val loaded: Boolean = try {
        System.loadLibrary("lc3")
        true
    } catch (e: Throwable) {
        false
    }

    @JvmStatic external fun open(): Long

    /** 200 bytes of LC3 (one G1 mic packet) -> 3200 bytes of 16 kHz 16-bit PCM. */
    @JvmStatic external fun decode(handle: Long, data: ByteArray): ByteArray

    @JvmStatic external fun close(handle: Long)
}
