package com.vox.evensuite

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PixelFormat
import android.graphics.Rect
import android.graphics.RectF
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.ImageReader
import android.media.projection.MediaProjection
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import android.util.DisplayMetrics
import android.util.Log
import android.view.WindowManager
import kotlin.math.min

/**
 * Captures the phone screen (Android's MediaProjection, no extra SDK) and
 * turns it into 576x136 grayscale frames for the glasses, at most once per
 * [intervalMs]. The screen is captured at a reduced size (max 720 px wide),
 * and an unchanged frame is reported as "same" without copying any pixels.
 * Conversion to 1-bit happens in Dart, where it is unit tested.
 */
class ScreenMirror(
    private val context: Context,
    private val projection: MediaProjection,
    private val onFrame: (Map<String, Any>) -> Unit,
    private val onStopped: () -> Unit,
) {
    companion object {
        const val W = 576
        const val H = 136
        private const val TAG = "ScreenMirror"
    }

    @Volatile var region = "strip" // strip, top, full, custom
    @Volatile var strip = 0 // 0..100, strip position from top to bottom
    @Volatile var intervalMs = 1000L
    // Custom area in percent of the screen (e.g. around a video player).
    @Volatile var cx = 0
    @Volatile var cy = 0
    @Volatile var cw = 100
    @Volatile var ch = 30
    @Volatile var fit = "fit" // fit (keep shape), fill (crop to fill), stretch
    @Volatile var rows = H // picture height sent to the glasses (136, 102, 68)

    private val thread = HandlerThread("screen-mirror").apply { start() }
    private val handler = Handler(thread.looper)
    private var reader: ImageReader? = null
    private var virtualDisplay: VirtualDisplay? = null
    private var src: Bitmap? = null
    private var padded: java.nio.ByteBuffer? = null
    private var out = Bitmap.createBitmap(W, H, Bitmap.Config.ARGB_8888)
    private var canvas = Canvas(out)
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG or Paint.ANTI_ALIAS_FLAG)
    private var pixels = IntArray(W * H)
    private var lastHash = 0
    private var capW = 0
    private var capH = 0
    @Volatile private var released = false

    // Diagnostics
    private var statFrames = 0
    private var statSame = 0
    private var statCaptureMs = 0L
    private var statCaptureMaxMs = 0L

    @Synchronized
    fun takeStats(): Map<String, Any> {
        val m = mapOf(
            "captured" to statFrames, "unchanged" to statSame,
            "captureAvgMs" to (if (statFrames == 0) 0.0 else statCaptureMs.toDouble() / statFrames),
            "captureMaxMs" to statCaptureMaxMs, "intervalMs" to intervalMs, "size" to "${capW}x$capH",
        )
        statFrames = 0; statSame = 0; statCaptureMs = 0; statCaptureMaxMs = 0
        return m
    }

    @Synchronized
    private fun countFrame(ms: Long, same: Boolean) {
        statFrames++
        if (same) statSame++
        statCaptureMs += ms
        if (ms > statCaptureMaxMs) statCaptureMaxMs = ms
    }

    fun start() {
        val (sw, sh, dpi) = screenSize()
        val scale = min(1f, 720f / sw)
        capW = ((sw * scale).toInt() / 2) * 2
        capH = ((sh * scale).toInt() / 2) * 2
        // Android 14 requires the callback before the virtual display.
        projection.registerCallback(object : MediaProjection.Callback() {
            override fun onStop() {
                handler.post {
                    release()
                    onStopped()
                }
            }
        }, handler)
        val r = ImageReader.newInstance(capW, capH, PixelFormat.RGBA_8888, 2)
        reader = r
        virtualDisplay = projection.createVirtualDisplay(
            "even-suite-glasses", capW, capH, dpi,
            DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR, r.surface, null, handler,
        )
        handler.postDelayed(tick, 400)
    }

    private val tick = object : Runnable {
        override fun run() {
            if (released) return
            try {
                grab()
            } catch (e: Exception) {
                Log.e(TAG, "grab: $e")
            }
            handler.postDelayed(this, intervalMs)
        }
    }

    private fun screenSize(): Triple<Int, Int, Int> {
        val wm = context.getSystemService(WindowManager::class.java)
        val dpi = context.resources.displayMetrics.densityDpi
        return if (Build.VERSION.SDK_INT >= 30) {
            val b = wm.maximumWindowMetrics.bounds
            Triple(b.width(), b.height(), dpi)
        } else {
            val m = DisplayMetrics()
            @Suppress("DEPRECATION")
            wm.defaultDisplay.getRealMetrics(m)
            Triple(m.widthPixels, m.heightPixels, dpi)
        }
    }

    private fun regionRect(w: Int, h: Int): Rect = when (region) {
        "full" -> Rect(0, 0, w, h)
        "top" -> Rect(0, 0, w, h / 2)
        "custom" -> {
            val l = (w * cx.coerceIn(0, 99)) / 100
            val t = (h * cy.coerceIn(0, 99)) / 100
            val r = (l + maxOf(1, w * cw.coerceIn(1, 100) / 100)).coerceAtMost(w)
            val b = (t + maxOf(1, h * ch.coerceIn(1, 100) / 100)).coerceAtMost(h)
            Rect(l, t, maxOf(r, l + 1), maxOf(b, t + 1))
        }
        else -> {
            // Full width, the glasses' own shape (576:136), so it fills the display.
            val sh = (w * H / W).coerceAtMost(h)
            val top = ((h - sh) * strip.coerceIn(0, 100)) / 100
            Rect(0, top, w, top + sh)
        }
    }

    private fun grab() {
        val t0 = SystemClock.uptimeMillis()
        val image = try {
            reader?.acquireLatestImage()
        } catch (e: Exception) {
            null
        }
        if (image == null) {
            onFrame(mapOf("same" to true))
            return
        }
        val bmp: Bitmap
        try {
            val plane = image.planes[0]
            val pixelStride = plane.pixelStride
            val rowPadding = plane.rowStride - pixelStride * image.width
            val bw = image.width + rowPadding / pixelStride
            val cached = src
            bmp = if (cached != null && cached.width == bw && cached.height == image.height) {
                cached
            } else {
                Bitmap.createBitmap(bw, image.height, Bitmap.Config.ARGB_8888).also { src = it }
            }
            val buf = plane.buffer
            if (buf.remaining() >= bmp.byteCount) {
                bmp.copyPixelsFromBuffer(buf)
            } else {
                // Some phones do not pad the last row; copy into a full-size buffer.
                val tmp = padded?.takeIf { it.capacity() == bmp.byteCount }
                    ?: java.nio.ByteBuffer.allocate(bmp.byteCount).also { padded = it }
                tmp.clear()
                tmp.put(buf)
                tmp.position(0)
                bmp.copyPixelsFromBuffer(tmp)
            }
        } finally {
            image.close()
        }
        val oh = rows.coerceIn(16, H)
        if (out.height != oh) {
            out = Bitmap.createBitmap(W, oh, Bitmap.Config.ARGB_8888)
            canvas = Canvas(out)
            pixels = IntArray(W * oh)
            lastHash = 0
        }
        var r = regionRect(capW, capH)
        val dst: RectF
        when (fit) {
            "stretch" -> dst = RectF(0f, 0f, W.toFloat(), oh.toFloat())
            "fill" -> {
                // Crop the area to the picture's shape, then fill it.
                val target = W.toFloat() / oh
                val cur = r.width().toFloat() / r.height()
                r = if (cur > target) {
                    val nw = (r.height() * target).toInt()
                    val l = r.left + (r.width() - nw) / 2
                    Rect(l, r.top, l + nw, r.bottom)
                } else {
                    val nh = (r.width() / target).toInt().coerceAtLeast(1)
                    val t = r.top + (r.height() - nh) / 2
                    Rect(r.left, t, r.right, t + nh)
                }
                dst = RectF(0f, 0f, W.toFloat(), oh.toFloat())
            }
            else -> {
                val s = min(W.toFloat() / r.width(), oh.toFloat() / r.height())
                val dw = r.width() * s
                val dh = r.height() * s
                dst = RectF((W - dw) / 2f, (oh - dh) / 2f, (W + dw) / 2f, (oh + dh) / 2f)
            }
        }
        canvas.drawColor(Color.BLACK)
        canvas.drawBitmap(bmp, r, dst, paint)
        out.getPixels(pixels, 0, W, 0, 0, W, oh)
        val gray = ByteArray(W * oh)
        var hash = 1
        for (i in pixels.indices) {
            val p = pixels[i]
            val g = (((p shr 16) and 0xff) * 299 + ((p shr 8) and 0xff) * 587 + (p and 0xff) * 114) / 1000
            gray[i] = g.toByte()
            hash = 31 * hash + g
        }
        val ms = SystemClock.uptimeMillis() - t0
        if (hash == lastHash) {
            countFrame(ms, true)
            onFrame(mapOf("same" to true))
            return
        }
        countFrame(ms, false)
        lastHash = hash
        onFrame(
            mapOf(
                "gray" to gray,
                "x0" to dst.left.toInt(), "y0" to dst.top.toInt(),
                "x1" to dst.right.toInt(), "y1" to dst.bottom.toInt(),
                "w" to W, "h" to oh,
                "ms" to (SystemClock.uptimeMillis() - t0).toInt(),
            )
        )
    }

    /** Force the next frame to be sent even if the screen did not change. */
    fun refresh() {
        handler.post { lastHash = 0 }
    }

    fun release() {
        if (released) return
        released = true
        handler.removeCallbacks(tick)
        try {
            virtualDisplay?.release()
        } catch (_: Exception) {
        }
        try {
            reader?.close()
        } catch (_: Exception) {
        }
        try {
            projection.stop()
        } catch (_: Exception) {
        }
        virtualDisplay = null
        reader = null
        thread.quitSafely()
    }
}
