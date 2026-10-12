package com.vox.evensuite.ble

import android.annotation.SuppressLint
import android.bluetooth.BluetoothGatt
import android.bluetooth.BluetoothGattCharacteristic
import android.bluetooth.BluetoothStatusCodes
import android.os.Build
import android.util.Log
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

/** One arm of the glasses (G1_<channel>_<L|R>_<id>). */
@SuppressLint("MissingPermission")
class BleDevice(
    val name: String,
    val address: String,
    val channelNumber: String,
) {
    @Volatile var gatt: BluetoothGatt? = null
    @Volatile var writeCharacteristic: BluetoothGattCharacteristic? = null
    @Volatile var isConnected: Boolean = false

    fun isLeft() = name.contains("_L_")
    fun isRight() = name.contains("_R_")

    //  One writer thread per arm. Android allows a single GATT write in flight
    //  per connection and refuses (busy) any write issued before
    //  onCharacteristicWrite, so packets are queued here and each one waits for
    //  that callback instead of a fixed sleep.
    private val writer = Executors.newSingleThreadExecutor()
    @Volatile private var writeDone: CountDownLatch? = null

    // Diagnostics (read and reset by takeStats): how long each write waited
    // for the Bluetooth stack, and how often the stack said "busy".
    private var statWrites = 0
    private var statBytes = 0L
    private var statBusy = 0
    private var statDropped = 0
    private var statWaitUs = 0L
    private var statWaitMaxUs = 0L

    @Synchronized
    fun takeStats(): Map<String, Any> {
        val m = mapOf(
            "writes" to statWrites, "bytes" to statBytes, "busy" to statBusy, "dropped" to statDropped,
            "waitAvgMs" to (if (statWrites == 0) 0.0 else statWaitUs / 1000.0 / statWrites),
            "waitMaxMs" to statWaitMaxUs / 1000.0,
        )
        statWrites = 0; statBytes = 0; statBusy = 0; statDropped = 0; statWaitUs = 0; statWaitMaxUs = 0
        return m
    }

    @Synchronized
    private fun record(bytes: Int, waitUs: Long, busy: Int) {
        statWrites++
        statBytes += bytes
        statBusy += busy
        statWaitUs += waitUs
        if (waitUs > statWaitMaxUs) statWaitMaxUs = waitUs
    }

    /** Called from BluetoothGattCallback.onCharacteristicWrite. */
    fun onWriteComplete() {
        writeDone?.countDown()
    }

    fun sendData(data: ByteArray): Boolean {
        if (gatt == null || writeCharacteristic == null) {
            Log.w(TAG, "$name: not ready, dropping ${data.size} bytes")
            return false
        }
        enqueue(listOf(data))
        return true
    }

    /**
     * Queue [packets] in order, [gapMs] apart (the demo spaces image packets
     * ~5 ms so the glasses can keep up); [done] gets how many were written.
     */
    fun enqueue(packets: List<ByteArray>, gapMs: Long = 0, done: ((Int) -> Unit)? = null) {
        writer.execute {
            var sent = 0
            for ((i, packet) in packets.withIndex()) {
                if (i > 0 && gapMs > 0) Thread.sleep(gapMs)
                if (!writeNow(packet)) break
                sent++
            }
            done?.invoke(sent)
        }
    }

    fun reset() {
        isConnected = false
        writeCharacteristic = null
        gatt = null
        writeDone?.countDown()
    }

    private fun writeNow(data: ByteArray): Boolean {
        val start = System.nanoTime()
        var busy = 0
        repeat(50) {
            val g = gatt ?: return false
            val c = writeCharacteristic ?: return false
            val latch = CountDownLatch(1)
            writeDone = latch
            val accepted = try {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                    g.writeCharacteristic(c, data, BluetoothGattCharacteristic.WRITE_TYPE_NO_RESPONSE) ==
                        BluetoothStatusCodes.SUCCESS
                } else {
                    //  The pre-13 API writes whatever value the characteristic
                    //  holds, so the data has to be set on it first.
                    @Suppress("DEPRECATION")
                    c.writeType = BluetoothGattCharacteristic.WRITE_TYPE_NO_RESPONSE
                    @Suppress("DEPRECATION")
                    c.value = data
                    @Suppress("DEPRECATION")
                    g.writeCharacteristic(c)
                }
            } catch (e: Exception) {
                Log.e(TAG, "$name: send error = $e")
                false
            }
            if (accepted) {
                latch.await(200, TimeUnit.MILLISECONDS)
                record(data.size, (System.nanoTime() - start) / 1000, busy)
                return true
            }
            //  Stack busy (a callback still pending): try again shortly.
            busy++
            Thread.sleep(2)
        }
        Log.e(TAG, "$name: write refused 50 times, dropping packet")
        synchronized(this) { statDropped++ }
        return false
    }

    companion object {
        private const val TAG = "BleDevice"
    }
}

data class BlePairDevice(
    val leftDevice: BleDevice,
    val rightDevice: BleDevice,
) {
    val channelNumber get() = leftDevice.channelNumber

    fun toInfoJson(): Map<String, String> = mapOf(
        "leftDeviceName" to leftDevice.name,
        "rightDeviceName" to rightDevice.name,
        "leftAddress" to leftDevice.address,
        "rightAddress" to rightDevice.address,
        "channelNumber" to channelNumber,
    )

    fun isBothConnected() = leftDevice.isConnected && rightDevice.isConnected

    fun armFor(address: String?): BleDevice? = when (address) {
        leftDevice.address -> leftDevice
        rightDevice.address -> rightDevice
        else -> null
    }
}
