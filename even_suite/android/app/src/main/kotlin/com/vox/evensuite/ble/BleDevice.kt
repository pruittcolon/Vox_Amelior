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

    /** Queue [packets] in order; [done] gets how many were written. */
    fun enqueue(packets: List<ByteArray>, done: ((Int) -> Unit)? = null) {
        writer.execute {
            var sent = 0
            for (packet in packets) {
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
                return true
            }
            //  Stack busy (a callback still pending): try again shortly.
            Thread.sleep(2)
        }
        Log.e(TAG, "$name: write refused 50 times, dropping packet")
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
