package com.example.demo_ai_even.model

import android.annotation.SuppressLint
import android.bluetooth.BluetoothGatt
import android.bluetooth.BluetoothGattCharacteristic
import android.bluetooth.BluetoothStatusCodes
import android.os.Build
import android.util.Log
import com.example.demo_ai_even.bluetooth.BleManager
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

@SuppressLint("MissingPermission")
data class BleDevice(
    val name: String,
    val address: String,
    var gatt: BluetoothGatt?,
    var writeCharacteristic: BluetoothGattCharacteristic?,
    var isConnect: Boolean,
    val channelNumber: String,
) {

    companion object {
        fun createByDevice(
            name: String,
            address: String,
            channelNumber: String,
        ) = BleDevice(name, address, null, null,false, channelNumber)
    }

    fun isLeft() = name.contains("_L_")

    fun isRight() = name.contains("_R_")

    //  One writer thread per arm. Android allows a single GATT write in flight
    //  per connection and silently refuses (busy) any write issued before
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
            Log.e(BleManager.LOG_TAG, "$name: Gatt or WriteCharacteristic is null")
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

    private fun writeNow(data: ByteArray): Boolean {
        val g = gatt ?: return false
        val c = writeCharacteristic ?: return false
        repeat(50) {
            val latch = CountDownLatch(1)
            writeDone = latch
            val accepted = try {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                    g.writeCharacteristic(c, data, BluetoothGattCharacteristic.WRITE_TYPE_NO_RESPONSE) ==
                        BluetoothStatusCodes.SUCCESS
                } else {
                    //  Pre-13 API writes whatever value the characteristic holds,
                    //  so the data has to be set on it first.
                    @Suppress("DEPRECATION")
                    c.writeType = BluetoothGattCharacteristic.WRITE_TYPE_NO_RESPONSE
                    @Suppress("DEPRECATION")
                    c.value = data
                    @Suppress("DEPRECATION")
                    g.writeCharacteristic(c)
                }
            } catch (e: Exception) {
                Log.e(BleManager.LOG_TAG, "$name: send error = $e")
                false
            }
            if (accepted) {
                latch.await(200, TimeUnit.MILLISECONDS)
                return true
            }
            //  Stack busy (e.g. a callback still pending): try again shortly.
            Thread.sleep(2)
        }
        Log.e(BleManager.LOG_TAG, "$name: write refused 50 times, dropping packet")
        return false
    }
}

