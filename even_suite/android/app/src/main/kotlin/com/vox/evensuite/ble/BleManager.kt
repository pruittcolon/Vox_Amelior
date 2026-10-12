package com.vox.evensuite.ble

import android.annotation.SuppressLint
import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothGatt
import android.bluetooth.BluetoothGattCallback
import android.bluetooth.BluetoothGattCharacteristic
import android.bluetooth.BluetoothGattDescriptor
import android.bluetooth.BluetoothManager
import android.bluetooth.BluetoothProfile
import android.bluetooth.le.ScanCallback
import android.bluetooth.le.ScanResult
import android.bluetooth.le.ScanSettings
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.util.Log
import java.util.UUID

/**
 * Bluetooth link to both arms of the G1, based on the Even Realities demo.
 *
 * Changes from the demo: GATT setup runs one step at a time (Android drops a
 * GATT call issued while another is pending), writes are flow controlled per
 * arm (see [BleDevice]), notifications also arrive on Android 12 and older,
 * disconnects are reported, and a pair the user connected to is reconnected
 * automatically until they disconnect it.
 */
@SuppressLint("MissingPermission")
object BleManager {

    interface Listener {
        fun onPairFound(info: Map<String, String>)
        fun onStatus(method: String, info: Map<String, Any>)
        fun onReceive(lr: String, data: ByteArray)
    }

    private const val TAG = "BleManager"
    private val SERVICE_UUID = UUID.fromString("6E400001-B5A3-F393-E0A9-E50E24DCCA9E")
    private val WRITE_UUID = UUID.fromString("6E400002-B5A3-F393-E0A9-E50E24DCCA9E")
    private val READ_UUID = UUID.fromString("6E400003-B5A3-F393-E0A9-E50E24DCCA9E")
    private val CCCD_UUID = UUID.fromString("00002902-0000-1000-8000-00805f9b34fb")

    var listener: Listener? = null

    private lateinit var appContext: Context
    private var manager: BluetoothManager? = null
    private val adapter: BluetoothAdapter? get() = manager?.adapter
    private val main = Handler(Looper.getMainLooper())

    private val found = mutableListOf<BleDevice>()
    @Volatile private var pair: BlePairDevice? = null
    @Volatile private var wanted = false // the user wants this pair connected
    private val retries = mutableMapOf<String, Int>()

    private val scanSettings = ScanSettings.Builder()
        .setScanMode(ScanSettings.SCAN_MODE_LOW_LATENCY)
        .build()

    fun init(context: Context) {
        if (manager != null) return
        appContext = context.applicationContext
        manager = appContext.getSystemService(BluetoothManager::class.java)
        val filter = IntentFilter(BluetoothAdapter.ACTION_STATE_CHANGED)
        if (Build.VERSION.SDK_INT >= 33) {
            appContext.registerReceiver(adapterReceiver, filter, Context.RECEIVER_NOT_EXPORTED)
        } else {
            appContext.registerReceiver(adapterReceiver, filter)
        }
    }

    // ------------------------------------------------------------- public

    /** Returns an error message, or null when scanning started. */
    fun startScan(): String? {
        val a = adapter ?: return "This phone has no Bluetooth"
        if (!a.isEnabled) return "Bluetooth is off. Turn it on and try again."
        val scanner = a.bluetoothLeScanner ?: return "Bluetooth is off. Turn it on and try again."
        found.clear()
        try {
            scanner.stopScan(scanCallback)
        } catch (_: Exception) {
        }
        scanner.startScan(null, scanSettings, scanCallback)
        return null
    }

    fun stopScan() {
        try {
            adapter?.bluetoothLeScanner?.stopScan(scanCallback)
        } catch (e: Exception) {
            Log.w(TAG, "stopScan: $e")
        }
    }

    /**
     * Connect to the pair on [channel]. Scan results are used when present;
     * otherwise the saved addresses (from an earlier session) are dialled directly.
     */
    fun connect(channel: String, saved: Map<*, *>?): String? {
        val a = adapter ?: return "This phone has no Bluetooth"
        if (!a.isEnabled) return "Bluetooth is off. Turn it on and try again."
        val left = found.firstOrNull { it.name.contains("_${channel}_L_") }
            ?: savedArm(saved, "left", channel)
        val right = found.firstOrNull { it.name.contains("_${channel}_R_") }
            ?: savedArm(saved, "right", channel)
        if (left == null || right == null) return "Glasses $channel not found. Open the case near the phone and scan again."
        stopScan()
        closeAll()
        val p = BlePairDevice(left, right)
        pair = p
        wanted = true
        retries.clear()
        emitStatus("glassesConnecting", p.toInfoJson())
        dial(left, autoConnect = false)
        dial(right, autoConnect = false)
        return null
    }

    fun disconnect() {
        wanted = false
        val p = pair
        closeAll()
        pair = null
        if (p != null) emitStatus("glassesDisconnected", p.toInfoJson() + ("reason" to "user"))
    }

    fun send(data: ByteArray, lr: String?) {
        val p = pair ?: return
        if (lr == null || lr == "L") p.leftDevice.sendData(data)
        if (lr == null || lr == "R") p.rightDevice.sendData(data)
    }

    /** Queue many packets to one arm in a single call; [done] gets how many were written. */
    fun sendBatch(packets: List<ByteArray>, lr: String?, done: (Int) -> Unit) {
        val arm = (if (lr == "R") pair?.rightDevice else pair?.leftDevice)
        if (arm == null || !arm.isConnected) {
            done(0)
            return
        }
        arm.enqueue(packets) { sent -> main.post { done(sent) } }
    }

    fun status(): Map<String, Any> {
        val p = pair ?: return mapOf("state" to "disconnected")
        val state = when {
            p.isBothConnected() -> "connected"
            wanted -> "connecting"
            else -> "disconnected"
        }
        return p.toInfoJson() + mapOf(
            "state" to state,
            "left" to p.leftDevice.isConnected,
            "right" to p.rightDevice.isConnected,
        )
    }

    // ------------------------------------------------------------ private

    private fun savedArm(saved: Map<*, *>?, side: String, channel: String): BleDevice? {
        val address = saved?.get("${side}Address") as? String ?: return null
        val name = saved?.get("${side}DeviceName") as? String ?: return null
        if (address.isEmpty() || !BluetoothAdapter.checkBluetoothAddress(address)) return null
        return BleDevice(name, address, channel)
    }

    private fun dial(arm: BleDevice, autoConnect: Boolean) {
        val device = adapter?.getRemoteDevice(arm.address) ?: return
        Log.i(TAG, "dial ${arm.name} autoConnect=$autoConnect")
        arm.gatt = device.connectGatt(appContext, autoConnect, gattCallback, BluetoothDevice.TRANSPORT_LE)
    }

    private fun closeAll() {
        val p = pair ?: return
        for (arm in listOf(p.leftDevice, p.rightDevice)) {
            try {
                arm.gatt?.disconnect()
                arm.gatt?.close()
            } catch (e: Exception) {
                Log.w(TAG, "close ${arm.name}: $e")
            }
            arm.reset()
        }
    }

    private fun emitStatus(method: String, info: Map<String, Any>) {
        main.post { listener?.onStatus(method, info) }
    }

    private fun armLabel(arm: BleDevice) = if (arm.isLeft()) "L" else "R"

    private val scanCallback = object : ScanCallback() {
        override fun onScanResult(callbackType: Int, result: ScanResult?) {
            val device = result?.device ?: return
            val name = device.name
            //  eg. G1_45_L_92333
            if (name.isNullOrEmpty() || !name.contains("G\\d+".toRegex()) ||
                name.split("_").size != 4 || found.any { it.address == device.address }) {
                return
            }
            val channel = name.split("_")[1]
            found.add(BleDevice(name, device.address, channel))
            val left = found.firstOrNull { it.channelNumber == channel && it.isLeft() }
            val right = found.firstOrNull { it.channelNumber == channel && it.isRight() }
            if (left != null && right != null) {
                val info = BlePairDevice(left, right).toInfoJson()
                main.post { listener?.onPairFound(info) }
            }
        }

        override fun onScanFailed(errorCode: Int) {
            Log.e(TAG, "scan failed: $errorCode")
        }
    }

    private val gattCallback = object : BluetoothGattCallback() {
        override fun onConnectionStateChange(gatt: BluetoothGatt, status: Int, newState: Int) {
            val p = pair
            val arm = p?.armFor(gatt.device.address)
            if (p == null || arm == null) {
                gatt.close()
                return
            }
            if (status == BluetoothGatt.GATT_SUCCESS && newState == BluetoothProfile.STATE_CONNECTED) {
                arm.gatt = gatt
                retries[arm.address] = 0
                // A short pause lets a fresh bond settle before discovery.
                main.postDelayed({ gatt.discoverServices() }, 300)
                return
            }
            // Disconnected, or the connect attempt failed (often status 133).
            Log.w(TAG, "${arm.name} disconnected: status=$status state=$newState")
            val wasConnected = arm.isConnected
            arm.reset()
            try {
                gatt.close()
            } catch (_: Exception) {
            }
            if (wasConnected) {
                emitStatus("glassesDisconnected", p.toInfoJson() + mapOf("arm" to armLabel(arm), "reason" to "link"))
            }
            if (wanted && pair === p) {
                val n = (retries[arm.address] ?: 0) + 1
                retries[arm.address] = n
                // Retry quickly a few times, then let Android wait for the
                // glasses to come back in range (autoConnect costs no power).
                val auto = n > 3
                main.postDelayed({
                    if (wanted && pair === p && !arm.isConnected) {
                        if (!wasConnected) emitStatus("glassesConnecting", p.toInfoJson())
                        dial(arm, autoConnect = auto)
                    }
                }, if (auto) 1000L else 800L * n)
            }
        }

        override fun onServicesDiscovered(gatt: BluetoothGatt, status: Int) {
            val arm = pair?.armFor(gatt.device.address) ?: return
            if (status != BluetoothGatt.GATT_SUCCESS) {
                Log.e(TAG, "${arm.name}: service discovery failed $status")
                gatt.disconnect()
                return
            }
            val service = gatt.getService(SERVICE_UUID)
            val read = service?.getCharacteristic(READ_UUID)
            val write = service?.getCharacteristic(WRITE_UUID)
            if (read == null || write == null) {
                Log.e(TAG, "${arm.name}: UART service not found")
                gatt.disconnect()
                return
            }
            arm.writeCharacteristic = write
            gatt.setCharacteristicNotification(read, true)
            val cccd = read.getDescriptor(CCCD_UUID)
            val started = if (cccd == null) {
                false
            } else if (Build.VERSION.SDK_INT >= 33) {
                gatt.writeDescriptor(cccd, BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE) == 0
            } else {
                @Suppress("DEPRECATION")
                cccd.value = BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE
                @Suppress("DEPRECATION")
                gatt.writeDescriptor(cccd)
            }
            // Next steps continue in onDescriptorWrite / onMtuChanged.
            if (!started) gatt.requestMtu(251)
        }

        override fun onDescriptorWrite(gatt: BluetoothGatt, descriptor: BluetoothGattDescriptor, status: Int) {
            gatt.requestMtu(251)
        }

        override fun onMtuChanged(gatt: BluetoothGatt, mtu: Int, status: Int) {
            val p = pair ?: return
            val arm = p.armFor(gatt.device.address) ?: return
            Log.i(TAG, "${arm.name}: mtu=$mtu status=$status")
            // Shortest connection interval: the main limit on packets per second.
            gatt.requestConnectionPriority(BluetoothGatt.CONNECTION_PRIORITY_HIGH)
            if (gatt.device.bondState == BluetoothDevice.BOND_NONE) gatt.device.createBond()
            arm.isConnected = true
            arm.sendData(byteArrayOf(0xf4.toByte(), 0x01.toByte()))
            if (p.isBothConnected()) {
                emitStatus("glassesConnected", p.toInfoJson() + ("status" to "connected"))
            }
        }

        override fun onCharacteristicWrite(gatt: BluetoothGatt, characteristic: BluetoothGattCharacteristic, status: Int) {
            pair?.armFor(gatt.device.address)?.onWriteComplete()
        }

        // Android 13+
        override fun onCharacteristicChanged(gatt: BluetoothGatt, characteristic: BluetoothGattCharacteristic, value: ByteArray) {
            deliver(gatt, value)
        }

        // Android 12 and older only call this one.
        @Deprecated("Deprecated in Java")
        override fun onCharacteristicChanged(gatt: BluetoothGatt, characteristic: BluetoothGattCharacteristic) {
            if (Build.VERSION.SDK_INT >= 33) return
            @Suppress("DEPRECATION")
            val value = characteristic.value ?: return
            deliver(gatt, value.copyOf())
        }
    }

    private fun deliver(gatt: BluetoothGatt, value: ByteArray) {
        if (value.isEmpty()) return
        // 0xF1 is microphone audio; this app listens with the phone instead.
        if (value[0] == 0xF1.toByte()) return
        val arm = pair?.armFor(gatt.device.address) ?: return
        val lr = armLabel(arm)
        main.post { listener?.onReceive(lr, value) }
    }

    private val adapterReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            val state = intent.getIntExtra(BluetoothAdapter.EXTRA_STATE, BluetoothAdapter.ERROR)
            val p = pair ?: return
            if (state == BluetoothAdapter.STATE_OFF) {
                val wasConnected = p.leftDevice.isConnected || p.rightDevice.isConnected
                closeAll()
                if (wasConnected) emitStatus("glassesDisconnected", p.toInfoJson() + ("reason" to "bluetooth-off"))
            } else if (state == BluetoothAdapter.STATE_ON && wanted) {
                emitStatus("glassesConnecting", p.toInfoJson())
                retries.clear()
                dial(p.leftDevice, autoConnect = false)
                dial(p.rightDevice, autoConnect = false)
            }
        }
    }
}
