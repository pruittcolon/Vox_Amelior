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
import android.os.HandlerThread
import android.os.Looper
import android.os.PowerManager
import android.util.Log
import com.vox.evensuite.SuiteService
import com.vox.evensuite.speech.GlassesMic
import java.util.UUID

/**
 * Bluetooth link to both arms of the G1, based on the Even Realities demo.
 *
 * Changes from the demo: GATT setup runs one step at a time (Android drops a
 * GATT call issued while another is pending), writes are flow controlled per
 * arm (see [BleDevice]), notifications also arrive on Android 12 and older,
 * disconnects are reported, and a pair the user connected to is reconnected
 * automatically until they disconnect it.
 *
 * Staying connected with the app out of sight:
 * - The link belongs to this object (and the foreground service), not to the
 *   screen: closing the app does not drop it, and opening it again finds it.
 * - The glasses drop a link nothing is written to for ~30 s. The app's own
 *   heartbeat (Flutter) can stall in the background, so this object sends one
 *   too whenever an arm has been quiet for [IDLE_BEFORE_BEAT_MS], holding a
 *   partial wake lock while linked so the phone does not sleep through it.
 * - The pair is saved, so a restarted process (the system restarts the
 *   service) dials it again with no screen at all.
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
    @Volatile var prefer2M = true // ask for Bluetooth 5's 2M radio mode
    private val retries = mutableMapOf<String, Int>()

    /** The user wants this pair connected (until they disconnect it). */
    val isWanted: Boolean get() = wanted

    private const val PREFS = "even_suite_ble"
    private const val BEAT_EVERY_MS = 4_000L
    private const val IDLE_BEFORE_BEAT_MS = 12_000L
    private val beatThread = HandlerThread("G1-heartbeat").apply { start() }
    private val beat = Handler(beatThread.looper)
    private var beatSeq = 0
    private var wakeLock: PowerManager.WakeLock? = null

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

    /** A link was wanted when the process last ran (it was killed, not disconnected). */
    fun hasSavedLink(context: Context): Boolean =
        context.applicationContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getBoolean("wanted", false)

    /**
     * The process was restarted (by the system, or at boot): dial the saved
     * pair again, with no app screen. Returns false when nothing was saved.
     */
    fun restore(context: Context): Boolean {
        init(context)
        if (wanted && pair != null) return true
        val sp = appContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        if (!sp.getBoolean("wanted", false)) return false
        val channel = sp.getString("channel", null) ?: return false
        val saved = mapOf(
            "leftDeviceName" to sp.getString("leftName", ""), "leftAddress" to sp.getString("leftAddress", ""),
            "rightDeviceName" to sp.getString("rightName", ""), "rightAddress" to sp.getString("rightAddress", ""),
        )
        return connect(channel, saved, fromRestore = true) == null
    }

    private fun persist(p: BlePairDevice?) {
        val e = appContext.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit()
        if (p == null) {
            e.clear()
        } else {
            e.putBoolean("wanted", true).putString("channel", p.channelNumber)
                .putString("leftName", p.leftDevice.name).putString("leftAddress", p.leftDevice.address)
                .putString("rightName", p.rightDevice.name).putString("rightAddress", p.rightDevice.address)
        }
        e.apply()
    }

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
    fun connect(channel: String, saved: Map<*, *>?, fromRestore: Boolean = false): String? {
        val a = adapter ?: return "This phone has no Bluetooth"
        if (!a.isEnabled) return "Bluetooth is off. Turn it on and try again."
        // Already linked to this pair (the app was closed and opened again):
        // keep the link, just tell the app where it stands. Arms that are
        // not up are dialled again.
        val existing = pair
        if (wanted && existing != null && existing.channelNumber == channel) {
            if (existing.isBothConnected()) {
                emitStatus("glassesConnected", existing.toInfoJson() + ("status" to "connected"))
            } else {
                emitStatus("glassesConnecting", existing.toInfoJson())
                for (arm in listOf(existing.leftDevice, existing.rightDevice)) {
                    if (arm.isConnected) continue
                    try {
                        arm.gatt?.disconnect()
                        arm.gatt?.close()
                    } catch (_: Exception) {
                    }
                    arm.gatt = null
                    retries[arm.address] = 0
                    dial(arm, autoConnect = false)
                }
            }
            startBeat()
            return null
        }
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
        persist(p)
        if (!fromRestore) SuiteService.linkChanged(appContext)
        startBeat()
        emitStatus("glassesConnecting", p.toInfoJson())
        dial(left, autoConnect = false)
        dial(right, autoConnect = false)
        return null
    }

    fun disconnect() {
        wanted = false
        val p = pair
        stopBeat()
        closeAll()
        pair = null
        persist(null)
        SuiteService.linkChanged(appContext)
        if (p != null) emitStatus("glassesDisconnected", p.toInfoJson() + ("reason" to "user"))
    }

    /** True once an app screen listens to this link (the notification says so when not). */
    val appAttached: Boolean get() = listener != null

    fun send(data: ByteArray, lr: String?) {
        val p = pair ?: return
        if (lr == null || lr == "L") p.leftDevice.sendData(data)
        if (lr == null || lr == "R") p.rightDevice.sendData(data)
    }

    /** Queue many packets to one arm in a single call; [done] gets how many were written. */
    fun sendBatch(packets: List<ByteArray>, lr: String?, gapMs: Long, done: (Int) -> Unit) {
        val arm = (if (lr == "R") pair?.rightDevice else pair?.leftDevice)
        if (arm == null || !arm.isConnected) {
            done(0)
            return
        }
        arm.enqueue(packets, gapMs) { sent -> main.post { done(sent) } }
    }

    /** Per-arm write statistics since the last call (diagnostics log). */
    fun stats(): Map<String, Any> {
        val p = pair ?: return emptyMap()
        return mapOf("L" to p.leftDevice.takeStats(), "R" to p.rightDevice.takeStats())
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

    // ------------------------------------------------- keeping the link alive

    private fun startBeat() {
        beat.removeCallbacks(beatTick)
        beat.postDelayed(beatTick, BEAT_EVERY_MS)
    }

    private fun stopBeat() {
        beat.removeCallbacks(beatTick)
        releaseAwake()
    }

    /** Every few seconds: an arm nothing was written to for a while gets a heartbeat (and the phone stays awake for it). */
    private val beatTick = object : Runnable {
        override fun run() {
            val p = pair
            if (p != null && wanted) {
                val up = listOf(p.leftDevice, p.rightDevice).filter { it.isConnected }
                if (up.isNotEmpty()) {
                    holdAwake()
                    for (arm in up) {
                        if (arm.idleMs() > IDLE_BEFORE_BEAT_MS) {
                            // 0x25 <len> 0 <seq> 4 <seq>, as the app's own heartbeat (its sequence numbers are below 0x80).
                            val s = (0x80 + (beatSeq++ % 0x7e)).toByte()
                            arm.sendData(byteArrayOf(0x25, 6, 0, s, 0x04, s))
                        }
                    }
                } else {
                    releaseAwake()
                }
                beat.postDelayed(this, BEAT_EVERY_MS)
            } else {
                releaseAwake()
            }
        }
    }

    /** Partial wake lock while linked; re-armed on every tick, so a hung process lets go by itself. */
    private fun holdAwake() {
        try {
            val wl = wakeLock ?: appContext.getSystemService(PowerManager::class.java)
                .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "EvenSuite:glasses")
                .apply { setReferenceCounted(false) }
                .also { wakeLock = it }
            wl.acquire(30_000L)
        } catch (e: Exception) {
            Log.w(TAG, "wake lock: $e")
        }
    }

    private fun releaseAwake() {
        try {
            wakeLock?.let { if (it.isHeld) it.release() }
        } catch (_: Exception) {
        }
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
                // status: 8 = the glasses stopped answering (supervision timeout), 19 = they hung up,
                // 22 = this phone hung up, 133 = a failed attempt.
                emitStatus("glassesDisconnected",
                    p.toInfoJson() + mapOf("arm" to armLabel(arm), "reason" to "link", "gattStatus" to status))
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
            arm.mtu = mtu
            // Shortest connection interval: the main limit on packets per second.
            gatt.requestConnectionPriority(BluetoothGatt.CONNECTION_PRIORITY_HIGH)
            // Bluetooth 5 "2M" radio mode doubles the over-the-air rate if the
            // glasses accept it; otherwise the link simply stays at 1M.
            try {
                if (prefer2M) {
                    gatt.setPreferredPhy(BluetoothDevice.PHY_LE_2M_MASK, BluetoothDevice.PHY_LE_2M_MASK,
                        BluetoothDevice.PHY_OPTION_NO_PREFERRED)
                }
            } catch (e: Exception) {
                Log.w(TAG, "${arm.name}: setPreferredPhy: $e")
            }
            if (gatt.device.bondState == BluetoothDevice.BOND_NONE) gatt.device.createBond()
            arm.isConnected = true
            arm.sendData(byteArrayOf(0xf4.toByte(), 0x01.toByte()))
            if (p.isBothConnected()) {
                emitStatus("glassesConnected", p.toInfoJson() + ("status" to "connected"))
            }
        }

        override fun onPhyUpdate(gatt: BluetoothGatt, txPhy: Int, rxPhy: Int, status: Int) {
            val arm = pair?.armFor(gatt.device.address) ?: return
            arm.phy = if (txPhy == BluetoothDevice.PHY_LE_2M) "2M" else if (txPhy == BluetoothDevice.PHY_LE_CODED) "coded" else "1M"
            Log.i(TAG, "${arm.name}: phy tx=$txPhy rx=$rxPhy status=$status")
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
        // 0xF1 = glasses microphone: [0xF1, seq, 200 bytes LC3]. Straight to the
        // speech recognizer, never through Flutter.
        if (value[0] == 0xF1.toByte()) {
            if (value.size == 202) GlassesMic.onPacket(value.copyOfRange(2, 202))
            return
        }
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
                startBeat()
                dial(p.leftDevice, autoConnect = false)
                dial(p.rightDevice, autoConnect = false)
            }
        }
    }
}
