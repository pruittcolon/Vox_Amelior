package com.example.demo_ai_even

import android.app.SearchManager
import android.content.Intent
import android.net.Uri
import android.provider.AlarmClock
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import android.app.Activity
import android.util.Log

/**
 * Platform channel for Google Assistant-like functionality.
 * 
 * Provides native Android intent integration for:
 * - Launching Google Assistant
 * - Setting timers and alarms
 * - Making calls and sending messages
 * - Web search and navigation
 */
class GoogleAssistantChannel(
    private val flutterEngine: FlutterEngine, 
    private val activity: Activity
) {
    companion object {
        const val CHANNEL = "com.example.demo_ai_even/google_assistant"
        private const val TAG = "GoogleAssistantChannel"
    }
    
    /**
     * Register the platform channel with Flutter.
     */
    fun register() {
        Log.i(TAG, "Registering Google Assistant platform channel")
        
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL)
            .setMethodCallHandler { call, result ->
                Log.d(TAG, "Method called: ${call.method}")
                
                when (call.method) {
                    "launchAssistant" -> launchGoogleAssistant(result)
                    "setTimer" -> {
                        @Suppress("UNCHECKED_CAST")
                        val args = call.arguments as? Map<String, Any>
                        setTimer(args, result)
                    }
                    "setAlarm" -> {
                        @Suppress("UNCHECKED_CAST")
                        val args = call.arguments as? Map<String, Any>
                        setAlarm(args, result)
                    }
                    "makeCall" -> {
                        @Suppress("UNCHECKED_CAST")
                        val args = call.arguments as? Map<String, Any>
                        makeCall(args, result)
                    }
                    "sendMessage" -> {
                        @Suppress("UNCHECKED_CAST")
                        val args = call.arguments as? Map<String, Any>
                        sendMessage(args, result)
                    }
                    "webSearch" -> {
                        @Suppress("UNCHECKED_CAST")
                        val args = call.arguments as? Map<String, Any>
                        webSearch(args, result)
                    }
                    "openMaps" -> {
                        @Suppress("UNCHECKED_CAST")
                        val args = call.arguments as? Map<String, Any>
                        openMaps(args, result)
                    }
                    else -> {
                        Log.w(TAG, "Unknown method: ${call.method}")
                        result.notImplemented()
                    }
                }
            }
    }
    
    /**
     * Launch Google Assistant in listening mode.
     */
    private fun launchGoogleAssistant(result: MethodChannel.Result) {
        try {
            // Try ACTION_VOICE_COMMAND first (opens Assistant in listening mode)
            val intent = Intent(Intent.ACTION_VOICE_COMMAND).apply {
                flags = Intent.FLAG_ACTIVITY_NEW_TASK
            }
            activity.startActivity(intent)
            Log.i(TAG, "Launched Google Assistant via ACTION_VOICE_COMMAND")
            result.success(true)
        } catch (e: Exception) {
            Log.w(TAG, "ACTION_VOICE_COMMAND failed, trying ACTION_ASSIST: ${e.message}")
            try {
                // Fallback to ACTION_ASSIST
                val fallback = Intent(Intent.ACTION_ASSIST).apply {
                    flags = Intent.FLAG_ACTIVITY_NEW_TASK
                }
                activity.startActivity(fallback)
                Log.i(TAG, "Launched Google Assistant via ACTION_ASSIST")
                result.success(true)
            } catch (e2: Exception) {
                Log.e(TAG, "Failed to launch assistant: ${e2.message}")
                result.error("ASSISTANT_ERROR", "Could not launch assistant: ${e2.message}", null)
            }
        }
    }
    
    /**
     * Set a countdown timer.
     * 
     * @param args Map containing 'seconds' (Int) and optional 'message' (String)
     */
    private fun setTimer(args: Map<String, Any>?, result: MethodChannel.Result) {
        if (args == null) {
            result.error("INVALID_ARGS", "Arguments required for setTimer", null)
            return
        }
        
        val seconds = (args["seconds"] as? Number)?.toInt() ?: 60
        val message = args["message"] as? String ?: "Timer"
        
        try {
            val intent = Intent(AlarmClock.ACTION_SET_TIMER).apply {
                putExtra(AlarmClock.EXTRA_LENGTH, seconds)
                putExtra(AlarmClock.EXTRA_MESSAGE, message)
                putExtra(AlarmClock.EXTRA_SKIP_UI, false) // Show Clock UI
                flags = Intent.FLAG_ACTIVITY_NEW_TASK
            }
            activity.startActivity(intent)
            Log.i(TAG, "Set timer for $seconds seconds with message '$message'")
            result.success(true)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to set timer: ${e.message}")
            result.error("TIMER_ERROR", "Could not set timer: ${e.message}", null)
        }
    }
    
    /**
     * Set an alarm.
     * 
     * @param args Map containing 'hour' (Int 0-23), 'minute' (Int 0-59), optional 'message' (String)
     */
    private fun setAlarm(args: Map<String, Any>?, result: MethodChannel.Result) {
        if (args == null) {
            result.error("INVALID_ARGS", "Arguments required for setAlarm", null)
            return
        }
        
        val hour = (args["hour"] as? Number)?.toInt() ?: 7
        val minute = (args["minute"] as? Number)?.toInt() ?: 0
        val message = args["message"] as? String ?: "Alarm"
        
        try {
            val intent = Intent(AlarmClock.ACTION_SET_ALARM).apply {
                putExtra(AlarmClock.EXTRA_HOUR, hour)
                putExtra(AlarmClock.EXTRA_MINUTES, minute)
                putExtra(AlarmClock.EXTRA_MESSAGE, message)
                putExtra(AlarmClock.EXTRA_SKIP_UI, false) // Show Clock UI
                flags = Intent.FLAG_ACTIVITY_NEW_TASK
            }
            activity.startActivity(intent)
            Log.i(TAG, "Set alarm for $hour:$minute with message '$message'")
            result.success(true)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to set alarm: ${e.message}")
            result.error("ALARM_ERROR", "Could not set alarm: ${e.message}", null)
        }
    }
    
    /**
     * Open the phone dialer with a number.
     * 
     * @param args Map containing 'number' (String)
     */
    private fun makeCall(args: Map<String, Any>?, result: MethodChannel.Result) {
        val number = args?.get("number") as? String
        if (number.isNullOrBlank()) {
            result.error("INVALID_ARGS", "Phone number required", null)
            return
        }
        
        try {
            val intent = Intent(Intent.ACTION_DIAL).apply {
                data = Uri.parse("tel:$number")
                flags = Intent.FLAG_ACTIVITY_NEW_TASK
            }
            activity.startActivity(intent)
            Log.i(TAG, "Opening dialer for $number")
            result.success(true)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to open dialer: ${e.message}")
            result.error("CALL_ERROR", "Could not open dialer: ${e.message}", null)
        }
    }
    
    /**
     * Open the SMS app with a recipient and optional message.
     * 
     * @param args Map containing optional 'number' (String) and 'message' (String)
     */
    private fun sendMessage(args: Map<String, Any>?, result: MethodChannel.Result) {
        val number = args?.get("number") as? String ?: ""
        val message = args?.get("message") as? String ?: ""
        
        try {
            val intent = Intent(Intent.ACTION_SENDTO).apply {
                data = Uri.parse("smsto:$number")
                putExtra("sms_body", message)
                flags = Intent.FLAG_ACTIVITY_NEW_TASK
            }
            activity.startActivity(intent)
            Log.i(TAG, "Opening messaging for $number")
            result.success(true)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to open messaging: ${e.message}")
            result.error("MESSAGE_ERROR", "Could not open messaging: ${e.message}", null)
        }
    }
    
    /**
     * Perform a web search.
     * 
     * @param args Map containing 'query' (String)
     */
    private fun webSearch(args: Map<String, Any>?, result: MethodChannel.Result) {
        val query = args?.get("query") as? String
        if (query.isNullOrBlank()) {
            result.error("INVALID_ARGS", "Search query required", null)
            return
        }
        
        try {
            val intent = Intent(Intent.ACTION_WEB_SEARCH).apply {
                putExtra(SearchManager.QUERY, query)
                flags = Intent.FLAG_ACTIVITY_NEW_TASK
            }
            activity.startActivity(intent)
            Log.i(TAG, "Performing web search for '$query'")
            result.success(true)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to perform search: ${e.message}")
            result.error("SEARCH_ERROR", "Could not perform search: ${e.message}", null)
        }
    }
    
    /**
     * Open Google Maps with a destination query.
     * 
     * @param args Map containing 'query' (String)
     */
    private fun openMaps(args: Map<String, Any>?, result: MethodChannel.Result) {
        val query = args?.get("query") as? String
        if (query.isNullOrBlank()) {
            result.error("INVALID_ARGS", "Destination required", null)
            return
        }
        
        try {
            val intent = Intent(Intent.ACTION_VIEW).apply {
                data = Uri.parse("geo:0,0?q=${Uri.encode(query)}")
                flags = Intent.FLAG_ACTIVITY_NEW_TASK
            }
            activity.startActivity(intent)
            Log.i(TAG, "Opening maps for '$query'")
            result.success(true)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to open maps: ${e.message}")
            result.error("MAPS_ERROR", "Could not open maps: ${e.message}", null)
        }
    }
}
