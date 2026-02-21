package com.brace.sideline

import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import org.json.JSONObject
import java.util.concurrent.TimeUnit

interface CollisionSignalListener {
    fun onCollisionStart()
    fun onCollisionEnd(stopAfterMs: Long)
}

class LiveStreamSocketClient(
    private val webSocketUrl: String,
    private val listener: CollisionSignalListener
) {
    private val client = OkHttpClient.Builder()
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .build()

    private var socket: WebSocket? = null

    fun connect() {
        val request = Request.Builder().url(webSocketUrl).build()
        socket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                val ping = JSONObject().put("type", "ping")
                webSocket.send(ping.toString())
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                val payload = JSONObject(text)
                val type = payload.optString("type")
                when (type) {
                    "collision_start" -> listener.onCollisionStart()
                    "collision_end" -> {
                        val stopAfter = payload.optLong("stop_after_ms", 1000L)
                        listener.onCollisionEnd(stopAfter)
                    }
                }
            }
        })
    }

    fun disconnect() {
        socket?.close(1000, "client disconnect")
        socket = null
    }

    fun sendLandmarkFrame(
        playId: String,
        playerId: String,
        timestampMs: Double,
        headX: Double,
        headY: Double,
        shoulderWidthPx: Double
    ) {
        val payload = JSONObject()
            .put("play_id", playId)
            .put("player_id", playerId)
            .put("timestamp_ms", timestampMs)
            .put("head", JSONObject().put("x", headX).put("y", headY))
            .put("shoulder_width_px", shoulderWidthPx)
        socket?.send(payload.toString())
    }
}
