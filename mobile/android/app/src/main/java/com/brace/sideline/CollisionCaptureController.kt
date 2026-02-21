package com.brace.sideline

import android.content.Context
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraConstrainedHighSpeedCaptureSession
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.media.MediaRecorder
import android.os.Handler
import android.os.Looper
import android.view.Surface
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.asRequestBody
import java.io.File
import java.io.IOException
import java.util.concurrent.TimeUnit

class CollisionCaptureController(
    private val context: Context,
    private val uploadBaseUrl: String
) {
    private val mainHandler = Handler(Looper.getMainLooper())
    private val cameraManager = context.getSystemService(CameraManager::class.java)
    private val uploadClient = ClipUploadClient(uploadBaseUrl)

    private var cameraDevice: CameraDevice? = null
    private var highSpeedSession: CameraConstrainedHighSpeedCaptureSession? = null
    private var previewSurface: Surface? = null
    private var recorderSurface: Surface? = null

    private var mediaRecorder: MediaRecorder? = null
    private var outputFile: File? = null
    private var isRecording: Boolean = false

    private var playId: String = ""
    private var playerId: String = ""
    private var whistleReceived = false

    fun setContext(playId: String, playerId: String) {
        this.playId = playId
        this.playerId = playerId
    }

    fun open240FpsCamera(previewSurface: Surface) {
        this.previewSurface = previewSurface
        val cameraId = findBackCameraWith240Fps() ?: return
        cameraManager.openCamera(cameraId, object : CameraDevice.StateCallback() {
            override fun onOpened(camera: CameraDevice) {
                cameraDevice = camera
                createHighSpeedSession()
            }

            override fun onDisconnected(camera: CameraDevice) {
                camera.close()
                cameraDevice = null
            }

            override fun onError(camera: CameraDevice, error: Int) {
                camera.close()
                cameraDevice = null
            }
        }, mainHandler)
    }

    fun onCollisionStart() {
        if (isRecording) return
        val camera = cameraDevice ?: return
        val preview = previewSurface ?: return
        val recorder = mediaRecorder ?: return
        recorder.prepare()
        recorder.start()
        isRecording = true

        val recorderOutputSurface = recorderSurface ?: recorder.surface
        val session = highSpeedSession ?: return

        val requestBuilder = camera.createCaptureRequest(CameraDevice.TEMPLATE_RECORD).apply {
            addTarget(preview)
            addTarget(recorderOutputSurface)
            set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, android.util.Range(240, 240))
        }
        val burst = session.createHighSpeedRequestList(requestBuilder.build())
        session.setRepeatingBurst(burst, null, mainHandler)
    }

    fun onCollisionEnd(stopAfterMs: Long = 1000L) {
        mainHandler.postDelayed({ stopRecordingIfNeeded() }, stopAfterMs)
    }

    fun onPlayWhistle() {
        whistleReceived = true
        if (!isRecording) {
            outputFile?.let { file ->
                uploadFile(file)
            }
        }
    }

    private fun stopRecordingIfNeeded() {
        if (!isRecording) return
        val recorder = mediaRecorder ?: return
        try {
            recorder.stop()
        } catch (_: Exception) {
        }
        isRecording = false
        recorder.reset()
        recorder.release()
        mediaRecorder = null
        recorderSurface = null

        if (whistleReceived) {
            outputFile?.let { uploadFile(it) }
        }
        createHighSpeedSession()
    }

    private fun createHighSpeedSession() {
        val camera = cameraDevice ?: return
        val preview = previewSurface ?: return
        if (mediaRecorder == null) {
            mediaRecorder = buildRecorder()
        }
        val recorder = mediaRecorder ?: return
        recorderSurface = recorder.surface

        val surfaces = mutableListOf(preview, recorder.surface)
        camera.createConstrainedHighSpeedCaptureSession(
            surfaces,
            object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(session: CameraCaptureSession) {
                    highSpeedSession = session as CameraConstrainedHighSpeedCaptureSession
                }

                override fun onConfigureFailed(session: CameraCaptureSession) {
                    highSpeedSession = null
                }
            },
            mainHandler
        )
    }

    private fun buildRecorder(): MediaRecorder {
        val file = File(context.cacheDir, "play-${System.currentTimeMillis()}.mp4")
        outputFile = file
        return MediaRecorder().apply {
            setVideoSource(MediaRecorder.VideoSource.SURFACE)
            setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
            setVideoEncoder(MediaRecorder.VideoEncoder.H264)
            setVideoEncodingBitRate(24_000_000)
            setVideoFrameRate(240)
            setCaptureRate(240.0)
            setVideoSize(1280, 720)
            setOutputFile(file.absolutePath)
        }
    }

    private fun findBackCameraWith240Fps(): String? {
        for (cameraId in cameraManager.cameraIdList) {
            val chars = cameraManager.getCameraCharacteristics(cameraId)
            val facing = chars.get(CameraCharacteristics.LENS_FACING)
            if (facing != CameraCharacteristics.LENS_FACING_BACK) continue

            val map = chars.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP) ?: continue
            val sizes = map.highSpeedVideoSizes ?: continue
            for (size in sizes) {
                val ranges = map.getHighSpeedVideoFpsRangesFor(size)
                if (ranges.any { it.upper >= 240 }) {
                    return cameraId
                }
            }
        }
        return null
    }

    private fun uploadFile(file: File) {
        Thread {
            try {
                uploadClient.uploadClip(file, playId, playerId)
            } catch (_: IOException) {
            }
        }.start()
    }
}

private class ClipUploadClient(private val baseUrl: String) {
    private val client = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(25, TimeUnit.SECONDS)
        .build()

    fun uploadClip(file: File, playId: String, playerId: String) {
        val mp4 = "video/mp4".toMediaType()
        val form = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart("play_id", playId)
            .addFormDataPart("player_id", playerId)
            .addFormDataPart(
                "file",
                file.name,
                file.asRequestBody(mp4)
            )
            .build()

        val request = Request.Builder()
            .url("$baseUrl/upload-clip")
            .post(form)
            .build()
        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                throw IOException("upload failed: ${response.code}")
            }
        }
    }
}
