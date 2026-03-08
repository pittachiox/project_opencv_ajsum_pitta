import re

with open('c:/Users/HP/source/repos/final/online1.h', 'r', encoding='utf-8') as f:
    text = f.read()

# Locate where the UI class begins
split_marker = 'namespace ConsoleApplication3 {'
if split_marker not in text:
    print("Error: Could not find WinForms namespace.")
    exit(1)

engine_part, ui_part = text.split(split_marker, 1)

# We want to replace calls to Engine variables and functions with `GetCam()->...`
# Wait, first we must define `GetCam()` inside `UploadForm` or globally.
global_defs = """
__declspec(selectany) std::map<int, CameraInstance*> g_cameras;
__declspec(selectany) int g_activeCameraId = 1;

static CameraInstance* GetCam(int id = -1) {
    if (id == -1) id = g_activeCameraId;
    if (g_cameras.find(id) == g_cameras.end()) {
        std::vector<CameraConfig> confs = CameraManager::LoadCameras();
        CameraConfig curr;
        curr.id = id;
        for (auto& c : confs) if(c.id == id) curr = c;
        g_cameras[id] = new CameraInstance(curr);
    }
    return g_cameras[id];
}

namespace ConsoleApplication3 {
"""

engine_part = engine_part + global_defs 

# Variables and functions to replace inside UI
# Let's list the known ones that were moved into CameraInstance
targets = [
    'g_onlineState', 'g_onlineStateMutex', 'g_net', 'g_onnx_net', 'g_classes', 'g_colors',
    'g_tracker', 'g_pm_logic_online', 'g_cap', 'g_latestRawFrame', 'g_frameSeq_online',
    'g_frameMutex', 'g_connectionAttemptId_online', 'g_cameraFPS', 'g_aiMutex_online',
    'g_videoWriter_online', 'g_videoWriterMutex_online', 'g_currentVideoRelPath',
    'g_videoClipStartTick', 'g_videoFramesWritten', 'g_lastClipActualFps',
    'VIDEO_CLIP_SECONDS', 'g_videoRecordingRunning', 'g_videoRecordingThread',
    'g_videoCurrentFrameMutex', 'g_videoCurrentFrame', 'g_modelReady', 'g_parkingEnabled_online',
    'templateSet_online', 'YOLO_INPUT_SIZE', 'CONF_THRESHOLD', 'NMS_THRESHOLD',
    'g_pm_display_online', 'g_cachedParkingOverlay_online', 'g_lastDrawnStatus_online',
    'g_drawingBuffer_online', 'g_processedFrame_online', 'g_processedSeq_online',
    'g_processedMutex_online', 'g_mjpegServer_online', 'g_lastViolationCheck_online',
    'VIOLATION_CHECK_INTERVAL_MS_ONLINE', 'g_droppedFrames_online', 'g_processedFramesCount_online',
    'MAX_FRAME_LAG_ONLINE', 'NETWORK_BUFFER_SIZE', 'g_labelCache_online', 'g_redOverlayBuffer_online',
    'g_fpsMonitor_online',
    # Functions
    'ResetParkingCache_Online', 'FormatToLetterbox', 'GetRawFrameOnline', 
    'GetProcessedFrameOnline', 'OpenGlobalCamera', 'OpenGlobalCameraFromIP',
    'StopCameraHeadless', 'LoadParkingTemplate_Online', 'CreateViolationVisualization',
    'ProcessFrameOnline', 'DrawSceneOnline', 'StartNewVideoClip_Online',
    'StopVideoRecording_Online', 'VideoRecordingThreadFunc_Online',
    'StartVideoRecordingThread_Online', 'StopVideoRecordingThread_Online'
]

# We should sort targets by length (longest first) to avoid partial matching bugs
targets.sort(key=len, reverse=True)

for t in targets:
    # Use negative lookbehind so we don't double replace: (?<!->)
    pattern = r'(?<!->)\b' + re.escape(t) + r'\b'
    ui_part = re.sub(pattern, f'GetCam()->{t}', ui_part)

with open('c:/Users/HP/source/repos/final/online1.h', 'w', encoding='utf-8') as f:
    f.write(engine_part + ui_part)

print("UI Code Patched!")
