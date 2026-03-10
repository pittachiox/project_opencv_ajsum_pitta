#pragma once
#define NOMINMAX // [PHASE 1 FIX] Move to top BEFORE any includes
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <windows.h>
#include <msclr/marshal_cppstd.h>
#include <string>
#include <vector>
#include <map>
#include <direct.h>  // For _getcwd
#include "BYTETracker.h"
#include "ParkingSlot.h"
#include "MjpegServer.h"  // [NEW] Added MjpegServer
#include "ViolationDetailForm.h"
#include "json.hpp" // [PHASE 3] MongoEngine JSON API integration

using json = nlohmann::json;

#pragma managed(push, off)
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <mutex>
#include <set>
#include <chrono> // [PHASE 1] Add for future use
#include <atomic> // [PHASE 1] Add for atomic operations
#include "OnnxYoloInference.h" // [GPU] ONNX Runtime GPU acceleration

// ==========================================
//  LAYER 1: SHARED CONSTANTS & STRUCTS
// ==========================================
#include <fstream>

const int YOLO_INPUT_SIZE = 640;
const float CONF_THRESHOLD = 0.25f;
const float NMS_THRESHOLD = 0.45f;
const int VIOLATION_CHECK_INTERVAL_MS_ONLINE = 500;

struct OnlineAppState {
	std::vector<TrackedObject> cars;
	std::set<int> violatingCarIds;
	std::map<int, SlotStatus> slotStatuses;
	std::map<int, float> slotOccupancy;
	std::map<int, std::string> slotTypes; // [NEW] Track Car/Motorcycle
	long long frameSequence = -1;
};

struct CachedLabel_Online {
	std::string text;
	cv::Size size;
	int baseline;
	bool isViolating;
	int classId;
	
	CachedLabel_Online() : baseline(0), isViolating(false), classId(-1) {}
};

struct FPSMonitor_Online {
	double alpha = 0.1;
	double avgFPS = 0.0;
	long long lastTick = 0;
	
	void update() {
		long long currentTick = cv::getTickCount();
		if (lastTick != 0) {
			double timeSec = (currentTick - lastTick) / cv::getTickFrequency();
			if (timeSec > 0) {
				double fps = 1.0 / timeSec;
				if (avgFPS == 0) avgFPS = fps;
				else avgFPS = avgFPS * (1.0 - alpha) + fps * alpha;
			}
		}
		lastTick = currentTick;
	}
};

// ==========================================
//  [PHASE 14] MULTI-CAMERA INFRASTRUCTURE
// ==========================================
struct CameraConfig {
	int id;
	std::string name;
	std::string rtspUrl;
};

class CameraManager {
public:
	static std::vector<CameraConfig> LoadCameras() {
		std::vector<CameraConfig> cameras;
		std::ifstream file("C:\\camera_ids\\cameras.json");
		if (!file.is_open()) return cameras;

		try {
			json j;
			file >> j;
			for (const auto& item : j) {
				CameraConfig cam;
				cam.id = item.value("id", 0);
				cam.name = item.value("name", "");
				cam.rtspUrl = item.value("rtspUrl", "");
				if (cam.id > 0) cameras.push_back(cam);
			}
		} catch (...) {
			// Catch JSON parsing errors
		}
		return cameras;
	}

	static void SaveCameras(const std::vector<CameraConfig>& cameras) {
		CreateDirectoryA("C:\\camera_ids", NULL);
		json j = json::array();
		for (const auto& cam : cameras) {
			j.push_back({
				{"id", cam.id},
				{"name", cam.name},
				{"rtspUrl", cam.rtspUrl}
			});
		}
		std::ofstream file("C:\\camera_ids\\cameras.json");
		if (file.is_open()) {
			file << j.dump(4);
		}
	}
};

extern MjpegServer* g_globalWebServer;
const int NETWORK_BUFFER_SIZE = 5;    // Network jitter buffer

// CAMERA INSTANCE CLASS DEFINITION REPLACED


__declspec(selectany) std::mutex g_aiMutex_online;
extern int g_selectedGpuId;
const int MAX_FRAME_LAG_ONLINE = 10; // Relaxed from 3 to 10 frames

class CameraInstance {
public:
    int camera_id = 0;
    CameraConfig config;
    
    CameraInstance(const CameraConfig& cfg) : config(cfg) {
        camera_id = cfg.id;
    }
    ~CameraInstance() {
        StopProcessing();
        if (g_cap) {
            if (g_cap->isOpened()) g_cap->release();
            delete g_cap;
            g_cap = nullptr;
        }
		if (g_onnx_net) { delete g_onnx_net; g_onnx_net = nullptr; }
		if (g_tracker) { delete g_tracker; g_tracker = nullptr; }
		if (g_pm_logic_online) { delete g_pm_logic_online; g_pm_logic_online = nullptr; }
		if (g_pm_display_online) { delete g_pm_display_online; g_pm_display_online = nullptr; }
		if (g_mjpegServer_online) { delete g_mjpegServer_online; g_mjpegServer_online = nullptr; }
    }

	OnlineAppState g_onlineState;
	std::mutex g_onlineStateMutex;

	// ==========================================
	//  LAYER 2: LOGIC & BACKEND
	// ==========================================

	// cv::dnn::Net* g_net = nullptr; // [DEPRECATED] Old OpenCV DNN
	OnnxYoloInference* g_onnx_net = nullptr; // [GPU] New ONNX Runtime
	std::vector<std::string> g_classes;
	std::vector<cv::Scalar> g_colors;
	BYTETracker* g_tracker = nullptr;

	ParkingManager* g_pm_logic_online = nullptr;

	cv::VideoCapture* g_cap = nullptr;
	cv::Mat g_latestRawFrame;
	long long g_frameSeq_online = 0;
	std::mutex g_frameMutex;
	std::atomic<int> g_connectionAttemptId_online{0}; // [NEW] Prevent race conditions on multiple connect clicks
	double g_cameraFPS = 30.0;

	// *** [NEW] PROCESSED FRAME SHARING ***
	cv::Mat g_processedFrame_online;
	long long g_processedSeq_online = 0;
	std::mutex g_processedMutex_online;
	int g_droppedFrames_online = 0;
	int g_processedFramesCount_online = 0;
	std::chrono::steady_clock::time_point g_lastViolationCheck_online;

	// *** Threading for Headless Mode (Multi-camera safety) ***
	std::thread* readerThread_online = nullptr;
	std::thread* processingThread_online = nullptr;
	std::atomic<bool> isProcessing{false};
	std::atomic<bool> shouldStop{false};
	long long lastProcessedSeq = -1;
	bool g_modelReady = false; // [PHASE 14] Model ready flag moved to instance

	inline void CameraReaderLoop() {
		while (!shouldStop) {
			cv::Mat tempFrame;
			bool success = false;

			if (g_cap && g_cap->isOpened()) {
				success = g_cap->read(tempFrame);
				if (success && !tempFrame.empty()) {
					long long currentSeq;
					{
						std::lock_guard<std::mutex> lock(g_frameMutex);
						g_latestRawFrame = tempFrame;
						g_frameSeq_online++;
						currentSeq = g_frameSeq_online;
					}
					std::this_thread::sleep_for(std::chrono::milliseconds(5));
					continue;
				}
			} else {
				break;
			}
		}
	}

	inline void ProcessingLoopHeadless();

	inline void StartProcessing() {
		shouldStop = false;
		isProcessing = true;
		
		g_droppedFrames_online = 0;
		g_processedFramesCount_online = 0;
		g_lastViolationCheck_online = std::chrono::steady_clock::now();
		
		{
			std::lock_guard<std::mutex> lock(g_onlineStateMutex);
			g_onlineState = OnlineAppState();
		}
		ResetParkingCache_Online();

		if (processingThread_online == nullptr) {
			processingThread_online = new std::thread(&CameraInstance::ProcessingLoopHeadless, this);
		}
		if (readerThread_online == nullptr) {
			readerThread_online = new std::thread(&CameraInstance::CameraReaderLoop, this);
		}
	}

	inline void StopProcessing() {
		shouldStop = true;
		isProcessing = false;
		
		if (readerThread_online) {
			if (readerThread_online->joinable()) readerThread_online->join();
			delete readerThread_online;
			readerThread_online = nullptr;
		}
		if (processingThread_online) {
			if (processingThread_online->joinable()) processingThread_online->join();
			delete processingThread_online;
			processingThread_online = nullptr;
		}

		if (g_mjpegServer_online) {
			g_mjpegServer_online->Stop();
			delete g_mjpegServer_online;
			g_mjpegServer_online = nullptr;
		}

		StopVideoRecordingThread_Online();
	}

	// ============================================================
	//  [PHASE 1] VIDEO DVR RECORDING (60-SECOND CHUNKS)
	// ============================================================
	cv::VideoWriter* g_videoWriter_online = nullptr;
	std::mutex g_videoWriterMutex_online;
	std::string g_currentVideoRelPath = ""; 
	long long g_videoClipStartTick = 0;
	int g_videoFramesWritten = 0;
	double lastClipActualFps = 0.0;
	const int VIDEO_CLIP_SECONDS = 60; 

	// *** Async Video Recording (Frame Duplication System) ***
	std::atomic<bool> g_videoRecordingRunning{false};
	std::thread* g_videoRecordingThread = nullptr;
	std::mutex g_videoCurrentFrameMutex;
	cv::Mat g_videoCurrentFrame; 
	std::atomic<bool> g_parkingEnabled_online{false}; 
	bool templateSet_online = false;  

	// ==========================================
	//  LAYER 3: PRESENTATION (Frontend)
	// ==========================================
	// [FIX] Move these variables to public area
	public: 
	ParkingManager* g_pm_display_online = nullptr;
	cv::Mat g_cachedParkingOverlay_online;
	std::map<int, SlotStatus> g_lastDrawnStatus_online;
	cv::Mat g_drawingBuffer_online; 

	// Memory pool
	std::map<int, CachedLabel_Online> g_labelCache_online;
	cv::Mat g_redOverlayBuffer_online;
	FPSMonitor_Online g_fpsMonitor_online;

	// *** [NEW] MJPEG SERVER ***
	MjpegServer* g_mjpegServer_online = nullptr;

	// --- Helper Functions ---

	void ResetParkingCache_Online() {
		g_cachedParkingOverlay_online = cv::Mat();
		g_lastDrawnStatus_online.clear();
		g_labelCache_online.clear(); // [PHASE 3] Clear label cache
		g_redOverlayBuffer_online = cv::Mat(); // [PHASE 3] Clear red overlay buffer
	}

	cv::Mat GetRawFrame() {
		std::lock_guard<std::mutex> lock(g_frameMutex);
		return g_latestRawFrame.clone();
	}

	cv::Mat GetProcessedFrame(long long& seq) {
		std::lock_guard<std::mutex> lock(g_processedMutex_online);
		seq = g_processedSeq_online;
		return g_processedFrame_online.clone();
	}

	void OpenGlobalCamera(int cameraIndex = 0) {
		cv::VideoCapture* temp_cap = new cv::VideoCapture(cameraIndex);
		double temp_fps = 30.0;
		if (temp_cap->isOpened()) {
			temp_fps = temp_cap->get(cv::CAP_PROP_FPS);
			if (temp_fps <= 0) temp_fps = 30.0;
		}

		{
			std::lock_guard<std::mutex> lock(g_frameMutex);
			if (g_cap) { delete g_cap; g_cap = nullptr; }
			g_cap = temp_cap;
			g_frameSeq_online = 0;
			g_cameraFPS = temp_fps;
		}

		ResetParkingCache_Online();
		
		std::lock_guard<std::mutex> slock(g_onlineStateMutex);
		g_onlineState = OnlineAppState();
	}

	void OpenGlobalCameraFromIP(const std::string& rtspUrl, int attemptId = -1) {
		cv::VideoCapture* temp_cap = nullptr;
		double temp_fps = 30.0;

		bool isNetwork = rtspUrl.find("http://") == 0 || rtspUrl.find("rtsp://") == 0 || rtspUrl.find("https://") == 0;
		if (isNetwork) {
			OutputDebugStringA(("[OPENCV] Opening network stream with timeout (Auto Backend) for camera " + std::to_string(camera_id) + ": " + rtspUrl + "\n").c_str());
			temp_cap = new cv::VideoCapture();
			
			// [FIX] Set connection timeout to prevent hanging (5 seconds)
			temp_cap->set(cv::CAP_PROP_OPEN_TIMEOUT_MSEC, 5000); // 5 second timeout
			
			bool opened = temp_cap->open(rtspUrl);
			
			if (attemptId != -1 && attemptId != g_connectionAttemptId_online.load()) {
				delete temp_cap; return;
			}

			if (!opened || !temp_cap->isOpened()) {
				OutputDebugStringA(("[OPENCV] Auto Backend failed or timeout for camera " + std::to_string(camera_id) + ". Trying CAP_FFMPEG explicitly...\n").c_str());
				delete temp_cap;
				
				if (attemptId != -1 && attemptId != g_connectionAttemptId_online.load()) return;
				
				temp_cap = new cv::VideoCapture();
				temp_cap->set(cv::CAP_PROP_OPEN_TIMEOUT_MSEC, 5000);
				temp_cap->open(rtspUrl, cv::CAP_FFMPEG);
			}
		} else {
			OutputDebugStringA(("[OPENCV] Opening local stream for camera " + std::to_string(camera_id) + ": " + rtspUrl + "\n").c_str());
			temp_cap = new cv::VideoCapture(rtspUrl);
		}
		
		if (attemptId != -1 && attemptId != g_connectionAttemptId_online.load()) {
			if (temp_cap) delete temp_cap;
			return;
		}

		// [FIX] Network stream optimization
		if (temp_cap && temp_cap->isOpened()) {
			OutputDebugStringA(("[OPENCV] Stream opened successfully for camera " + std::to_string(camera_id) + "! Backend API used: " + temp_cap->getBackendName() + "\n").c_str());
			temp_cap->set(cv::CAP_PROP_BUFFERSIZE, NETWORK_BUFFER_SIZE); // Reduce latency
			temp_fps = temp_cap->get(cv::CAP_PROP_FPS);
			if (temp_fps <= 0) temp_fps = 30.0;
		} else {
			OutputDebugStringA(("[OPENCV ERROR] Failed to open stream (connection timeout or invalid URL) for camera " + std::to_string(camera_id) + ": " + rtspUrl + "\n").c_str());
		}
		
		{
			std::lock_guard<std::mutex> lock(g_frameMutex);
			if (g_cap) { delete g_cap; g_cap = nullptr; }
			g_cap = temp_cap;
			g_frameSeq_online = 0;
			g_cameraFPS = temp_fps;
		}

		ResetParkingCache_Online();
		
		std::lock_guard<std::mutex> slock(g_onlineStateMutex);
		g_onlineState = OnlineAppState();
	}

	void InitGlobalModel(const std::string& modelPath) {
		// Use the global ai mutex, it is defined far below
		g_modelReady = false;
		if (g_onnx_net) { delete g_onnx_net; g_onnx_net = nullptr; }
		if (g_tracker) { delete g_tracker; g_tracker = nullptr; }

		try {
			// Use ONNX Runtime with GPU acceleration and user-selected Device ID
			g_onnx_net = new OnnxYoloInference();
			if (!g_onnx_net->loadModel(modelPath, true, g_selectedGpuId)) { // true = use GPU, pass ID
				delete g_onnx_net;
				g_onnx_net = nullptr;
				OutputDebugStringA(("[ERROR] Failed to load ONNX model with GPU for camera " + std::to_string(camera_id) + "\n").c_str());
				return;
			}
			
			g_tracker = new BYTETracker(90, 0.25f);
			
			OutputDebugStringA(("[INFO] Online mode with ONNX Runtime GPU + ByteTrack for camera " + std::to_string(camera_id) + "\n").c_str());

			g_classes = {
				"bicycle", "car", "motorcycle", "bus", "train", "truck"
			};

			g_colors.clear();
			for (size_t i = 0; i < g_classes.size(); i++) {
				g_colors.push_back(cv::Scalar(rand() % 255, rand() % 255, rand() % 255));
			}
			g_modelReady = true;
		}
		catch (...) {
			if (g_onnx_net) { delete g_onnx_net; g_onnx_net = nullptr; }
			if (g_tracker) { delete g_tracker; g_tracker = nullptr; }
		}
	}

	// [NEW] โหลด Parking Template
	bool LoadParkingTemplate_Online(const std::string& filename) {
		ResetParkingCache_Online();
		templateSet_online = false; // [FIX] Force the engine to register the new template frame geometry

		if (!g_pm_logic_online) g_pm_logic_online = new ParkingManager();
		if (!g_pm_display_online) g_pm_display_online = new ParkingManager();

		bool s1 = g_pm_logic_online->loadTemplate(filename);
		bool s2 = g_pm_display_online->loadTemplate(filename);

		if (s1 && s2) {
			g_parkingEnabled_online.store(true); // [PHASE 1 FIX] Use atomic store
			return true;
		}
		return false;
	}

// ==========================================
//  LAYER 3: PRESENTATION (Frontend)
// ==========================================

// *** [NEW] PROCESSED FRAME SHARING (Pipeline Output) moved to CameraInstance ***

// *** [PHASE 2] PERFORMANCE OPTIMIZATION ***
// We initialize g_lastViolationCheck_online below where the definition runs.

// *** [PHASE 3] MEMORY OPTIMIZATION ***

// --- Helper Functions ---

inline cv::Mat CameraInstance::FormatToLetterbox(const cv::Mat& source, int width, int height, float& ratio, int& dw, int& dh) {
	if (source.empty()) return cv::Mat();

	float r = (std::min)((float)width / source.cols, (float)height / source.rows);
	int new_unpad_w = (int)round(source.cols * r);
	int new_unpad_h = (int)round(source.rows * r);

	dw = (width - new_unpad_w) / 2;
    dh = (height - new_unpad_h) / 2;

	cv::Mat resized;
	if (source.cols != new_unpad_w || source.rows != new_unpad_h) {
		cv::resize(source, resized, cv::Size(new_unpad_w, new_unpad_h));
	}
	else {
		resized = source.clone();
	}

	cv::Mat result(height, width, CV_8UC3, cv::Scalar(114, 114, 114));
	resized.copyTo(result(cv::Rect(dw, dh, new_unpad_w, new_unpad_h)));
	ratio = r;
	return result;
}

// [FIX] Moved the stray code away because it was causing compile errors.
	
/*
	cv::Mat result = fullFrame.clone();
	result = result * 0.3;
	
	cv::Rect safeBbox = carBox & cv::Rect(0, 0, fullFrame.cols, fullFrame.rows);
	if (safeBbox.area() > 0) {
		cv::Mat carROI = fullFrame(safeBbox).clone();
		carROI.copyTo(result(safeBbox));
		cv::rectangle(result, safeBbox, cv::Scalar(0, 255, 255), 3);
	}
	
	return result;
}
*/

// *** WORKER PROCESS (AI Thread) ***

inline void CameraInstance::ProcessFrameOnline(const cv::Mat& inputFrame, long long frameSeq) {
	{
		std::lock_guard<std::mutex> lock(g_aiMutex_online);
		if (inputFrame.empty() || !g_onnx_net || !g_modelReady || !g_tracker) return;
	}

	try {
		cv::Mat workingImage = inputFrame.clone();
		float ratio; int dw, dh;
		cv::Mat input_image = FormatToLetterbox(workingImage, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE, ratio, dw, dh);
		if (input_image.empty()) return;

		cv::Mat blob;
		cv::dnn::blobFromImage(input_image, blob, 1.0 / 255.0, cv::Size(YOLO_INPUT_SIZE, YOLO_INPUT_SIZE), cv::Scalar(), true, false);

		std::vector<cv::Mat> outputs;
		{
			std::lock_guard<std::mutex> lock(g_aiMutex_online);
			if (!g_onnx_net->forward(blob, outputs)) return; // [GPU] ONNX Runtime inference
		}

		if (outputs.empty() || outputs[0].empty()) return;

		cv::Mat output_data = outputs[0];
		int rows = output_data.size[1];
		int dimensions = output_data.size[2];

		if (output_data.dims == 3) {
			output_data = output_data.reshape(1, rows);
			// Only transpose if we have a format like YOLOv8 (e.g., 84x8400)
			if (dimensions > rows) {
				cv::transpose(output_data, output_data);
			}
			rows = output_data.rows;
			dimensions = output_data.cols;
		}
		else {
			output_data = output_data.reshape(1, output_data.size[1]);
			if (output_data.cols > output_data.rows) {
				cv::Mat output_t;
				cv::transpose(output_data, output_t);
				output_data = output_t;
			}
			rows = output_data.rows;
			dimensions = output_data.cols;
		}

		float* data = (float*)output_data.data;
		std::vector<int> class_ids;
		std::vector<float> confs;
		std::vector<cv::Rect> boxes;

		// Check output format: yolo26s uses [cx, cy, w, h, conf, class] (6 cols)
		bool is_yolo26s_format = (dimensions == 6);

		for (int i = 0; i < rows; i++) {
			if (is_yolo26s_format) {
				// yolo26s format: [x1, y1, x2, y2, conf, class_id] (Absolute Coordinates)
				float x1_lb = data[0];
				float y1_lb = data[1];
				float x2_lb  = data[2];
				float y2_lb  = data[3];
				float conf  = data[4];
				int cls     = (int)data[5];

				// The Python test showed class_id can be 0 or 60. We accept any reasonable vehicle class.
				// For the custom car model, we'll accept classes: 0, 1, 2, 3, 5, 7.
				// However, if the user explicitly trained a custom model with only 1-2 classes, it might output 0.
				bool is_vehicle = (cls == 0 || cls == 1 || cls == 2 || cls == 3 || cls == 5 || cls == 7);

				if (conf > CONF_THRESHOLD && is_vehicle) {
					// Convert from letterbox corner coordinates back to original image coordinates
					float left   = (x1_lb - dw) / ratio;
					float top    = (y1_lb - dh) / ratio;
					float right  = (x2_lb - dw) / ratio;
					float bottom = (y2_lb - dh) / ratio;

					float width  = right - left;
					float height = bottom - top;

					// Sanity check
					if (width > 0 && height > 0) {
						boxes.push_back(cv::Rect((int)left, (int)top, (int)width, (int)height));
						confs.push_back(conf);
						class_ids.push_back(cls);
					}
				}
			}
			else {
				// Standard YOLO format: [x_center, y_center, w, h, class_0_conf, class_1_conf, ...]
				int num_classes = dimensions - 4;
				if (num_classes > 0) {
					float* classes_scores = data + 4;
					cv::Mat scores(1, num_classes, CV_32FC1, classes_scores);
					cv::Point class_id;
					double max_class_score;
					cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

					bool is_vehicle = (class_id.x == 0 || class_id.x == 1 || class_id.x == 2 || class_id.x == 3 || class_id.x == 5 || class_id.x == 7);

					if (max_class_score > CONF_THRESHOLD && is_vehicle) {
						float x = data[0]; 
						float y = data[1]; 
						float w = data[2]; 
						float h = data[3];

						float left = (float)((x - 0.5 * w - dw) / ratio);
						float top = (float)((y - 0.5 * h - dh) / ratio);
						float width = w / ratio;
						float height = h / ratio;

						if (width > 0 && height > 0) {
							boxes.push_back(cv::Rect((int)left, (int)top, (int)width, (int)height));
							confs.push_back((float)max_class_score);
							class_ids.push_back(class_id.x);
						}
					}
				}
			}
			data += dimensions;
		}

		std::vector<int> nms;
		cv::dnn::NMSBoxes(boxes, confs, CONF_THRESHOLD, NMS_THRESHOLD, nms);

		std::vector<cv::Rect> nms_boxes;
		std::vector<int> nms_class_ids;
		std::vector<float> nms_confs;
		
		for (int idx : nms) {
			nms_boxes.push_back(boxes[idx]);
			nms_class_ids.push_back(class_ids[idx]);
			nms_confs.push_back(confs[idx]);
		}

		std::vector<TrackedObject> trackedObjs;
		{
			std::lock_guard<std::mutex> lock(g_aiMutex_online);
			trackedObjs = g_tracker->update(nms_boxes, nms_class_ids, nms_confs);
		}

		std::map<int, SlotStatus> calculatedStatuses;
		std::map<int, float> calculatedOccupancy;
		std::map<int, std::string> calculatedTypes;
		std::set<int> violations;

		bool parkingEnabled = g_parkingEnabled_online.load(); // [PHASE 1 FIX] Use atomic load
		if (parkingEnabled && g_pm_logic_online) {
			if (!templateSet_online) {
				g_pm_logic_online->setTemplateFrame(inputFrame);
				templateSet_online = true;
			}

			g_pm_logic_online->updateSlotStatus(trackedObjs);

			for (const auto& slot : g_pm_logic_online->getSlots()) {
				calculatedStatuses[slot.id] = slot.status;
				calculatedOccupancy[slot.id] = slot.occupancyPercent;
				calculatedTypes[slot.id] = slot.type;
			}

			// ตรวจจับรถจอดผิด (จอดนอกช่อง หรือ จอดผิดประเภท)
			for (const auto& car : trackedObjs) {
				if (car.framesStill > 30) {
					bool inAnySlot = false;
                    bool inWrongTypeSlot = false;
                    
					for (const auto& slot : g_pm_logic_online->getSlots()) {
						cv::Point center = (car.bbox.tl() + car.bbox.br()) * 0.5;
						if (cv::pointPolygonTest(slot.polygon, center, false) >= 0) {
							inAnySlot = true;
                            if (slot.status == SlotStatus::ILLEGAL && slot.occupiedByTrackId == car.id) {
                                inWrongTypeSlot = true;
                            }
							break;
						}
					}
                    
                    // Violation if either not in any slot OR parked in a wrong-type slot
					if (!inAnySlot || inWrongTypeSlot) {
						violations.insert(car.id);
					}
				}
			}
		}

		{
			std::lock_guard<std::mutex> stateLock(g_onlineStateMutex);
			g_onlineState.cars = trackedObjs;
			g_onlineState.slotStatuses = calculatedStatuses;
			g_onlineState.slotOccupancy = calculatedOccupancy;
			g_onlineState.slotTypes = calculatedTypes;
			g_onlineState.violatingCarIds = violations;
			g_onlineState.frameSequence = frameSeq;
		}
	}
	catch (...) {}
}

// *** DRAWING FUNCTION (UI Thread) - เหมือนออฟไลน์ ***

inline void CameraInstance::DrawSceneOnline(const cv::Mat& frame, long long displaySeq, cv::Mat& outResult) {
	if (frame.empty()) return;

	// [PHASE 3] Update FPS
	g_fpsMonitor_online.update();

	if (g_drawingBuffer_online.size() != frame.size() || g_drawingBuffer_online.type() != frame.type()) {
		g_drawingBuffer_online.create(frame.size(), frame.type());
	}
	frame.copyTo(g_drawingBuffer_online);
	outResult = g_drawingBuffer_online;

	OnlineAppState state;
	{
		std::lock_guard<std::mutex> lock(g_onlineStateMutex);
		state = g_onlineState;
	}

	bool isFuture = (state.frameSequence > displaySeq);

	// Parking Layer
	bool parkingEnabled = g_parkingEnabled_online.load(); // [PHASE 1 FIX] Use atomic load
	if (parkingEnabled && g_pm_display_online) {
		bool statusChanged = (state.slotStatuses != g_lastDrawnStatus_online);
		bool noCache = g_cachedParkingOverlay_online.empty() || g_cachedParkingOverlay_online.size() != outResult.size();

		if (statusChanged || noCache) {
			g_cachedParkingOverlay_online = cv::Mat::zeros(outResult.size(), CV_8UC3);
			if (!state.slotStatuses.empty()) {
				auto& displaySlots = g_pm_display_online->getSlots();
				for (auto& slot : displaySlots) {
					if (state.slotStatuses.count(slot.id)) {
						slot.status = state.slotStatuses[slot.id];
						slot.occupancyPercent = state.slotOccupancy[slot.id];
					}
				}
			}
			g_cachedParkingOverlay_online = g_pm_display_online->drawSlots(g_cachedParkingOverlay_online);
			g_lastDrawnStatus_online = state.slotStatuses;
		}

		if (!g_cachedParkingOverlay_online.empty()) {
			cv::add(outResult, g_cachedParkingOverlay_online, outResult);
		}
	}

	// Car Layer
	if (!isFuture) {
		// [PHASE 3] Track cars in current frame for GC
		std::set<int> currentFrameCarIds;
		
		for (const auto& obj : state.cars) {
			if (obj.classId >= 0 && obj.classId < g_classes.size()) {
				currentFrameCarIds.insert(obj.id);
				
				cv::Rect box = obj.bbox;
				bool isViolating = (state.violatingCarIds.count(obj.id) > 0);

				if (isViolating) {
					cv::Rect roi = box & cv::Rect(0, 0, outResult.cols, outResult.rows);
					if (roi.area() > 0) {
						cv::Mat roiMat = outResult(roi);
						cv::Mat redBuf(roi.size(), CV_8UC3, cv::Scalar(0, 0, 255));
						cv::addWeighted(roiMat, 0.6, redBuf, 0.4, 0, roiMat);
					}
					cv::rectangle(outResult, box, cv::Scalar(0, 0, 255), 2);
				}
				else {
					cv::rectangle(outResult, box, g_colors[obj.classId], 2);
				}

				// [PHASE 3] Label Caching Logic
				bool needsUpdate = false;
				auto it = g_labelCache_online.find(obj.id);
				
				if (it == g_labelCache_online.end()) {
					needsUpdate = true;
				}
				else {
					CachedLabel_Online& cached = it->second;
					if (cached.isViolating != isViolating || cached.classId != obj.classId) {
						needsUpdate = true;
					}
					if (cached.text.empty()) needsUpdate = true;
				}
				
				if (needsUpdate) {
					CachedLabel_Online cl;
					cl.isViolating = isViolating;
					cl.classId = obj.classId;
					cl.text = "ID:" + std::to_string(obj.id);
					if (isViolating) cl.text += " [VIOLATION]";
					else if (!parkingEnabled) cl.text += " " + g_classes[obj.classId];
					
					cl.size = cv::getTextSize(cl.text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &cl.baseline);
					g_labelCache_online[obj.id] = cl;
				}
				
				// Draw using Cache
				CachedLabel_Online& labelInfo = g_labelCache_online[obj.id];
				cv::Scalar labelBg = isViolating ? cv::Scalar(0, 0, 255) : g_colors[obj.classId];

				cv::rectangle(outResult, cv::Point(box.x, box.y - labelInfo.size.height - 5), 
							  cv::Point(box.x + labelInfo.size.width, box.y), labelBg, -1);
				cv::putText(outResult, labelInfo.text, cv::Point(box.x, box.y - 5), 
							cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
			}
		}
		
		// [PHASE 3] Cache Garbage Collection
		if (g_labelCache_online.size() > 100 && g_labelCache_online.size() > currentFrameCarIds.size() * 2) {
			auto it = g_labelCache_online.begin();
			while (it != g_labelCache_online.end()) {
				if (currentFrameCarIds.find(it->first) == currentFrameCarIds.end()) {
					it = g_labelCache_online.erase(it);
				}
				else {
					++it;
				}
			}
		}
	}

	// [PHASE 3] Draw Stats (Obj count + FPS)
	std::string stats = "Obj: " + std::to_string(state.cars.size()) + " | FPS: " + std::to_string((int)g_fpsMonitor_online.avgFPS);
	cv::putText(outResult, stats, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
}

// ============================================================
//  [PHASE 1] [UNMANAGED] VIDEO RECORDING (DVR) & FPS SYNC
// ============================================================

inline void CameraInstance::StartNewVideoClip_Online(int width, int height) {
	if (width <= 0 || height <= 0) return;
	SYSTEMTIME st;
	GetLocalTime(&st);

	// Date folder: C:\locvideo\YYYYMMDD\camera_{ID}
	char dateFolder[128];
	sprintf_s(dateFolder, sizeof(dateFolder),
		"C:\\locvideo\\%04d%02d%02d\\camera_%d", st.wYear, st.wMonth, st.wDay, camera_id);
	CreateDirectoryA("C:\\locvideo", NULL);
	
	char parentFolder[64];
	sprintf_s(parentFolder, sizeof(parentFolder), "C:\\locvideo\\%04d%02d%02d", st.wYear, st.wMonth, st.wDay);
	CreateDirectoryA(parentFolder, NULL);
	
	CreateDirectoryA(dateFolder, NULL);

	// Filename: HHMMSS.webm (start time of clip)
	char relPath[128];
	sprintf_s(relPath, sizeof(relPath),
		"%04d%02d%02d/camera_%d/%02d%02d%02d.webm",
		st.wYear, st.wMonth, st.wDay, camera_id,
		st.wHour, st.wMinute, st.wSecond);

	std::string fullPath = std::string("C:\\locvideo\\") + relPath;
	std::replace(fullPath.begin(), fullPath.end(), '/', '\\');

	std::lock_guard<std::mutex> lock(g_videoWriterMutex_online);
	if (g_videoWriter_online) {
		g_videoWriter_online->release();
		delete g_videoWriter_online;
		g_videoWriter_online = nullptr;
	}

	// We force the writer to expect strictly 10.0 FPS
	cv::VideoWriter* writer2 = new cv::VideoWriter(
		fullPath,
		cv::VideoWriter::fourcc('V', 'P', '8', '0'),
		10.0,
		cv::Size(width, height));

	if (writer2->isOpened()) {
		g_videoWriter_online = writer2;
		g_currentVideoRelPath = std::string(relPath);
		g_videoClipStartTick = cv::getTickCount();
		g_videoFramesWritten = 0; // reset counter for new clip
		OutputDebugStringA(("[VIDEO] New clip (10 FPS DVR): " + fullPath + "\n").c_str());
	}
	else {
		delete writer2;
		OutputDebugStringA("[VIDEO] Failed to open VideoWriter!\n");
	}
}

inline void CameraInstance::StopVideoRecording_Online() {
	std::lock_guard<std::mutex> lock(g_videoWriterMutex_online);
	if (g_videoWriter_online) {
		g_videoWriter_online->release();
		delete g_videoWriter_online;
		g_videoWriter_online = nullptr;
	}
	g_currentVideoRelPath = "";
}

inline void CameraInstance::VideoRecordingThreadFunc_Online(int width, int height) {
	const int targetFPS = 10;
	const int frameDelayMs = 1000 / targetFPS; // 100ms per frame

	StartNewVideoClip_Online(width, height);
	
	auto nextFrameTime = std::chrono::steady_clock::now();

	cv::Mat lastValidFrame;

	while (g_videoRecordingRunning.load()) {
		cv::Mat frameToWrite;
		{
			std::lock_guard<std::mutex> lock(g_videoCurrentFrameMutex);
			if (!g_videoCurrentFrame.empty()) {
				frameToWrite = g_videoCurrentFrame.clone(); 
				lastValidFrame = frameToWrite.clone();
			} else if (!lastValidFrame.empty()) {
				frameToWrite = lastValidFrame.clone(); // Duplicate last frame to pad the timeline
			}
		}

		bool needsNewClip = false;
		if (!frameToWrite.empty()) {
			std::lock_guard<std::mutex> vl(g_videoWriterMutex_online);
			if (g_videoWriter_online && g_videoWriter_online->isOpened()) {
				g_videoWriter_online->write(frameToWrite);
				g_videoFramesWritten++;

				// Force split strictly by clock time
				double secondsElapsed = ((double)cv::getTickCount() - (double)g_videoClipStartTick) / cv::getTickFrequency();
				if (secondsElapsed >= VIDEO_CLIP_SECONDS) {
					needsNewClip = true;
				}
			}
		}

		if (needsNewClip) {
			StartNewVideoClip_Online(width, height);
			nextFrameTime = std::chrono::steady_clock::now(); // Reset anchor for the next brand new clip
		}

		// ALWAYS advance the clock by exactly 100ms and wait. This forces frame duplication if AI is slow
		nextFrameTime += std::chrono::milliseconds(frameDelayMs);
		auto now = std::chrono::steady_clock::now();
		if (now < nextFrameTime) {
			std::this_thread::sleep_until(nextFrameTime);
		}
	}

	StopVideoRecording_Online();
}

inline void CameraInstance::StartVideoRecordingThread_Online(int width, int height) {
	if (g_videoRecordingThread) return;
	g_videoRecordingRunning.store(true);
	g_videoRecordingThread = new std::thread(&CameraInstance::VideoRecordingThreadFunc_Online, this, width, height);
}

inline void CameraInstance::StopVideoRecordingThread_Online() {
	g_videoRecordingRunning.store(false);
	if (g_videoRecordingThread && g_videoRecordingThread->joinable()) {
		g_videoRecordingThread->join();
	}
	delete g_videoRecordingThread;
	g_videoRecordingThread = nullptr;
}

// *** GET RAW FRAME ***
inline void CameraInstance::GetRawFrameOnline(cv::Mat& outFrame, long long& outSeq) {
	std::lock_guard<std::mutex> lock(g_frameMutex);
	extern void DumpLog(const std::string& msg);
	
	if (!g_latestRawFrame.empty()) {
		outFrame = g_latestRawFrame; // [FIX] Shallow copy for speed (AI thread clones if needed)
		outSeq = g_frameSeq_online;
	}
}

// *** [NEW] GET PROCESSED FRAME (For UI) ***
inline void CameraInstance::GetProcessedFrameOnline(cv::Mat& outFrame, long long& outSeq) {
	std::lock_guard<std::mutex> lock(g_processedMutex_online);
	if (!g_processedFrame_online.empty()) {
		outFrame = g_processedFrame_online; // [FIX] Shallow copy for speed
		outSeq = g_processedSeq_online;
	}
}

// *** [NEW] CREATE VIOLATION VISUALIZATION ***
inline cv::Mat CameraInstance::CreateViolationVisualization(cv::Mat fullFrame, cv::Rect carBox) {
	if (fullFrame.empty()) return cv::Mat();
	
	cv::Mat result = fullFrame.clone();
	result = result * 0.3;
	
	cv::Rect safeBbox = carBox & cv::Rect(0, 0, fullFrame.cols, fullFrame.rows);
	if (safeBbox.area() > 0) {
		cv::Mat carROI = fullFrame(safeBbox).clone();
		carROI.copyTo(result(safeBbox));
		cv::rectangle(result, safeBbox, cv::Scalar(0, 255, 255), 3);
	}
	
	return result;
}

}; // ---- END OF CameraInstance CLASS ----

#pragma managed(pop)


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

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Windows::Forms;
	using namespace System::Drawing;
	using namespace System::Threading;
	using namespace System::Net;
	using namespace System::Net::Sockets;

	public ref class UploadForm : public System::Windows::Forms::Form
	{
	public:
		static UploadForm^ Instance = nullptr;

		UploadForm(void) {
			InitializeComponent();
			bufferLock = gcnew Object();
			bmpBuffer1 = nullptr;
			bmpBuffer2 = nullptr;
			useBuffer1 = true;
			
			isProcessing = false;
			shouldStop = false;
			violationsList_online = gcnew System::Collections::Generic::List<ViolationRecord_Online^>();
			violatingCarTimers_online = gcnew System::Collections::Generic::Dictionary<int, System::DateTime>();

			// [UI FIX] Disable Live Camera until template is loaded
			btnLiveCamera->Enabled = false;
			btnLiveCamera->BackColor = System::Drawing::Color::Gray;

			// Assign self to static property for web api
			UploadForm::Instance = this;

			BackgroundWorker^ modelLoader = gcnew BackgroundWorker();
			modelLoader->DoWork += gcnew DoWorkEventHandler(this, &UploadForm::LoadModel_DoWork);
			modelLoader->RunWorkerCompleted += gcnew RunWorkerCompletedEventHandler(this, &UploadForm::LoadModel_Completed);
			modelLoader->RunWorkerAsync();
		}

	public: bool StartCameraHeadless(int cameraId, String^ ip, String^ port, String^ path) {
		int currentAttemptId = ++GetCam(cameraId)->g_connectionAttemptId_online;
		
		// [FIX CRASH] Stop any existing reader threads to prevent AccessViolation
		// when GetCam(cameraId)->OpenGlobalCameraFromIP deletes GetCam(cameraId)->g_cap while the thread is still reading.
		StopProcessingPublic(cameraId);
		
		if (!path->StartsWith("/")) {
			path = "/" + path;
		}
		array<String^>^ urlFormats = gcnew array<String^> {
			String::Format("http://{0}:{1}{2}", ip, port, path),
			String::Format("http://{0}:{1}/videofeed", ip, port),
			String::Format("http://{0}:{1}/video", ip, port),
			String::Format("rtsp://{0}:{1}", ip, port)
		};

		bool connected = false;
		for each (String^ streamUrl in urlFormats) {
			if (currentAttemptId != GetCam(cameraId)->g_connectionAttemptId_online.load()) {
				DumpLog("[INFO] Aborting previous connection attempt because a new one started.");
				return false;
			}
			
			std::string url = msclr::interop::marshal_as<std::string>(streamUrl);
			DumpLog("[INFO] Headless trying to connect: " + url);
			
			// [FIX] Use attempt ID to allow early abort
			GetCam(cameraId)->OpenGlobalCameraFromIP(url, currentAttemptId);
			
			for(int i = 0; i < 5; i++) {
				if (currentAttemptId != GetCam(cameraId)->g_connectionAttemptId_online.load()) return false;
				Threading::Thread::Sleep(100);
			}

			if (GetCam(cameraId)->g_cap && GetCam(cameraId)->g_cap->isOpened()) {
				cv::Mat testFrame;
				bool canRead = false;
				for (int i = 0; i < 10; ++i) {
					if (currentAttemptId != GetCam(cameraId)->g_connectionAttemptId_online.load()) {
						// Abort! Another thread will clean up GetCam(cameraId)->g_cap if needed, or it's holding valid stream.
						return false;
					}
					{
						std::lock_guard<std::mutex> lock(GetCam(cameraId)->g_frameMutex);
						if (GetCam(cameraId)->g_cap && GetCam(cameraId)->g_cap->read(testFrame) && !testFrame.empty()) { 
                            canRead = true; 
                            GetCam(cameraId)->g_latestRawFrame = testFrame.clone();
                            GetCam(cameraId)->g_frameSeq_online++;
                        }
					}
					if (canRead) {
						connected = true;
						DumpLog("[SUCCESS] Headless Connected successfully!");
						break;
					}
					Threading::Thread::Sleep(100);
				}
				if (!connected) {
					DumpLog("[WARNING] Camera opened but failed to read first frame after 1 second.");
				}
				if (connected) break;
			}
			{
				if (!connected) {
                    std::lock_guard<std::mutex> lock(GetCam(cameraId)->g_frameMutex);
                    if (GetCam(cameraId)->g_cap) { delete GetCam(cameraId)->g_cap; GetCam(cameraId)->g_cap = nullptr; }
                }
			}
		}

		if (connected) {
			GetCam(cameraId)->StartProcessing();
			timer1->Start(); // Ensure UI tick still runs
			return true;
		} else {
			DumpLog("[ERROR] Headless could not connect to camera (timeout or invalid IP).");
			return false;
		}
	}

	protected:
		~UploadForm() {
			StopProcessing(1);
			if (components) delete components;
			if (GetCam(1)->g_pm_logic_online) { delete GetCam(1)->g_pm_logic_online; GetCam(1)->g_pm_logic_online = nullptr; }
			if (GetCam(1)->g_pm_display_online) { delete GetCam(1)->g_pm_display_online; GetCam(1)->g_pm_display_online = nullptr; }
		}

	private: System::Windows::Forms::Button^ button1;
	private: System::Windows::Forms::Button^ button2;
	private: System::Windows::Forms::Timer^ timer1;
	private: System::Windows::Forms::PictureBox^ pictureBox1;
	private: BackgroundWorker^ processingWorker;
	private: Thread^ readerThread;
	private: System::ComponentModel::IContainer^ components;
	private: Bitmap^ currentFrame;
	private: Object^ bufferLock;
	private: bool isProcessing;
	private: System::Windows::Forms::Label^ lblCameraName;
	private: System::Windows::Forms::Button^ btnOnlineMode;
	private: System::Windows::Forms::Panel^ panel2;
private: System::Windows::Forms::Label^ lblViolation;
private: System::Windows::Forms::Label^ lblNormal;
private: System::Windows::Forms::Label^ lblEmpty;
private: System::Windows::Forms::Button^ btnLiveCamera;
private: System::Windows::Forms::Button^ btnLoadParkingTemplate;
private: System::Windows::Forms::CheckBox^ chkParkingMode;

private: System::Windows::Forms::Label^ lblLogs;
private: bool shouldStop;
private: long long lastProcessedSeq = -1;
private: long long lastDisplaySeq = -1;
private: System::Windows::Forms::SplitContainer^ splitContainer1;
private: Bitmap^ bmpBuffer1;
private: Bitmap^ bmpBuffer2;
private: bool useBuffer1;
private: System::Windows::Forms::Label^ label1;

	// *** [NEW] PARKING STATISTICS LABELS ***

	private: System::Windows::Forms::Label^ label5_online;
	private: System::Windows::Forms::Label^ label6_online;
	private: System::Windows::Forms::Label^ label7_online;

	// *** [NEW] VIOLATION ALERTS SYSTEM ***

	private: ref struct ViolationRecord_Online {
		int carId;
		Bitmap^ screenshot;
		Bitmap^ visualizationBitmap;
		System::String^ violationType;
		System::DateTime captureTime;
		int durationSeconds;
	};

	private: System::Collections::Generic::List<ViolationRecord_Online^>^ violationsList_online;
	private: System::Windows::Forms::Panel^ pnlViolationContainer_online;
	private: System::Windows::Forms::FlowLayoutPanel^ flpViolations_online;
	private: System::Windows::Forms::Label^ lblViolationTitle_online;
	private: System::Windows::Forms::Label^ lblViolationCount_online;
	private: System::Windows::Forms::Button^ btnClearViolations_online;
	private: System::Collections::Generic::Dictionary<int, System::DateTime>^ violatingCarTimers_online;

	// [NEW] Background mode controls
	private: System::Windows::Forms::Button^ btnRunInBackground;
	private: System::Windows::Forms::NotifyIcon^ notifyIcon;
	private: System::Windows::Forms::ContextMenuStrip^ trayMenu;
	private: System::Windows::Forms::ToolStripMenuItem^ menuShow;
	private: System::Windows::Forms::ToolStripMenuItem^ menuExit;
	private: System::Windows::Forms::Label^ lblNetworkStream;
	private: bool isBackgroundMode = false;

#pragma region Windows Form Designer generated code
		   void InitializeComponent(void)
		   {
			   this->components = (gcnew System::ComponentModel::Container());
			   this->timer1 = (gcnew System::Windows::Forms::Timer(this->components));
			   this->pictureBox1 = (gcnew System::Windows::Forms::PictureBox());
			   this->btnOnlineMode = (gcnew System::Windows::Forms::Button());
			   this->lblCameraName = (gcnew System::Windows::Forms::Label());
			   this->panel2 = (gcnew System::Windows::Forms::Panel());
			   this->lblLogs = (gcnew System::Windows::Forms::Label());
			   this->chkParkingMode = (gcnew System::Windows::Forms::CheckBox());
			   this->btnLoadParkingTemplate = (gcnew System::Windows::Forms::Button());
			   this->lblViolation = (gcnew System::Windows::Forms::Label());
			   this->lblNormal = (gcnew System::Windows::Forms::Label());
			   this->lblEmpty = (gcnew System::Windows::Forms::Label());
			   this->btnLiveCamera = (gcnew System::Windows::Forms::Button());
			   this->label5_online = (gcnew System::Windows::Forms::Label());
			   this->label6_online = (gcnew System::Windows::Forms::Label());
				this->label7_online = (gcnew System::Windows::Forms::Label());
			   this->pnlViolationContainer_online = (gcnew System::Windows::Forms::Panel());
			   this->flpViolations_online = (gcnew System::Windows::Forms::FlowLayoutPanel());
			   this->btnClearViolations_online = (gcnew System::Windows::Forms::Button());
			   this->lblViolationCount_online = (gcnew System::Windows::Forms::Label());
			   this->lblViolationTitle_online = (gcnew System::Windows::Forms::Label());
			   this->splitContainer1 = (gcnew System::Windows::Forms::SplitContainer());
			   this->label1 = (gcnew System::Windows::Forms::Label());
			   
			   // Background additions
			   this->btnRunInBackground = (gcnew System::Windows::Forms::Button());
			   this->notifyIcon = (gcnew System::Windows::Forms::NotifyIcon(this->components));
			   this->trayMenu = (gcnew System::Windows::Forms::ContextMenuStrip(this->components));
			   this->menuShow = (gcnew System::Windows::Forms::ToolStripMenuItem());
			   this->menuExit = (gcnew System::Windows::Forms::ToolStripMenuItem());
			   this->lblNetworkStream = (gcnew System::Windows::Forms::Label());

			   (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox1))->BeginInit();
			   (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->splitContainer1))->BeginInit();
			   this->splitContainer1->Panel1->SuspendLayout();
			   this->splitContainer1->Panel2->SuspendLayout();
			   this->splitContainer1->SuspendLayout();
			   this->SuspendLayout();
			   // 
			   // timer1
			   // 
			   this->timer1->Enabled = true;
			   this->timer1->Interval = 33; // [FIX] Changed from 15ms to 33ms (~30 FPS UI, matches camera FPS)
			   this->timer1->Tick += gcnew System::EventHandler(this, &UploadForm::timer1_Tick);
			   // 
			   // pictureBox1
			   // 
			   this->pictureBox1->BackColor = System::Drawing::Color::White;
			   this->pictureBox1->Dock = System::Windows::Forms::DockStyle::Fill;
			   this->pictureBox1->Location = System::Drawing::Point(30, 30);
			   this->pictureBox1->Name = L"pictureBox1";
			   this->pictureBox1->Size = System::Drawing::Size(949, 699);
			   this->pictureBox1->SizeMode = System::Windows::Forms::PictureBoxSizeMode::Zoom;
			   this->pictureBox1->TabIndex = 1;
			   this->pictureBox1->TabStop = false;
			   // 
			   // btnOnlineMode
			   // 
			   this->btnOnlineMode->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(40)), static_cast<System::Int32>(static_cast<System::Byte>(167)),
				   static_cast<System::Int32>(static_cast<System::Byte>(69)));
			   this->btnOnlineMode->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 12, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				   static_cast<System::Byte>(0)));
			   this->btnOnlineMode->ForeColor = System::Drawing::SystemColors::ButtonHighlight;
			   this->btnOnlineMode->Location = System::Drawing::Point(851, 46);
			   this->btnOnlineMode->Name = L"btnOnlineMode";
			   this->btnOnlineMode->Size = System::Drawing::Size(112, 46);
			   this->btnOnlineMode->TabIndex = 2;
			   this->btnOnlineMode->Text = L"Online";
			   this->btnOnlineMode->UseVisualStyleBackColor = false;
			   // 
			   // lblCameraName
			   // 
			   this->lblCameraName->AutoSize = true;
			   this->lblCameraName->BackColor = System::Drawing::Color::White;
			   this->lblCameraName->Font = (gcnew System::Drawing::Font(L"Segoe UI", 16.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				   static_cast<System::Byte>(0)));
			   this->lblCameraName->ForeColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(45)), static_cast<System::Int32>(static_cast<System::Byte>(45)),
				   static_cast<System::Int32>(static_cast<System::Byte>(48)));
			   this->lblCameraName->Location = System::Drawing::Point(76, 41);
			   this->lblCameraName->Name = L"lblCameraName";
			   this->lblCameraName->Size = System::Drawing::Size(102, 30);
			   this->lblCameraName->TabIndex = 6;
			   this->lblCameraName->Text = L"camera1";
			   // 
			   // panel2
			   // 
			   this->panel2->Location = System::Drawing::Point(0, 0);
			   this->panel2->Name = L"panel2";
			   this->panel2->Size = System::Drawing::Size(200, 100);
			   this->panel2->TabIndex = 0;
			   // 
			   // lblLogs
			   // 
			   this->lblLogs->AutoSize = true;
			   this->lblLogs->Font = (gcnew System::Drawing::Font(L"Segoe UI", 16.75F, System::Drawing::FontStyle::Bold));
			   this->lblLogs->Location = System::Drawing::Point(144, 30);
			   this->lblLogs->Name = L"lblLogs";
			   this->lblLogs->Size = System::Drawing::Size(163, 31);
			   this->lblLogs->TabIndex = 0;
			   this->lblLogs->Text = L"logs 25/12/67";
			   this->lblLogs->Click += gcnew System::EventHandler(this, &UploadForm::lblLogs_Click);
			   // 
			   // chkParkingMode
			   // 
			   this->chkParkingMode->AutoSize = true;
			   this->chkParkingMode->Location = System::Drawing::Point(17, 95);
			   this->chkParkingMode->Name = L"chkParkingMode";
			   this->chkParkingMode->Size = System::Drawing::Size(98, 17);
			   this->chkParkingMode->TabIndex = 8;
			   this->chkParkingMode->Text = L"Enable Parking";
			   this->chkParkingMode->CheckedChanged += gcnew System::EventHandler(this, &UploadForm::chkParkingMode_CheckedChanged);
			   // 
			   // btnLoadParkingTemplate
			   // 
			   this->btnLoadParkingTemplate->BackColor = System::Drawing::Color::LightGreen;
			   this->btnLoadParkingTemplate->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 11.25F, System::Drawing::FontStyle::Bold,
				   System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(0)));
			   this->btnLoadParkingTemplate->Location = System::Drawing::Point(223, 131);
			   this->btnLoadParkingTemplate->Name = L"btnLoadParkingTemplate";
			   this->btnLoadParkingTemplate->Size = System::Drawing::Size(141, 57);
			   this->btnLoadParkingTemplate->TabIndex = 7;
			   this->btnLoadParkingTemplate->Text = L"Load Template";
			   this->btnLoadParkingTemplate->UseVisualStyleBackColor = false;
			   this->btnLoadParkingTemplate->Click += gcnew System::EventHandler(this, &UploadForm::btnLoadParkingTemplate_Click);
			   // 
			   // lblViolation
			   // 
			   this->lblViolation->Location = System::Drawing::Point(0, 0);
			   this->lblViolation->Name = L"lblViolation";
			   this->lblViolation->Size = System::Drawing::Size(100, 23);
			   this->lblViolation->TabIndex = 0;
			   // 
			   // lblNormal
			   // 
			   this->lblNormal->Location = System::Drawing::Point(0, 0);
			   this->lblNormal->Name = L"lblNormal";
			   this->lblNormal->Size = System::Drawing::Size(100, 23);
			   this->lblNormal->TabIndex = 0;
			   // 
			   // lblEmpty
			   // 
			   this->lblEmpty->Location = System::Drawing::Point(0, 0);
			   this->lblEmpty->Name = L"lblEmpty";
			   this->lblEmpty->Size = System::Drawing::Size(100, 23);
			   this->lblEmpty->TabIndex = 0;
			   // 
			   // btnLiveCamera
			   // 
			   this->btnLiveCamera->BackColor = System::Drawing::Color::Tomato;
			   this->btnLiveCamera->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 11.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				   static_cast<System::Byte>(0)));
			   this->btnLiveCamera->ForeColor = System::Drawing::SystemColors::ButtonHighlight;
			   this->btnLiveCamera->Location = System::Drawing::Point(74, 131);
			   this->btnLiveCamera->Name = L"btnLiveCamera";
			   this->btnLiveCamera->Size = System::Drawing::Size(143, 57);
			   this->btnLiveCamera->TabIndex = 0;
			   this->btnLiveCamera->Text = L"📹 Live Camera";
			   this->btnLiveCamera->UseVisualStyleBackColor = false;
			   this->btnLiveCamera->Click += gcnew System::EventHandler(this, &UploadForm::btnLiveCamera_Click);
			   // 
			   // label5_online
			   // 
			   this->label5_online->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(76)), static_cast<System::Int32>(static_cast<System::Byte>(175)),
				   static_cast<System::Int32>(static_cast<System::Byte>(80)));
			   this->label5_online->Font = (gcnew System::Drawing::Font(L"Arial", 26, System::Drawing::FontStyle::Bold));
			   this->label5_online->ForeColor = System::Drawing::Color::White;
			   this->label5_online->Location = System::Drawing::Point(60, 218);
			   this->label5_online->Name = L"label5_online";
			   this->label5_online->Size = System::Drawing::Size(320, 50);
			   this->label5_online->TabIndex = 14;
			   this->label5_online->Text = L"Empty: 0";
			   this->label5_online->TextAlign = System::Drawing::ContentAlignment::MiddleCenter;
			   // 
			   // label6_online
			   // 
			   this->label6_online->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(255)), static_cast<System::Int32>(static_cast<System::Byte>(193)),
				   static_cast<System::Int32>(static_cast<System::Byte>(7)));
			   this->label6_online->Font = (gcnew System::Drawing::Font(L"Arial", 26, System::Drawing::FontStyle::Bold));
			   this->label6_online->ForeColor = System::Drawing::Color::White;
			   this->label6_online->Location = System::Drawing::Point(60, 286);
			   this->label6_online->Name = L"label6_online";
			   this->label6_online->Size = System::Drawing::Size(320, 50);
			   this->label6_online->TabIndex = 15;
			   this->label6_online->Text = L"Normal: 0";
			   this->label6_online->TextAlign = System::Drawing::ContentAlignment::MiddleCenter;
			   // 
			   // label7_online
			   // 
			   this->label7_online->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(244)), static_cast<System::Int32>(static_cast<System::Byte>(67)),
				   static_cast<System::Int32>(static_cast<System::Byte>(54)));
			   this->label7_online->Font = (gcnew System::Drawing::Font(L"Arial", 26, System::Drawing::FontStyle::Bold));
			   this->label7_online->ForeColor = System::Drawing::Color::White;
			   this->label7_online->Location = System::Drawing::Point(60, 355);
			   this->label7_online->Name = L"label7_online";
			   this->label7_online->Size = System::Drawing::Size(320, 50);
			   this->label7_online->TabIndex = 16;
			   this->label7_online->Text = L"Violation: 0";
			   this->label7_online->TextAlign = System::Drawing::ContentAlignment::MiddleCenter;
			   // 
			   // pnlViolationContainer_online
			   // 
			   this->pnlViolationContainer_online->BackColor = System::Drawing::Color::LightSteelBlue;
			   this->pnlViolationContainer_online->Controls->Add(this->flpViolations_online);
			   this->pnlViolationContainer_online->Controls->Add(this->btnClearViolations_online);
			   this->pnlViolationContainer_online->Controls->Add(this->lblViolationCount_online);
			   this->pnlViolationContainer_online->Controls->Add(this->lblViolationTitle_online);
			   this->pnlViolationContainer_online->Location = System::Drawing::Point(37, 456);
			   this->pnlViolationContainer_online->Name = L"pnlViolationContainer_online";
			   this->pnlViolationContainer_online->Size = System::Drawing::Size(352, 450);
			   this->pnlViolationContainer_online->TabIndex = 13;
			   // 
			   // flpViolations_online
			   // 
			   this->flpViolations_online->AutoScroll = true;
			   this->flpViolations_online->Location = System::Drawing::Point(30, 52);
			   this->flpViolations_online->Name = L"flpViolations_online";
			   this->flpViolations_online->Size = System::Drawing::Size(286, 385);
			   this->flpViolations_online->TabIndex = 3;
			   // 
			   // btnClearViolations_online
			   // 
			   this->btnClearViolations_online->BackColor = System::Drawing::Color::Tomato;
			   this->btnClearViolations_online->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
			   this->btnClearViolations_online->Font = (gcnew System::Drawing::Font(L"Segoe UI", 10));
			   this->btnClearViolations_online->Location = System::Drawing::Point(270, 5);
			   this->btnClearViolations_online->Name = L"btnClearViolations_online";
			   this->btnClearViolations_online->Size = System::Drawing::Size(75, 27);
			   this->btnClearViolations_online->TabIndex = 2;
			   this->btnClearViolations_online->Text = L"Clear All";
			   this->btnClearViolations_online->UseVisualStyleBackColor = false;
			   this->btnClearViolations_online->Click += gcnew System::EventHandler(this, &UploadForm::btnClearViolations_online_Click);
			   // 
			   // lblViolationCount_online
			   // 
			   this->lblViolationCount_online->AutoSize = true;
			   this->lblViolationCount_online->Font = (gcnew System::Drawing::Font(L"Segoe UI", 10));
			   this->lblViolationCount_online->Location = System::Drawing::Point(3, 30);
			   this->lblViolationCount_online->Name = L"lblViolationCount_online";
			   this->lblViolationCount_online->Size = System::Drawing::Size(84, 19);
			   this->lblViolationCount_online->TabIndex = 1;
			   this->lblViolationCount_online->Text = L"Violations: 0";
			   // 
			   // lblViolationTitle_online
			   // 
			   this->lblViolationTitle_online->AutoSize = true;
			   this->lblViolationTitle_online->Font = (gcnew System::Drawing::Font(L"Segoe UI", 12));
			   this->lblViolationTitle_online->Location = System::Drawing::Point(3, 5);
			   this->lblViolationTitle_online->Name = L"lblViolationTitle_online";
			   this->lblViolationTitle_online->Size = System::Drawing::Size(139, 21);
			   this->lblViolationTitle_online->TabIndex = 0;
			   this->lblViolationTitle_online->Text = L"Violation Alerts (0)";
			   // 
			   // splitContainer1
			   // 
			   this->splitContainer1->Dock = System::Windows::Forms::DockStyle::Fill;
			   this->splitContainer1->Location = System::Drawing::Point(0, 0);
			   this->splitContainer1->Name = L"splitContainer1";
			   // 
			   // splitContainer1.Panel1
			   // 
			   this->splitContainer1->Panel1->BackColor = System::Drawing::Color::LightSteelBlue;
			   this->splitContainer1->Panel1->Controls->Add(this->btnOnlineMode);
			   this->splitContainer1->Panel1->Controls->Add(this->pictureBox1);
			   this->splitContainer1->Panel1->Padding = System::Windows::Forms::Padding(30);
			   // 
			   // splitContainer1.Panel2
			   // 
			   this->splitContainer1->Panel2->BackColor = System::Drawing::Color::LightSteelBlue;
			   this->splitContainer1->Panel2->Controls->Add(this->btnRunInBackground);
			   this->splitContainer1->Panel2->Controls->Add(this->lblNetworkStream);
			   this->splitContainer1->Panel2->Controls->Add(this->pnlViolationContainer_online);
			   this->splitContainer1->Panel2->Controls->Add(this->label7_online);
			   this->splitContainer1->Panel2->Controls->Add(this->label6_online);
			   this->splitContainer1->Panel2->Controls->Add(this->label5_online);
			   this->splitContainer1->Panel2->Controls->Add(this->btnLiveCamera);
			   this->splitContainer1->Panel2->Controls->Add(this->btnLoadParkingTemplate);
			   this->splitContainer1->Panel2->Controls->Add(this->chkParkingMode);
			   this->splitContainer1->Panel2->Controls->Add(this->lblLogs);
			   this->splitContainer1->Panel2->Padding = System::Windows::Forms::Padding(14);
			   this->splitContainer1->Size = System::Drawing::Size(1443, 759);
			   this->splitContainer1->SplitterDistance = 1009;
			   this->splitContainer1->TabIndex = 5;
			   // 
			   // label1
			   // 
			   this->label1->AutoSize = true;
			   this->label1->BackColor = System::Drawing::Color::White;
			   this->label1->Font = (gcnew System::Drawing::Font(L"Segoe UI", 16.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				   static_cast<System::Byte>(0)));
			   this->label1->ForeColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(45)), static_cast<System::Int32>(static_cast<System::Byte>(45)),
				   static_cast<System::Int32>(static_cast<System::Byte>(48)));
			   this->label1->Location = System::Drawing::Point(76, 41);
			   this->label1->Name = L"label1";
			   this->label1->Size = System::Drawing::Size(102, 30);
			   this->label1->TabIndex = 6;
			   this->label1->Text = L"camera1";
			   // 
			   // btnRunInBackground
			   // 
			   this->btnRunInBackground->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(155)), static_cast<System::Int32>(static_cast<System::Byte>(89)), static_cast<System::Int32>(static_cast<System::Byte>(182)));
			   this->btnRunInBackground->FlatAppearance->BorderSize = 0;
			   this->btnRunInBackground->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
			   this->btnRunInBackground->Font = (gcnew System::Drawing::Font(L"Segoe UI", 10, System::Drawing::FontStyle::Bold));
			   this->btnRunInBackground->ForeColor = System::Drawing::Color::White;
			   this->btnRunInBackground->Location = System::Drawing::Point(37, 410); // [FIX] Changed Y from 910
			   this->btnRunInBackground->Name = L"btnRunInBackground";
			   this->btnRunInBackground->Size = System::Drawing::Size(352, 35);
			   this->btnRunInBackground->TabIndex = 11;
			   this->btnRunInBackground->Text = L"⬇️ Run in Background";
			   this->btnRunInBackground->UseVisualStyleBackColor = false;
			   this->btnRunInBackground->Click += gcnew System::EventHandler(this, &UploadForm::btnRunInBackground_Click);
			   this->btnRunInBackground->Enabled = false;
			   // 
			   // trayMenu
			   // 
			   this->trayMenu->Items->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(2) {
				   this->menuShow,
					   this->menuExit
			   });
			   this->trayMenu->Name = L"trayMenu";
			   this->trayMenu->Size = System::Drawing::Size(104, 48);
			   // 
			   // menuShow
			   // 
			   this->menuShow->Name = L"menuShow";
			   this->menuShow->Size = System::Drawing::Size(103, 22);
			   this->menuShow->Text = L"Show";
			   this->menuShow->Click += gcnew System::EventHandler(this, &UploadForm::menuShow_Click);
			   // 
			   // menuExit
			   // 
			   this->menuExit->Name = L"menuExit";
			   this->menuExit->Size = System::Drawing::Size(103, 22);
			   this->menuExit->Text = L"Exit";
			   this->menuExit->Click += gcnew System::EventHandler(this, &UploadForm::menuExit_Click);
			   // 
			   // notifyIcon
			   // 
			   this->notifyIcon->ContextMenuStrip = this->trayMenu;
			   this->notifyIcon->Text = L"Smart Parking Monitoring";
			   this->notifyIcon->Visible = false;
			   this->notifyIcon->DoubleClick += gcnew System::EventHandler(this, &UploadForm::notifyIcon_DoubleClick);
			   // 
			   // lblNetworkStream
			   // 
			   this->lblNetworkStream->AutoSize = true;
			   this->lblNetworkStream->Font = (gcnew System::Drawing::Font(L"Segoe UI", 10, System::Drawing::FontStyle::Bold));
			   this->lblNetworkStream->ForeColor = System::Drawing::Color::DarkGreen;
			   this->lblNetworkStream->Location = System::Drawing::Point(100, 100); // [FIX] Position it reasonably under the other buttons
			   this->lblNetworkStream->Name = L"lblNetworkStream";
			   this->lblNetworkStream->Size = System::Drawing::Size(120, 19);
			   this->lblNetworkStream->TabIndex = 1;
			   this->lblNetworkStream->Text = L"Stream: Offline";
			   // 
			   // UploadForm
			   // 
			   this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			   this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			   this->ClientSize = System::Drawing::Size(1443, 759);
			   this->Controls->Add(this->splitContainer1);
			   this->Name = L"UploadForm";
			   this->Text = L"Online Mode - Loading Model...";
			   this->FormClosing += gcnew System::Windows::Forms::FormClosingEventHandler(this, &UploadForm::UploadForm_FormClosing);
			   (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox1))->EndInit();
			   this->pnlViolationContainer_online->ResumeLayout(false);
			   this->pnlViolationContainer_online->PerformLayout();
			   this->splitContainer1->Panel1->ResumeLayout(false);
			   this->splitContainer1->Panel1->PerformLayout();
			   this->splitContainer1->Panel2->ResumeLayout(false);
			   this->splitContainer1->Panel2->PerformLayout();
			   (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->splitContainer1))->EndInit();
			   this->splitContainer1->ResumeLayout(false);
			   this->ResumeLayout(false);

		   }
#pragma endregion

	private: Bitmap^ MatToBitmap(cv::Mat& mat) {
		if (mat.empty() || mat.type() != CV_8UC3) return nullptr;
		try {
			int w = mat.cols; int h = mat.rows;
			Bitmap^ bmp = gcnew Bitmap(w, h, System::Drawing::Imaging::PixelFormat::Format24bppRgb);
			System::Drawing::Rectangle rect = System::Drawing::Rectangle(0, 0, w, h);
			System::Drawing::Imaging::BitmapData^ bmpData = bmp->LockBits(rect, System::Drawing::Imaging::ImageLockMode::WriteOnly, bmp->PixelFormat);
			for (int y = 0; y < h; y++) {
				memcpy((unsigned char*)bmpData->Scan0.ToPointer() + y * bmpData->Stride, mat.data + y * mat.step, w * 3);
			}
			bmp->UnlockBits(bmpData);
			return bmp;
		}
		catch (...) { return nullptr; }
	}

	private: void UpdatePictureBox(cv::Mat& mat) {
		if (mat.empty() || mat.type() != CV_8UC3) return;
		int w = mat.cols;
		int h = mat.rows;

		Bitmap^ targetBmp = useBuffer1 ? bmpBuffer1 : bmpBuffer2;

		// [PHASE 1 FIX] Don't manually delete managed Bitmap - let GC handle it
		if (targetBmp == nullptr || targetBmp->Width != w || targetBmp->Height != h) {
			targetBmp = gcnew Bitmap(w, h, System::Drawing::Imaging::PixelFormat::Format24bppRgb);
			if (useBuffer1) bmpBuffer1 = targetBmp; 
			else bmpBuffer2 = targetBmp;
		}

		System::Drawing::Rectangle rect = System::Drawing::Rectangle(0, 0, w, h);
		System::Drawing::Imaging::BitmapData^ bmpData = targetBmp->LockBits(rect, System::Drawing::Imaging::ImageLockMode::WriteOnly, targetBmp->PixelFormat);
		
		// [PHASE 2] Optimized memcpy - single copy when possible
		if (bmpData->Stride == mat.step) {
			memcpy((unsigned char*)bmpData->Scan0.ToPointer(), mat.data, (size_t)h * mat.step);
		} else {
			for (int y = 0; y < h; y++) {
				memcpy((unsigned char*)bmpData->Scan0.ToPointer() + y * bmpData->Stride, mat.data + y * mat.step, w * 3);
			}
		}
		
		targetBmp->UnlockBits(bmpData);

		pictureBox1->Image = targetBmp;
		useBuffer1 = !useBuffer1;
	}

	// *** [OPTIMIZED] UI TIMER - หยิบภาพมาโชว์อย่างเดียว (0.1ms) ***
	private: System::Void timer1_Tick(System::Object^ sender, System::EventArgs^ e) {
		try {
			cv::Mat finalFrame;
			long long seq = 0;
			GetCam()->GetProcessedFrameOnline(finalFrame, seq);

			if (seq == lastDisplaySeq) return;
			lastDisplaySeq = seq;

			if (!finalFrame.empty()) {
				// Only update picture box when NOT in background mode — saves CPU/GDI
				if (!isBackgroundMode) {
					UpdatePictureBox(finalFrame);
				}
				// Violation checks always run — needed for logging even in background mode
				bool parkingEnabled = GetCam()->g_parkingEnabled_online.load();
				if (parkingEnabled) {
					CheckViolations_Online(g_activeCameraId, finalFrame);
				}
			}

			// *** [NEW] UPDATE LOGS WITH CURRENT DATETIME ***
			System::DateTime now = System::DateTime::Now;
			System::String^ dateTimeStr = now.ToString(L"dd/MM/yy");
			lblLogs->Text = dateTimeStr;

			// *** [NEW] UPDATE PARKING STATISTICS LABELS ***
			OnlineAppState state;
			{
				std::lock_guard<std::mutex> lock(GetCam()->g_onlineStateMutex);
				state = GetCam()->g_onlineState;
			}

			bool parkingEnabledForStats = GetCam()->g_parkingEnabled_online.load();
			if (parkingEnabledForStats && !state.slotStatuses.empty()) {
				int emptyCount = 0;
				int occupiedCount = 0;
				int carEmpty = 0, carNormal = 0;
				int motoEmpty = 0, motoNormal = 0;

				for (const auto& slotEntry : state.slotStatuses) {
					int slotId = slotEntry.first;
					SlotStatus status = slotEntry.second;
					
					// Get the slot type
					std::string type = "Car"; // Default
					if (state.slotTypes.find(slotId) != state.slotTypes.end()) {
						type = state.slotTypes[slotId];
					}

					if (status == SlotStatus::EMPTY) {
						emptyCount++;
						if (type == "Motorcycle") motoEmpty++;
						else carEmpty++;
					}
					else {
						occupiedCount++;
						if (type == "Motorcycle") motoNormal++;
						else carNormal++;
					}
				}

				int violationCount = (int)state.violatingCarIds.size();

				label5_online->Text = System::String::Format(L"Empty: {0}", emptyCount);
				label6_online->Text = System::String::Format(L"Normal: {0}", occupiedCount);
				label7_online->Text = System::String::Format(L"Violation: {0}", violationCount);

				if (GetCam()->g_mjpegServer_online) {
					std::string json = "{\"empty\":" + std::to_string(emptyCount) + 
					                   ",\"normal\":" + std::to_string(occupiedCount) + 
					                   ",\"carEmpty\":" + std::to_string(carEmpty) + 
					                   ",\"carNormal\":" + std::to_string(carNormal) + 
					                   ",\"motoEmpty\":" + std::to_string(motoEmpty) + 
					                   ",\"motoNormal\":" + std::to_string(motoNormal) + 
					                   ",\"violation\":" + std::to_string(violationCount) + ",\"logs\":[";
					bool first = true;
					for each (ViolationRecord_Online^ rec in violationsList_online) {
						if (!first) json += ",";
						System::String^ ts = rec->captureTime.ToString("HH:mm:ss");
						System::String^ vt = rec->violationType;
						std::string timeStr = msclr::interop::marshal_as<std::string>(ts);
						std::string typeStr = msclr::interop::marshal_as<std::string>(vt);
						json += "{\"id\":" + std::to_string(rec->carId) + ",\"time\":\"" + timeStr + "\",\"type\":\"" + typeStr + "\"}";
						first = false;
					}
					json += "]}";
					GetCam(1)->g_mjpegServer_online->SetStats(1, json);
				}
			}
		}
		catch (...) {}
	}
	// Called from ProcessingLoopHeadless to keep the web dashboard fed with parking stats
	// This is needed because timer1_Tick (WinForms UI timer) may not fire in headless mode.
	private: void UpdateWebStats(int cameraId) {
		if (!GetCam(cameraId)->g_mjpegServer_online) return;
		OnlineAppState state;
		{
			std::lock_guard<std::mutex> lock(GetCam(cameraId)->g_onlineStateMutex);
			state = GetCam(cameraId)->g_onlineState;
		}
		if (state.slotStatuses.empty()) return;

		int emptyCount = 0, occupiedCount = 0, carEmpty = 0, carNormal = 0, motoEmpty = 0, motoNormal = 0;
		for (const auto& slotEntry : state.slotStatuses) {
			int slotId = slotEntry.first;
			std::string type = "Car";
			if (state.slotTypes.find(slotId) != state.slotTypes.end())
				type = state.slotTypes.at(slotId);
			if (slotEntry.second == SlotStatus::EMPTY) {
				emptyCount++;
				if (type == "Motorcycle") motoEmpty++; else carEmpty++;
			} else {
				occupiedCount++;
				if (type == "Motorcycle") motoNormal++; else carNormal++;
			}
		}
		int violationCount = (int)state.violatingCarIds.size();

		std::string json = "{\"empty\":" + std::to_string(emptyCount) +
		                   ",\"normal\":" + std::to_string(occupiedCount) +
		                   ",\"carEmpty\":" + std::to_string(carEmpty) +
		                   ",\"carNormal\":" + std::to_string(carNormal) +
		                   ",\"motoEmpty\":" + std::to_string(motoEmpty) +
		                   ",\"motoNormal\":" + std::to_string(motoNormal) +
		                   ",\"violation\":" + std::to_string(violationCount) + ",\"logs\":[";
		bool first = true;
		int logCount = 0;
		// Send up to the 5 most recent logs
		for (int i = violationsList_online->Count - 1; i >= 0; i--) {
			if (logCount >= 5) break;
			ViolationRecord_Online^ rec = violationsList_online[i];
			if (!first) json += ",";
			System::String^ ts = rec->captureTime.ToString("HH:mm:ss");
			System::String^ vt = rec->violationType;
			std::string timeStr = msclr::interop::marshal_as<std::string>(ts);
			std::string typeStr = msclr::interop::marshal_as<std::string>(vt);
			json += "{\"id\":" + std::to_string(rec->carId) + ",\"time\":\"" + timeStr + "\",\"type\":\"" + typeStr + "\"}";
			first = false;
			logCount++;
		}
		json += "]}";

		static std::map<int, std::string> last_sent_jsons;
		if (json != last_sent_jsons[cameraId]) {
			last_sent_jsons[cameraId] = json;
			GetCam(cameraId)->g_mjpegServer_online->SetStats(cameraId, json);
		}
	}

	// Called by web API disconnect — stops camera threads but keeps the web server running
	public: void StopProcessingPublic(int cameraId) {
		GetCam(cameraId)->shouldStop = true;
		GetCam(cameraId)->isProcessing = false;
		
		// NOTE: Do NOT delete GetCam(cameraId)->g_mjpegServer_online here — it is owned by main.cpp global web server
		
		// Only stop AI loop and recording, avoid stopping Web Server
		if (GetCam(cameraId)->readerThread_online) {
			if (GetCam(cameraId)->readerThread_online->joinable()) GetCam(cameraId)->readerThread_online->join();
			delete GetCam(cameraId)->readerThread_online;
			GetCam(cameraId)->readerThread_online = nullptr;
		}
		if (GetCam(cameraId)->processingThread_online) {
			if (GetCam(cameraId)->processingThread_online->joinable()) GetCam(cameraId)->processingThread_online->join();
			delete GetCam(cameraId)->processingThread_online;
			GetCam(cameraId)->processingThread_online = nullptr;
		}

		GetCam(cameraId)->StopVideoRecordingThread_Online();
		DumpLog("[DISCONNECT] Camera threads stopped. Web server and timer preserved. (ID: " + std::to_string(cameraId) + ")");
	}
	
	private: void StopProcessing(int cameraId) {
		timer1->Stop();
		GetCam(cameraId)->StopProcessing();

		if (btnRunInBackground) btnRunInBackground->Enabled = false;
		if (lblNetworkStream) lblNetworkStream->Text = "Stream: Offline";
	}

	private: System::Void LoadModel_DoWork(System::Object^ sender, DoWorkEventArgs^ e) {
		try {
			std::string modelPath = "models/test/yolo26s.onnx";
			std::vector<CameraConfig> confs = CameraManager::LoadCameras();
			for (auto& c : confs) {
				GetCam(c.id)->InitGlobalModel(modelPath);
			}
			// If no cameras are configured yet, at least initialize the default one
			if (confs.empty()) {
				GetCam(1)->InitGlobalModel(modelPath);
			}
			e->Result = true;
		}
		catch (const std::exception& ex) { e->Result = gcnew System::String(ex.what()); }
	}

	private: System::Void LoadModel_Completed(System::Object^ sender, RunWorkerCompletedEventArgs^ e) {
		if (e->Result != nullptr && e->Result->GetType() == bool::typeid && safe_cast<bool>(e->Result)) {
			this->Text = L"Online Mode - YOLO Detection (Ready)";
			// [UI FIX] Only enable template button, Live Camera stays disabled
			btnLoadParkingTemplate->Enabled = true;
			MessageBox::Show("Model loaded!\n\nNote: Please load a parking template before starting live camera.", "Success", MessageBoxButtons::OK, MessageBoxIcon::Information);
		}
		else {
			MessageBox::Show("Error loading model", "Error", MessageBoxButtons::OK, MessageBoxIcon::Error);
		}
	}

	// ===================== Managed: get local IP =====================
	private: std::string GetLocalIP() {
		System::String^ bestIP = "0.0.0.0";
		try {
			cli::array<System::Net::NetworkInformation::NetworkInterface^>^ interfaces = System::Net::NetworkInformation::NetworkInterface::GetAllNetworkInterfaces();
			for each (System::Net::NetworkInformation::NetworkInterface^ adapter in interfaces) {
				if (adapter->OperationalStatus == System::Net::NetworkInformation::OperationalStatus::Up) {
					System::String^ desc = adapter->Description->ToLower();
					// Skip VPNs, VMware, VirtualBox, Radmin
					if (desc->Contains("virtual") || desc->Contains("vpn") || desc->Contains("vmware") || desc->Contains("radmin") || desc->Contains("hamachi")) continue;
					
					System::Net::NetworkInformation::IPInterfaceProperties^ properties = adapter->GetIPProperties();
					for each (System::Net::NetworkInformation::UnicastIPAddressInformation^ ip in properties->UnicastAddresses) {
						if (ip->Address->AddressFamily == System::Net::Sockets::AddressFamily::InterNetwork) {
							System::String^ ipStr = ip->Address->ToString();
							if (ipStr->StartsWith("26.") || ipStr->StartsWith("169.254.")) continue;
							bestIP = ipStr;
							// Prefer physical connections that look like classic local IPs
							if (ipStr->StartsWith("192.168.") || ipStr->StartsWith("172.") || ipStr->StartsWith("10.")) {
								return msclr::interop::marshal_as<std::string>(bestIP);
							}
						}
					}
				}
			}
		} catch (...) {}
		return msclr::interop::marshal_as<std::string>(bestIP);
	}

	private: System::Void btnLiveCamera_Click(System::Object^ sender, System::EventArgs^ e) {
	// [UI FIX] Check if template is loaded before proceeding
	if (!GetCam()->g_parkingEnabled_online.load() || !GetCam()->g_pm_logic_online || !GetCam()->g_pm_display_online) {
		MessageBox::Show(
			"[!] Parking template not loaded!\n\n" +
			"Please click 'Load Template' button first before starting live camera.\n\n" +
			"Steps:\n" +
			"1. Click 'Load Template' button\n" +
			"2. Select a parking template (.xml file)\n" +
			"3. Then start Live Camera",
			"Template Required",
			MessageBoxButtons::OK,
			MessageBoxIcon::Warning
		);
		return;
	}

	StopProcessingPublic(1);
	
	Form^ ipForm = gcnew Form();
	ipForm->Text = L"Connect to Mobile Camera";
	ipForm->Size = System::Drawing::Size(450, 320);
	ipForm->StartPosition = FormStartPosition::CenterParent;
	ipForm->FormBorderStyle = System::Windows::Forms::FormBorderStyle::FixedDialog;
	ipForm->MaximizeBox = false;
	ipForm->MinimizeBox = false;

	Label^ labelTitle = gcnew Label();
	labelTitle->Text = L"Enter Mobile Phone IP Address and Port";
	labelTitle->Location = System::Drawing::Point(20, 20);
	labelTitle->Size = System::Drawing::Size(400, 25);
	labelTitle->Font = gcnew System::Drawing::Font(L"Segoe UI", 10, FontStyle::Bold);

	Label^ labelIP = gcnew Label();
	labelIP->Text = L"IP Address:";
labelIP->Location = System::Drawing::Point(20, 60);
labelIP->Size = System::Drawing::Size(100, 20);

TextBox^ textBoxIP = gcnew TextBox();
textBoxIP->Location = System::Drawing::Point(120, 58);
textBoxIP->Size = System::Drawing::Size(290, 25);
textBoxIP->Text = L"192.168.1.100";

Label^ labelPort = gcnew Label();
labelPort->Text = L"Port:";
labelPort->Location = System::Drawing::Point(20, 95);
labelPort->Size = System::Drawing::Size(100, 20);

TextBox^ textBoxPort = gcnew TextBox();
textBoxPort->Location = System::Drawing::Point(120, 93);
textBoxPort->Size = System::Drawing::Size(290, 25);
textBoxPort->Text = L"8080";

Label^ labelPath = gcnew Label();
labelPath->Text = L"Path:";
labelPath->Location = System::Drawing::Point(20, 130);
labelPath->Size = System::Drawing::Size(100, 20);

TextBox^ textBoxPath = gcnew TextBox();
textBoxPath->Location = System::Drawing::Point(120, 128);
textBoxPath->Size = System::Drawing::Size(290, 25);
textBoxPath->Text = L"/video";

Label^ labelExample = gcnew Label();
labelExample->Text = L"Example apps: IP Webcam, DroidCam, or iVCam\nMake sure both devices are on the same WiFi network";
labelExample->Location = System::Drawing::Point(20, 165);
labelExample->Size = System::Drawing::Size(400, 35);
labelExample->Font = gcnew System::Drawing::Font(L"Segoe UI", 8, FontStyle::Italic);
labelExample->ForeColor = System::Drawing::Color::Gray;

Button^ btnConnect = gcnew Button();
btnConnect->Text = L"Connect";
btnConnect->Location = System::Drawing::Point(120, 215);
btnConnect->Size = System::Drawing::Size(100, 35);
btnConnect->BackColor = System::Drawing::Color::FromArgb(40, 167, 69);
btnConnect->ForeColor = System::Drawing::Color::White;
btnConnect->FlatStyle = FlatStyle::Flat;
btnConnect->DialogResult = System::Windows::Forms::DialogResult::OK;

Button^ btnCancel = gcnew Button();
btnCancel->Text = L"Cancel";
btnCancel->Location = System::Drawing::Point(230, 215);
btnCancel->Size = System::Drawing::Size(100, 35);
btnCancel->BackColor = System::Drawing::Color::FromArgb(220, 53, 69);
btnCancel->ForeColor = System::Drawing::Color::White;
btnCancel->FlatStyle = FlatStyle::Flat;
btnCancel->DialogResult = System::Windows::Forms::DialogResult::Cancel;

	ipForm->Controls->Add(labelTitle);
	ipForm->Controls->Add(labelIP);
	ipForm->Controls->Add(textBoxIP);
	ipForm->Controls->Add(labelPort);
	ipForm->Controls->Add(textBoxPort);
	ipForm->Controls->Add(labelPath);
	ipForm->Controls->Add(textBoxPath);
	ipForm->Controls->Add(labelExample);
	ipForm->Controls->Add(btnConnect);
	ipForm->Controls->Add(btnCancel);
	ipForm->AcceptButton = btnConnect;
	ipForm->CancelButton = btnCancel;

	if (ipForm->ShowDialog() == System::Windows::Forms::DialogResult::OK) {
		String^ ip = textBoxIP->Text->Trim();
		String^ port = textBoxPort->Text->Trim();
		String^ path = textBoxPath->Text->Trim();

		if (String::IsNullOrEmpty(ip) || String::IsNullOrEmpty(port)) {
			MessageBox::Show(
				"Please enter both IP Address and Port",
				"Input Required",
				MessageBoxButtons::OK,
				MessageBoxIcon::Warning
			);
			return;
		}

		if (!path->StartsWith("/")) {
			path = "/" + path;
		}

		try {
			array<String^>^ urlFormats = gcnew array<String^> {
				String::Format("http://{0}:{1}{2}", ip, port, path),
				String::Format("http://{0}:{1}/videofeed", ip, port),
				String::Format("http://{0}:{1}/video", ip, port),
				String::Format("rtsp://{0}:{1}", ip, port)
			};

			bool connected = false;
			String^ successUrl = "";
			int attemptNum = 0;

			for each (String^ streamUrl in urlFormats) {
				attemptNum++;
				this->Text = String::Format("Connecting... ({0}/4) - {1}", attemptNum, streamUrl);
				Application::DoEvents();
				
				std::string url = msclr::interop::marshal_as<std::string>(streamUrl);
				
				OutputDebugStringA(("[INFO] Trying to connect: " + url + "\n").c_str());
				
				GetCam()->OpenGlobalCameraFromIP(url);
				
				// [FIX] Reduced wait time from 1000ms to 500ms (timeout is already 5s in GetCam()->OpenGlobalCameraFromIP)
				Threading::Thread::Sleep(500);

				if (GetCam()->g_cap && GetCam()->g_cap->isOpened()) {
					cv::Mat testFrame;
					bool canRead = false;
					{
						std::lock_guard<std::mutex> lock(GetCam()->g_frameMutex);
						if (GetCam()->g_cap->read(testFrame)) {
							canRead = !testFrame.empty();
						}
					}

					if (canRead) {
						connected = true;
						successUrl = streamUrl;
						OutputDebugStringA("[SUCCESS] Connected successfully!\n");
						break;
					}
				}
				
				{
					std::lock_guard<std::mutex> lock(GetCam()->g_frameMutex);
					if (GetCam()->g_cap) {
						delete GetCam()->g_cap;
						GetCam()->g_cap = nullptr;
					}
				}
			}

			if (connected) {
				GetCam()->StartProcessing();
				this->Text = L"Online Mode - Live Camera Connected";
				MessageBox::Show(
					"Successfully connected to mobile camera!\n\n" +
					"Stream URL: " + successUrl + "\n\n" +
					"Press OK to start detection.",
					"Connection Successful",
					MessageBoxButtons::OK,
					MessageBoxIcon::Information
				);
			}
			else {
				// [FIX] Clean up state after failed connection to allow retry
				GetCam(1)->StopProcessing();
				
				// Reset global camera state
				{
					std::lock_guard<std::mutex> lock(GetCam()->g_frameMutex);
					if (GetCam()->g_cap) {
						delete GetCam()->g_cap;
						GetCam()->g_cap = nullptr;
					}
				}
				
				this->Text = L"Online Mode - Connection Failed (Ready to Retry)";
				
				String^ errorMsg = "[X] Failed to connect to mobile camera!\n\n";
				errorMsg += "Troubleshooting Steps:\n";
				errorMsg += "1. Check IP Address: " + ip + " (is it correct?)\n";
				errorMsg += "2. Verify Port: " + port + "\n";
				errorMsg += "3. Check if camera app is running on mobile\n";
				errorMsg += "4. Ensure both devices are on the same WiFi network\n";
				errorMsg += "5. Check firewall settings\n";
				errorMsg += "6. Try disabling antivirus temporarily\n\n";
				errorMsg += "Attempted URLs:\n";
				for each (String^ url in urlFormats) {
					errorMsg += "  - " + url + "\n";
				}
				errorMsg += "\n[OK] You can try connecting again with a different IP/Port.";
				
				MessageBox::Show(
					errorMsg,
					"Connection Error",
					MessageBoxButtons::OK,
					MessageBoxIcon::Error
				);
				
				// Re-enable button for retry
				if (btnLiveCamera) btnLiveCamera->Enabled = true;
			}
		}
		catch (Exception^ ex) {
			// [FIX] Clean up state after exception to allow retry
			GetCam(1)->StopProcessing();
			
			// Reset global camera state
			{
				std::lock_guard<std::mutex> lock(GetCam()->g_frameMutex);
				if (GetCam()->g_cap) {
					delete GetCam()->g_cap;
					GetCam()->g_cap = nullptr;
				}
			}
			
			this->Text = L"Online Mode - Error Occurred (Ready to Retry)";
			
			MessageBox::Show(
				"[X] An error occurred while connecting:\n\n" + 
				ex->Message + "\n\n" +
				"Stack Trace:\n" + ex->StackTrace + "\n\n" +
				"[OK] You can try connecting again.",
				"Exception Error",
				MessageBoxButtons::OK,
				MessageBoxIcon::Error
			);
			
			// Re-enable button for retry
			if (btnLiveCamera) btnLiveCamera->Enabled = true;
		}
	}
}

private: System::Void btnLoadParkingTemplate_Click(System::Object^ sender, System::EventArgs^ e) {
	OpenFileDialog^ ofd = gcnew OpenFileDialog();
	ofd->Filter = "Parking Template|*.xml";
	
	char buffer[MAX_PATH];
	_getcwd(buffer, MAX_PATH);
	std::string currentDir(buffer);
	std::string folder = currentDir + "\\parking_templates";
	
	ofd->InitialDirectory = gcnew String(folder.c_str());
	ofd->Title = "Load Parking Template";
	
	if (ofd->ShowDialog() == System::Windows::Forms::DialogResult::OK) {
		std::string fileName = msclr::interop::marshal_as<std::string>(ofd->FileName);
		bool anyLoaded = false;
		
		std::vector<CameraConfig> confs = CameraManager::LoadCameras();
		for (auto& c : confs) {
			if (GetCam(c.id)->LoadParkingTemplate_Online(fileName)) {
				anyLoaded = true;
			}
		}
		// Fallback for single camera without config
		if (confs.empty()) {
			anyLoaded = GetCam(1)->LoadParkingTemplate_Online(fileName);
		}

		if (anyLoaded) {
			chkParkingMode->Checked = true;
			
			// [UI FIX] Change button color to indicate template is loaded
			btnLoadParkingTemplate->BackColor = System::Drawing::Color::DarkGreen;
			btnLoadParkingTemplate->ForeColor = System::Drawing::Color::White;
			btnLoadParkingTemplate->Text = L"✓ Template Loaded";
			
			// [UI FIX] Enable Live Camera button after successful template load
			btnLiveCamera->Enabled = true;
			btnLiveCamera->BackColor = System::Drawing::Color::Tomato;
			
			MessageBox::Show(
				"[OK] Template loaded successfully!\n\n" +
				"Parking slot detection is now active.\n" +
				"Violations (cars parked outside slots) will be marked in RED.\n\n" +
				"You can now start Live Camera.", 
				"Success", 
				MessageBoxButtons::OK, 
				MessageBoxIcon::Information
			);
		}
		else {
			MessageBox::Show("Failed to load template!", "Error", MessageBoxButtons::OK, MessageBoxIcon::Error);
		}
	}
}

private: System::Void chkParkingMode_CheckedChanged(System::Object^ sender, System::EventArgs^ e) {
	GetCam()->g_parkingEnabled_online.store(chkParkingMode->Checked); // [PHASE 1 FIX] Use atomic store
	if (chkParkingMode->Checked) {
		label1->Text = L"Parking Mode ON";
		label1->BackColor = System::Drawing::Color::LightGreen;
	}
	else {
		label1->Text = L"Camera 1";
		label1->BackColor = System::Drawing::Color::Yellow;
	}
}

private: System::Void UploadForm_FormClosing(System::Object^ sender, FormClosingEventArgs^ e) {
	StopProcessing(1);
}

	// ==========================================================
	// [PHASE 3] JSON REGISTRY & MONGOENGINE EXPORT (1 FILE PER EVENT/CHANGE)
	// ==========================================================
	private: void SaveAnomalyEventJson(int cameraId, int carId, System::String^ violationType, long long epochMs, std::string snapshotPath) {
		SYSTEMTIME st;
		GetLocalTime(&st);
		char dateFolder[256];
		sprintf_s(dateFolder, sizeof(dateFolder), "C:\\loc_json\\anomaly_events\\%04d%02d%02d\\camera_%d", st.wYear, st.wMonth, st.wDay, cameraId);
		CreateDirectoryA("C:\\loc_json", NULL);
		CreateDirectoryA("C:\\loc_json\\anomaly_events", NULL);
		
		char parentFolder[128];
		sprintf_s(parentFolder, sizeof(parentFolder), "C:\\loc_json\\anomaly_events\\%04d%02d%02d", st.wYear, st.wMonth, st.wDay);
		CreateDirectoryA(parentFolder, NULL);

		CreateDirectoryA(dateFolder, NULL);

		// [1 File Per Event]
		std::string jsonPath = std::string(dateFolder) + "\\event_" + std::to_string(epochMs) + "_car_" + std::to_string(carId) + ".json";
		
		// Calculate seconds into the clip for media_seek_time_seconds based on exact frame count (10 FPS)
		int seekSeconds = 0;
		{
			std::lock_guard<std::mutex> vl(GetCam(cameraId)->g_videoWriterMutex_online);
			seekSeconds = GetCam(cameraId)->g_videoFramesWritten / 10;
		}
		if (seekSeconds < 0) seekSeconds = 0;

		// Format timestamp as ISO-8601 string for MongoEngine DateTimeField
		char timeBuf[64];
		sprintf_s(timeBuf, sizeof(timeBuf), "%04d-%02d-%02dT%02d:%02d:%02d", 
                  st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond);

		// Dynamic Camera ID string representation
		std::string camId = "camera_" + std::to_string(cameraId);

		// Replace backslashes with forward slashes for cross-platform JSON API usage
		std::replace(snapshotPath.begin(), snapshotPath.end(), '\\', '/');
		std::string videoPath = "C:/locvideo/" + GetCam(cameraId)->g_currentVideoRelPath;

		json newEvent = {
			{"camera_id", camId},
			{"timestamp", std::string(timeBuf)},
			{"event_type", msclr::interop::marshal_as<std::string>(violationType)},
			{"confidence", 0.95},
			{"media_snapshot_url", snapshotPath},
			{"media_video_url", videoPath},
			{"media_seek_time_seconds", seekSeconds},
			{"is_reviewed", false}
		};

		std::ofstream outFile(jsonPath);
		if (outFile.is_open()) {
			outFile << newEvent.dump(4);
			outFile.close();
		}
	}

	private: void SaveParkingAreaJson(int cameraId) {
		OnlineAppState state;
		{
			std::lock_guard<std::mutex> lock(GetCam(cameraId)->g_onlineStateMutex);
			state = GetCam(cameraId)->g_onlineState;
		}

		int carEmptyCount = 0;
		int carOccupiedCount = 0;
		int motoEmptyCount = 0;
		int motoOccupiedCount = 0;

		for (const auto& slotEntry : state.slotStatuses) {
			int slotId = slotEntry.first;
			std::string type = "Car";
			if (state.slotTypes.find(slotId) != state.slotTypes.end()) {
				type = state.slotTypes[slotId];
			}

			if (type == "Motorcycle") {
				if (slotEntry.second == SlotStatus::EMPTY) motoEmptyCount++;
				else motoOccupiedCount++;
			} else {
				if (slotEntry.second == SlotStatus::EMPTY) carEmptyCount++;
				else carOccupiedCount++;
			}
		}
		
		int totalSlots = carEmptyCount + carOccupiedCount + motoEmptyCount + motoOccupiedCount;
		int violationCount = (int)state.violatingCarIds.size();

		// Track state changes to prevent bloat
		static int lastCarEmpty = -1;
		static int lastCarOccupied = -1;
		static int lastMotoEmpty = -1;
		static int lastMotoOccupied = -1;
		static int lastViolation = -1;

		if (carEmptyCount == lastCarEmpty && carOccupiedCount == lastCarOccupied && 
			motoEmptyCount == lastMotoEmpty && motoOccupiedCount == lastMotoOccupied && 
			violationCount == lastViolation) {
			return; // No change in numbers, do not generate a file
		}

		lastCarEmpty = carEmptyCount;
		lastCarOccupied = carOccupiedCount;
		lastMotoEmpty = motoEmptyCount;
		lastMotoOccupied = motoOccupiedCount;
		lastViolation = violationCount;

		SYSTEMTIME st;
		GetLocalTime(&st);
		char dateFolder[256];
		sprintf_s(dateFolder, sizeof(dateFolder), "C:\\loc_json\\parking_areas\\%04d%02d%02d\\camera_%d", st.wYear, st.wMonth, st.wDay, cameraId);
		CreateDirectoryA("C:\\loc_json", NULL);
		CreateDirectoryA("C:\\loc_json\\parking_areas", NULL);
		
		char parentFolder[128];
		sprintf_s(parentFolder, sizeof(parentFolder), "C:\\loc_json\\parking_areas\\%04d%02d%02d", st.wYear, st.wMonth, st.wDay);
		CreateDirectoryA(parentFolder, NULL);
		
		CreateDirectoryA(dateFolder, NULL);

		long long epoch = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
		std::string jsonPath = std::string(dateFolder) + "\\stats_" + std::to_string(epoch) + ".json";

		char timeBuf[64];
		sprintf_s(timeBuf, sizeof(timeBuf), "%04d-%02d-%02dT%02d:%02d:%02d", 
                  st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond);

		std::string camId = "camera_" + std::to_string(cameraId);

		// MongoEngine ParkingArea representation
		json areaStats = {
			{"name", "Main Zone A"},
			{"description", "Front Parking Monitoring"},
			{"camera_id", camId},
			{"total_slots", totalSlots},
			{"total_car_slots", carEmptyCount + carOccupiedCount},
			{"available_car_slots", carEmptyCount},
			{"occupied_car_slots", carOccupiedCount},
			{"total_motorcycle_slots", motoEmptyCount + motoOccupiedCount},
			{"available_motorcycle_slots", motoEmptyCount},
			{"occupied_motorcycle_slots", motoOccupiedCount},
			{"violation_slots", violationCount},
			{"created_date", std::string(timeBuf)},
			{"updated_date", std::string(timeBuf)}
		};

		std::ofstream outFile(jsonPath);
		if (outFile.is_open()) {
			outFile << areaStats.dump(4);
			outFile.close();
		}
	}

	// *** [NEW] VIOLATION ALERTS METHODS ***
	private: void AddViolationRecord_Online(int cameraId, int carId, cv::Mat& frameCapture, System::String^ violationType, 
									 cv::Mat fullFrame, cv::Rect carBox) {
		DumpLog("[VIOLATION] AddViolationRecord_Online called! carId=" + std::to_string(carId) + " frameEmpty=" + std::string(frameCapture.empty() ? "yes" : "no") + " fullEmpty=" + std::string(fullFrame.empty() ? "yes" : "no"));
		if (frameCapture.empty()) return;

		Bitmap^ screenshot = gcnew Bitmap(frameCapture.cols, frameCapture.rows, System::Drawing::Imaging::PixelFormat::Format24bppRgb);
		System::Drawing::Rectangle rect(0, 0, frameCapture.cols, frameCapture.rows);
		System::Drawing::Imaging::BitmapData^ bmpData = screenshot->LockBits(rect, System::Drawing::Imaging::ImageLockMode::WriteOnly, screenshot->PixelFormat);

		for (int y = 0; y < frameCapture.rows; y++) {
			memcpy((unsigned char*)bmpData->Scan0.ToPointer() + y * bmpData->Stride, frameCapture.data + y * frameCapture.step, frameCapture.cols * 3);
		}
		screenshot->UnlockBits(bmpData);

		cv::Mat visualizationMat = GetCam(cameraId)->CreateViolationVisualization(fullFrame, carBox);
		Bitmap^ visualizationBitmap = nullptr;
		if (!visualizationMat.empty()) {
			visualizationBitmap = gcnew Bitmap(visualizationMat.cols, visualizationMat.rows, System::Drawing::Imaging::PixelFormat::Format24bppRgb);
			System::Drawing::Rectangle visRect(0, 0, visualizationMat.cols, visualizationMat.rows);
			System::Drawing::Imaging::BitmapData^ visBmpData = visualizationBitmap->LockBits(visRect, System::Drawing::Imaging::ImageLockMode::WriteOnly, visualizationBitmap->PixelFormat);

			for (int y = 0; y < visualizationMat.rows; y++) {
				memcpy((unsigned char*)visBmpData->Scan0.ToPointer() + y * visBmpData->Stride, visualizationMat.data + y * visualizationMat.step, visualizationMat.cols * 3);
			}
			visualizationBitmap->UnlockBits(visBmpData);

			// ==========================================================
			// [PHASE 2] SAVE DARKENED SNAPSHOT TO DISK (WITH DATE FOLDER)
			// ==========================================================
			SYSTEMTIME st;
			GetLocalTime(&st);
			char dateFolder[256];
			sprintf_s(dateFolder, sizeof(dateFolder),
				"C:\\smart_parking_violations\\%04d%02d%02d\\camera_%d", st.wYear, st.wMonth, st.wDay, cameraId);

			char parentFolder[128];
			sprintf_s(parentFolder, sizeof(parentFolder), "C:\\smart_parking_violations\\%04d%02d%02d", st.wYear, st.wMonth, st.wDay);

			CreateDirectoryA("C:\\smart_parking_violations", NULL);
			CreateDirectoryA(parentFolder, NULL);
			CreateDirectoryA(dateFolder, NULL);

			long long epoch = std::chrono::duration_cast<std::chrono::milliseconds>(
				std::chrono::system_clock::now().time_since_epoch()
			).count();
			
			std::string violationFileName = "camera_" + std::to_string(cameraId) + "_event_" + std::to_string(epoch) + "_car_" + std::to_string(carId) + ".jpg";
			std::string violationFilePath = std::string(dateFolder) + "\\" + violationFileName;
			
			cv::imwrite(violationFilePath, visualizationMat);
			OutputDebugStringA(("[SNAPSHOT] Saved violation snapshot: " + violationFilePath + "\n").c_str());

			// [PHASE 3] Trigger JSON generation
			SaveAnomalyEventJson(cameraId, carId, violationType, epoch, violationFilePath);
			SaveParkingAreaJson(cameraId);
		}

		ViolationRecord_Online^ record = gcnew ViolationRecord_Online();
		record->carId = carId;
		record->screenshot = screenshot;
		record->visualizationBitmap = visualizationBitmap;
		record->violationType = violationType;
		record->captureTime = System::DateTime::Now;
		record->durationSeconds = 0;

		std::string vTypeStr = msclr::interop::marshal_as<std::string>(violationType);
		DumpLog("[LOG-APPEND] Adding Violation Record for Car ID: " + std::to_string(carId) + " | Type: " + vTypeStr + " | Current List Size: " + std::to_string(violationsList_online->Count));

		violationsList_online->Add(record);
		RefreshViolationPanel_Online();
	}

	private: void RefreshViolationPanel_Online() {
		if (this->InvokeRequired) {
			this->Invoke(gcnew System::Action(this, &UploadForm::RefreshViolationPanel_Online));
			return;
		}

		if (!flpViolations_online) return;

		flpViolations_online->Controls->Clear();

		for each(ViolationRecord_Online^ record in violationsList_online) {
			Panel^ itemPanel = gcnew Panel();
			itemPanel->BackColor = System::Drawing::Color::White;
			itemPanel->Size = System::Drawing::Size(250, 180);
			itemPanel->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
			itemPanel->Margin = System::Windows::Forms::Padding(5);
			itemPanel->Cursor = System::Windows::Forms::Cursors::Hand;
			
			itemPanel->Tag = record;
			itemPanel->Click += gcnew System::EventHandler(this, &UploadForm::OnViolationItemClick_Online);

			PictureBox^ pbScreenshot = gcnew PictureBox();
			pbScreenshot->Image = record->screenshot;
			pbScreenshot->SizeMode = System::Windows::Forms::PictureBoxSizeMode::Zoom;
			pbScreenshot->Location = System::Drawing::Point(5, 5);
			pbScreenshot->Size = System::Drawing::Size(240, 100);
			pbScreenshot->Cursor = System::Windows::Forms::Cursors::Hand;
					pbScreenshot->Click += gcnew System::EventHandler(this, &UploadForm::OnViolationItemClick_Online);
			itemPanel->Controls->Add(pbScreenshot);

			Label^ lblInfo = gcnew Label();
			lblInfo->Font = gcnew System::Drawing::Font(L"Segoe UI", 8);
			lblInfo->Location = System::Drawing::Point(5, 110);
			lblInfo->Size = System::Drawing::Size(240, 65);
			lblInfo->Text = System::String::Format(L"ID: {0}\nType: {1}\nTime: {2:HH:mm:ss}\nDuration: {3}s",
				record->carId, record->violationType, record->captureTime, record->durationSeconds);
			lblInfo->Cursor = System::Windows::Forms::Cursors::Hand;
			lblInfo->Click += gcnew System::EventHandler(this, &UploadForm::OnViolationItemClick_Online);
			itemPanel->Controls->Add(lblInfo);

			flpViolations_online->Controls->Add(itemPanel);
		}

		lblViolationCount_online->Text = System::String::Format(L"Violations: {0}", violationsList_online->Count);
	}
	
	private: System::Void OnViolationItemClick_Online(System::Object^ sender, System::EventArgs^ e) {
		Control^ clickedControl = safe_cast<Control^>(sender);
		Panel^ itemPanel = nullptr;
		
		if (clickedControl->GetType() == Panel::typeid) {
			itemPanel = safe_cast<Panel^>(clickedControl);
		}
		else {
			itemPanel = safe_cast<Panel^>(clickedControl->Parent);
		}
		
		if (itemPanel && itemPanel->Tag != nullptr) {
			ViolationRecord_Online^ record = safe_cast<ViolationRecord_Online^>(itemPanel->Tag);
			
			ViolationDetailForm^ detailForm = gcnew ViolationDetailForm(
				record->carId,
				record->screenshot,
				record->visualizationBitmap,
				record->violationType,
				record->captureTime,
				record->durationSeconds
			);
			detailForm->ShowDialog(this);
		}
	}

	public: void CheckViolations_Online(int cameraId, cv::Mat& currentFrame) {
			// [PHASE 2] Throttle violation checks to 500ms
		auto now = std::chrono::steady_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - GetCam(cameraId)->g_lastViolationCheck_online).count();
		
		if (elapsed < VIOLATION_CHECK_INTERVAL_MS_ONLINE) {
			return;
		}
		GetCam(cameraId)->g_lastViolationCheck_online = now;

		OnlineAppState state;
		{
			std::lock_guard<std::mutex> lock(GetCam(cameraId)->g_onlineStateMutex);
			state = GetCam(cameraId)->g_onlineState;
		}

		for each(auto car in state.cars) {
			if (car.framesStill > 300) {
				if (!violatingCarTimers_online->ContainsKey(car.id)) {
					violatingCarTimers_online->Add(car.id, System::DateTime::Now);
					
					cv::Rect safeBbox = car.bbox & cv::Rect(0, 0, currentFrame.cols, currentFrame.rows);
					if (safeBbox.area() > 0) {
						cv::Mat croppedFrame = currentFrame(safeBbox).clone();
						AddViolationRecord_Online(cameraId, car.id, croppedFrame, L"Overstay", currentFrame, car.bbox);
					}
				}
			}
		}

		for each(int violatingId in state.violatingCarIds) {
			// Determine specific violation type
			System::String^ specificViolation = L"Wrong Parking"; // Default: parked outside slots
			
			for (const auto& slot : GetCam(cameraId)->g_pm_logic_online->getSlots()) {
				if (slot.status == SlotStatus::ILLEGAL && slot.occupiedByTrackId == violatingId) {
					specificViolation = L"Wrong Vehicle Type";
					break;
				}
			}

			bool already_captured = false;
			for each(ViolationRecord_Online^ record in violationsList_online) {
				if (record->carId == violatingId && record->violationType == specificViolation) {
					already_captured = true;
					break;
				}
			}

			if (!already_captured) {
				for each(auto car in state.cars) {
					if (car.id == violatingId) {
						cv::Rect safeBbox = car.bbox & cv::Rect(0, 0, currentFrame.cols, currentFrame.rows);
						if (safeBbox.area() > 0) {
							cv::Mat croppedFrame = currentFrame(safeBbox).clone(); // [FIX] Clone only when capturing
							AddViolationRecord_Online(cameraId, violatingId, croppedFrame, specificViolation, currentFrame, car.bbox);
						}
						break;
					}
				}
			}
		}

		for each(ViolationRecord_Online^ record in violationsList_online) {
			System::TimeSpan duration = System::DateTime::Now - record->captureTime;
			record->durationSeconds = (int)duration.TotalSeconds;
		}

		UpdateWebStats(cameraId);
	}

	private: System::Void btnClearViolations_online_Click(System::Object^ sender, System::EventArgs^ e) {
		violationsList_online->Clear();
		violatingCarTimers_online->Clear();
		RefreshViolationPanel_Online();
	}

	// *** [NEW] SHOW LOG DATETIME OR DEFAULT MESSAGE ***
	private: System::Void lblLogs_Click(System::Object^ sender, System::EventArgs^ e) {
		if (isProcessing) {
			System::DateTime now = System::DateTime::Now;
			System::String^ dateTimeStr = now.ToString(L"dd/MM/yy");
			lblLogs->Text = dateTimeStr;
		}
		else {
			lblLogs->Text = L"logs 25/12/67";
		}
	}

	// *** [NEW] BACKGROUND MODE LOGIC ***
	private: System::Void btnRunInBackground_Click(System::Object^ sender, System::EventArgs^ e) {
		this->Hide();
		isBackgroundMode = true;
		notifyIcon->Visible = true;
		
		// Set notification icon if available, otherwise use a default icon
		try {
			notifyIcon->Icon = gcnew System::Drawing::Icon("app.ico");
		} catch (...) {
			notifyIcon->Icon = SystemIcons::Application;
		}

		// Show the stream URL if it's available
		System::String^ balloonText = "Application is running in the background. MJPEG Stream is active.";
		if (lblNetworkStream != nullptr && lblNetworkStream->Text->StartsWith("Stream: http")) {
			balloonText += "\n" + lblNetworkStream->Text;
		}

		notifyIcon->ShowBalloonTip(3000, "Smart Parking", balloonText, ToolTipIcon::Info);
	}

	private: System::Void notifyIcon_DoubleClick(System::Object^ sender, System::EventArgs^ e) {
		RestoreFromBackground();
	}

	private: System::Void menuShow_Click(System::Object^ sender, System::EventArgs^ e) {
		RestoreFromBackground();
	}

	private: System::Void RestoreFromBackground() {
		this->Show();
		this->WindowState = FormWindowState::Normal;
		isBackgroundMode = false;
		notifyIcon->Visible = false;
	}

	private: System::Void menuExit_Click(System::Object^ sender, System::EventArgs^ e) {
		// Stop processing cleanly before exiting
		StopProcessing(1);
		this->Close();
	}
	};
} // End of namespace ConsoleApplication3

inline void CameraInstance::ProcessingLoopHeadless() {
	lastProcessedSeq = -1;
	while (!shouldStop) {
		try {
			cv::Mat frameToProcess;
			long long seq = 0;
			GetRawFrameOnline(frameToProcess, seq);

			if (!frameToProcess.empty() && seq > lastProcessedSeq) {
				ProcessFrameOnline(frameToProcess, seq);
				
				cv::Mat renderedFrame;
				DrawSceneOnline(frameToProcess, seq, renderedFrame);

				if (!renderedFrame.empty()) {
					{
						std::lock_guard<std::mutex> lock(g_processedMutex_online);
						g_processedFrame_online = renderedFrame;
						g_processedSeq_online = seq;
						g_processedFramesCount_online++;
					}

					// [PHASE 3] Allow background headless AI threads to process violations natively
					if (ConsoleApplication3::UploadForm::Instance != nullptr && g_parkingEnabled_online.load()) {
						ConsoleApplication3::UploadForm::Instance->CheckViolations_Online(camera_id, frameToProcess);
					}

					if (g_mjpegServer_online) {
						g_mjpegServer_online->SetLatestFrame(camera_id, renderedFrame);
					}

					cv::Mat scaledFrame;
					double maxW = 1280.0;
					if (renderedFrame.cols > maxW) {
						double scale = maxW / renderedFrame.cols;
						cv::resize(renderedFrame, scaledFrame, cv::Size(), scale, scale);
					} else {
						scaledFrame = renderedFrame;
					}

					StartVideoRecordingThread_Online(scaledFrame.cols, scaledFrame.rows);

					{
						std::lock_guard<std::mutex> vidLock(g_videoCurrentFrameMutex);
						g_videoCurrentFrame = scaledFrame.clone();
					}
				}
				lastProcessedSeq = seq;
			} else {
				std::this_thread::sleep_for(std::chrono::milliseconds(2));
			}
		}
		catch (...) { std::this_thread::sleep_for(std::chrono::milliseconds(5)); }
	}
}
