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
#include "json.hpp"

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

using json = nlohmann::json;

// ==========================================
//  LAYER 1: SHARED DATA
// ==========================================
struct OnlineAppState {
	std::vector<TrackedObject> cars;
	std::set<int> violatingCarIds;
	std::map<int, SlotStatus> slotStatuses;
	std::map<int, float> slotOccupancy;
	long long frameSequence = -1;
};

static OnlineAppState g_onlineState;
static std::mutex g_onlineStateMutex;

// ==========================================
//  LAYER 2: LOGIC & BACKEND
// ==========================================

static cv::dnn::Net* g_net = nullptr;
static std::vector<std::string> g_classes;
static std::vector<cv::Scalar> g_colors;
static BYTETracker* g_tracker = nullptr;

static ParkingManager* g_pm_logic_online = nullptr;

static cv::VideoCapture* g_cap = nullptr;
static cv::Mat g_latestRawFrame;
static long long g_frameSeq_online = 0;
static std::mutex g_frameMutex;
static double g_cameraFPS = 30.0;

static std::mutex g_aiMutex_online;
static bool g_modelReady = false;
static std::atomic<bool> g_parkingEnabled_online(false); // [PHASE 1 FIX] Use atomic for thread safety

static const int YOLO_INPUT_SIZE = 640;
static const float CONF_THRESHOLD = 0.25f;
static const float NMS_THRESHOLD = 0.45f;

// ==========================================
//  LAYER 3: PRESENTATION (Frontend)
// ==========================================
static ParkingManager* g_pm_display_online = nullptr;

static cv::Mat g_cachedParkingOverlay_online;
static std::map<int, SlotStatus> g_lastDrawnStatus_online;
static cv::Mat g_drawingBuffer_online; // Memory Pool

// *** [NEW] PROCESSED FRAME SHARING (Pipeline Output) ***
static cv::Mat g_processedFrame_online;
static long long g_processedSeq_online = 0;
static std::mutex g_processedMutex_online;

// *** [NEW] MJPEG SERVER ***
static MjpegServer* g_mjpegServer_online = nullptr;

// *** [PHASE 2] PERFORMANCE OPTIMIZATION ***
static std::chrono::steady_clock::time_point g_lastViolationCheck_online = std::chrono::steady_clock::now();
static const int VIOLATION_CHECK_INTERVAL_MS_ONLINE = 500;

static std::atomic<int> g_droppedFrames_online(0);
static std::atomic<int> g_processedFramesCount_online(0);

// [FIX] Stream stability improvements
static const int MAX_FRAME_LAG_ONLINE = 10; // Relaxed from 3 to 10 frames
static const int NETWORK_BUFFER_SIZE = 5;    // Network jitter buffer

// *** [PHASE 3] MEMORY OPTIMIZATION ***

struct CachedLabel_Online {
	std::string text;
	cv::Size size;
	int baseline;
	bool isViolating;
	int classId;
	
	CachedLabel_Online() : baseline(0), isViolating(false), classId(-1) {}
};

static std::map<int, CachedLabel_Online> g_labelCache_online;
static cv::Mat g_redOverlayBuffer_online; // [PHASE 3] Reusable red overlay buffer

// [PHASE 3] FPS Monitor
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

static FPSMonitor_Online g_fpsMonitor_online;

// *** [STREAMING LOG] Mutex-protected stats updated every frame, sent every 500ms ***
struct LogStats_Online {
	int empty = 0;
	int occupied = 0;
	int violation = 0;
	cv::Mat lastFrame; // shallow copy of processed frame (for base64 on violation)
};
static LogStats_Online g_logStats_online;
static std::mutex g_logStatsMutex_online;
static long long g_lastLogSendTick_online = 0;

// *** [STREAMING LOG] Last-sent snapshot — send SSE only when values change ***
struct LastSentStats_Online {
	int empty = -1;
	int occupied = -1;
	int violation = -1;
};
static LastSentStats_Online g_lastSentStats_online;

// --- Helper Functions ---

// *** [VIOLATION LOG] Save cv::Mat as JPEG to C:\logpic, return filename ***
static std::string SaveViolationImage(const cv::Mat& frame, int carId, const std::string& typeStr) {
	if (frame.empty()) return "";
	CreateDirectoryA("C:\\logpic", NULL);
	SYSTEMTIME st;
	GetLocalTime(&st);

	// Replace spaces with underscores so URL has no spaces (e.g. "Wrong_Slot")
	std::string safeType = typeStr;
	std::replace(safeType.begin(), safeType.end(), ' ', '_');

	char fname[128];
	sprintf_s(fname, sizeof(fname),
		"vio_%d_%s_%04d%02d%02d_%02d%02d%02d.jpg",
		carId, safeType.c_str(),
		st.wYear, st.wMonth, st.wDay,
		st.wHour, st.wMinute, st.wSecond);
	std::string fullPath = std::string("C:\\logpic\\") + fname;
	cv::imwrite(fullPath, frame);
	return std::string(fname);
}

static void ResetParkingCache_Online() {
	g_cachedParkingOverlay_online = cv::Mat();
	g_lastDrawnStatus_online.clear();
	g_labelCache_online.clear(); // [PHASE 3] Clear label cache
	g_redOverlayBuffer_online = cv::Mat(); // [PHASE 3] Clear red overlay buffer
}

static cv::Mat FormatToLetterbox(const cv::Mat& source, int width, int height, float& ratio, int& dw, int& dh) {
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

// *** GET RAW FRAME ***
static void GetRawFrameOnline(cv::Mat& outFrame, long long& outSeq) {
	std::lock_guard<std::mutex> lock(g_frameMutex);
	if (!g_latestRawFrame.empty()) {
		outFrame = g_latestRawFrame; // [FIX] Shallow copy for speed (AI thread clones if needed)
		outSeq = g_frameSeq_online;
	}
}

// *** [NEW] GET PROCESSED FRAME (For UI) ***
static void GetProcessedFrameOnline(cv::Mat& outFrame, long long& outSeq) {
	std::lock_guard<std::mutex> lock(g_processedMutex_online);
	if (!g_processedFrame_online.empty()) {
		outFrame = g_processedFrame_online; // [FIX] Shallow copy for speed
		outSeq = g_processedSeq_online;
	}
}

static void OpenGlobalCamera(int cameraIndex = 0) {
	std::lock_guard<std::mutex> lock(g_frameMutex);
	if (g_cap) { delete g_cap; g_cap = nullptr; }
	g_cap = new cv::VideoCapture(cameraIndex);
	g_frameSeq_online = 0;
	if (g_cap->isOpened()) {
		g_cameraFPS = g_cap->get(cv::CAP_PROP_FPS);
		if (g_cameraFPS <= 0) g_cameraFPS = 30.0;
	}
	ResetParkingCache_Online();
	
	std::lock_guard<std::mutex> slock(g_onlineStateMutex);
	g_onlineState = OnlineAppState();
}

static void OpenGlobalCameraFromIP(const std::string& rtspUrl) {
	std::lock_guard<std::mutex> lock(g_frameMutex);
	if (g_cap) { delete g_cap; g_cap = nullptr; }
	g_cap = new cv::VideoCapture(rtspUrl);
	
	// [FIX] Network stream optimization
	if (g_cap->isOpened()) {
		g_cap->set(cv::CAP_PROP_BUFFERSIZE, NETWORK_BUFFER_SIZE); // Reduce latency
		g_cameraFPS = g_cap->get(cv::CAP_PROP_FPS);
		if (g_cameraFPS <= 0) g_cameraFPS = 30.0;
	}
	
	g_frameSeq_online = 0;
	ResetParkingCache_Online();
	
	std::lock_guard<std::mutex> slock(g_onlineStateMutex);
	g_onlineState = OnlineAppState();
}

static void InitGlobalModel(const std::string& modelPath) {
	std::lock_guard<std::mutex> lock(g_aiMutex_online);
	g_modelReady = false;
	if (g_net) { delete g_net; g_net = nullptr; }
	if (g_tracker) { delete g_tracker; g_tracker = nullptr; }

	try {
		g_net = new cv::dnn::Net(cv::dnn::readNetFromONNX(modelPath));
		g_net->setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
		g_net->setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
		g_tracker = new BYTETracker(90, 0.25f);
		
		OutputDebugStringA("[INFO] Online mode with improved ByteTrack (90 frames tolerance)\n");

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
		if (g_tracker) { delete g_tracker; g_tracker = nullptr; }
	}
}

// [NEW] โหลด Parking Template
static bool LoadParkingTemplate_Online(const std::string& filename) {
	ResetParkingCache_Online();

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

// *** [NEW] CREATE VIOLATION VISUALIZATION ***
static cv::Mat CreateViolationVisualization(cv::Mat fullFrame, cv::Rect carBox) {
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

// *** WORKER PROCESS (AI Thread) ***

static void ProcessFrameOnline(const cv::Mat& inputFrame, long long frameSeq) {
	{
		std::lock_guard<std::mutex> lock(g_aiMutex_online);
		if (inputFrame.empty() || !g_net || !g_modelReady || !g_tracker) return;
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
			g_net->setInput(blob);
			g_net->forward(outputs, g_net->getUnconnectedOutLayersNames());
		}

		if (outputs.empty() || outputs[0].empty()) return;

		cv::Mat output_data = outputs[0];
		int rows = output_data.size[1];
		int dimensions = output_data.size[2];

		if (output_data.dims == 3) {
			output_data = output_data.reshape(1, rows);
			cv::transpose(output_data, output_data);
			rows = output_data.rows;
			dimensions = output_data.cols;
		}
		else {
			cv::Mat output_t;
			cv::transpose(output_data.reshape(1, output_data.size[1]), output_t);
			output_data = output_t;
			rows = output_data.rows;
			dimensions = output_data.cols;
		}

		float* data = (float*)output_data.data;
		std::vector<int> class_ids;
		std::vector<float> confs;
		std::vector<cv::Rect> boxes;

		for (int i = 0; i < rows; i++) {
			float* classes_scores = data + 4;
			if (dimensions >= 4 + (int)g_classes.size()) {
				cv::Mat scores(1, (int)g_classes.size(), CV_32FC1, classes_scores);
				cv::Point class_id;
				double max_class_score;
				cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

				if (max_class_score > CONF_THRESHOLD && class_id.x == 2) {
					float x = data[0]; float y = data[1]; float w = data[2]; float h = data[3];
					float left = (x - 0.5 * w - dw) / ratio;
					float top = (y - 0.5 * h - dh) / ratio;
					float width = w / ratio;
					float height = h / ratio;
				 boxes.push_back(cv::Rect((int)left, (int)top, (int)width, (int)height));
					confs.push_back((float)max_class_score);
					class_ids.push_back(class_id.x);
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
		std::set<int> violations;

		bool parkingEnabled = g_parkingEnabled_online.load(); // [PHASE 1 FIX] Use atomic load
		if (parkingEnabled && g_pm_logic_online) {
			static bool templateSet_online = false;
			if (!templateSet_online) {
				g_pm_logic_online->setTemplateFrame(inputFrame);
				templateSet_online = true;
			}

			g_pm_logic_online->updateSlotStatus(trackedObjs);

			for (const auto& slot : g_pm_logic_online->getSlots()) {
				calculatedStatuses[slot.id] = slot.status;
				calculatedOccupancy[slot.id] = slot.occupancyPercent;
			}

			// ตรวจจับรถจอดผิด
			for (const auto& car : trackedObjs) {
				if (car.framesStill > 30) {
					bool inAnySlot = false;
					for (const auto& slot : g_pm_logic_online->getSlots()) {
						cv::Point center = (car.bbox.tl() + car.bbox.br()) * 0.5;
						if (cv::pointPolygonTest(slot.polygon, center, false) >= 0) {
							inAnySlot = true;
							break;
						}
					}
					if (!inAnySlot) {
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
			g_onlineState.violatingCarIds = violations;
			g_onlineState.frameSequence = frameSeq;
		}
	}
	catch (...) {}
}

// EMA bounding box smoothing — alpha=0.35: lower=smoother, higher=more responsive
static std::map<int, cv::Rect> g_emaBoxes_online;
static std::map<int, int>      g_emaMissCount_online; // consecutive frames not detected
static const float             EMA_ALPHA = 0.35f;
static const int               EMA_MISS_THRESHOLD = 30; // delete after N missed frames

static void DrawSceneOnline(const cv::Mat& frame, long long displaySeq, cv::Mat& outResult) {
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

				// *** EMA smoothing: position only (x,y) — size stays raw ***
				auto& ema = g_emaBoxes_online[obj.id];
				if (ema.area() == 0) {
					ema = box; // first detection — seed with raw
				} else {
					ema.x = (int)(EMA_ALPHA * box.x + (1-EMA_ALPHA) * ema.x);
					ema.y = (int)(EMA_ALPHA * box.y + (1-EMA_ALPHA) * ema.y);
					ema.width  = box.width;  // always use latest size
					ema.height = box.height; // always use latest size
				}
				box = ema; // use smoothed position for drawing

				bool isViolating = (state.violatingCarIds.count(obj.id) > 0);

				if (isViolating) {
					cv::Rect roi = box & cv::Rect(0, 0, outResult.cols, outResult.rows);
					if (roi.area() > 0) {
						cv::Mat roiMat = outResult(roi);
						// [PHASE 3] Reuse red overlay buffer
						if (g_redOverlayBuffer_online.size() != roi.size() || g_redOverlayBuffer_online.type() != CV_8UC3) {
							g_redOverlayBuffer_online = cv::Mat(roi.size(), CV_8UC3, cv::Scalar(0, 0, 255));
						}
						cv::addWeighted(roiMat, 0.6, g_redOverlayBuffer_online, 0.4, 0, roiMat);
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
			// EMA GC: increment miss counter for disappeared cars, erase after threshold
			for (auto it = g_emaMissCount_online.begin(); it != g_emaMissCount_online.end();) {
				if (!currentFrameCarIds.count(it->first)) {
					it->second++;
					if (it->second > EMA_MISS_THRESHOLD) {
						g_emaBoxes_online.erase(it->first);
						it = g_emaMissCount_online.erase(it);
						continue;
					}
				} else {
					it->second = 0; // reset counter — car is visible
				}
				++it;
			}
			// Register new cars in miss counter
			for (int id : currentFrameCarIds) {
				if (!g_emaMissCount_online.count(id))
					g_emaMissCount_online[id] = 0;
			}
		}
	}

	// [PHASE 3] Draw Stats (Obj count + FPS)
	std::string stats = "Obj: " + std::to_string(state.cars.size()) + " | FPS: " + std::to_string((int)g_fpsMonitor_online.avgFPS);
	cv::putText(outResult, stats, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

	// *** CCTV Timestamp Overlay ***
	{
		SYSTEMTIME st;
		GetLocalTime(&st);
		char timeBuf[64];
		sprintf_s(timeBuf, sizeof(timeBuf), "%04d-%02d-%02d  %02d:%02d:%02d",
			st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond);
		std::string timeStr(timeBuf);
		int fontFace = cv::FONT_HERSHEY_SIMPLEX;
		double fontScale = 0.7;
		int thickness = 2;
		int baseline = 0;
		cv::Size textSize = cv::getTextSize(timeStr, fontFace, fontScale, thickness, &baseline);
		// Bottom-right corner with padding
		int x = outResult.cols - textSize.width - 15;
		int y = outResult.rows - 15;
		// Dark background for readability
		cv::rectangle(outResult, cv::Point(x - 5, y - textSize.height - 5),
			cv::Point(x + textSize.width + 5, y + baseline + 5), cv::Scalar(0, 0, 0), -1);
		cv::putText(outResult, timeStr, cv::Point(x, y), fontFace, fontScale, cv::Scalar(255, 255, 255), thickness);
	}
}

// ==========================================
//  STEP 2: BASE64 ENCODING FUNCTIONS
// ==========================================

static std::string Base64Encode(const unsigned char* buffer, size_t length) {
	static const char base64_table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

	std::string result;
	result.reserve((length + 2) / 3 * 4);
	
	for (size_t i = 0; i < length; i += 3) {
		unsigned int b1 = buffer[i];
		unsigned int b2 = (i + 1 < length) ? buffer[i + 1] : 0;
		unsigned int b3 = (i + 2 < length) ? buffer[i + 2] : 0;
		
		unsigned int triplet = (b1 << 16) | (b2 << 8) | b3;
		
		result += base64_table[(triplet >> 18) & 0x3F];
		result += base64_table[(triplet >> 12) & 0x3F];
		result += (i + 1 < length) ? base64_table[(triplet >> 6) & 0x3F] : '=';
		result += (i + 2 < length) ? base64_table[triplet & 0x3F] : '=';
	}
	
	return result;
}

static std::string MatToBase64Jpeg(const cv::Mat& frame) {
	// Check if frame is empty
	if (frame.empty()) {
		return std::string();
	}
	
	// Encode frame to JPEG with quality 85
	std::vector<uchar> buffer;
	std::vector<int> params = { cv::IMWRITE_JPEG_QUALITY, 85 };
	bool success = cv::imencode(".jpg", frame, buffer, params);
	
	if (!success || buffer.empty()) {
		OutputDebugStringA("[ERROR] Failed to encode frame to JPEG\n");
		return std::string();
	}
	
	// Debug output
	OutputDebugStringA(("[IMG] Encoded JPEG: " + std::to_string(buffer.size()) + " bytes\n").c_str());
	
	// Convert JPEG buffer to Base64
	std::string base64String = Base64Encode(buffer.data(), buffer.size());
	
	return base64String;
}

// ==========================================
//  STEP 3: ASYNC HTTP POST FUNCTIONS
// ==========================================

static void SendHttpPostAsync(const std::string& jsonPayload, 
                              const std::string& serverIP = "127.0.0.1", 
                              int port = 9000) {
	// Fire-and-forget: detach thread to send HTTP POST asynchronously
	std::thread([jsonPayload, serverIP, port]() {
		try {
			// Initialize WinSock2
			WSADATA wsaData;
			int wsaResult = WSAStartup(MAKEWORD(2, 2), &wsaData);
			if (wsaResult != 0) {
				OutputDebugStringA(("[HTTP] ❌ WSAStartup failed with error: " + std::to_string(wsaResult) + "\n").c_str());
				return;
			}

			// Create socket
			SOCKET sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
			if (sock == INVALID_SOCKET) {
				int err = WSAGetLastError();
				OutputDebugStringA(("[HTTP] ❌ socket() failed with error: " + std::to_string(err) + "\n").c_str());
				WSACleanup();
				return;
			}

			// Set socket timeout to 5 seconds
			DWORD timeout = 5000;
			setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout, sizeof(timeout));
			setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, (const char*)&timeout, sizeof(timeout));

			// Prepare server address
			sockaddr_in serverAddr;
			serverAddr.sin_family = AF_INET;
			serverAddr.sin_port = htons(port);
			serverAddr.sin_addr.s_addr = inet_addr(serverIP.c_str());

			// Log connection attempt
			OutputDebugStringA(("[HTTP] Sending to " + serverIP + ":" + std::to_string(port) + "\n").c_str());
			OutputDebugStringA(("[HTTP] Payload size: " + std::to_string(jsonPayload.size()) + " bytes\n").c_str());

			// Connect to server
			if (connect(sock, (sockaddr*)&serverAddr, sizeof(serverAddr)) == SOCKET_ERROR) {
				int err = WSAGetLastError();
				OutputDebugStringA(("[HTTP] ❌ connect() failed with error: " + std::to_string(err) + "\n").c_str());
				closesocket(sock);
				WSACleanup();
				return;
			}

			// Build HTTP POST request
			std::string httpRequest = "POST /api/violations HTTP/1.1\r\n";
			httpRequest += "Host: " + serverIP + ":" + std::to_string(port) + "\r\n";
			httpRequest += "Content-Type: application/json\r\n";
			httpRequest += "Content-Length: " + std::to_string(jsonPayload.size()) + "\r\n";
			httpRequest += "Connection: close\r\n";
			httpRequest += "\r\n";
			httpRequest += jsonPayload;

			// Send HTTP request
			int sendResult = send(sock, httpRequest.c_str(), (int)httpRequest.size(), 0);
			if (sendResult == SOCKET_ERROR) {
				int err = WSAGetLastError();
				OutputDebugStringA(("[HTTP] ❌ send() failed with error: " + std::to_string(err) + "\n").c_str());
				closesocket(sock);
				WSACleanup();
				return;
			}

			// Receive response
			char recvbuf[512] = {0};
			int recvResult = recv(sock, recvbuf, sizeof(recvbuf) - 1, 0);
			
			if (recvResult > 0) {
				recvbuf[recvResult] = '\0';
				std::string response(recvbuf);
				
				// Check for 200 OK response
				if (response.find("200 OK") != std::string::npos) {
					OutputDebugStringA("[HTTP] ✅ Success - 200 OK\n");
				} else {
					OutputDebugStringA(("[HTTP] ⚠️  Received response: " + response.substr(0, 100) + "\n").c_str());
				}
			} else if (recvResult == 0) {
				OutputDebugStringA("[HTTP] ⚠️  Connection closed by server\n");
			} else {
				int err = WSAGetLastError();
				if (err == WSAETIMEDOUT) {
					OutputDebugStringA("[HTTP] ⏱️ Timeout after 5s\n");
				} else {
					OutputDebugStringA(("[HTTP] ❌ recv() failed with error: " + std::to_string(err) + "\n").c_str());
				}
			}

			// Close socket
			closesocket(sock);
			WSACleanup();
		}
		catch (const std::exception& ex) {
			OutputDebugStringA(("[HTTP] ❌ Exception: " + std::string(ex.what()) + "\n").c_str());
		}
		catch (...) {
			OutputDebugStringA("[HTTP] ❌ Unknown exception occurred\n");
		}
	}).detach(); // Fire-and-forget: detach thread
}

// ==========================================
//  STEP 4: TIMESTAMP & STATUS UPDATE
// ==========================================

static std::string GetISO8601Timestamp() {
	SYSTEMTIME st;
	GetSystemTime(&st);
	
	char buffer[32];
	sprintf_s(buffer, sizeof(buffer), "%04d-%02d-%02dT%02d:%02d:%02d.%03dZ",
			  st.wYear, st.wMonth, st.wDay,
			  st.wHour, st.wMinute, st.wSecond, st.wMilliseconds);
	
	return std::string(buffer);
}

static void SendAggregateStatusUpdate(int emptyCount, 
                                      int occupiedCount, 
                                      int violationCount,
                                      const cv::Mat& frame,
                                      const std::string& cameraId = "cam_01",
                                      const std::string& serverIP = "127.0.0.1",
                                      int port = 9000) {
	try {
		// Log the status update
		OutputDebugStringA(("[STATS] Sending: " + std::to_string(violationCount) + 
						  " violations, " + std::to_string(occupiedCount) + 
						  " occupied, " + std::to_string(emptyCount) + " empty\n").c_str());

		// Get current timestamp in ISO8601 format
		std::string timestamp = GetISO8601Timestamp();

		// Convert frame to base64 JPEG
		std::string base64Image = MatToBase64Jpeg(frame);

		// Build JSON payload
		json payload;
		payload["camera_id"] = cameraId;
		payload["timestamp"] = timestamp;
		payload["total_slots"] = emptyCount + occupiedCount;
		payload["available_slots"] = emptyCount;
		payload["occupied_slots"] = occupiedCount;
		payload["violation_slots"] = violationCount;
		
		// Only include image if we have a valid base64 string
		if (!base64Image.empty()) {
			payload["latest_violation_image_base64"] = base64Image;
		}

		// Convert JSON to string
		std::string jsonStr = payload.dump();
		
		OutputDebugStringA(("[STATS] JSON payload size: " + std::to_string(jsonStr.size()) + " bytes\n").c_str());

		// Send HTTP POST request asynchronously
		SendHttpPostAsync(jsonStr, serverIP, port);
	}
	catch (const std::exception& ex) {
		OutputDebugStringA(("[STATS] ❌ Exception: " + std::string(ex.what()) + "\n").c_str());
	}
	catch (...) {
		OutputDebugStringA("[STATS] ❌ Unknown exception occurred\n");
	}
}

// ============================================================
//  [UNMANAGED] VIDEO RECORDING — 20s MP4 clips
//  C:\locvideo\YYYYMMDD\HHMMSS.mp4
// ============================================================

static cv::VideoWriter* g_videoWriter_online      = nullptr;
static std::mutex        g_videoWriterMutex_online;
static std::string       g_currentVideoRelPath     = ""; // current clip being written
static std::string       g_lastCompletedRelPath    = ""; // last fully-finished clip (ready to serve)
static long long         g_videoClipStartTick      = 0;
static int               g_videoFramesWritten      = 0;  // frames written since clip start
static double            g_lastClipActualFps       = 0;  // measured fps of last completed clip
static const int         VIDEO_CLIP_SECONDS        = 60; // 1 minute per clip

// *** Async Video Write Queue ***
#include <queue>
#include <condition_variable>
static std::queue<cv::Mat>     g_videoWriteQueue;
static std::mutex              g_videoQueueMutex;
static std::condition_variable g_videoQueueCV;
static std::atomic<bool>       g_videoQueueRunning(false);
static std::thread*            g_videoWriteThread = nullptr;
static const int               VIDEO_QUEUE_MAX = 30; // drop frames if queue too deep

static void VideoWriteThreadFunc() {
	while (g_videoQueueRunning.load()) {
		cv::Mat frame;
		{
			std::unique_lock<std::mutex> ul(g_videoQueueMutex);
			g_videoQueueCV.wait_for(ul, std::chrono::milliseconds(100),
				[]{ return !g_videoWriteQueue.empty() || !g_videoQueueRunning.load(); });
			if (g_videoWriteQueue.empty()) continue;
			frame = g_videoWriteQueue.front();
			g_videoWriteQueue.pop();
		}
		if (!frame.empty()) {
			std::lock_guard<std::mutex> vl(g_videoWriterMutex_online);
			if (g_videoWriter_online && g_videoWriter_online->isOpened()) {
				g_videoWriter_online->write(frame);
				g_videoFramesWritten++;
			}
		}
	}
}

static void StartVideoWriteThread() {
	if (g_videoWriteThread) return;
	g_videoQueueRunning.store(true);
	g_videoWriteThread = new std::thread(VideoWriteThreadFunc);
}

static void StopVideoWriteThread() {
	g_videoQueueRunning.store(false);
	g_videoQueueCV.notify_all();
	if (g_videoWriteThread && g_videoWriteThread->joinable())
		g_videoWriteThread->join();
	delete g_videoWriteThread;
	g_videoWriteThread = nullptr;
}

static void EnqueueVideoFrame(const cv::Mat& frame) {
	std::lock_guard<std::mutex> ql(g_videoQueueMutex);
	if ((int)g_videoWriteQueue.size() < VIDEO_QUEUE_MAX)
		g_videoWriteQueue.push(frame.clone());
	// else: drop frame silently (queue full)
	g_videoQueueCV.notify_one();
}

static void StartNewVideoClip(int width, int height, double fps) {
	if (width <= 0 || height <= 0) return;
	SYSTEMTIME st;
	GetLocalTime(&st);

	// Date folder: C:\locvideo\YYYYMMDD
	char dateFolder[64];
	sprintf_s(dateFolder, sizeof(dateFolder),
		"C:\\locvideo\\%04d%02d%02d", st.wYear, st.wMonth, st.wDay);
	CreateDirectoryA("C:\\locvideo", NULL);
	CreateDirectoryA(dateFolder, NULL);

	// Filename: HHMMSS.webm (start time of clip)
	char relPath[64];
	sprintf_s(relPath, sizeof(relPath),
		"%04d%02d%02d/%02d%02d%02d.webm",
		st.wYear, st.wMonth, st.wDay,
		st.wHour, st.wMinute, st.wSecond);

	std::string fullPath = std::string("C:\\locvideo\\") + relPath;
	// Fix backslashes in relPath for file system
	std::replace(fullPath.begin(), fullPath.end(), '/', '\\');

	std::lock_guard<std::mutex> lock(g_videoWriterMutex_online);
	if (g_videoWriter_online) {
		// Measure actual fps of the clip we're closing
		double elapsed = (cv::getTickCount() - g_videoClipStartTick) / cv::getTickFrequency();
		if (elapsed > 1.0 && g_videoFramesWritten > 0)
			g_lastClipActualFps = g_videoFramesWritten / elapsed;
		// Save completed clip path
		if (!g_currentVideoRelPath.empty())
			g_lastCompletedRelPath = g_currentVideoRelPath;
		g_videoWriter_online->release();
		delete g_videoWriter_online;
		g_videoWriter_online = nullptr;
	}

	// Use measured fps if available; fall back to 6fps for first clip
	double useFps = (g_lastClipActualFps > 1.0) ? g_lastClipActualFps : 6.0;

	cv::VideoWriter* writer2 = new cv::VideoWriter(
		fullPath,
		cv::VideoWriter::fourcc('V','P','8','0'),
		useFps,
		cv::Size(width, height));

	if (writer2->isOpened()) {
		g_videoWriter_online  = writer2;
		g_currentVideoRelPath = std::string(relPath);
		g_videoClipStartTick  = cv::getTickCount();
		g_videoFramesWritten  = 0; // reset counter for new clip
		OutputDebugStringA(("[VIDEO] New clip (" + std::to_string((int)useFps) + "fps): " + fullPath + "\n").c_str());
	} else {
		delete writer2;
		OutputDebugStringA("[VIDEO] Failed to open VideoWriter!\n");
	}
}

static void StopVideoRecording() {
	std::lock_guard<std::mutex> lock(g_videoWriterMutex_online);
	if (g_videoWriter_online) {
		g_videoWriter_online->release();
		delete g_videoWriter_online;
		g_videoWriter_online = nullptr;
	}
	g_currentVideoRelPath = "";
}

// Close current clip (finalizes MP4 moov atom) and return its URL.
// Starts fresh — next frame will lazily open a new clip.
static std::string FinalizeAndGetVideoClip(const std::string& serverIp) {
	std::lock_guard<std::mutex> lock(g_videoWriterMutex_online);
	if (!g_videoWriter_online || g_currentVideoRelPath.empty()) return "";

	// Release finalizes the MP4 (writes moov atom)
	g_videoWriter_online->release();
	delete g_videoWriter_online;
	g_videoWriter_online = nullptr;

	std::string relPath = g_currentVideoRelPath;
	g_currentVideoRelPath = ""; // next frame will start a new clip
	OutputDebugStringA(("[VIDEO] Finalized clip: " + relPath + "\n").c_str());
	return "http://" + serverIp + ":8080/locvideo/" + relPath;
}

// ============================================================
//  [UNMANAGED] SSE Push Helpers — JSON must be built here,
//  NOT inside a managed ref class (nlohmann JSON incompatible)
// ============================================================

static void PushStatsSSEHelper(int emptyCount, int occupiedCount, int violationCount) {
	if (!g_mjpegServer_online) return;
	try {
		json p;
		p["event"]           = "stats";
		p["camera_id"]       = "cam_01";
		p["timestamp"]       = GetISO8601Timestamp();
		p["total_slots"]     = emptyCount + occupiedCount;
		p["available_slots"] = emptyCount;
		p["occupied_slots"]  = occupiedCount;
		p["violation_slots"] = violationCount;
		g_mjpegServer_online->PushLogEvent(p.dump());
	} catch (...) {}
}

static void PushViolationSSEHelper(int carId, const std::string& typeStr,
                                   const std::string& timeStr,
                                   int clipOffsetSec, int parkingDurationSec,
                                   const std::string& imgUrl, const std::string& savedPath,
                                   const std::string& videoUrl) {
	if (!g_mjpegServer_online) return;
	try {
		json p;
		p["event"]                = "violation";
		p["car_id"]               = carId;
		p["type"]                 = typeStr;
		p["time"]                 = timeStr;
		p["clip_offset_sec"]      = clipOffsetSec;      // seek to this second in video
		p["parking_duration_sec"] = parkingDurationSec; // how long car has been parked
		p["img_url"]              = imgUrl;
		p["saved_path"]           = savedPath;
		p["video_url"]            = videoUrl.empty() ? "" : videoUrl;
		g_mjpegServer_online->PushLogEvent(p.dump());
		OutputDebugStringA(("[VIO] SSE sent car_id=" + std::to_string(carId) + "\n").c_str());
	} catch (...) {}
}

#pragma managed(pop)

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

			BackgroundWorker^ modelLoader = gcnew BackgroundWorker();
			modelLoader->DoWork += gcnew DoWorkEventHandler(this, &UploadForm::LoadModel_DoWork);
			modelLoader->RunWorkerCompleted += gcnew RunWorkerCompletedEventHandler(this, &UploadForm::LoadModel_Completed);
			modelLoader->RunWorkerAsync();
		}

	protected:
		~UploadForm() {
			StopProcessing();
			if (components) delete components;
			if (g_pm_logic_online) { delete g_pm_logic_online; g_pm_logic_online = nullptr; }
			if (g_pm_display_online) { delete g_pm_display_online; g_pm_display_online = nullptr; }
			// [PHASE 1 FIX] Don't delete managed Bitmaps - GC will handle them
			// bmpBuffer1 and bmpBuffer2 are Bitmap^ (managed objects)
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
			   this->processingWorker = (gcnew System::ComponentModel::BackgroundWorker());
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
			   // processingWorker
			   // 
			   this->processingWorker->WorkerSupportsCancellation = true;
			   this->processingWorker->DoWork += gcnew System::ComponentModel::DoWorkEventHandler(this, &UploadForm::processingWorker_DoWork);
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
		if (isBackgroundMode) return; // [NEW] Skip updating UI if in background

		// *** [STREAMING LOG] Timer counters ***
		static int sendCounter = 0;
		static int tickCounter = 0;

		try {
			cv::Mat finalFrame;
			long long seq = 0;
			GetProcessedFrameOnline(finalFrame, seq);

			if (seq == lastDisplaySeq) return;
			lastDisplaySeq = seq;

			if (!finalFrame.empty()) {
				UpdatePictureBox(finalFrame);
				// Check for violations (no need to clone, just read-only access)
				bool parkingEnabled = g_parkingEnabled_online.load();
				if (parkingEnabled) {
					CheckViolations_Online(finalFrame); // [FIX] Pass by reference, no clone
				}
			}

			// *** [NEW] UPDATE LOGS WITH CURRENT DATETIME ***
			System::DateTime now = System::DateTime::Now;
			System::String^ dateTimeStr = now.ToString(L"dd/MM/yy");
			lblLogs->Text = dateTimeStr;

			// *** UPDATE PARKING STATISTICS LABELS ***
			OnlineAppState state;
			{
				std::lock_guard<std::mutex> lock(g_onlineStateMutex);
				state = g_onlineState;
			}

			bool parkingEnabledForStats = g_parkingEnabled_online.load();
			if (parkingEnabledForStats && !state.slotStatuses.empty()) {
				int emptyCount = 0;
				int occupiedCount = 0;

				for (const auto& slotEntry : state.slotStatuses) {
					SlotStatus status = slotEntry.second;
					if (status == SlotStatus::EMPTY) emptyCount++;
					else occupiedCount++;
				}

				int violationCount = (int)state.violatingCarIds.size();

				label5_online->Text = System::String::Format(L"Empty: {0}", emptyCount);
				label6_online->Text = System::String::Format(L"Normal: {0}", occupiedCount);
				label7_online->Text = System::String::Format(L"Violation: {0}", violationCount);

				// *** [STREAMING LOG] SEND HTTP POST EVERY 500ms using mutex-protected globals ***
				long long nowTick = cv::getTickCount();
				double elapsedMs = (nowTick - g_lastLogSendTick_online) /
					               cv::getTickFrequency() * 1000.0;

				if (elapsedMs >= 500.0) {
					g_lastLogSendTick_online = nowTick;
					sendCounter++;

					// Read latest stats + frame from globals under lock
					int snapEmpty, snapOccupied, snapViolation;
					cv::Mat snapFrame;
					{
						std::lock_guard<std::mutex> lock(g_logStatsMutex_online);
						snapEmpty     = g_logStats_online.empty;
						snapOccupied  = g_logStats_online.occupied;
						snapViolation = g_logStats_online.violation;
						// Only clone frame when there is a violation (CPU optimization)
						if (snapViolation > 0 && !g_logStats_online.lastFrame.empty()) {
							snapFrame = g_logStats_online.lastFrame.clone();
						}
					}

					// Async HTTP POST - snapFrame is empty when no violations (no base64)
					SendAggregateStatusUpdate(snapEmpty, snapOccupied,
											  snapViolation,
											  snapFrame,   // empty Mat → no base64 encoded
											  "cam_01",
											  "127.0.0.1",
											  9000);

					// *** [SSE] Push JSON log to all browser clients — ONLY if values changed ***
					if (g_mjpegServer_online) {
						bool changed = (snapEmpty    != g_lastSentStats_online.empty ||
										snapOccupied != g_lastSentStats_online.occupied ||
										snapViolation!= g_lastSentStats_online.violation);
						if (changed) {
							g_lastSentStats_online.empty    = snapEmpty;
							g_lastSentStats_online.occupied = snapOccupied;
							g_lastSentStats_online.violation= snapViolation;

							// Call unmanaged helper to build+push JSON (nlohmann JSON not allowed in managed class)
							PushStatsSSEHelper(snapEmpty, snapOccupied, snapViolation);
						}
					}

					OutputDebugStringA(("[STREAM] POST #" + std::to_string(sendCounter) +
									    " | E:" + std::to_string(snapEmpty) +
									    " O:" + std::to_string(snapOccupied) +
									    " V:" + std::to_string(snapViolation) +
									    (snapViolation > 0 ? " [+img]" : "") +
									    "\n").c_str());
				}
			}

			// *** DEBUG LOGGING - Every ~1 second ***
			tickCounter++;
			if (tickCounter % 30 == 0) {
				OutputDebugStringA(("[TIMER] FPS: " +
									std::to_string((int)g_fpsMonitor_online.avgFPS) + "\n").c_str());
			}
		}
		catch (...) {}
	}
	private: void StopProcessing() {
		shouldStop = true;
		isProcessing = false;
		timer1->Stop();
		if (processingWorker->IsBusy) processingWorker->CancelAsync();

		// Stop video recording
		StopVideoRecording();

		// *** [NEW] Stop MJPEG Server ***
		if (g_mjpegServer_online) {
			g_mjpegServer_online->Stop();
			delete g_mjpegServer_online;
			g_mjpegServer_online = nullptr;
		}

		if (btnRunInBackground) btnRunInBackground->Enabled = false;
		if (lblNetworkStream) lblNetworkStream->Text = "Stream: Offline";
	}

	private: System::Void LoadModel_DoWork(System::Object^ sender, DoWorkEventArgs^ e) {
		try {
			std::string modelPath = "models/test/yolo11n.onnx";
			InitGlobalModel(modelPath);
			e->Result = true;
		}
		catch (const std::exception& ex) { e->Result = gcnew System::String(ex.what()); }
	}

	private: System::Void LoadModel_Completed(System::Object^ sender, RunWorkerCompletedEventArgs^ e) {
		if (e->Result != nullptr && e->Result->GetType() == bool::typeid && safe_cast<bool>(e->Result)) {
			this->Text = L"Online Mode - YOLO Detection (Ready)";
			// [UI FIX] Only enable template button, Live Camera stays disabled
			btnLoadParkingTemplate->Enabled = true;
			MessageBox::Show("Model loaded!\n\n⚠️ Please load a parking template before starting live camera.", "Success", MessageBoxButtons::OK, MessageBoxIcon::Information);
		}
		else {
			MessageBox::Show("Error loading model", "Error", MessageBoxButtons::OK, MessageBoxIcon::Error);
		}
	}

	// *** [NEW] Get Local IP helper ***
	private: std::string GetLocalIP() {
		System::String^ bestIP = "127.0.0.1";
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

	private: void StartProcessing() {
		shouldStop = false;
		isProcessing = true;
		
		// [PHASE 2] Reset performance counters
		g_droppedFrames_online = 0;
		g_processedFramesCount_online = 0;
		g_lastViolationCheck_online = std::chrono::steady_clock::now();
		
		{
			std::lock_guard<std::mutex> lock(g_onlineStateMutex);
			g_onlineState = OnlineAppState();
		}
		ResetParkingCache_Online();
		violationsList_online->Clear();
		violatingCarTimers_online->Clear();

		if (!processingWorker->IsBusy) processingWorker->RunWorkerAsync();

		if (readerThread == nullptr || !readerThread->IsAlive) {
			readerThread = gcnew Thread(gcnew ThreadStart(this, &UploadForm::CameraReaderLoop));
			readerThread->IsBackground = true;
			readerThread->Start();
		}

		timer1->Start();
		
		btnRunInBackground->Enabled = true;

		// *** [NEW] Start MJPEG Server ***
		if (!g_mjpegServer_online) {
			g_mjpegServer_online = new MjpegServer(8080);
			if (g_mjpegServer_online->Start()) {
				std::string ip = GetLocalIP();
				lblNetworkStream->Text = gcnew String(("Open: http://" + ip + ":8080  |  Stream: /stream  |  Logs: /events").c_str());
			} else {
				lblNetworkStream->Text = "Stream: Failed to start";
				delete g_mjpegServer_online;
				g_mjpegServer_online = nullptr;
			}
		}
	}

	private: void CameraReaderLoop() {
		double ticksPerFrame = 1000.0 / 30.0;
		if (g_cameraFPS > 0) ticksPerFrame = 1000.0 / g_cameraFPS;

		long long nextTick = cv::getTickCount();
		double tickFreq = cv::getTickFrequency();

		while (!shouldStop) {
			long long currentTick = cv::getTickCount();
			if (currentTick < nextTick) {
				Threading::Thread::Sleep(1);
				continue;
			}

			cv::Mat tempFrame;
			bool success = false;

			if (g_cap && g_cap->isOpened()) {
				success = g_cap->read(tempFrame);
			}
			else {
				break;
			}

			if (success && !tempFrame.empty()) {
				long long currentSeq;
				{
					std::lock_guard<std::mutex> lock(g_frameMutex);
					g_latestRawFrame = tempFrame; // [FIX] Use shallow copy (swap later)
					g_frameSeq_online++;
					currentSeq = g_frameSeq_online;
				}

				// [FIX] Relaxed frame drop logic (10 frames tolerance instead of 3)
				bool shouldDrop = false;
				{
					std::lock_guard<std::mutex> lock(g_processedMutex_online);
					if (currentSeq - g_processedSeq_online > MAX_FRAME_LAG_ONLINE) {
						shouldDrop = true;
						g_droppedFrames_online++;
					}
				}

				if (!shouldDrop) {
					cv::Mat renderedFrame;
					DrawSceneOnline(tempFrame, currentSeq, renderedFrame);

					if (!renderedFrame.empty()) {
						std::lock_guard<std::mutex> lock(g_processedMutex_online);
						g_processedFrame_online = renderedFrame; // [FIX] Shallow copy
						g_processedSeq_online = currentSeq;
						g_processedFramesCount_online++;

						// [FIX] Update MJPEG Server inside the loop so it streams out!
						if (g_mjpegServer_online) {
							g_mjpegServer_online->SetLatestFrame(g_processedFrame_online);
						}

						// *** [STREAMING LOG] Write latest stats to globals under mutex (every frame, no HTTP) ***
						{
							OnlineAppState snapState;
							{
								std::lock_guard<std::mutex> sLock(g_onlineStateMutex);
								snapState = g_onlineState;
							}
							int em = 0, oc = 0;
							for (const auto& s : snapState.slotStatuses) {
								if (s.second == SlotStatus::EMPTY) em++; else oc++;
							}
							int vi = (int)snapState.violatingCarIds.size();

							std::lock_guard<std::mutex> lg(g_logStatsMutex_online);
							g_logStats_online.empty     = em;
							g_logStats_online.occupied  = oc;
							g_logStats_online.violation = vi;
							// Store latest frame (shallow copy, cheap)
							g_logStats_online.lastFrame = g_processedFrame_online;
						}

						// *** [VIDEO] Write frame to current clip ***
					{
						bool needNewClip = false;
						{
							std::lock_guard<std::mutex> vl(g_videoWriterMutex_online);
							double elapsed = (cv::getTickCount() - g_videoClipStartTick) / cv::getTickFrequency();
							needNewClip = (!g_videoWriter_online || !g_videoWriter_online->isOpened() || elapsed >= VIDEO_CLIP_SECONDS);
						}
						if (needNewClip) {
							double fps = g_cameraFPS > 0 ? g_cameraFPS : 30.0;
							StartNewVideoClip(renderedFrame.cols, renderedFrame.rows, fps);
						}
						std::lock_guard<std::mutex> vl2(g_videoWriterMutex_online);
						if (g_videoWriter_online && g_videoWriter_online->isOpened()) {
							g_videoWriter_online->write(renderedFrame);
							g_videoFramesWritten++;
						}
					}
					}
				}

				nextTick += (long long)(ticksPerFrame * tickFreq / 1000.0);
				if (cv::getTickCount() > nextTick) nextTick = cv::getTickCount();
			}
			else {
				if (g_cap && g_cap->isOpened()) {
					// [FIX] Don't break immediately, might be temporary network hiccup
					Threading::Thread::Sleep(50);
					continue;
				}
				break;
			}
		}
	}

private: System::Void processingWorker_DoWork(System::Object^ sender, DoWorkEventArgs^ e) {
	BackgroundWorker^ worker = safe_cast<BackgroundWorker^>(sender);
	lastProcessedSeq = -1;
	while (!shouldStop && !worker->CancellationPending) {
		try {
			cv::Mat frameToProcess;
			long long seq = 0;
			GetRawFrameOnline(frameToProcess, seq);

			if (!frameToProcess.empty() && seq > lastProcessedSeq) {
				ProcessFrameOnline(frameToProcess, seq);
				lastProcessedSeq = seq;
			}
			else {
				Threading::Thread::Sleep(10);
			}
		}
		catch (...) { Threading::Thread::Sleep(50); }
	}
}

private: System::Void btnLiveCamera_Click(System::Object^ sender, System::EventArgs^ e) {
	// [UI FIX] Check if template is loaded before proceeding
	if (!g_parkingEnabled_online.load() || !g_pm_logic_online || !g_pm_display_online) {
		MessageBox::Show(
			"⚠️ Parking template not loaded!\n\n" +
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

	StopProcessing();
	
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

			this->Text = L"Online Mode - Connecting to camera...";
			Application::DoEvents();

			bool connected = false;
			String^ successUrl = "";

			for each (String^ streamUrl in urlFormats) {
				std::string url = msclr::interop::marshal_as<std::string>(streamUrl);
				
				OutputDebugStringA(("[INFO] Trying to connect: " + url + "\n").c_str());
				
				OpenGlobalCameraFromIP(url);
				Threading::Thread::Sleep(1000);

				if (g_cap && g_cap->isOpened()) {
					cv::Mat testFrame;
					bool canRead = false;
					{
						std::lock_guard<std::mutex> lock(g_frameMutex);
						if (g_cap->read(testFrame)) {
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
					std::lock_guard<std::mutex> lock(g_frameMutex);
					if (g_cap) {
						delete g_cap;
						g_cap = nullptr;
					}
				}
			}

			if (connected) {
				StartProcessing();
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
				this->Text = L"Online Mode - Connection Failed";
				
				String^ errorMsg = "Failed to connect to mobile camera!\n\n";
				errorMsg += "Troubleshooting Steps:\n";
				errorMsg += "1. Verify IP Address: " + ip + "\n";
				errorMsg += "2. Verify Port: " + port + "\n";
				errorMsg += "3. Check if camera app is running on mobile\n";
				errorMsg += "4. Ensure both devices are on the same WiFi network\n";
				errorMsg += "5. Check firewall settings\n";
				errorMsg += "6. Try disabling antivirus temporarily\n\n";
				errorMsg += "Attempted URLs:\n";
				for each (String^ url in urlFormats) {
					errorMsg += "  - " + url + "\n";
				}
				
				MessageBox::Show(
					errorMsg,
					"Connection Error",
					MessageBoxButtons::OK,
					MessageBoxIcon::Error
				);
			}
		}
		catch (Exception^ ex) {
			this->Text = L"Online Mode - Error Occurred";
			MessageBox::Show(
				"An error occurred while connecting:\n\n" + 
				ex->Message + "\n\n" +
				"Stack Trace:\n" + ex->StackTrace,
				"Exception Error",
				MessageBoxButtons::OK,
				MessageBoxIcon::Error
			);
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
		if (LoadParkingTemplate_Online(fileName)) {
			chkParkingMode->Checked = true;
			
			// [UI FIX] Enable Live Camera button after successful template load
			btnLiveCamera->Enabled = true;
			btnLiveCamera->BackColor = System::Drawing::Color::Tomato;
			
			MessageBox::Show(
				"✅ Template loaded successfully!\n\n" +
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
	g_parkingEnabled_online.store(chkParkingMode->Checked); // [PHASE 1 FIX] Use atomic store
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
	StopProcessing();
}

	// *** [NEW] VIOLATION ALERTS METHODS ***
	private: void AddViolationRecord_Online(int carId, cv::Mat& frameCapture, System::String^ violationType, 
									 cv::Mat fullFrame, cv::Rect carBox,
									 int parkingDurationSec, int clipOffsetSec) {
		if (frameCapture.empty()) return;

		Bitmap^ screenshot = gcnew Bitmap(frameCapture.cols, frameCapture.rows, System::Drawing::Imaging::PixelFormat::Format24bppRgb);
		System::Drawing::Rectangle rect(0, 0, frameCapture.cols, frameCapture.rows);
		System::Drawing::Imaging::BitmapData^ bmpData = screenshot->LockBits(rect, System::Drawing::Imaging::ImageLockMode::WriteOnly, screenshot->PixelFormat);

		for (int y = 0; y < frameCapture.rows; y++) {
			memcpy((unsigned char*)bmpData->Scan0.ToPointer() + y * bmpData->Stride, frameCapture.data + y * frameCapture.step, frameCapture.cols * 3);
		}
		screenshot->UnlockBits(bmpData);

		cv::Mat visualizationMat = CreateViolationVisualization(fullFrame, carBox);
		Bitmap^ visualizationBitmap = nullptr;
		if (!visualizationMat.empty()) {
			visualizationBitmap = gcnew Bitmap(visualizationMat.cols, visualizationMat.rows, System::Drawing::Imaging::PixelFormat::Format24bppRgb);
			System::Drawing::Rectangle visRect(0, 0, visualizationMat.cols, visualizationMat.rows);
			System::Drawing::Imaging::BitmapData^ visBmpData = visualizationBitmap->LockBits(visRect, System::Drawing::Imaging::ImageLockMode::WriteOnly, visualizationBitmap->PixelFormat);

			for (int y = 0; y < visualizationMat.rows; y++) {
				memcpy((unsigned char*)visBmpData->Scan0.ToPointer() + y * visBmpData->Stride, visualizationMat.data + y * visualizationMat.step, visualizationMat.cols * 3);
			}
			visualizationBitmap->UnlockBits(visBmpData);
		}

		ViolationRecord_Online^ record = gcnew ViolationRecord_Online();
		record->carId = carId;
		record->screenshot = screenshot;
		record->visualizationBitmap = visualizationBitmap;
		record->violationType = violationType;
		record->captureTime = System::DateTime::Now;
		record->durationSeconds = parkingDurationSec;

		violationsList_online->Add(record);
		RefreshViolationPanel_Online();

		// *** [VIOLATION LOG] Save image + push SSE event ***
		std::string typeStr = msclr::interop::marshal_as<std::string>(violationType);
		cv::Mat frameToSave = visualizationMat.empty() ? fullFrame : visualizationMat;
		std::string savedFile = SaveViolationImage(frameToSave, carId, typeStr);

		if (!savedFile.empty() && g_mjpegServer_online) {
			std::string serverIp = GetLocalIP();
			std::string imgUrl   = "http://" + serverIp + ":8080/violations/" + savedFile;
			std::string timeStr  = msclr::interop::marshal_as<std::string>(record->captureTime.ToString("HH:mm:ss"));
			std::string savedPath = "C:\\logpic\\" + savedFile;

			// video_url = CURRENT clip (contains this violation at clip_offset_sec)
			// Link becomes playable when clip rotates after 60 seconds
			std::string videoUrl = "";
			{
				std::lock_guard<std::mutex> vlock(g_videoWriterMutex_online);
				if (!g_currentVideoRelPath.empty())
					videoUrl = "http://" + serverIp + ":8080/locvideo/" + g_currentVideoRelPath;
			}

			PushViolationSSEHelper(carId, typeStr, timeStr,
				clipOffsetSec, parkingDurationSec, imgUrl, savedPath, videoUrl);
		}
	}

	private: void RefreshViolationPanel_Online() {
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

private: void CheckViolations_Online(cv::Mat& currentFrame) {
			// [PHASE 2] Throttle violation checks to 500ms
		auto now = std::chrono::steady_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - g_lastViolationCheck_online).count();
		
		if (elapsed < VIOLATION_CHECK_INTERVAL_MS_ONLINE) {
			return;
		}
		g_lastViolationCheck_online = now;

		OnlineAppState state;
		{
			std::lock_guard<std::mutex> lock(g_onlineStateMutex);
			state = g_onlineState;
		}

		for each(auto car in state.cars) {
			if (car.framesStill > 300) {
				if (!violatingCarTimers_online->ContainsKey(car.id)) {
					violatingCarTimers_online->Add(car.id, System::DateTime::Now);
					
					cv::Rect safeBbox = car.bbox & cv::Rect(0, 0, currentFrame.cols, currentFrame.rows);
				if (safeBbox.area() > 0) {
						cv::Mat croppedFrame = currentFrame(safeBbox).clone();
						int parkingSec = (int)(car.framesStill / (g_cameraFPS > 0 ? g_cameraFPS : 30.0));
						int clipOffset = (g_videoClipStartTick > 0)
							? (int)((cv::getTickCount() - g_videoClipStartTick) / cv::getTickFrequency())
							: 0;
						AddViolationRecord_Online(car.id, croppedFrame, L"Overstay", currentFrame, car.bbox, parkingSec, clipOffset);
					}
				}
			}
		}

		for each(int violatingId in state.violatingCarIds) {
			bool already_captured = false;
			for each(ViolationRecord_Online^ record in violationsList_online) {
				if (record->carId == violatingId && record->violationType == L"Wrong Slot") {
					already_captured = true;
					break;
				}
			}

			if (!already_captured) {
				for each(auto car in state.cars) {
					if (car.id == violatingId) {
						cv::Rect safeBbox = car.bbox & cv::Rect(0, 0, currentFrame.cols, currentFrame.rows);
						if (safeBbox.area() > 0) {
							cv::Mat croppedFrame = currentFrame(safeBbox).clone();
							int parkingSec = (int)(car.framesStill / (g_cameraFPS > 0 ? g_cameraFPS : 30.0));
							int clipOffset = (g_videoClipStartTick > 0)
								? (int)((cv::getTickCount() - g_videoClipStartTick) / cv::getTickFrequency())
								: 0;
							AddViolationRecord_Online(violatingId, croppedFrame, L"Wrong Slot", currentFrame, car.bbox, parkingSec, clipOffset);
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
		StopProcessing();
		this->Close();
	}
};
}