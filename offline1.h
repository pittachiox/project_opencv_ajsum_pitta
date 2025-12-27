#pragma once
#include <msclr/marshal_cppstd.h>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <direct.h>  // For _getcwd
#include "BYTETracker.h"
#include "ParkingSlot.h"

#pragma managed(push, off)
#define NOMINMAX
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <mutex>
#include <thread>
#include <chrono>

// ==========================================
//  PART 1: GLOBAL VARIABLES & SETTINGS
// ==========================================

// Settings
static const int YOLO_SIZE = 640;
static const float CONF_THRESH = 0.25f;
static const float NMS_THRESH = 0.45f;

// Shared State Structure
struct AppState {
	std::vector<TrackedObject> cars;
	std::set<int> violatingCarIds;
	std::map<int, SlotStatus> slotStatuses;
	std::map<int, float> slotOccupancy;
	long long frameSequence = -1;
};

static struct {
	cv::Mat frame;
	long long sequence;
}g_latestFrameInfo;

// Global Objects
static AppState g_appState;
static std::mutex g_stateMutex;

static cv::dnn::Net* g_net_offline = nullptr;
static std::vector<std::string> g_classes_offline;
static std::vector<cv::Scalar> g_colors_offline;
static BYTETracker* g_tracker_offline = nullptr;
static ParkingManager* g_pm_logic = nullptr;
static ParkingManager* g_pm_display = nullptr;

// Video & Sync Resources
static cv::VideoCapture* g_cap_offline = nullptr;
static cv::Mat g_latestRawFrame_offline;
static long long g_frameSeq_offline = 0;
static std::mutex g_frameMutex_offline;
static double g_videoFPS = 30.0;

static std::mutex g_aiMutex_offline;
static bool g_modelReady_offline = false;
static bool g_parkingEnabled_offline = false;

// Drawing Cache
static cv::Mat g_cachedParkingOverlay;
static std::map<int, SlotStatus> g_lastDrawnStatus;
static cv::Mat g_drawingBuffer; // Memory Pool

// *** [NEW] PROCESSED FRAME SHARING (Pipeline Output) ***
static cv::Mat g_processedFrame_shared;
static long long g_processedSeq_shared = 0;
static std::mutex g_processedMutex;

// ==========================================
//  PART 2: HELPER FUNCTIONS (STATIC)
// ==========================================

static void ResetParkingCache() {
	g_cachedParkingOverlay = cv::Mat();
	g_lastDrawnStatus.clear();
}

static cv::Mat FormatToLetterboxOffline(const cv::Mat& source, int width, int height, float& ratio, int& dw, int& dh) {
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

static void InitBackend(const std::string& modelPath) {
	std::lock_guard<std::mutex> lock(g_aiMutex_offline);
	g_modelReady_offline = false;
	if (g_net_offline) { delete g_net_offline; g_net_offline = nullptr; }
	if (g_tracker_offline) { delete g_tracker_offline; g_tracker_offline = nullptr; }

	try {
		g_net_offline = new cv::dnn::Net(cv::dnn::readNetFromONNX(modelPath));
		g_net_offline->setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
		g_net_offline->setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
		g_tracker_offline = new BYTETracker(90, 0.25f);
		g_classes_offline = { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow" };
		g_colors_offline.clear();
		for (size_t i = 0; i < 100; i++) g_colors_offline.push_back(cv::Scalar(rand() % 255, rand() % 255, rand() % 255));
		g_modelReady_offline = true;
	}
	catch (...) {}
}

static bool LoadParkingTemplate(const std::string& filename) {
	ResetParkingCache();
	if (!g_pm_logic) g_pm_logic = new ParkingManager();
	if (!g_pm_display) g_pm_display = new ParkingManager();
	bool s1 = g_pm_logic->loadTemplate(filename);
	bool s2 = g_pm_display->loadTemplate(filename);
	if (s1 && s2) {
		g_parkingEnabled_offline = true;
		return true;
	}
	return false;
}

static void GetRawFrame(cv::Mat& outFrame, long long& outSeq) {
	std::lock_guard<std::mutex> lock(g_frameMutex_offline);
	if (!g_latestRawFrame_offline.empty()) {
		outFrame = g_latestRawFrame_offline;
		outSeq = g_frameSeq_offline;
	}
}

// *** [NEW] GET PROCESSED FRAME (For UI) ***
static void GetProcessedFrame(cv::Mat& outFrame, long long& outSeq) {
	std::lock_guard<std::mutex> lock(g_processedMutex);
	if (!g_processedFrame_shared.empty()) {
		outFrame = g_processedFrame_shared;
		outSeq = g_processedSeq_shared;
	}
}

static void OpenCamera(const std::string& filename) {
	std::lock_guard<std::mutex> lock(g_frameMutex_offline);
	if (g_cap_offline) { delete g_cap_offline; g_cap_offline = nullptr; }
	g_cap_offline = new cv::VideoCapture(filename);
	if (g_cap_offline->isOpened()) {
		g_videoFPS = g_cap_offline->get(cv::CAP_PROP_FPS);
		if (g_videoFPS <= 0 || g_videoFPS > 60) g_videoFPS = 30.0;
	}
	g_frameSeq_offline = 0;

	std::lock_guard<std::mutex> slock(g_stateMutex);
	g_appState = AppState();
	ResetParkingCache();
}

// *** WORKER PROCESS ***
static void ProcessFrame(const cv::Mat& inputFrame, long long frameSeq) {
	{
		std::lock_guard<std::mutex> lock(g_aiMutex_offline);
		if (inputFrame.empty() || !g_net_offline || !g_modelReady_offline || !g_tracker_offline) return;
	}

	try {
		cv::Mat workingImage = inputFrame.clone();
		float ratio; int dw, dh;
		cv::Mat input_image = FormatToLetterboxOffline(workingImage, YOLO_SIZE, YOLO_SIZE, ratio, dw, dh);
		if (input_image.empty()) return;

		cv::Mat blob;
		cv::dnn::blobFromImage(input_image, blob, 1.0 / 255.0, cv::Size(YOLO_SIZE, YOLO_SIZE), cv::Scalar(), true, false);

		std::vector<cv::Mat> outputs;
		{
			std::lock_guard<std::mutex> lock(g_aiMutex_offline);
			g_net_offline->setInput(blob);
			g_net_offline->forward(outputs, g_net_offline->getUnconnectedOutLayersNames());
		}

		if (outputs.empty() || outputs[0].empty()) return;

		cv::Mat output_data = outputs[0];
		int rows = output_data.size[1];
		int dimensions = output_data.size[2];
		if (output_data.dims == 3) {
			output_data = output_data.reshape(1, rows);
			cv::transpose(output_data, output_data);
			rows = output_data.rows; dimensions = output_data.cols;
		}
		else {
			cv::Mat output_t;
			cv::transpose(output_data.reshape(1, output_data.size[1]), output_t);
			output_data = output_t;
		rows = output_data.rows; dimensions = output_data.cols;
		}

		float* data = (float*)output_data.data;
		std::vector<int> class_ids; std::vector<float> confs; std::vector<cv::Rect> boxes;
		for (int i = 0; i < rows; i++) {
			float* classes_scores = data + 4;
			if (dimensions >= 4 + (int)g_classes_offline.size()) {
				cv::Mat scores(1, (int)g_classes_offline.size(), CV_32FC1, classes_scores);
				cv::Point class_id; double max_class_score;
				cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
				if (max_class_score > CONF_THRESH) {
					float x = data[0]; float y = data[1]; float w = data[2]; float h = data[3];
					float left = (x - 0.5 * w - dw) / ratio; float top = (y - 0.5 * h - dh) / ratio;
					float width = w / ratio; float height = h / ratio;
					boxes.push_back(cv::Rect((int)left, (int)top, (int)width, (int)height));
					confs.push_back((float)max_class_score);
					class_ids.push_back(class_id.x);
				}
			}
		 data += dimensions;
		}
	 std::vector<int> nms;
	 cv::dnn::NMSBoxes(boxes, confs, CONF_THRESH, NMS_THRESH, nms);

		std::vector<cv::Rect> nms_boxes; std::vector<int> nms_class_ids; std::vector<float> nms_confs;
		for (int idx : nms) {
			nms_boxes.push_back(boxes[idx]); nms_class_ids.push_back(class_ids[idx]); nms_confs.push_back(confs[idx]);
		}

		std::vector<TrackedObject> trackedObjs;
		{
			std::lock_guard<std::mutex> lock(g_aiMutex_offline);
			trackedObjs = g_tracker_offline->update(nms_boxes, nms_class_ids, nms_confs);
		}

		std::map<int, SlotStatus> calculatedStatuses;
		std::map<int, float> calculatedOccupancy;
		std::set<int> violations;

		if (g_parkingEnabled_offline && g_pm_logic) {
			static bool templateSet = false;
			if (!templateSet) {
				g_pm_logic->setTemplateFrame(inputFrame);
				templateSet = true;
			}

			g_pm_logic->updateSlotStatus(trackedObjs);
			for (const auto& slot : g_pm_logic->getSlots()) {
				calculatedStatuses[slot.id] = slot.status;
				calculatedOccupancy[slot.id] = slot.occupancyPercent;
			}

			for (const auto& car : trackedObjs) {
				if (car.framesStill > 30) {
					bool inAnySlot = false;
					for (const auto& slot : g_pm_logic->getSlots()) {
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
			std::lock_guard<std::mutex> stateLock(g_stateMutex);
			g_appState.cars = trackedObjs;
		g_appState.slotStatuses = calculatedStatuses;
			g_appState.slotOccupancy = calculatedOccupancy;
			g_appState.violatingCarIds = violations;
			g_appState.frameSequence = frameSeq;
		}
	}
	catch (...) {}
}

// *** DRAWING FUNCTION ***
static void DrawScene(const cv::Mat& frame, long long displaySeq, cv::Mat& outResult) {
	if (frame.empty()) return;

	// Reuse Buffer
	if (g_drawingBuffer.size() != frame.size() || g_drawingBuffer.type() != frame.type()) {
		g_drawingBuffer.create(frame.size(), frame.type());
	}
	frame.copyTo(g_drawingBuffer);
	outResult = g_drawingBuffer;

	AppState state;
	{
		std::lock_guard<std::mutex> lock(g_stateMutex);
		state = g_appState;
	}

	bool isFuture = (state.frameSequence > displaySeq);

	// Parking Layer
	if (g_parkingEnabled_offline && g_pm_display) {
		bool statusChanged = (state.slotStatuses != g_lastDrawnStatus);
		bool noCache = g_cachedParkingOverlay.empty() || g_cachedParkingOverlay.size() != outResult.size();

		if (statusChanged || noCache) {
			g_cachedParkingOverlay = cv::Mat::zeros(outResult.size(), CV_8UC3);
			if (!state.slotStatuses.empty()) {
				auto& displaySlots = g_pm_display->getSlots();
				for (auto& slot : displaySlots) {
					if (state.slotStatuses.count(slot.id)) {
						slot.status = state.slotStatuses[slot.id];
						slot.occupancyPercent = state.slotOccupancy[slot.id];
					}
				}
			}
			g_cachedParkingOverlay = g_pm_display->drawSlots(g_cachedParkingOverlay);
			g_lastDrawnStatus = state.slotStatuses;
		}

		if (!g_cachedParkingOverlay.empty()) {
			cv::add(outResult, g_cachedParkingOverlay, outResult);
		}
	}

	// Car Layer
	if (!isFuture) {
		for (const auto& obj : state.cars) {
			if (obj.classId >= 0 && obj.classId < g_classes_offline.size()) {
				cv::Rect box = obj.bbox;
				bool isViolating = (state.violatingCarIds.count(obj.id) > 0);

				if (isViolating) {
					// Filled Violation Box
					cv::Rect roi = box & cv::Rect(0, 0, outResult.cols, outResult.rows);
					if (roi.area() > 0) {
						cv::Mat roiMat = outResult(roi);
						cv::Mat colorBlock(roi.size(), CV_8UC3, cv::Scalar(0, 0, 255));
						cv::addWeighted(roiMat, 0.6, colorBlock, 0.4, 0, roiMat);
					}
					cv::rectangle(outResult, box, cv::Scalar(0, 0, 255), 2);
				}
				else {
					cv::rectangle(outResult, box, g_colors_offline[obj.classId], 2);
				}

				std::string label = "ID:" + std::to_string(obj.id);
				if (isViolating) label += " [!]";
				else if (!g_parkingEnabled_offline) label += " " + g_classes_offline[obj.classId];

				int baseline;
				cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
				cv::Scalar labelBg = isViolating ? cv::Scalar(0, 0, 255) : g_colors_offline[obj.classId];

				cv::rectangle(outResult, cv::Point(box.x, box.y - textSize.height - 5), cv::Point(box.x + textSize.width, box.y), labelBg, -1);
				cv::putText(outResult, label, cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
			}
		}
	}

	std::string stats = "Obj: " + std::to_string(state.cars.size());
	cv::putText(outResult, stats, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
}

#pragma managed(pop)

namespace ConsoleApplication3 {
	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Windows::Forms;
	using namespace System::Drawing;
	using namespace System::Threading;

	public ref class OfflineUploadForm : public System::Windows::Forms::Form
	{
	public:
		OfflineUploadForm(void) {
			InitializeComponent();
			bufferLock = gcnew Object();
			bmpBuffer1 = nullptr;
			bmpBuffer2 = nullptr;
			useBuffer1 = true;
			
			isProcessing = false;
			shouldStop = false;
			violationsList = gcnew System::Collections::Generic::List<ViolationRecord^>();
			violatingCarTimers = gcnew System::Collections::Generic::Dictionary<int, System::DateTime>();

			BackgroundWorker^ modelLoader = gcnew BackgroundWorker();
			modelLoader->DoWork += gcnew DoWorkEventHandler(this, &OfflineUploadForm::LoadModel_DoWork);
			modelLoader->RunWorkerCompleted += gcnew RunWorkerCompletedEventHandler(this, &OfflineUploadForm::LoadModel_Completed);
			modelLoader->RunWorkerAsync();
		}

	protected:
		~OfflineUploadForm() {
			StopProcessing();
			if (components) delete components;
			if (g_pm_logic) { delete g_pm_logic; g_pm_logic = nullptr; }
			if (g_pm_display) { delete g_pm_display; g_pm_display = nullptr; }
			if (bmpBuffer1) delete bmpBuffer1;
			if (bmpBuffer2) delete bmpBuffer2;
		}

	private: System::Windows::Forms::Timer^ timer1;

	private: BackgroundWorker^ processingWorker;
	private: Thread^ readerThread;

	private: System::ComponentModel::IContainer^ components;

	private: Bitmap^ bmpBuffer1;
	private: Bitmap^ bmpBuffer2;
	private: bool useBuffer1;

	private: Object^ bufferLock;
	private: bool isProcessing;


	private: System::Windows::Forms::Button^ btnPlayPause;
	private: System::Windows::Forms::TrackBar^ trackBar1;
	private: System::Windows::Forms::Button^ btnNextFrame;
	private: System::Windows::Forms::Button^ btnOfflineMode;
	private: System::Windows::Forms::Button^ btnPrevFrame;


	private: System::Windows::Forms::Button^ btnUploadImage;
	private: System::Windows::Forms::Button^ btnUploadVideo;
	private: System::Windows::Forms::Button^ btnLoadParkingTemplate;
	private: System::Windows::Forms::CheckBox^ chkParkingMode;

	private: System::Windows::Forms::Label^ lblLogs;
	private: bool shouldStop;
	private: long long lastProcessedSeq = -1;
	private: System::Windows::Forms::SplitContainer^ splitContainer1;
	private: System::Windows::Forms::PictureBox^ pictureBox1;
	private: System::Windows::Forms::Panel^ panel1;
	private: System::Windows::Forms::Label^ label1;





	private: long long lastDisplaySeq = -1;
	private: System::Windows::Forms::Timer^ cameraInitTimer;
	private: int cameraInitCountdown = 0;

	// *** [NEW] TRACKBAR CONTROL VARIABLES ***
	private: long long totalFrames = 0;
	private: bool isTrackBarDragging = false;
	private: bool isPlayingVideo = false;

	// *** [NEW] VIOLATION ALERTS SYSTEM ***
	private: ref struct ViolationRecord {
		int carId;
		Bitmap^ screenshot;
		System::String^ violationType;
		System::DateTime captureTime;
		int durationSeconds;
	};

	private: System::Collections::Generic::List<ViolationRecord^>^ violationsList;
	private: System::Windows::Forms::Panel^ pnlViolationContainer;
	private: System::Windows::Forms::FlowLayoutPanel^ flpViolations;
	private: System::Windows::Forms::Label^ lblViolationTitle;
	private: System::Windows::Forms::Label^ lblViolationCount;
	private: System::Windows::Forms::Button^ btnClearViolations;
	private: System::Windows::Forms::Label^ label7;
	private: System::Windows::Forms::Label^ label6;
	private: System::Windows::Forms::Label^ label5;


	private: System::Collections::Generic::Dictionary<int, System::DateTime>^ violatingCarTimers;

#pragma region Windows Form Designer generated code
		   void InitializeComponent(void) {
			   this->components = (gcnew System::ComponentModel::Container());
			   this->timer1 = (gcnew System::Windows::Forms::Timer(this->components));
			   this->processingWorker = (gcnew System::ComponentModel::BackgroundWorker());
			   this->btnPrevFrame = (gcnew System::Windows::Forms::Button());
			   this->btnNextFrame = (gcnew System::Windows::Forms::Button());
			   this->btnOfflineMode = (gcnew System::Windows::Forms::Button());
			   this->btnPlayPause = (gcnew System::Windows::Forms::Button());
			   this->trackBar1 = (gcnew System::Windows::Forms::TrackBar());
			   this->lblLogs = (gcnew System::Windows::Forms::Label());
			   this->btnUploadImage = (gcnew System::Windows::Forms::Button());
			   this->btnUploadVideo = (gcnew System::Windows::Forms::Button());
			   this->btnLoadParkingTemplate = (gcnew System::Windows::Forms::Button());
			   this->chkParkingMode = (gcnew System::Windows::Forms::CheckBox());
			   this->splitContainer1 = (gcnew System::Windows::Forms::SplitContainer());
			   this->label1 = (gcnew System::Windows::Forms::Label());
			   this->pictureBox1 = (gcnew System::Windows::Forms::PictureBox());
			   this->label7 = (gcnew System::Windows::Forms::Label());
			   this->label6 = (gcnew System::Windows::Forms::Label());
			   this->label5 = (gcnew System::Windows::Forms::Label());
			   this->pnlViolationContainer = (gcnew System::Windows::Forms::Panel());
			   this->flpViolations = (gcnew System::Windows::Forms::FlowLayoutPanel());
			   this->btnClearViolations = (gcnew System::Windows::Forms::Button());
			   this->lblViolationCount = (gcnew System::Windows::Forms::Label());
			   this->lblViolationTitle = (gcnew System::Windows::Forms::Label());
			   this->panel1 = (gcnew System::Windows::Forms::Panel());
			   (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->trackBar1))->BeginInit();
			   (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->splitContainer1))->BeginInit();
			   this->splitContainer1->Panel1->SuspendLayout();
			   this->splitContainer1->Panel2->SuspendLayout();
			   this->splitContainer1->SuspendLayout();
			   (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox1))->BeginInit();
			   this->pnlViolationContainer->SuspendLayout();
			   this->SuspendLayout();
			   // 
			   // timer1
			   // 
			   this->timer1->Interval = 15;
			   this->timer1->Tick += gcnew System::EventHandler(this, &OfflineUploadForm::timer1_Tick);
			   // 
			   // processingWorker
			   // 
			   this->processingWorker->WorkerSupportsCancellation = true;
			   this->processingWorker->DoWork += gcnew System::ComponentModel::DoWorkEventHandler(this, &OfflineUploadForm::processingWorker_DoWork);
			   // 
			   // btnPrevFrame
			   // 
			   this->btnPrevFrame->BackColor = System::Drawing::Color::Yellow;
			   this->btnPrevFrame->Location = System::Drawing::Point(42, 47);
			   this->btnPrevFrame->Name = L"btnPrevFrame";
			   this->btnPrevFrame->Size = System::Drawing::Size(28, 23);
			   this->btnPrevFrame->TabIndex = 0;
			   this->btnPrevFrame->Text = L"<";
			   this->btnPrevFrame->UseVisualStyleBackColor = false;
			   // 
			   // btnNextFrame
			   // 
			   this->btnNextFrame->BackColor = System::Drawing::Color::Yellow;
			   this->btnNextFrame->Location = System::Drawing::Point(184, 47);
			   this->btnNextFrame->Name = L"btnNextFrame";
			   this->btnNextFrame->Size = System::Drawing::Size(27, 23);
			   this->btnNextFrame->TabIndex = 1;
			   this->btnNextFrame->Text = L">";
			   this->btnNextFrame->UseVisualStyleBackColor = false;
			   // 
			   // btnOfflineMode
			   // 
			   this->btnOfflineMode->BackColor = System::Drawing::Color::OrangeRed;
			   this->btnOfflineMode->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 12, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				   static_cast<System::Byte>(0)));
			   this->btnOfflineMode->ForeColor = System::Drawing::SystemColors::ButtonHighlight;
			   this->btnOfflineMode->Location = System::Drawing::Point(851, 46);
			   this->btnOfflineMode->Name = L"btnOfflineMode";
			   this->btnOfflineMode->Size = System::Drawing::Size(112, 46);
			   this->btnOfflineMode->TabIndex = 2;
			   this->btnOfflineMode->Text = L"Offline";
			   this->btnOfflineMode->UseVisualStyleBackColor = false;
			   // 
			   // btnPlayPause
			   // 
			   this->btnPlayPause->BackColor = System::Drawing::Color::Gold;
			   this->btnPlayPause->Location = System::Drawing::Point(554, 47);
			   this->btnPlayPause->Name = L"btnPlayPause";
			   this->btnPlayPause->Size = System::Drawing::Size(45, 44);
			   this->btnPlayPause->TabIndex = 3;
			   this->btnPlayPause->Text = L"▶";
			   this->btnPlayPause->UseVisualStyleBackColor = false;
			   this->btnPlayPause->Click += gcnew System::EventHandler(this, &OfflineUploadForm::btnPlayPause_Click);
			   // 
			   // trackBar1
			   // 
			   this->trackBar1->Location = System::Drawing::Point(605, 47);
			   this->trackBar1->Name = L"trackBar1";
			   this->trackBar1->Size = System::Drawing::Size(217, 45);
			   this->trackBar1->TabIndex = 4;
			   this->trackBar1->Scroll += gcnew System::EventHandler(this, &OfflineUploadForm::trackBar1_Scroll);
			   this->trackBar1->MouseDown += gcnew System::Windows::Forms::MouseEventHandler(this, &OfflineUploadForm::trackBar1_MouseDown);
			   this->trackBar1->MouseUp += gcnew System::Windows::Forms::MouseEventHandler(this, &OfflineUploadForm::trackBar1_MouseUp);
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
			   // 
			   // btnUploadImage
			   // 
			   this->btnUploadImage->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(255)), static_cast<System::Int32>(static_cast<System::Byte>(255)),
				   static_cast<System::Int32>(static_cast<System::Byte>(192)));
			   this->btnUploadImage->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 12, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				   static_cast<System::Byte>(0)));
			   this->btnUploadImage->Location = System::Drawing::Point(97, 142);
			   this->btnUploadImage->Name = L"btnUploadImage";
			   this->btnUploadImage->Size = System::Drawing::Size(125, 59);
			   this->btnUploadImage->TabIndex = 5;
			   this->btnUploadImage->Text = L"Upload Image 📷";
			   this->btnUploadImage->UseVisualStyleBackColor = false;
			   this->btnUploadImage->Click += gcnew System::EventHandler(this, &OfflineUploadForm::btnUploadImage_Click);
			   // 
			   // btnUploadVideo
			   // 
			   this->btnUploadVideo->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(255)), static_cast<System::Int32>(static_cast<System::Byte>(255)),
				   static_cast<System::Int32>(static_cast<System::Byte>(192)));
			   this->btnUploadVideo->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 12, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				   static_cast<System::Byte>(0)));
			   this->btnUploadVideo->Location = System::Drawing::Point(228, 142);
			   this->btnUploadVideo->Name = L"btnUploadVideo";
			   this->btnUploadVideo->Size = System::Drawing::Size(125, 59);
			   this->btnUploadVideo->TabIndex = 6;
			   this->btnUploadVideo->Text = L"Upload Video 🎬";
			   this->btnUploadVideo->UseVisualStyleBackColor = false;
			   this->btnUploadVideo->Click += gcnew System::EventHandler(this, &OfflineUploadForm::btnUploadVideo_Click);
			   // 
			   // btnLoadParkingTemplate
			   // 
			   this->btnLoadParkingTemplate->BackColor = System::Drawing::Color::LightGreen;
			   this->btnLoadParkingTemplate->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 11.25F, System::Drawing::FontStyle::Bold,
				   System::Drawing::GraphicsUnit::Point, static_cast<System::Byte>(0)));
			   this->btnLoadParkingTemplate->Location = System::Drawing::Point(288, 83);
			   this->btnLoadParkingTemplate->Name = L"btnLoadParkingTemplate";
			   this->btnLoadParkingTemplate->Size = System::Drawing::Size(119, 47);
			   this->btnLoadParkingTemplate->TabIndex = 7;
			   this->btnLoadParkingTemplate->Text = L"Load Template";
			   this->btnLoadParkingTemplate->UseVisualStyleBackColor = false;
			   this->btnLoadParkingTemplate->Click += gcnew System::EventHandler(this, &OfflineUploadForm::btnLoadParkingTemplate_Click);
			   // 
			   // chkParkingMode
			   // 
			   this->chkParkingMode->AutoSize = true;
			   this->chkParkingMode->Location = System::Drawing::Point(14, 96);
			   this->chkParkingMode->Name = L"chkParkingMode";
			   this->chkParkingMode->Size = System::Drawing::Size(98, 17);
			   this->chkParkingMode->TabIndex = 8;
			   this->chkParkingMode->Text = L"Enable Parking";
			   this->chkParkingMode->CheckedChanged += gcnew System::EventHandler(this, &OfflineUploadForm::chkParkingMode_CheckedChanged);
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
			   this->splitContainer1->Panel1->Controls->Add(this->label1);
			   this->splitContainer1->Panel1->Controls->Add(this->btnOfflineMode);
			   this->splitContainer1->Panel1->Controls->Add(this->trackBar1);
			   this->splitContainer1->Panel1->Controls->Add(this->btnPlayPause);
			   this->splitContainer1->Panel1->Controls->Add(this->btnNextFrame);
			   this->splitContainer1->Panel1->Controls->Add(this->btnPrevFrame);
			   this->splitContainer1->Panel1->Controls->Add(this->pictureBox1);
			   this->splitContainer1->Panel1->Padding = System::Windows::Forms::Padding(30);
			   // 
			   // splitContainer1.Panel2
			   // 
			   this->splitContainer1->Panel2->BackColor = System::Drawing::Color::LightSteelBlue;
			   this->splitContainer1->Panel2->Controls->Add(this->label7);
			   this->splitContainer1->Panel2->Controls->Add(this->label6);
			   this->splitContainer1->Panel2->Controls->Add(this->label5);
			   this->splitContainer1->Panel2->Controls->Add(this->btnLoadParkingTemplate);
			   this->splitContainer1->Panel2->Controls->Add(this->chkParkingMode);
			   this->splitContainer1->Panel2->Controls->Add(this->pnlViolationContainer);
			   this->splitContainer1->Panel2->Controls->Add(this->btnUploadVideo);
			   this->splitContainer1->Panel2->Controls->Add(this->btnUploadImage);
			   this->splitContainer1->Panel2->Controls->Add(this->lblLogs);
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
			   // label7
			   // 
			   this->label7->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(244)), static_cast<System::Int32>(static_cast<System::Byte>(67)),
				   static_cast<System::Int32>(static_cast<System::Byte>(54)));
			   this->label7->Font = (gcnew System::Drawing::Font(L"Arial", 26, System::Drawing::FontStyle::Bold));
			   this->label7->ForeColor = System::Drawing::Color::White;
			   this->label7->Location = System::Drawing::Point(60, 356);
			   this->label7->Name = L"label7";
			   this->label7->Size = System::Drawing::Size(320, 50);
			   this->label7->TabIndex = 16;
			   this->label7->Text = L"0";
			   this->label7->TextAlign = System::Drawing::ContentAlignment::MiddleCenter;
			   // 
			   // label6
			   // 
			   this->label6->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(255)), static_cast<System::Int32>(static_cast<System::Byte>(193)),
				   static_cast<System::Int32>(static_cast<System::Byte>(7)));
			   this->label6->Font = (gcnew System::Drawing::Font(L"Arial", 26, System::Drawing::FontStyle::Bold));
			   this->label6->ForeColor = System::Drawing::Color::White;
			   this->label6->Location = System::Drawing::Point(60, 291);
			   this->label6->Name = L"label6";
			   this->label6->Size = System::Drawing::Size(320, 50);
			   this->label6->TabIndex = 15;
			   this->label6->Text = L"0";
			   this->label6->TextAlign = System::Drawing::ContentAlignment::MiddleCenter;
			   this->label6->Click += gcnew System::EventHandler(this, &OfflineUploadForm::label6_Click);
			   // 
			   // label5
			   // 
			   this->label5->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(76)), static_cast<System::Int32>(static_cast<System::Byte>(175)),
				   static_cast<System::Int32>(static_cast<System::Byte>(80)));
			   this->label5->Font = (gcnew System::Drawing::Font(L"Arial", 26, System::Drawing::FontStyle::Bold));
			   this->label5->ForeColor = System::Drawing::Color::White;
			   this->label5->Location = System::Drawing::Point(60, 225);
			   this->label5->Name = L"label5";
			   this->label5->Size = System::Drawing::Size(320, 50);
			   this->label5->TabIndex = 14;
			   this->label5->Text = L"0";
			   this->label5->TextAlign = System::Drawing::ContentAlignment::MiddleCenter;
			   // 
			   // pnlViolationContainer
			   // 
			   this->pnlViolationContainer->BackColor = System::Drawing::Color::LightSteelBlue;
			   this->pnlViolationContainer->Controls->Add(this->flpViolations);
			   this->pnlViolationContainer->Controls->Add(this->btnClearViolations);
			   this->pnlViolationContainer->Controls->Add(this->lblViolationCount);
			   this->pnlViolationContainer->Controls->Add(this->lblViolationTitle);
			   this->pnlViolationContainer->Location = System::Drawing::Point(37, 456);
			   this->pnlViolationContainer->Name = L"pnlViolationContainer";
			   this->pnlViolationContainer->Size = System::Drawing::Size(352, 225);
			   this->pnlViolationContainer->TabIndex = 13;
			   // 
			   // flpViolations
			   // 
			   this->flpViolations->AutoScroll = true;
			   this->flpViolations->Location = System::Drawing::Point(30, 52);
			   this->flpViolations->Name = L"flpViolations";
			   this->flpViolations->Size = System::Drawing::Size(286, 162);
			   this->flpViolations->TabIndex = 3;
			   // 
			   // btnClearViolations
			   // 
			   this->btnClearViolations->BackColor = System::Drawing::Color::Tomato;
			   this->btnClearViolations->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
			   this->btnClearViolations->Font = (gcnew System::Drawing::Font(L"Segoe UI", 10));
			   this->btnClearViolations->Location = System::Drawing::Point(757, 5);
			   this->btnClearViolations->Name = L"btnClearViolations";
			   this->btnClearViolations->Size = System::Drawing::Size(91, 27);
			   this->btnClearViolations->TabIndex = 2;
			   this->btnClearViolations->Text = L"Clear All";
			   this->btnClearViolations->UseVisualStyleBackColor = false;
			   this->btnClearViolations->Click += gcnew System::EventHandler(this, &OfflineUploadForm::btnClearViolations_Click);
			   // 
			   // lblViolationCount
			   // 
			   this->lblViolationCount->AutoSize = true;
			   this->lblViolationCount->Font = (gcnew System::Drawing::Font(L"Segoe UI", 10));
			   this->lblViolationCount->Location = System::Drawing::Point(3, 30);
			   this->lblViolationCount->Name = L"lblViolationCount";
			   this->lblViolationCount->Size = System::Drawing::Size(117, 19);
			   this->lblViolationCount->TabIndex = 1;
			   this->lblViolationCount->Text = L"Total Violations: 0";
			   // 
			   // lblViolationTitle
			   // 
			   this->lblViolationTitle->AutoSize = true;
			   this->lblViolationTitle->Font = (gcnew System::Drawing::Font(L"Segoe UI", 12));
			   this->lblViolationTitle->Location = System::Drawing::Point(3, 5);
			   this->lblViolationTitle->Name = L"lblViolationTitle";
			   this->lblViolationTitle->Size = System::Drawing::Size(139, 21);
			   this->lblViolationTitle->TabIndex = 0;
			   this->lblViolationTitle->Text = L"Violation Alerts (0)";
			   // 
			   // panel1
			   // 
			   this->panel1->BackColor = System::Drawing::Color::LightSteelBlue;
			   this->panel1->Location = System::Drawing::Point(12, 12);
			   this->panel1->Name = L"panel1";
			   this->panel1->Size = System::Drawing::Size(851, 484);
			   this->panel1->TabIndex = 4;
			   // 
			   // OfflineUploadForm
			   // 
			   this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			   this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			   this->ClientSize = System::Drawing::Size(1443, 759);
			   this->Controls->Add(this->splitContainer1);
			   this->Name = L"OfflineUploadForm";
			   this->Text = L"Offline Mode - Loading Model...";
			   this->FormClosing += gcnew System::Windows::Forms::FormClosingEventHandler(this, &OfflineUploadForm::OfflineUploadForm_FormClosing);
			   (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->trackBar1))->EndInit();
			   this->splitContainer1->Panel1->ResumeLayout(false);
			   this->splitContainer1->Panel1->PerformLayout();
			   this->splitContainer1->Panel2->ResumeLayout(false);
			   this->splitContainer1->Panel2->PerformLayout();
			   (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->splitContainer1))->EndInit();
			   this->splitContainer1->ResumeLayout(false);
			   (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox1))->EndInit();
			   this->pnlViolationContainer->ResumeLayout(false);
			   this->pnlViolationContainer->PerformLayout();
			   this->ResumeLayout(false);

		   }
#pragma endregion

	private: void UpdatePictureBox(cv::Mat& mat) {
		if (mat.empty() || mat.type() != CV_8UC3) return;
		int w = mat.cols;
		int h = mat.rows;

		Bitmap^ targetBmp = useBuffer1 ? bmpBuffer1 : bmpBuffer2;

		if (targetBmp == nullptr || targetBmp->Width != w || targetBmp->Height != h) {
			if (targetBmp != nullptr) delete targetBmp;
			targetBmp = gcnew Bitmap(w, h, System::Drawing::Imaging::PixelFormat::Format24bppRgb);
			if (useBuffer1) bmpBuffer1 = targetBmp; else bmpBuffer2 = targetBmp;
		}

		System::Drawing::Rectangle rect = System::Drawing::Rectangle(0, 0, w, h);
		System::Drawing::Imaging::BitmapData^ bmpData = targetBmp->LockBits(rect, System::Drawing::Imaging::ImageLockMode::WriteOnly, targetBmp->PixelFormat);
		for (int y = 0; y < h; y++) {
			memcpy((unsigned char*)bmpData->Scan0.ToPointer() + y * bmpData->Stride, mat.data + y * mat.step, w * 3);
		}
		targetBmp->UnlockBits(bmpData);

		pictureBox1->Image = targetBmp;
		useBuffer1 = !useBuffer1;
	}

	// *** [OPTIMIZED] READER THREAD - ?????????????????? ***
	private: void VideoReaderLoop() {
		double ticksPerFrame = 1000.0 / 30.0;
		if (g_videoFPS > 0) ticksPerFrame = 1000.0 / g_videoFPS;

		long long nextTick = cv::getTickCount();
		double tickFreq = cv::getTickFrequency();

		while (!shouldStop) {
			long long currentTick = cv::getTickCount();
			double timeRemaining = (nextTick - currentTick) / tickFreq * 1000.0;

			if (timeRemaining > 2.0) {
				Threading::Thread::Sleep(1);
				continue;
			}

			if (currentTick >= nextTick) {
				cv::Mat tempFrame;
				bool success = false;

				if (g_cap_offline && g_cap_offline->isOpened()) {
					success = g_cap_offline->read(tempFrame);
				}
				else {
					break;
				}

				if (success && !tempFrame.empty()) {
					long long currentSeq;
					{
						std::lock_guard<std::mutex> lock(g_frameMutex_offline);
						g_latestRawFrame_offline = tempFrame;
						g_frameSeq_offline++;
						currentSeq = g_frameSeq_offline;
					}

					// *** [NEW] ??????????? (Reader ??) ***
					cv::Mat renderedFrame;
					DrawScene(tempFrame, currentSeq, renderedFrame);

					// ???????????????
					if (!renderedFrame.empty()) {
						std::lock_guard<std::mutex> lock(g_processedMutex);
						g_processedFrame_shared = renderedFrame.clone();
						g_processedSeq_shared = currentSeq;
					}

					nextTick += (long long)(ticksPerFrame * tickFreq / 1000.0);
					if (cv::getTickCount() > nextTick) nextTick = cv::getTickCount();
				}
				else {
					if (g_cap_offline && g_cap_offline->isOpened()) break;
					Threading::Thread::Sleep(100);
				}
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
				GetRawFrame(frameToProcess, seq);
				if (!frameToProcess.empty() && seq > lastProcessedSeq) {
					ProcessFrame(frameToProcess, seq);
					lastProcessedSeq = seq;
				}
				else {
					Threading::Thread::Sleep(10);
				}
			}
			catch (...) { Threading::Thread::Sleep(50); }
		}
	}

	// *** [OPTIMIZED] UI TIMER - ??????????????????????? (0.1ms) ***
	private: System::Void timer1_Tick(System::Object^ sender, System::EventArgs^ e) {
		try {
			cv::Mat finalFrame;
			long long seq = 0;
			GetProcessedFrame(finalFrame, seq);

			if (seq == lastDisplaySeq) return;
			lastDisplaySeq = seq;

			if (!finalFrame.empty()) {
				UpdatePictureBox(finalFrame);
				// Check for violations
				if (g_parkingEnabled_offline) {
					cv::Mat frameCopy = finalFrame.clone();
					CheckViolations(frameCopy);
				}
			}

			// *** [NEW] SYNC TRACKBAR WITH CURRENT FRAME POSITION ***
			if (!isTrackBarDragging && g_cap_offline && g_cap_offline->isOpened() && isProcessing) {
				long long currentFrame = (long long)g_cap_offline->get(cv::CAP_PROP_POS_FRAMES);
				if (currentFrame <= trackBar1->Maximum) {
					trackBar1->Value = (int)currentFrame;
				}
			}

			// *** [NEW] UPDATE PARKING STATISTICS LABELS ***
			AppState state;
			{
				std::lock_guard<std::mutex> lock(g_stateMutex);
				state = g_appState;
			}

			if (g_parkingEnabled_offline && !state.slotStatuses.empty()) {
				int emptyCount = 0;
				int occupiedCount = 0;

				for (const auto& slotEntry : state.slotStatuses) {
					SlotStatus status = slotEntry.second;
					if (status == SlotStatus::EMPTY) {
						emptyCount++;
					}
					else {
						occupiedCount++;
					}
				}

				int violationCount = state.violatingCarIds.size();

				label5->Text = System::String::Format(L"Empty: {0}", emptyCount);
				label6->Text = System::String::Format(L"Normal: {0}", occupiedCount);
				label7->Text = System::String::Format(L"Violation: {0}", violationCount);
			}
		}
		catch (...) {}
	}

	private: void StartProcessing() {
		shouldStop = false;
		isProcessing = true;
		{
			std::lock_guard<std::mutex> lock(g_stateMutex);
			g_appState = AppState();
		}
		ResetParkingCache();
		if (!processingWorker->IsBusy) processingWorker->RunWorkerAsync();

		if (readerThread == nullptr || !readerThread->IsAlive) {
			readerThread = gcnew Thread(gcnew ThreadStart(this, &OfflineUploadForm::VideoReaderLoop));
			readerThread->IsBackground = true;
			readerThread->Start();
		}
		timer1->Start();
	}

	private: void StopProcessing() {
		shouldStop = true;
		isProcessing = false;
		timer1->Stop();
		if (processingWorker->IsBusy) processingWorker->CancelAsync();
	}

	private: System::Void LoadModel_DoWork(System::Object^ sender, DoWorkEventArgs^ e) {
		try {
			std::string modelPath = "models/test/yolo11n.onnx";
			InitBackend(modelPath);
			e->Result = true;
		}
		catch (const std::exception& ex) { e->Result = gcnew System::String(ex.what()); }
	}

	private: System::Void LoadModel_Completed(System::Object^ sender, RunWorkerCompletedEventArgs^ e) {
		if (e->Result != nullptr && safe_cast<bool>(e->Result)) {
			this->Text = L"Offline Mode - Ready";
			btnUploadImage->Enabled = true; btnUploadVideo->Enabled = true;
			MessageBox::Show("System Ready!", "Success", MessageBoxButtons::OK, MessageBoxIcon::Information);
		}
		else {
			System::String^ errorMsg = e->Result != nullptr ? safe_cast<System::String^>(e->Result) : L"Unknown error";
			MessageBox::Show("Error loading model: " + errorMsg, "Error", MessageBoxButtons::OK, MessageBoxIcon::Error);
		}
	}

	private: System::Void btnUploadImage_Click(System::Object^ sender, System::EventArgs^ e) {
		StopProcessing();
		OpenFileDialog^ ofd = gcnew OpenFileDialog();
		ofd->Filter = "Image Files|*.jpg;*.png;*.jpeg;*.bmp";
		if (ofd->ShowDialog() == System::Windows::Forms::DialogResult::OK) {
			std::string fileName = msclr::interop::marshal_as<std::string>(ofd->FileName);
			cv::Mat img = cv::imread(fileName);
			if (!img.empty()) {
				// Process the image
				ProcessFrame(img, 999999);
				// Draw the result
				cv::Mat result;
				DrawScene(img, 999999, result);
				// Update pictureBox to display the result
				UpdatePictureBox(result);
				MessageBox::Show("Image loaded and processed successfully!", "Success", MessageBoxButtons::OK, MessageBoxIcon::Information);
			}
			else {
				MessageBox::Show("Error loading image!", "Error", MessageBoxButtons::OK, MessageBoxIcon::Error);
			}
		}
	}

	private: System::Void btnUploadVideo_Click(System::Object^ sender, System::EventArgs^ e) {
		StopProcessing();
		OpenFileDialog^ ofd = gcnew OpenFileDialog();
		ofd->Filter = "Video Files|*.mp4;*.avi";
		if (ofd->ShowDialog() == System::Windows::Forms::DialogResult::OK) {
			std::string fileName = msclr::interop::marshal_as<std::string>(ofd->FileName);
			OpenCamera(fileName);
			InitializeTrackBar();
			if (g_cap_offline && g_cap_offline->isOpened()) StartProcessing();
		}
	}

	private: System::Void OfflineUploadForm_FormClosing(System::Object^ sender, FormClosingEventArgs^ e) {
		StopProcessing();
	}

	private: System::Void btnPlayPause_Click(System::Object^ sender, System::EventArgs^ e) {
		if (isProcessing) {
			StopProcessing();
			btnPlayPause->Text = L"▶";
		}
		else {
			isPlayingVideo = true;
			StartProcessing();
			btnPlayPause->Text = L"⏸";
		}
	}

	private: System::Void btnLoadParkingTemplate_Click(System::Object^ sender, System::EventArgs^ e) {
		OpenFileDialog^ ofd = gcnew OpenFileDialog();
		ofd->Filter = "Parking Template|*.xml";
		
		// ??? full path
		char buffer[MAX_PATH];
		_getcwd(buffer, MAX_PATH);
		std::string currentDir(buffer);
		std::string folder = currentDir + "\\parking_templates";
		
		ofd->InitialDirectory = gcnew String(folder.c_str());
		ofd->Title = "Load Parking Template";
		
		if (ofd->ShowDialog() == System::Windows::Forms::DialogResult::OK) {
			std::string fileName = msclr::interop::marshal_as<std::string>(ofd->FileName);
			if (LoadParkingTemplate(fileName)) {
				chkParkingMode->Checked = true;
				MessageBox::Show("Template loaded!", "Success", MessageBoxButtons::OK, MessageBoxIcon::Information);
			}
			else MessageBox::Show("Failed to load!", "Error", MessageBoxButtons::OK, MessageBoxIcon::Error);
		}
	}

	private: System::Void chkParkingMode_CheckedChanged(System::Object^ sender, System::EventArgs^ e) {
		g_parkingEnabled_offline = chkParkingMode->Checked;
		if (chkParkingMode->Checked) {
			label1->Text = L"Parking Mode ON";
			label1->BackColor = System::Drawing::Color::LightGreen;
		}
		else {
			label1->Text = L"Camera 1";
			label1->BackColor = System::Drawing::Color::Yellow;
		}
	}

	// *** [NEW] VIOLATION ALERTS METHODS ***
	private: void AddViolationRecord(int carId, cv::Mat& frameCapture, System::String^ violationType) {
		if (frameCapture.empty()) return;

		// Convert cv::Mat to Bitmap
		Bitmap^ screenshot = gcnew Bitmap(frameCapture.cols, frameCapture.rows, System::Drawing::Imaging::PixelFormat::Format24bppRgb);
		System::Drawing::Rectangle rect(0, 0, frameCapture.cols, frameCapture.rows);
		System::Drawing::Imaging::BitmapData^ bmpData = screenshot->LockBits(rect, System::Drawing::Imaging::ImageLockMode::WriteOnly, screenshot->PixelFormat);

		for (int y = 0; y < frameCapture.rows; y++) {
			memcpy((unsigned char*)bmpData->Scan0.ToPointer() + y * bmpData->Stride, frameCapture.data + y * frameCapture.step, frameCapture.cols * 3);
		}
		screenshot->UnlockBits(bmpData);

		// Create violation record
		ViolationRecord^ record = gcnew ViolationRecord();
		record->carId = carId;
		record->screenshot = screenshot;
		record->violationType = violationType;
		record->captureTime = System::DateTime::Now;
		record->durationSeconds = 0;

		violationsList->Add(record);
		RefreshViolationPanel();
	}

	private: void RefreshViolationPanel() {
		if (!flpViolations) return;

		flpViolations->Controls->Clear();

		for each(ViolationRecord^ record in violationsList) {
			// Create item panel
			Panel^ itemPanel = gcnew Panel();
			itemPanel->BackColor = System::Drawing::Color::White;
			itemPanel->Size = System::Drawing::Size(250, 180);
			itemPanel->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
			itemPanel->Margin = System::Windows::Forms::Padding(5);

			// Screenshot
			PictureBox^ pbScreenshot = gcnew PictureBox();
			pbScreenshot->Image = record->screenshot;
			pbScreenshot->SizeMode = System::Windows::Forms::PictureBoxSizeMode::Zoom;
			pbScreenshot->Location = System::Drawing::Point(5, 5);
			pbScreenshot->Size = System::Drawing::Size(240, 100);
			itemPanel->Controls->Add(pbScreenshot);

			// Info label
			Label^ lblInfo = gcnew Label();
			lblInfo->Font = gcnew System::Drawing::Font(L"Segoe UI", 8);
			lblInfo->Location = System::Drawing::Point(5, 110);
			lblInfo->Size = System::Drawing::Size(240, 65);
			lblInfo->Text = System::String::Format(L"ID: {0}\nType: {1}\nTime: {2:HH:mm:ss}\nDuration: {3}s",
				record->carId, record->violationType, record->captureTime, record->durationSeconds);
			itemPanel->Controls->Add(lblInfo);

			flpViolations->Controls->Add(itemPanel);
		}

		lblViolationCount->Text = System::String::Format(L"Violations: {0}", violationsList->Count);
	}

	private: void CheckViolations(cv::Mat& currentFrame) {
		AppState state;
		{
			std::lock_guard<std::mutex> lock(g_stateMutex);
			state = g_appState;
		}

		// Check for overstay (> 10 seconds = 300 frames at 30fps)
		for each(auto car in state.cars) {
			if (car.framesStill > 300) { // 10 seconds
				if (!violatingCarTimers->ContainsKey(car.id)) {
					violatingCarTimers->Add(car.id, System::DateTime::Now);
					
					// Capture only the bounding box region
					cv::Rect safeBbox = car.bbox & cv::Rect(0, 0, currentFrame.cols, currentFrame.rows);
					if (safeBbox.area() > 0) {
						cv::Mat croppedFrame = currentFrame(safeBbox).clone();
						AddViolationRecord(car.id, croppedFrame, L"Overstay");
					}
				}
			}
		}

		// Check for wrong slot (parking violation)
		for each(int violatingId in state.violatingCarIds) {
			bool already_captured = false;
			for each(ViolationRecord^ record in violationsList) {
				if (record->carId == violatingId && record->violationType == L"Wrong Slot") {
					already_captured = true;
					break;
				}
			}

			if (!already_captured) {
				// Find the car with this ID and capture its bbox
			 for each(auto car in state.cars) {
					if (car.id == violatingId) {
						// Capture only the bounding box region
						cv::Rect safeBbox = car.bbox & cv::Rect(0, 0, currentFrame.cols, currentFrame.rows);
						if (safeBbox.area() > 0) {
							cv::Mat croppedFrame = currentFrame(safeBbox).clone();
							AddViolationRecord(violatingId, croppedFrame, L"Wrong Slot");
						}
						break;
					}
				}
			}
		}

		// Update durations
		for each(ViolationRecord^ record in violationsList) {
			System::TimeSpan duration = System::DateTime::Now - record->captureTime;
			record->durationSeconds = (int)duration.TotalSeconds;
		}
	}

	private: System::Void btnClearViolations_Click(System::Object^ sender, System::EventArgs^ e) {
		violationsList->Clear();
		violatingCarTimers->Clear();
		RefreshViolationPanel();
	}

	// *** [NEW] TRACKBAR EVENT HANDLERS ***
	private: System::Void trackBar1_MouseDown(System::Object^ sender, System::Windows::Forms::MouseEventArgs^ e) {
		isTrackBarDragging = true;
		StopProcessing();
	}

	private: System::Void trackBar1_MouseUp(System::Object^ sender, System::Windows::Forms::MouseEventArgs^ e) {
		isTrackBarDragging = false;
		if (g_cap_offline && g_cap_offline->isOpened()) {
			long long framePos = trackBar1->Value;
			g_cap_offline->set(cv::CAP_PROP_POS_FRAMES, framePos);
			StartProcessing();
		}
	}

	private: System::Void trackBar1_Scroll(System::Object^ sender, System::EventArgs^ e) {
		if (isTrackBarDragging && g_cap_offline && g_cap_offline->isOpened()) {
			long long framePos = trackBar1->Value;
			g_cap_offline->set(cv::CAP_PROP_POS_FRAMES, framePos);
		}
	}

	// *** INITIALIZE TRACKBAR AFTER VIDEO LOAD ***
	private: void InitializeTrackBar() {
		if (g_cap_offline && g_cap_offline->isOpened()) {
			double frameCount = g_cap_offline->get(cv::CAP_PROP_FRAME_COUNT);
			totalFrames = (long long)frameCount;
			trackBar1->Minimum = 0;
			trackBar1->Maximum = (int)((totalFrames > 0) ? totalFrames - 1 : 0);
			trackBar1->Value = 0;
		}
	}
private: System::Void label6_Click(System::Object^ sender, System::EventArgs^ e) {
}
private: System::Void label4_Click(System::Object^ sender, System::EventArgs^ e) {
}
private: System::Void label2_Click(System::Object^ sender, System::EventArgs^ e) {
}
};
}
