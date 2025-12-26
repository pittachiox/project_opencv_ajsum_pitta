#pragma once
#include <msclr/marshal_cppstd.h>
#include <string>
#include <vector>
#include <map>
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
#include <set>

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
static bool g_parkingEnabled_online = false;

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

// --- Helper Functions ---

static void ResetParkingCache_Online() {
	g_cachedParkingOverlay_online = cv::Mat();
	g_lastDrawnStatus_online.clear();
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
		outFrame = g_latestRawFrame;
		outSeq = g_frameSeq_online;
	}
}

// *** [NEW] GET PROCESSED FRAME (For UI) ***
static void GetProcessedFrameOnline(cv::Mat& outFrame, long long& outSeq) {
	std::lock_guard<std::mutex> lock(g_processedMutex_online);
	if (!g_processedFrame_online.empty()) {
		outFrame = g_processedFrame_online;
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
	g_frameSeq_online = 0;
	if (g_cap->isOpened()) {
		g_cameraFPS = g_cap->get(cv::CAP_PROP_FPS);
		if (g_cameraFPS <= 0) g_cameraFPS = 30.0;
	}
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
			"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
			"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
			"elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
			"skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
			"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
			"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
			"potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
			"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
			"hair drier", "toothbrush"
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
		g_parkingEnabled_online = true;
		return true;
	}
	return false;
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

				if (max_class_score > CONF_THRESHOLD) {
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

		if (g_parkingEnabled_online && g_pm_logic_online) {
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

// *** DRAWING FUNCTION (UI Thread) - เหมือนออฟไลน์ ***

static void DrawSceneOnline(const cv::Mat& frame, long long displaySeq, cv::Mat& outResult) {
	if (frame.empty()) return;

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
	if (g_parkingEnabled_online && g_pm_display_online) {
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
		for (const auto& obj : state.cars) {
			if (obj.classId >= 0 && obj.classId < g_classes.size()) {
				cv::Rect box = obj.bbox;
				bool isViolating = (state.violatingCarIds.count(obj.id) > 0);

				if (isViolating) {
					cv::Rect roi = box & cv::Rect(0, 0, outResult.cols, outResult.rows);
					if (roi.area() > 0) {
						cv::Mat roiMat = outResult(roi);
						cv::Mat colorBlock(roi.size(), CV_8UC3, cv::Scalar(0, 0, 255));
						cv::addWeighted(roiMat, 0.6, colorBlock, 0.4, 0, roiMat);
					}
					cv::rectangle(outResult, box, cv::Scalar(0, 0, 255), 2);
				}
				else {
					cv::rectangle(outResult, box, g_colors[obj.classId], 2);
				}

				std::string label = "ID:" + std::to_string(obj.id);
				if (isViolating) label += " [VIOLATION]";
				else if (!g_parkingEnabled_online) label += " " + g_classes[obj.classId];

				int baseline;
				cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
				cv::Scalar labelBg = isViolating ? cv::Scalar(0, 0, 255) : g_colors[obj.classId];

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
			if (bmpBuffer1) delete bmpBuffer1;
			if (bmpBuffer2) delete bmpBuffer2;
		}

	private: System::Windows::Forms::Button^ button1;
	private: System::Windows::Forms::Button^ button2;
	private: System::Windows::Forms::Timer^ timer1;
	private: System::Windows::Forms::PictureBox^ pictureBox1;
	private: BackgroundWorker^ processingWorker;
		   // [READER THREAD] - เหมือนออฟไลน์
	private: Thread^ readerThread;
	private: System::ComponentModel::IContainer^ components;
	private: Bitmap^ currentFrame;
	private: Object^ bufferLock;
	private: bool isProcessing;
	private: System::Windows::Forms::Label^ lblCameraName;
	private: System::Windows::Forms::Button^ btnPlayPause;
	private: System::Windows::Forms::TrackBar^ trackBar1;
	private: System::Windows::Forms::Button^ btnNextFrame;
	private: System::Windows::Forms::Button^ btnOnlineMode;
	private: System::Windows::Forms::Button^ btnPrevFrame;
	private: System::Windows::Forms::Panel^ panel2;
	private: System::Windows::Forms::Label^ lblViolation;
	private: System::Windows::Forms::Label^ lblNormal;
	private: System::Windows::Forms::Label^ lblEmpty;
	private: System::Windows::Forms::Button^ btnLiveCamera;
	private: System::Windows::Forms::Button^ btnLoadParkingTemplate;
	private: System::Windows::Forms::CheckBox^ chkParkingMode;
	private: System::Windows::Forms::Panel^ panel3;
	private: System::Windows::Forms::Label^ lblLogs;
	private: bool shouldStop;
	private: long long lastProcessedSeq = -1;
	private: long long lastDisplaySeq = -1;
	private: System::Windows::Forms::SplitContainer^ splitContainer1;
	private: Bitmap^ bmpBuffer1;
	private: Bitmap^ bmpBuffer2;
	private: bool useBuffer1;
	private: System::Windows::Forms::Label^ label1;

#pragma region Windows Form Designer generated code
		   void InitializeComponent(void)
		   {
			   this->components = (gcnew System::ComponentModel::Container());
			   this->timer1 = (gcnew System::Windows::Forms::Timer(this->components));
			   this->pictureBox1 = (gcnew System::Windows::Forms::PictureBox());
			   this->processingWorker = (gcnew System::ComponentModel::BackgroundWorker());
			   this->btnPrevFrame = (gcnew System::Windows::Forms::Button());
			   this->btnNextFrame = (gcnew System::Windows::Forms::Button());
			   this->btnOnlineMode = (gcnew System::Windows::Forms::Button());
			   this->btnPlayPause = (gcnew System::Windows::Forms::Button());
			   this->trackBar1 = (gcnew System::Windows::Forms::TrackBar());
			   this->lblCameraName = (gcnew System::Windows::Forms::Label());
			   this->panel2 = (gcnew System::Windows::Forms::Panel());
			   this->chkParkingMode = (gcnew System::Windows::Forms::CheckBox());
			   this->btnLoadParkingTemplate = (gcnew System::Windows::Forms::Button());
			   this->lblLogs = (gcnew System::Windows::Forms::Label());
			   this->panel3 = (gcnew System::Windows::Forms::Panel());
			   this->lblViolation = (gcnew System::Windows::Forms::Label());
			   this->lblNormal = (gcnew System::Windows::Forms::Label());
			   this->lblEmpty = (gcnew System::Windows::Forms::Label());
			   this->btnLiveCamera = (gcnew System::Windows::Forms::Button());
			   this->splitContainer1 = (gcnew System::Windows::Forms::SplitContainer());
			   this->label1 = (gcnew System::Windows::Forms::Label());
			   (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox1))->BeginInit();
			   (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->trackBar1))->BeginInit();
			   (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->splitContainer1))->BeginInit();
			   this->splitContainer1->Panel1->SuspendLayout();
			   this->splitContainer1->Panel2->SuspendLayout();
			   this->splitContainer1->SuspendLayout();
			   this->SuspendLayout();
			   // 
			   // timer1
			   // 
			   this->timer1->Enabled = true;
			   this->timer1->Interval = 15;
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
			   // btnOnlineMode
			   // 
			   this->btnOnlineMode->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(40)), static_cast<System::Int32>(static_cast<System::Byte>(167)),
				   static_cast<System::Int32>(static_cast<System::Byte>(69)));
			   this->btnOnlineMode->ForeColor = System::Drawing::SystemColors::ButtonHighlight;
			   this->btnOnlineMode->Location = System::Drawing::Point(851, 46);
			   this->btnOnlineMode->Name = L"btnOnlineMode";
			   this->btnOnlineMode->Size = System::Drawing::Size(112, 46);
			   this->btnOnlineMode->TabIndex = 2;
			   this->btnOnlineMode->Text = L"Online";
			   this->btnOnlineMode->UseVisualStyleBackColor = false;
			   // 
			   // btnPlayPause
			   // 
			   this->btnPlayPause->Location = System::Drawing::Point(554, 47);
			   this->btnPlayPause->Name = L"btnPlayPause";
			   this->btnPlayPause->Size = System::Drawing::Size(45, 44);
			   this->btnPlayPause->TabIndex = 3;
			   this->btnPlayPause->Text = L"▶";
			   this->btnPlayPause->Click += gcnew System::EventHandler(this, &UploadForm::btnPlayPause_Click);
			   // 
			   // trackBar1
			   // 
			   this->trackBar1->Location = System::Drawing::Point(605, 47);
			   this->trackBar1->Name = L"trackBar1";
			   this->trackBar1->Size = System::Drawing::Size(217, 45);
			   this->trackBar1->TabIndex = 4;
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
			   this->panel2->BackColor = System::Drawing::Color::LightSteelBlue;
			   this->panel2->Location = System::Drawing::Point(869, 12);
			   this->panel2->Name = L"panel2";
			   this->panel2->Size = System::Drawing::Size(541, 484);
			   this->panel2->TabIndex = 5;
			   // 
			   // chkParkingMode
			   // 
			   this->chkParkingMode->AutoSize = true;
			   this->chkParkingMode->Location = System::Drawing::Point(14, 96);
			   this->chkParkingMode->Name = L"chkParkingMode";
			   this->chkParkingMode->Size = System::Drawing::Size(98, 17);
			   this->chkParkingMode->TabIndex = 8;
			   this->chkParkingMode->Text = L"Enable Parking";
			   this->chkParkingMode->CheckedChanged += gcnew System::EventHandler(this, &UploadForm::chkParkingMode_CheckedChanged);
			   // 
			   // btnLoadParkingTemplate
			   // 
			   this->btnLoadParkingTemplate->BackColor = System::Drawing::Color::LightGreen;
			   this->btnLoadParkingTemplate->Location = System::Drawing::Point(177, 334);
			   this->btnLoadParkingTemplate->Name = L"btnLoadParkingTemplate";
			   this->btnLoadParkingTemplate->Size = System::Drawing::Size(100, 25);
			   this->btnLoadParkingTemplate->TabIndex = 7;
			   this->btnLoadParkingTemplate->Text = L"Load Template";
			   this->btnLoadParkingTemplate->UseVisualStyleBackColor = false;
			   this->btnLoadParkingTemplate->Click += gcnew System::EventHandler(this, &UploadForm::btnLoadParkingTemplate_Click);
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
			   // panel3
			   // 
			   this->panel3->BackColor = System::Drawing::Color::LemonChiffon;
			   this->panel3->Location = System::Drawing::Point(42, 229);
			   this->panel3->Name = L"panel3";
			   this->panel3->Size = System::Drawing::Size(346, 38);
			   this->panel3->TabIndex = 1;
			   // 
			   // lblViolation
			   // 
			   this->lblViolation->AutoSize = true;
			   this->lblViolation->BackColor = System::Drawing::Color::Red;
			   this->lblViolation->Font = (gcnew System::Drawing::Font(L"Segoe UI", 16.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				   static_cast<System::Byte>(0)));
			   this->lblViolation->ForeColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(45)), static_cast<System::Int32>(static_cast<System::Byte>(45)),
				   static_cast<System::Int32>(static_cast<System::Byte>(48)));
			   this->lblViolation->Location = System::Drawing::Point(282, 168);
			   this->lblViolation->Name = L"lblViolation";
			   this->lblViolation->Size = System::Drawing::Size(106, 30);
			   this->lblViolation->TabIndex = 11;
			   this->lblViolation->Text = L"Violation";
			   // 
			   // lblNormal
			   // 
			   this->lblNormal->AutoSize = true;
			   this->lblNormal->BackColor = System::Drawing::Color::Yellow;
			   this->lblNormal->Font = (gcnew System::Drawing::Font(L"Segoe UI", 16.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				   static_cast<System::Byte>(0)));
			   this->lblNormal->ForeColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(45)), static_cast<System::Int32>(static_cast<System::Byte>(45)),
				   static_cast<System::Int32>(static_cast<System::Byte>(48)));
			   this->lblNormal->Location = System::Drawing::Point(157, 168);
			   this->lblNormal->Name = L"lblNormal";
			   this->lblNormal->Size = System::Drawing::Size(90, 30);
			   this->lblNormal->TabIndex = 10;
			   this->lblNormal->Text = L"Normal";
			   // 
			   // lblEmpty
			   // 
			   this->lblEmpty->AutoSize = true;
			   this->lblEmpty->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(128)), static_cast<System::Int32>(static_cast<System::Byte>(255)),
				   static_cast<System::Int32>(static_cast<System::Byte>(128)));
			   this->lblEmpty->Font = (gcnew System::Drawing::Font(L"Segoe UI", 16.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				   static_cast<System::Byte>(0)));
			   this->lblEmpty->ForeColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(45)), static_cast<System::Int32>(static_cast<System::Byte>(45)),
				   static_cast<System::Int32>(static_cast<System::Byte>(48)));
			   this->lblEmpty->Location = System::Drawing::Point(44, 168);
			   this->lblEmpty->Name = L"lblEmpty";
			   this->lblEmpty->Size = System::Drawing::Size(80, 30);
			   this->lblEmpty->TabIndex = 9;
			   this->lblEmpty->Text = L"Empty";
			   // 
			   // btnLiveCamera
			   // 
			   this->btnLiveCamera->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(40)), static_cast<System::Int32>(static_cast<System::Byte>(167)),
				   static_cast<System::Int32>(static_cast<System::Byte>(69)));
			   this->btnLiveCamera->ForeColor = System::Drawing::SystemColors::ButtonHighlight;
			   this->btnLiveCamera->Location = System::Drawing::Point(14, 25);
			   this->btnLiveCamera->Name = L"btnLiveCamera";
			   this->btnLiveCamera->Size = System::Drawing::Size(100, 56);
			   this->btnLiveCamera->TabIndex = 0;
			   this->btnLiveCamera->Text = L"📹 Live Camera";
			   this->btnLiveCamera->UseVisualStyleBackColor = false;
			   this->btnLiveCamera->Click += gcnew System::EventHandler(this, &UploadForm::btnLiveCamera_Click);
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
			   this->splitContainer1->Panel1->Controls->Add(this->btnOnlineMode);
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
			   this->splitContainer1->Panel2->Controls->Add(this->lblViolation);
			   this->splitContainer1->Panel2->Controls->Add(this->lblNormal);
			   this->splitContainer1->Panel2->Controls->Add(this->lblEmpty);
			   this->splitContainer1->Panel2->Controls->Add(this->btnLoadParkingTemplate);
			   this->splitContainer1->Panel2->Controls->Add(this->chkParkingMode);
			   this->splitContainer1->Panel2->Controls->Add(this->btnLiveCamera);
			   this->splitContainer1->Panel2->Controls->Add(this->lblLogs);
			   this->splitContainer1->Panel2->Controls->Add(this->panel3);
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
			   (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->trackBar1))->EndInit();
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

	// *** [OPTIMIZED] UI TIMER - หยิบภาพมาโชว์อย่างเดียว (0.1ms) ***
	private: System::Void timer1_Tick(System::Object^ sender, System::EventArgs^ e) {
		try {
			cv::Mat finalFrame;
			long long seq = 0;
			GetProcessedFrameOnline(finalFrame, seq);

			if (seq == lastDisplaySeq) return;
			lastDisplaySeq = seq;

			if (!finalFrame.empty()) {
				UpdatePictureBox(finalFrame);
			}
		}
		catch (...) {}
	}
	private: void StopProcessing() {
		shouldStop = true;
		isProcessing = false;
		timer1->Stop();
		if (processingWorker->IsBusy) processingWorker->CancelAsync();
	}

	private: System::Void btnPlayPause_Click(System::Object^ sender, System::EventArgs^ e) {
		if (isProcessing) StopProcessing(); else StartProcessing();
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
			btnLiveCamera->Enabled = true;
			btnLoadParkingTemplate->Enabled = true;
			MessageBox::Show("Model loaded!", "Success", MessageBoxButtons::OK, MessageBoxIcon::Information);
		}
		else {
			MessageBox::Show("Error loading model", "Error", MessageBoxButtons::OK, MessageBoxIcon::Error);
		}
	}

	private: void StartProcessing() {
		shouldStop = false;
		isProcessing = true;
		{
			std::lock_guard<std::mutex> lock(g_onlineStateMutex);
			g_onlineState = OnlineAppState();
		}
		ResetParkingCache_Online();

		if (!processingWorker->IsBusy) processingWorker->RunWorkerAsync();

		if (readerThread == nullptr || !readerThread->IsAlive) {
			readerThread = gcnew Thread(gcnew ThreadStart(this, &UploadForm::CameraReaderLoop));
			readerThread->IsBackground = true;
			readerThread->Start();
		}

		timer1->Start();
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
					g_latestRawFrame = tempFrame;
					g_frameSeq_online++;
					currentSeq = g_frameSeq_online;
				}

				cv::Mat renderedFrame;
				DrawSceneOnline(tempFrame, currentSeq, renderedFrame);

				if (!renderedFrame.empty()) {
					std::lock_guard<std::mutex> lock(g_processedMutex_online);
					g_processedFrame_online = renderedFrame.clone();
					g_processedSeq_online = currentSeq;
				}

				nextTick += (long long)(ticksPerFrame * tickFreq / 1000.0);
				if (cv::getTickCount() > nextTick) nextTick = cv::getTickCount();
			}
			else {
				if (g_cap && g_cap->isOpened()) break;
				Threading::Thread::Sleep(100);
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
				MessageBox::Show("Template loaded!\n\nParking slot detection is now active.\nViolations (cars parked outside slots) will be marked in RED.", 
					"Success", MessageBoxButtons::OK, MessageBoxIcon::Information);
			}
			else {
				MessageBox::Show("Failed to load template!", "Error", MessageBoxButtons::OK, MessageBoxIcon::Error);
			}
		}
	}

	private: System::Void chkParkingMode_CheckedChanged(System::Object^ sender, System::EventArgs^ e) {
		g_parkingEnabled_online = chkParkingMode->Checked;
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
	};
}