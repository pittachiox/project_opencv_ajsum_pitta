#pragma once
#include <msclr/marshal_cppstd.h>
#include <string>
#include <vector>

#pragma managed(push, off)
#define NOMINMAX
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <mutex>

// --- Global Variables ---
static cv::dnn::Net* g_net = nullptr;
static std::vector<std::string> g_classes;
static std::vector<cv::Scalar> g_colors;
static cv::VideoCapture* g_cap = nullptr;

// --- Frame Sync Management (ส่วนที่เพิ่มใหม่) ---
static cv::Mat g_latestRawFrame;
static long long g_currentFrameSeq = 0; // เลขบัตรคิวของเฟรมปัจจุบัน
static std::mutex g_frameMutex;

// --- AI Status ---
static std::mutex g_processMutex;
static bool g_modelReady = false;

// --- Detection Results ---
static std::vector<int> g_persistentClassIds;
static std::vector<cv::Rect> g_persistentBoxes;
static long long g_detectionSourceSeq = -1; // บอกว่ากล่องนี้มาจากเฟรมเลขที่เท่าไหร่
static std::mutex g_detectionMutex;

// --- Settings ---
static const int YOLO_INPUT_SIZE = 640;
static const float CONF_THRESHOLD = 0.25f;
static const float NMS_THRESHOLD = 0.45f;

// --- Helper Functions ---

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

static void InitGlobalModel(const std::string& modelPath) {
	std::lock_guard<std::mutex> lock(g_processMutex);
	g_modelReady = false;
	if (g_net) { delete g_net; g_net = nullptr; }

	try {
		g_net = new cv::dnn::Net(cv::dnn::readNetFromONNX(modelPath));

		// ใช้ CPU เท่านั้น
		g_net->setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
		g_net->setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
		OutputDebugStringA("[INFO] Using CPU for inference\n");

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
	catch (...) {}
}

// [FIX] เพิ่ม Parameter frameSeq เพื่อระบุตัวตนของเฟรม
static void DetectObjectsOnFrame(const cv::Mat& inputFrame, long long frameSeq) {
	{
		std::lock_guard<std::mutex> lock(g_processMutex);
		if (inputFrame.empty() || !g_net || !g_modelReady) return;
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
			std::lock_guard<std::mutex> lock(g_processMutex);
			g_net->setInput(blob);
			g_net->forward(outputs, g_net->getUnconnectedOutLayersNames());
		}

		if (outputs.empty() || outputs[0].empty()) return;

		cv::Mat output_data = outputs[0];
		// Handle transpose for YOLO format
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

		// *** อัปเดตข้อมูลกล่อง พร้อมกับ Frame Sequence ***
		{
			std::lock_guard<std::mutex> detLock(g_detectionMutex);
			g_persistentClassIds.clear();
			g_persistentBoxes.clear();

			for (int idx : nms) {
				g_persistentBoxes.push_back(boxes[idx]);
				g_persistentClassIds.push_back(class_ids[idx]);
			}
			// [FIX] บันทึกว่ากล่องนี้มาจากเฟรมไหน
			g_detectionSourceSeq = frameSeq;
		}
	}
	catch (...) {}
}

// [FIX] รับ displaySeq มาเช็ค
static cv::Mat DrawPersistentDetections(const cv::Mat& frame, long long displaySeq) {
	if (frame.empty()) return cv::Mat();
	cv::Mat result = frame.clone();

	std::lock_guard<std::mutex> lock(g_detectionMutex);

	// [FIX Logic] ถ้าข้อมูลกล่อง มาจากเฟรมที่ใหม่กว่าภาพที่กำลังแสดง (Future Detection) -> ห้ามวาด!
	if (g_detectionSourceSeq > displaySeq) {
		// คุณอาจจะเลือก return result เปล่าๆ หรือวาด text debug ก็ได้
		// cv::putText(result, "Syncing...", cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,0,255), 2);
		return result;
	}

	for (size_t i = 0; i < g_persistentBoxes.size(); i++) {
		if (i < g_persistentClassIds.size()) {
			cv::Rect box = g_persistentBoxes[i];
			int classId = g_persistentClassIds[i];
			if (classId >= 0 && classId < g_classes.size()) {
				cv::rectangle(result, box, g_colors[classId], 2);
				std::string label = g_classes[classId];
				int baseline;
				cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
				int y_label = (std::max)(0, box.y - textSize.height - 5);
				cv::rectangle(result, cv::Point(box.x, y_label), cv::Point(box.x + textSize.width, y_label + textSize.height + 5), g_colors[classId], -1);
				cv::putText(result, label, cv::Point(box.x, y_label + textSize.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
			}
		}
	}
	return result;
}

static void OpenGlobalVideo(const std::string& filename) {
	std::lock_guard<std::mutex> lock(g_frameMutex);
	if (g_cap) { delete g_cap; g_cap = nullptr; }
	g_cap = new cv::VideoCapture(filename);
	g_currentFrameSeq = 0; // Reset ตัวนับ
	g_detectionSourceSeq = -1;
}

static bool ReadNextVideoFrame() {
	std::lock_guard<std::mutex> lock(g_frameMutex);
	if (!g_cap || !g_cap->isOpened()) return false;
	cv::Mat frame;
	*g_cap >> frame;
	if (frame.empty()) return false;
	g_latestRawFrame = frame;
	g_currentFrameSeq++; // [FIX] เพิ่มตัวนับทุกครั้งที่มีเฟรมใหม่
	return true;
}

static void GetLatestRawFrameCopy(cv::Mat& outFrame, long long& outSeq) {
	std::lock_guard<std::mutex> lock(g_frameMutex);
	if (!g_latestRawFrame.empty()) {
		outFrame = g_latestRawFrame.clone();
		outSeq = g_currentFrameSeq;
	}
}

#pragma managed(pop)

namespace ConsoleApplication3 {
	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;
	using namespace System::Threading;

	public ref class UploadForm : public System::Windows::Forms::Form
	{
	public:
		UploadForm(void)
		{
			InitializeComponent();
			bufferLock = gcnew Object();
			currentFrame = nullptr;
			isProcessing = false;
			shouldStop = false;

			BackgroundWorker^ modelLoader = gcnew BackgroundWorker();
			modelLoader->DoWork += gcnew DoWorkEventHandler(this, &UploadForm::LoadModel_DoWork);
			modelLoader->RunWorkerCompleted += gcnew RunWorkerCompletedEventHandler(this, &UploadForm::LoadModel_Completed);
			modelLoader->RunWorkerAsync();
		}

	protected:
		~UploadForm()
		{
			StopProcessing();
			if (components) delete components;
		}

	private: System::Windows::Forms::Button^ button1;
	private: System::Windows::Forms::Button^ button2;
	private: System::Windows::Forms::Timer^ timer1;
	private: System::Windows::Forms::PictureBox^ pictureBox1;
	private: BackgroundWorker^ processingWorker;
	private: System::ComponentModel::IContainer^ components;
	private: Bitmap^ currentFrame;
	private: Object^ bufferLock;
	private: bool isProcessing;
	private: System::Windows::Forms::Button^ button3;
	private: bool shouldStop;

#pragma region Windows Form Designer generated code
		   void InitializeComponent(void)
		   {
			   this->components = (gcnew System::ComponentModel::Container());
			   this->button1 = (gcnew System::Windows::Forms::Button());
			   this->button2 = (gcnew System::Windows::Forms::Button());
			   this->timer1 = (gcnew System::Windows::Forms::Timer(this->components));
			   this->pictureBox1 = (gcnew System::Windows::Forms::PictureBox());
			   this->processingWorker = (gcnew System::ComponentModel::BackgroundWorker());
			   this->button3 = (gcnew System::Windows::Forms::Button());
			   (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox1))->BeginInit();
			   this->SuspendLayout();
			   // 
			   // button1
			   // 
			   this->button1->Enabled = false;
			   this->button1->Location = System::Drawing::Point(1155, 280);
			   this->button1->Name = L"button1";
			   this->button1->Size = System::Drawing::Size(100, 25);
			   this->button1->TabIndex = 0;
			   this->button1->Text = L"Upload Image";
			   this->button1->UseVisualStyleBackColor = true;
			   this->button1->Click += gcnew System::EventHandler(this, &UploadForm::button1_Click);
			   // 
			   // button2
			   // 
			   this->button2->Enabled = false;
			   this->button2->Location = System::Drawing::Point(1271, 280);
			   this->button2->Name = L"button2";
			   this->button2->Size = System::Drawing::Size(100, 25);
			   this->button2->TabIndex = 2;
			   this->button2->Text = L"Upload Video";
			   this->button2->UseVisualStyleBackColor = true;
			   this->button2->Click += gcnew System::EventHandler(this, &UploadForm::button2_Click);
			   // 
			   // timer1
			   // 
			   this->timer1->Interval = 30;
			   this->timer1->Tick += gcnew System::EventHandler(this, &UploadForm::timer1_Tick);
			   // 
			   // pictureBox1
			   // 
			   this->pictureBox1->Location = System::Drawing::Point(16, 20);
			   this->pictureBox1->Name = L"pictureBox1";
			   this->pictureBox1->Size = System::Drawing::Size(937, 300);
			   this->pictureBox1->SizeMode = System::Windows::Forms::PictureBoxSizeMode::Zoom;
			   this->pictureBox1->TabIndex = 1;
			   this->pictureBox1->TabStop = false;
			   // 
			   // processingWorker
			   // 
			   this->processingWorker->WorkerSupportsCancellation = true;
			   this->processingWorker->DoWork += gcnew System::ComponentModel::DoWorkEventHandler(this, &UploadForm::processingWorker_DoWork);
			   // 
			   // button3
			   // 
			   this->button3->BackColor = System::Drawing::Color::IndianRed;
			   this->button3->Location = System::Drawing::Point(869, 31);
			   this->button3->Name = L"button3";
			   this->button3->Size = System::Drawing::Size(70, 29);
			   this->button3->TabIndex = 3;
			   this->button3->Text = L"live";
			   this->button3->UseVisualStyleBackColor = false;
			   // 
			   // UploadForm
			   // 
			   this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			   this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			   this->ClientSize = System::Drawing::Size(1422, 332);
			   this->Controls->Add(this->button3);
			   this->Controls->Add(this->pictureBox1);
			   this->Controls->Add(this->button1);
			   this->Controls->Add(this->button2);
			   this->Name = L"UploadForm";
			   this->Text = L"Upload Window - Loading Model...";
			   this->FormClosing += gcnew System::Windows::Forms::FormClosingEventHandler(this, &UploadForm::UploadForm_FormClosing);
			   (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox1))->EndInit();
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

		   // [FIX] Worker Loop: ส่ง Frame Sequence ไปด้วย
	private: System::Void processingWorker_DoWork(System::Object^ sender, DoWorkEventArgs^ e) {
		BackgroundWorker^ worker = safe_cast<BackgroundWorker^>(sender);
		while (!shouldStop && !worker->CancellationPending) {
			try {
				cv::Mat frameToProcess;
				long long seq = 0;
				GetLatestRawFrameCopy(frameToProcess, seq); // เอาเลข Seq มา

				if (!frameToProcess.empty()) {
					DetectObjectsOnFrame(frameToProcess, seq); // ส่งเลข Seq ไป
				}
				else {
					Threading::Thread::Sleep(10);
				}
			}
			catch (...) { Threading::Thread::Sleep(50); }
		}
	}

		   // [FIX] Timer Loop: เช็ค Frame Sequence ก่อนวาด
	private: System::Void timer1_Tick(System::Object^ sender, System::EventArgs^ e) {
		try {
			if (!ReadNextVideoFrame()) { StopProcessing(); return; }

			cv::Mat displayFrame;
			long long displaySeq = 0;
			GetLatestRawFrameCopy(displayFrame, displaySeq); // เอาเลข Seq ของภาพที่จะฉายมา

			if (!displayFrame.empty()) {
				// ส่ง displaySeq เข้าไปเช็ค ถ้ากล่องเป็นอนาคต (seq มากกว่า) มันจะไม่วาด
				cv::Mat result = DrawPersistentDetections(displayFrame, displaySeq);

				if (!result.empty()) {
					Bitmap^ newFrame = MatToBitmap(result);
					if (newFrame != nullptr) {
						Monitor::Enter(bufferLock);
						try {
							if (currentFrame != nullptr) delete currentFrame;
							currentFrame = newFrame;
							if (pictureBox1->Image) delete pictureBox1->Image;
							pictureBox1->Image = gcnew Bitmap(currentFrame);
						}
						finally { Monitor::Exit(bufferLock); }
					}
				}
			}
		}
		catch (...) {}
	}

	private: void StartProcessing() {
		shouldStop = false;
		isProcessing = true;
		{
			std::lock_guard<std::mutex> lock(g_detectionMutex);
			g_persistentBoxes.clear();
			g_persistentClassIds.clear();
			g_detectionSourceSeq = -1;
		}
		if (!processingWorker->IsBusy) processingWorker->RunWorkerAsync();
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
			InitGlobalModel(modelPath);
			e->Result = true;
		}
		catch (const std::exception& ex) { e->Result = gcnew System::String(ex.what()); }
	}

	private: System::Void LoadModel_Completed(System::Object^ sender, RunWorkerCompletedEventArgs^ e) {
		if (e->Result != nullptr && e->Result->GetType() == bool::typeid && safe_cast<bool>(e->Result)) {
			this->Text = L"Upload Window - YOLO Detection (Ready)";
			button1->Enabled = true; button2->Enabled = true;
			MessageBox::Show("Model loaded!", "Success", MessageBoxButtons::OK, MessageBoxIcon::Information);
		}
		else {
			MessageBox::Show("Error loading model", "Error", MessageBoxButtons::OK, MessageBoxIcon::Error);
		}
	}

	private: System::Void button1_Click(System::Object^ sender, System::EventArgs^ e) {
		StopProcessing();
		OpenFileDialog^ ofd = gcnew OpenFileDialog();
		ofd->Filter = "Image Files|*.jpg;*.png;*.jpeg;*.bmp";
		if (ofd->ShowDialog() == System::Windows::Forms::DialogResult::OK) {
			std::string fileName = msclr::interop::marshal_as<std::string>(ofd->FileName);
			cv::Mat img = cv::imread(fileName);
			if (!img.empty()) {
				// สำหรับรูปนิ่ง ส่ง seq สูงๆ ไปเลยเพื่อให้วาดได้เสมอ
				DetectObjectsOnFrame(img, 999999);
				cv::Mat result = DrawPersistentDetections(img, 999999);
				pictureBox1->Image = MatToBitmap(result);
			}
		}
	}

	private: System::Void button2_Click(System::Object^ sender, System::EventArgs^ e) {
		StopProcessing();
		OpenFileDialog^ ofd = gcnew OpenFileDialog();
		ofd->Filter = "Video Files|*.mp4;*.avi;*.mkv";
		if (ofd->ShowDialog() == System::Windows::Forms::DialogResult::OK) {
			std::string fileName = msclr::interop::marshal_as<std::string>(ofd->FileName);
			OpenGlobalVideo(fileName);
			if (g_cap && g_cap->isOpened()) StartProcessing();
		}
	}

	private: System::Void UploadForm_FormClosing(System::Object^ sender, FormClosingEventArgs^ e) {
		StopProcessing();
	}
	};
}