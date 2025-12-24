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

static cv::dnn::Net* g_net = nullptr;
static std::vector<std::string> g_classes;
static std::vector<cv::Scalar> g_colors;
static cv::VideoCapture* g_cap = nullptr;
static cv::Mat g_latestRawFrame;
static cv::Mat g_latestProcessedFrame;
static bool g_hasNewProcessedFrame = false;
static std::mutex g_frameMutex;
static std::mutex g_processMutex;

static const int YOLO_INPUT_SIZE = 256;
static const float CONF_THRESHOLD = 0.3f;
static const float NMS_THRESHOLD = 0.5f;

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
	
	if (g_net) {
		delete g_net;
		g_net = nullptr;
	}
	g_net = new cv::dnn::Net(cv::dnn::readNetFromONNX(modelPath));
	g_net->setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	g_net->setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

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
}

static cv::Mat DetectObjectsOnFrame(const cv::Mat& inputFrame) {
	std::lock_guard<std::mutex> lock(g_processMutex);
	
	if (inputFrame.empty() || !g_net) return inputFrame.clone();

	try {
		cv::Mat workingImage = inputFrame.clone();
		
		float ratio;
		int dw, dh;
		cv::Mat input_image = FormatToLetterbox(workingImage, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE, ratio, dw, dh);
		
		if (input_image.empty()) return workingImage;

		cv::Mat blob;
		cv::dnn::blobFromImage(input_image, blob, 1.0 / 255.0, cv::Size(YOLO_INPUT_SIZE, YOLO_INPUT_SIZE), cv::Scalar(), true, false);
		g_net->setInput(blob);

		std::vector<cv::Mat> outputs;
		g_net->forward(outputs, g_net->getUnconnectedOutLayersNames());

		if (outputs.empty() || outputs[0].empty()) return workingImage;

		cv::Mat output_data = outputs[0];
		cv::Mat output_t;
		cv::transpose(output_data.reshape(1, output_data.size[1]), output_t);
		output_data = output_t;

		float* data = (float*)output_data.data;
		int rows = output_data.rows;
		int dims = output_data.cols;

		std::vector<int> class_ids;
		std::vector<float> confs;
		std::vector<cv::Rect> boxes;

		for (int i = 0; i < rows; i++) {
			float* classes_scores = data + 4;
			if (dims >= 4 + (int)g_classes.size()) {
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

					left = (std::max)(0.0f, left);
					top = (std::max)(0.0f, top);
					if (left + width > workingImage.cols) width = workingImage.cols - left;
					if (top + height > workingImage.rows) height = workingImage.rows - top;

					boxes.push_back(cv::Rect((int)left, (int)top, (int)width, (int)height));
					confs.push_back((float)max_class_score);
					class_ids.push_back(class_id.x);
				}
			}
			data += dims;
		}

		std::vector<int> nms;
		cv::dnn::NMSBoxes(boxes, confs, CONF_THRESHOLD, NMS_THRESHOLD, nms);

		for (int idx : nms) {
			if (idx >= 0 && idx < (int)boxes.size()) {
				cv::rectangle(workingImage, boxes[idx], g_colors[class_ids[idx]], 2);
				
				std::string label = g_classes[class_ids[idx]];
				int baseline;
				cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
				
				cv::rectangle(workingImage, 
				             cv::Point(boxes[idx].x, boxes[idx].y - textSize.height - 10),
				             cv::Point(boxes[idx].x + textSize.width, boxes[idx].y),
				             g_colors[class_ids[idx]], -1);
				
				cv::putText(workingImage, label, cv::Point(boxes[idx].x, boxes[idx].y - 5), 
				           cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
			}
		}
		
		return workingImage;
	}
	catch (...) {
		return inputFrame.clone();
	}
}

static void OpenGlobalVideo(const std::string& filename) {
	std::lock_guard<std::mutex> lock(g_frameMutex);
	if (g_cap) {
		g_cap->release();
		delete g_cap;
		g_cap = nullptr;
	}
	g_cap = new cv::VideoCapture(filename);
}

static bool ReadNextVideoFrame() {
	std::lock_guard<std::mutex> lock(g_frameMutex);
	if (!g_cap || !g_cap->isOpened()) return false;
	
	cv::Mat frame;
	*g_cap >> frame;
	if (frame.empty()) return false;
	
	g_latestRawFrame = frame;
	return true;
}

static cv::Mat GetLatestRawFrameCopy() {
	std::lock_guard<std::mutex> lock(g_frameMutex);
	return g_latestRawFrame.clone();
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

	public ref class UploadForm1 : public System::Windows::Forms::Form
	{
	public:
		UploadForm1(void)
		{
			InitializeComponent();
			
			bufferLock = gcnew Object();
			currentFrame = nullptr;
			isProcessing = false;
			shouldStop = false;
			
			try {
				std::string modelPath = "C:/Users/HP/source/repos/project_opencv_ajsum_fina/models/test/yolo11m.onnx";
				InitGlobalModel(modelPath);
			}
			catch (...) {
				MessageBox::Show("Error loading YOLO model", "Error", MessageBoxButtons::OK, MessageBoxIcon::Error);
			}
		}

	protected:
		~UploadForm1()
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
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox1))->BeginInit();
			this->SuspendLayout();
			// 
			// button1
			// 
			this->button1->Location = System::Drawing::Point(30, 170);
			this->button1->Name = L"button1";
			this->button1->Size = System::Drawing::Size(100, 25);
			this->button1->TabIndex = 0;
			this->button1->Text = L"Upload Image";
			this->button1->UseVisualStyleBackColor = true;
			this->button1->Click += gcnew System::EventHandler(this, &UploadForm1::button1_Click);
			// 
			// button2
			// 
			this->button2->Location = System::Drawing::Point(150, 170);
			this->button2->Name = L"button2";
			this->button2->Size = System::Drawing::Size(100, 25);
			this->button2->TabIndex = 2;
			this->button2->Text = L"Upload Video";
			this->button2->UseVisualStyleBackColor = true;
			this->button2->Click += gcnew System::EventHandler(this, &UploadForm1::button2_Click);
			// 
			// timer1
			// 
			this->timer1->Interval = 30;
			this->timer1->Tick += gcnew System::EventHandler(this, &UploadForm1::timer1_Tick);
			// 
			// pictureBox1
			// 
			this->pictureBox1->Location = System::Drawing::Point(16, 20);
			this->pictureBox1->Name = L"pictureBox1";
			this->pictureBox1->Size = System::Drawing::Size(247, 145);
			this->pictureBox1->SizeMode = System::Windows::Forms::PictureBoxSizeMode::Zoom;
			this->pictureBox1->TabIndex = 1;
			this->pictureBox1->TabStop = false;
			// 
			// processingWorker
			// 
			this->processingWorker->WorkerSupportsCancellation = true;
			this->processingWorker->DoWork += gcnew System::ComponentModel::DoWorkEventHandler(this, &UploadForm1::processingWorker_DoWork);
			// 
			// UploadForm1
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(280, 210);
			this->Controls->Add(this->pictureBox1);
			this->Controls->Add(this->button1);
			this->Controls->Add(this->button2);
			this->Name = L"UploadForm1";
			this->Text = L"Upload Window";
			this->FormClosing += gcnew System::Windows::Forms::FormClosingEventHandler(this, &UploadForm1::UploadForm1_FormClosing);
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox1))->EndInit();
			this->ResumeLayout(false);

		}
#pragma endregion

	private: Bitmap^ MatToBitmap(cv::Mat& mat) {
		if (mat.empty()) return nullptr;
		
		try {
			int w = mat.cols;
			int h = mat.rows;
			if (w <= 0 || h <= 0) return nullptr;
			
			Bitmap^ bmp = gcnew Bitmap(w, h, System::Drawing::Imaging::PixelFormat::Format24bppRgb);
			System::Drawing::Rectangle rect = System::Drawing::Rectangle(0, 0, w, h);
			System::Drawing::Imaging::BitmapData^ bmpData = bmp->LockBits(rect, System::Drawing::Imaging::ImageLockMode::WriteOnly, bmp->PixelFormat);
			
			for (int y = 0; y < h; y++) {
				memcpy((unsigned char*)bmpData->Scan0.ToPointer() + y * bmpData->Stride,
				       mat.data + y * mat.step, w * 3);
			}
			
			bmp->UnlockBits(bmpData);
			return bmp;
		}
		catch (...) {
			return nullptr;
		}
	}

	private: System::Void processingWorker_DoWork(System::Object^ sender, DoWorkEventArgs^ e) {
		BackgroundWorker^ worker = safe_cast<BackgroundWorker^>(sender);
		
		while (!shouldStop && !worker->CancellationPending) {
			try {
				cv::Mat rawFrame = GetLatestRawFrameCopy();
				
				if (!rawFrame.empty()) {
					cv::Mat processedFrame = DetectObjectsOnFrame(rawFrame);
					
					if (!processedFrame.empty()) {
						g_latestProcessedFrame = processedFrame;
						g_hasNewProcessedFrame = true;
					}
				}
				else {
					Threading::Thread::Sleep(10);
				}
			}
			catch (...) {
				Threading::Thread::Sleep(50);
			}
		}
	}

	private: System::Void timer1_Tick(System::Object^ sender, System::EventArgs^ e) {
		try {
			if (!ReadNextVideoFrame()) {
				StopProcessing();
				return;
			}
			
			cv::Mat displayFrame;
			
			if (g_hasNewProcessedFrame && !g_latestProcessedFrame.empty()) {
				displayFrame = g_latestProcessedFrame.clone();
				g_hasNewProcessedFrame = false;
			}
			else {
				displayFrame = GetLatestRawFrameCopy();
			}
			
			if (!displayFrame.empty()) {
				Bitmap^ newFrame = MatToBitmap(displayFrame);
				
				if (newFrame != nullptr) {
					Monitor::Enter(bufferLock);
					try {
						if (currentFrame != nullptr) delete currentFrame;
						currentFrame = newFrame;
						
						if (pictureBox1->Image) delete pictureBox1->Image;
						pictureBox1->Image = gcnew Bitmap(currentFrame);
					}
					finally {
						Monitor::Exit(bufferLock);
					}
				}
			}
		}
		catch (...) {
		}
	}

	private: void StartProcessing() {
		shouldStop = false;
		isProcessing = true;
		g_hasNewProcessedFrame = false;
		
		timer1->Start();
		if (!processingWorker->IsBusy) {
			processingWorker->RunWorkerAsync();
		}
	}

	private: void StopProcessing() {
		shouldStop = true;
		isProcessing = false;
		timer1->Stop();
		
		if (processingWorker->IsBusy) {
			processingWorker->CancelAsync();
			Threading::Thread::Sleep(100);
		}
	}

	private: System::Void button1_Click(System::Object^ sender, System::EventArgs^ e) {
		StopProcessing();
		
		OpenFileDialog^ ofd = gcnew OpenFileDialog();
		ofd->Filter = "Image Files (*.jpg;*.png;*.jpeg;*.bmp)|*.jpg;*.png;*.jpeg;*.bmp";
		ofd->Title = "Select an Image";

		if (ofd->ShowDialog() == System::Windows::Forms::DialogResult::OK) {
			if (g_net == nullptr) {
				MessageBox::Show("AI Model not loaded. Please check model path.", "Warning", MessageBoxButtons::OK, MessageBoxIcon::Warning);
				return;
			}

			try {
				std::string fileName = msclr::interop::marshal_as<std::string>(ofd->FileName);
				cv::Mat img = cv::imread(fileName);
				
				if (img.empty()) {
					MessageBox::Show("Failed to load image. File may be corrupted.", "Error", MessageBoxButtons::OK, MessageBoxIcon::Error);
					return;
				}
				
				cv::Mat processed = DetectObjectsOnFrame(img);
				Bitmap^ bmp = MatToBitmap(processed);
				
				if (bmp != nullptr) {
					Monitor::Enter(bufferLock);
					try {
						if (currentFrame != nullptr) delete currentFrame;
						currentFrame = bmp;
						
						if (pictureBox1->Image) delete pictureBox1->Image;
						pictureBox1->Image = gcnew Bitmap(currentFrame);
					}
					finally {
						Monitor::Exit(bufferLock);
					}
				}
			}
			catch (const cv::Exception& ex) {
				MessageBox::Show("OpenCV Error: " + gcnew System::String(ex.what()), "Error", MessageBoxButtons::OK, MessageBoxIcon::Error);
			}
			catch (...) {
				MessageBox::Show("Error processing image", "Error", MessageBoxButtons::OK, MessageBoxIcon::Error);
			}
		}
	}

	private: System::Void button2_Click(System::Object^ sender, System::EventArgs^ e) {
		StopProcessing();
		
		OpenFileDialog^ ofd = gcnew OpenFileDialog();
		ofd->Filter = "Video Files (*.mp4;*.avi;*.mkv)|*.mp4;*.avi;*.mkv";
		ofd->Title = "Select a Video";
		
		if (ofd->ShowDialog() == System::Windows::Forms::DialogResult::OK) {
			if (g_net == nullptr) {
				MessageBox::Show("AI Model not loaded. Please check model path.", "Warning", MessageBoxButtons::OK, MessageBoxIcon::Warning);
				return;
			}

			try {
				std::string fileName = msclr::interop::marshal_as<std::string>(ofd->FileName);
				OpenGlobalVideo(fileName);
				
				if (!g_cap || !g_cap->isOpened()) {
					MessageBox::Show("Failed to open video file.", "Error", MessageBoxButtons::OK, MessageBoxIcon::Error);
					return;
				}
				
				StartProcessing();
			}
			catch (const cv::Exception& ex) {
				MessageBox::Show("OpenCV Error: " + gcnew System::String(ex.what()), "Error", MessageBoxButtons::OK, MessageBoxIcon::Error);
			}
			catch (...) {
				MessageBox::Show("Error opening video", "Error", MessageBoxButtons::OK, MessageBoxIcon::Error);
			}
		}
	}

	private: System::Void UploadForm1_FormClosing(System::Object^ sender, FormClosingEventArgs^ e) {
		StopProcessing();
	}
	};
}
