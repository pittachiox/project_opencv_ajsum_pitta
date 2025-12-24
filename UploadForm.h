#pragma once
#include <msclr/marshal_cppstd.h>
#include <string>
#include <vector>

// Use global variables in unmanaged block to avoid struct metadata issues
#pragma managed(push, off)
// Prevent Windows.h min/max macros from interfering with std::min/std::max
#define NOMINMAX
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <cstdlib> // For rand()
#include <algorithm> // For min, max
#include <cmath>     // For round

// Use static to avoid linker errors if included in multiple translation units
static cv::dnn::Net* g_net = nullptr;
static std::vector<std::string> g_classes;
static std::vector<cv::Scalar> g_colors;
static cv::Mat g_currentImage;
static cv::VideoCapture* g_cap = nullptr; // Add VideoCapture

// Helper function for Letterbox resizing (maintains aspect ratio)
static cv::Mat FormatToLetterbox(const cv::Mat& source, int width, int height, float& ratio, int& dw, int& dh) {
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
		resized = source;
	}

	// Create canvas with gray padding (114 is standard for YOLO)
	cv::Mat result(height, width, CV_8UC3, cv::Scalar(114, 114, 114));
	resized.copyTo(result(cv::Rect(dw, dh, new_unpad_w, new_unpad_h)));

	ratio = r;
	return result;
}

static void InitGlobalModel(const std::string& modelPath) {
	if (g_net) {
		delete g_net;
		g_net = nullptr;
	}
	g_net = new cv::dnn::Net(cv::dnn::readNetFromONNX(modelPath));
	g_net->setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	g_net->setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

	// Default to COCO classes (80 classes) which is standard for yolo11n.onnx
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

	// Generate random colors
	g_colors.clear();
	for (size_t i = 0; i < g_classes.size(); i++) {
		g_colors.push_back(cv::Scalar(rand() % 255, rand() % 255, rand() % 255));
	}
}

// Refactored: Core detection logic separated from image loading
static void DetectObjects() {
	if (g_currentImage.empty()) return;
	if (!g_net) return;

	// Use Letterbox resizing instead of direct stretching
	float ratio;
	int dw, dh;
	cv::Mat input_image = FormatToLetterbox(g_currentImage, 640, 640, ratio, dw, dh);

	cv::Mat blob;
	cv::dnn::blobFromImage(input_image, blob, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true, false);
	g_net->setInput(blob);

	std::vector<cv::Mat> outputs;
	g_net->forward(outputs, g_net->getUnconnectedOutLayersNames());

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

			if (max_class_score > 0.25) {
				float x = data[0]; float y = data[1]; float w = data[2]; float h = data[3];

				// Calculate coordinates relative to the original image (undo letterbox)
				float left = (x - 0.5 * w - dw) / ratio;
				float top = (y - 0.5 * h - dh) / ratio;
				float width = w / ratio;
				float height = h / ratio;

				// Clamp to image boundaries
				left = (std::max)(0.0f, left);
				top = (std::max)(0.0f, top);
				if (left + width > g_currentImage.cols) width = g_currentImage.cols - left;
				if (top + height > g_currentImage.rows) height = g_currentImage.rows - top;

				boxes.push_back(cv::Rect((int)left, (int)top, (int)width, (int)height));
				confs.push_back((float)max_class_score);
				class_ids.push_back(class_id.x);
			}
		}
		data += dims;
	}

	std::vector<int> nms;
	// Score Threshold (0.25) and NMS Threshold (0.45)
	cv::dnn::NMSBoxes(boxes, confs, 0.25f, 0.45f, nms);

	for (int idx : nms) {
		cv::rectangle(g_currentImage, boxes[idx], g_colors[class_ids[idx]], 2);
		cv::putText(g_currentImage, g_classes[class_ids[idx]], cv::Point(boxes[idx].x, boxes[idx].y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.6, g_colors[class_ids[idx]], 2);
	}
}

static void ProcessGlobalImage(const std::string& filename) {
	g_currentImage = cv::imread(filename);
	DetectObjects();
}

static void OpenGlobalVideo(const std::string& filename) {
	if (g_cap) {
		delete g_cap;
		g_cap = nullptr;
	}
	g_cap = new cv::VideoCapture(filename);
}

static bool ProcessNextFrame() {
	if (!g_cap || !g_cap->isOpened()) return false;
	
	cv::Mat frame;
	*g_cap >> frame;
	if (frame.empty()) return false;

	g_currentImage = frame;
	DetectObjects();
	return true;
}

static bool IsGlobalImageEmpty() { return g_currentImage.empty(); }
static int GetGlobalWidth() { return g_currentImage.cols; }
static int GetGlobalHeight() { return g_currentImage.rows; }
static int GetGlobalStep() { return (int)g_currentImage.step; }
static void* GetGlobalData() { return g_currentImage.data; }

static void CopyToBuffer(void* buffer, int stride) {
	if (g_currentImage.empty()) return;
	cv::Mat dst(g_currentImage.rows, g_currentImage.cols, CV_8UC3, buffer, stride);
	g_currentImage.copyTo(dst);
}

#pragma managed(pop)

namespace ConsoleApplication3 {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;

	public ref class UploadForm : public System::Windows::Forms::Form
	{
	public:
		UploadForm(void)
		{
			InitializeComponent();
			
			// Initialize YOLO Model
			try {
				std::string modelPath = "C:/Users/kt856/Downloads/yolo11m.onnx";
				InitGlobalModel(modelPath);
			}
			catch (const std::exception& e) {
				System::String^ errorMsg = gcnew System::String(e.what());
				if (errorMsg->Contains("ConcatLayer")) {
					errorMsg += "\n\n[Tip] This error is caused by 'dynamic=True' in your Python export code.\nPlease change it to 'dynamic=False' and re-export the model.";
				}
				MessageBox::Show("Error Model: " + errorMsg);
			}
			catch (...) {
				MessageBox::Show("Error Model: Unknown error");
			}
		}

	protected:
		~UploadForm()
		{
			if (components)
			{
				delete components;
			}
		}
	private: System::Windows::Forms::Button^ button1;
	private: System::Windows::Forms::Button^ button2; // New Button for Video
	private: System::Windows::Forms::Timer^ timer1;   // Timer for Video Playback
	private: System::Windows::Forms::PictureBox^ pictureBox1;
	protected:

	private:
		System::ComponentModel::Container^ components;

#pragma region Windows Form Designer generated code
		void InitializeComponent(void)
		{
			this->components = (gcnew System::ComponentModel::Container());
			this->button1 = (gcnew System::Windows::Forms::Button());
			this->button2 = (gcnew System::Windows::Forms::Button());
			this->timer1 = (gcnew System::Windows::Forms::Timer(this->components));
			this->pictureBox1 = (gcnew System::Windows::Forms::PictureBox());
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox1))->BeginInit();
			this->SuspendLayout();
			// 
			// button1
			// 
			this->button1->Location = System::Drawing::Point(50, 209);
			this->button1->Name = L"button1";
			this->button1->Size = System::Drawing::Size(100, 23);
			this->button1->TabIndex = 0;
			this->button1->Text = L"Upload Image";
			this->button1->UseVisualStyleBackColor = true;
			this->button1->Click += gcnew System::EventHandler(this, &UploadForm::button1_Click);
			// 
			// button2
			// 
			this->button2->Location = System::Drawing::Point(180, 209);
			this->button2->Name = L"button2";
			this->button2->Size = System::Drawing::Size(100, 23);
			this->button2->TabIndex = 2;
			this->button2->Text = L"Upload Video";
			this->button2->UseVisualStyleBackColor = true;
			this->button2->Click += gcnew System::EventHandler(this, &UploadForm::button2_Click);
			// 
			// timer1
			// 
			this->timer1->Interval = 33;
			this->timer1->Tick += gcnew System::EventHandler(this, &UploadForm::timer1_Tick);
			// 
			// pictureBox1
			// 
			this->pictureBox1->Location = System::Drawing::Point(21, 24);
			this->pictureBox1->Name = L"pictureBox1";
			this->pictureBox1->Size = System::Drawing::Size(329, 179);
			this->pictureBox1->SizeMode = System::Windows::Forms::PictureBoxSizeMode::Zoom;
			this->pictureBox1->TabIndex = 1;
			this->pictureBox1->TabStop = false;
			// 
			// UploadForm
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(8, 16);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(382, 253);
			this->Controls->Add(this->pictureBox1);
			this->Controls->Add(this->button1);
			this->Controls->Add(this->button2);
			this->Name = L"UploadForm";
			this->Text = L"Upload Window";
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox1))->EndInit();
			this->ResumeLayout(false);

		}
#pragma endregion
	// Helper to update PictureBox from global image
	private: void UpdateDisplay() {
		if (IsGlobalImageEmpty()) return;

		int w = GetGlobalWidth();
		int h = GetGlobalHeight();
		Bitmap^ bmp = gcnew Bitmap(w, h, System::Drawing::Imaging::PixelFormat::Format24bppRgb);

		System::Drawing::Rectangle rect = System::Drawing::Rectangle(0, 0, w, h);
		System::Drawing::Imaging::BitmapData^ bmpData = bmp->LockBits(rect, System::Drawing::Imaging::ImageLockMode::WriteOnly, bmp->PixelFormat);

		CopyToBuffer(bmpData->Scan0.ToPointer(), bmpData->Stride);

		bmp->UnlockBits(bmpData);

		// Clean up old image to prevent memory leaks
		if (pictureBox1->Image) delete pictureBox1->Image;
		pictureBox1->Image = bmp;
	}

	private: System::Void button1_Click(System::Object^ sender, System::EventArgs^ e) {
		timer1->Stop(); // Stop video if running
		OpenFileDialog^ ofd = gcnew OpenFileDialog();
		ofd->Filter = "Image Files|*.jpg;*.png;*.jpeg";

		if (ofd->ShowDialog() == System::Windows::Forms::DialogResult::OK) {
			if (g_net == nullptr) {
				MessageBox::Show("Warning: AI Model failed to load.", "Warning", MessageBoxButtons::OK, MessageBoxIcon::Warning);
			}

			std::string fileName = msclr::interop::marshal_as<std::string>(ofd->FileName);

			try {
				ProcessGlobalImage(fileName);
				UpdateDisplay();
			}
			catch (const std::exception& ex) {
				MessageBox::Show("Error Processing: " + gcnew System::String(ex.what()));
			}
		}
	}

	private: System::Void button2_Click(System::Object^ sender, System::EventArgs^ e) {
		OpenFileDialog^ ofd = gcnew OpenFileDialog();
		ofd->Filter = "Video Files|*.mp4;*.avi;*.mkv";

		if (ofd->ShowDialog() == System::Windows::Forms::DialogResult::OK) {
			if (g_net == nullptr) {
				MessageBox::Show("Warning: AI Model failed to load.", "Warning", MessageBoxButtons::OK, MessageBoxIcon::Warning);
			}

			std::string fileName = msclr::interop::marshal_as<std::string>(ofd->FileName);

			try {
				OpenGlobalVideo(fileName);
				timer1->Start();
			}
			catch (const std::exception& ex) {
				MessageBox::Show("Error Opening Video: " + gcnew System::String(ex.what()));
			}
		}
	}

	private: System::Void timer1_Tick(System::Object^ sender, System::EventArgs^ e) {
		if (!ProcessNextFrame()) {
			timer1->Stop();
			return;
		}
		UpdateDisplay();
	}
	};
}