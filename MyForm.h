#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include "online1.h"
#include "popup1.h"
#include "ParkingSetupForm.h"

namespace ConsoleApplication3 {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;

	public ref class MyForm : public System::Windows::Forms::Form
	{
	public:
		MyForm(void)
		{
			InitializeComponent();
		}

	protected:
		~MyForm()
		{
			if (components)
			{
				delete components;
			}
		}

	private: System::Windows::Forms::Button^ button1;
	private: System::Windows::Forms::Button^ button2;
	private: System::Windows::Forms::Timer^ timer1;
	private: System::Windows::Forms::PictureBox^ pictureBox1;
	private: BackgroundWorker^ processingWorker;
	private: System::Windows::Forms::MenuStrip^ menuStrip1;
	private: System::Windows::Forms::OpenFileDialog^ openFileDialog;
	private: System::Windows::Forms::SaveFileDialog^ saveFileDialog;
	private: System::Windows::Forms::ToolStripMenuItem^ uploadToolStripMenuItem;
	private: System::Windows::Forms::SplitContainer^ splitContainer1;
	private: System::Windows::Forms::Label^ label2;
	private: System::Windows::Forms::Label^ label1;
	private: System::Windows::Forms::Label^ label3;
	private: System::Windows::Forms::Label^ label4;
	private: System::Windows::Forms::Button^ btnID3NormalZone;
	private: System::Windows::Forms::Button^ btnID2NormalZone;


	private: System::Windows::Forms::Button^ btnSeeCamera;


	private: System::Windows::Forms::Button^ btnParkingSetup;
	private: System::ComponentModel::Container^ components;


	private:
		Bitmap^ bmp;

	private: System::Windows::Forms::Label^ lblAppName;
	private: System::Windows::Forms::Label^ lblAppDescription;
	private: System::Windows::Forms::PictureBox^ pictureBoxLogo;

#pragma region Windows Form Designer generated code
		void InitializeComponent(void)
		{
			this->btnSeeCamera = (gcnew System::Windows::Forms::Button());
			this->btnParkingSetup = (gcnew System::Windows::Forms::Button());
			this->lblAppName = (gcnew System::Windows::Forms::Label());
			this->lblAppDescription = (gcnew System::Windows::Forms::Label());
			this->pictureBoxLogo = (gcnew System::Windows::Forms::PictureBox());
			this->openFileDialog = (gcnew System::Windows::Forms::OpenFileDialog());
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBoxLogo))->BeginInit();
			this->SuspendLayout();
			// 
			// btnSeeCamera
			// 
			this->btnSeeCamera->Anchor = System::Windows::Forms::AnchorStyles::None;
			this->btnSeeCamera->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(76)), static_cast<System::Int32>(static_cast<System::Byte>(175)),
				static_cast<System::Int32>(static_cast<System::Byte>(80)));
			this->btnSeeCamera->FlatAppearance->BorderSize = 0;
			this->btnSeeCamera->FlatAppearance->MouseOverBackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(56)),
				static_cast<System::Int32>(static_cast<System::Byte>(142)), static_cast<System::Int32>(static_cast<System::Byte>(60)));
			this->btnSeeCamera->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
			this->btnSeeCamera->Font = (gcnew System::Drawing::Font(L"Segoe UI", 18, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->btnSeeCamera->ForeColor = System::Drawing::Color::White;
			this->btnSeeCamera->Location = System::Drawing::Point(420, 550);
			this->btnSeeCamera->Name = L"btnSeeCamera";
			this->btnSeeCamera->Size = System::Drawing::Size(320, 80);
			this->btnSeeCamera->TabIndex = 3;
			this->btnSeeCamera->Text = L"📹 See Camera";
			this->btnSeeCamera->UseVisualStyleBackColor = false;
			this->btnSeeCamera->Click += gcnew System::EventHandler(this, &MyForm::btnSeeCamera_Click);
			// 
			// btnParkingSetup
			// 
			this->btnParkingSetup->Anchor = System::Windows::Forms::AnchorStyles::None;
			this->btnParkingSetup->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(255)), static_cast<System::Int32>(static_cast<System::Byte>(152)),
				static_cast<System::Int32>(static_cast<System::Byte>(0)));
			this->btnParkingSetup->FlatAppearance->BorderSize = 0;
			this->btnParkingSetup->FlatAppearance->MouseOverBackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(230)),
				static_cast<System::Int32>(static_cast<System::Byte>(124)), static_cast<System::Int32>(static_cast<System::Byte>(0)));
			this->btnParkingSetup->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
			this->btnParkingSetup->Font = (gcnew System::Drawing::Font(L"Segoe UI", 18, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->btnParkingSetup->ForeColor = System::Drawing::Color::White;
			this->btnParkingSetup->Location = System::Drawing::Point(703, 550);
			this->btnParkingSetup->Name = L"btnParkingSetup";
			this->btnParkingSetup->Size = System::Drawing::Size(320, 80);
			this->btnParkingSetup->TabIndex = 4;
			this->btnParkingSetup->Text = L"🅿️ Parking Setup";
			this->btnParkingSetup->UseVisualStyleBackColor = false;
			this->btnParkingSetup->Click += gcnew System::EventHandler(this, &MyForm::btnParkingSetup_Click);
			// 
			// lblAppName
			// 
			this->lblAppName->Anchor = System::Windows::Forms::AnchorStyles::Top;
			this->lblAppName->Font = (gcnew System::Drawing::Font(L"Segoe UI", 48, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->lblAppName->ForeColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(33)), static_cast<System::Int32>(static_cast<System::Byte>(150)),
				static_cast<System::Int32>(static_cast<System::Byte>(243)));
			this->lblAppName->Location = System::Drawing::Point(400, 210);
			this->lblAppName->Name = L"lblAppName";
			this->lblAppName->Size = System::Drawing::Size(650, 80);
			this->lblAppName->TabIndex = 1;
			this->lblAppName->Text = L"ParkEnforcer";
			this->lblAppName->TextAlign = System::Drawing::ContentAlignment::MiddleCenter;
			// 
			// lblAppDescription
			// 
			this->lblAppDescription->Anchor = System::Windows::Forms::AnchorStyles::Top;
			this->lblAppDescription->Font = (gcnew System::Drawing::Font(L"Segoe UI", 16, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->lblAppDescription->ForeColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(80)),
				static_cast<System::Int32>(static_cast<System::Byte>(80)), static_cast<System::Int32>(static_cast<System::Byte>(80)));
			this->lblAppDescription->Location = System::Drawing::Point(250, 300);
			this->lblAppDescription->Name = L"lblAppDescription";
			this->lblAppDescription->Size = System::Drawing::Size(950, 120);
			this->lblAppDescription->TabIndex = 2;
			this->lblAppDescription->Text = L"🚗 แอพพลิเคชันตรวจจับรถที่จอดผิดที่และจอดเกินเวลา\r\n\r\n📍 ระบุพื้นที่จอดรถด้วยการวา"
				L"ดบนแผนที่ (Parking Setup)\r\n📹 ตรวจสอบสถานะเรียลไทม์ผ่านกล้องวิดีโอ (See Camera)";
			this->lblAppDescription->TextAlign = System::Drawing::ContentAlignment::MiddleCenter;
			// 
			// pictureBoxLogo
			// 
			this->pictureBoxLogo->Anchor = System::Windows::Forms::AnchorStyles::Top;
			this->pictureBoxLogo->BackColor = System::Drawing::Color::Transparent;
			this->pictureBoxLogo->Font = (gcnew System::Drawing::Font(L"Arial", 80, System::Drawing::FontStyle::Bold));
			this->pictureBoxLogo->ForeColor = System::Drawing::Color::White;
			this->pictureBoxLogo->Location = System::Drawing::Point(600, 80);
			this->pictureBoxLogo->Name = L"pictureBoxLogo";
			this->pictureBoxLogo->Size = System::Drawing::Size(250, 120);
			this->pictureBoxLogo->SizeMode = System::Windows::Forms::PictureBoxSizeMode::CenterImage;
			this->pictureBoxLogo->TabIndex = 0;
			this->pictureBoxLogo->TabStop = false;
			this->pictureBoxLogo->Text = L"🅿️";
			this->pictureBoxLogo->Click += gcnew System::EventHandler(this, &MyForm::pictureBoxLogo_Click);
			// 
			// openFileDialog
			// 
			this->openFileDialog->FileName = L"openFileDialog";
			this->openFileDialog->Filter = L"Image files|*.jpg;*.png;*.jpeg;*.bmp";
			// 
			// MyForm
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(245)), static_cast<System::Int32>(static_cast<System::Byte>(245)),
				static_cast<System::Int32>(static_cast<System::Byte>(245)));
			this->ClientSize = System::Drawing::Size(1443, 759);
			this->Controls->Add(this->btnParkingSetup);
			this->Controls->Add(this->btnSeeCamera);
			this->Controls->Add(this->lblAppDescription);
			this->Controls->Add(this->lblAppName);
			this->Controls->Add(this->pictureBoxLogo);
			this->Name = L"MyForm";
			this->StartPosition = System::Windows::Forms::FormStartPosition::CenterScreen;
			this->Text = L"ParkEnforcer - Parking Violation Detection System";
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBoxLogo))->EndInit();
			this->ResumeLayout(false);

		}
#pragma endregion

	private: System::Void btnSeeCamera_Click(System::Object^ sender, System::EventArgs^ e) {
		popup1^ form = gcnew popup1();
		form->ShowDialog();
	}

	private: System::Void btnParkingSetup_Click(System::Object^ sender, System::EventArgs^ e) {
		ParkingSetupForm^ form = gcnew ParkingSetupForm();
		form->ShowDialog();
	}
	private: System::Void pictureBoxLogo_Click(System::Object^ sender, System::EventArgs^ e) {
	}
};
}
