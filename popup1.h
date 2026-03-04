#pragma once
#include "online1.h"
#include "offline1.h"
#include <msclr/marshal_cppstd.h>

#pragma managed(push, off)
inline bool TriggerOnlineCameraHeadlessWrapper(std::string ip, std::string port, std::string path);
#pragma managed(pop)

inline bool TriggerOnlineCameraHeadlessWrapper(std::string ip, std::string port, std::string path) {
	if (ConsoleApplication3::UploadForm::Instance != nullptr) {
		System::String^ sysIp = msclr::interop::marshal_as<System::String^>(ip);
		System::String^ sysPort = msclr::interop::marshal_as<System::String^>(port);
		System::String^ sysPath = msclr::interop::marshal_as<System::String^>(path);
		return ConsoleApplication3::UploadForm::Instance->StartCameraHeadless(sysIp, sysPort, sysPath);
	}
	return false;
}

namespace ConsoleApplication3 {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;

	public ref class popup1 : public System::Windows::Forms::Form
	{
	public:
		popup1(void)
		{
			InitializeComponent();
			this->Load += gcnew System::EventHandler(this, &popup1::popup1_Load);
		}

	private: System::Void popup1_Load(System::Object^ sender, System::EventArgs^ e) {
		// Hide the GUI completely and run as a service
		this->Opacity = 0;
		this->ShowInTaskbar = false;
		
		// Wait a tiny bit and then hide properly
		System::Windows::Forms::Timer^ t = gcnew System::Windows::Forms::Timer();
		t->Interval = 100;
		t->Tick += gcnew System::EventHandler(this, &popup1::HideSelf);
		t->Start();

		// Start the global MjpegServer on port 8080
		if (!g_globalWebServer) {
			g_globalWebServer = new MjpegServer(8080);
			g_globalWebServer->Start();
		}

		// Initialize UploadForm silently
		if (UploadForm::Instance == nullptr) {
			UploadForm^ onlineForm = gcnew UploadForm();
			onlineForm->Hide();
		}

		// Connect the web API to the form's headless start method
		g_globalWebServer->SetConnectOnlineCallback(&TriggerOnlineCameraHeadlessWrapper);
		
		// Open the default browser to the web interface
		System::Diagnostics::Process::Start("http://localhost:8080/");
	}
	
	private: System::Void HideSelf(System::Object^ sender, System::EventArgs^ e) {
		System::Windows::Forms::Timer^ t = safe_cast<System::Windows::Forms::Timer^>(sender);
		t->Stop();
		this->Hide();
	}

	protected:
		~popup1()
		{
			if (components)
			{
				delete components;
			}
		}
	private: System::Windows::Forms::Panel^ panel1;
	private: System::Windows::Forms::Label^ label1;
	private: System::Windows::Forms::Button^ btnOffline;
	private: System::Windows::Forms::Button^ btnOnline;
	protected:

	private:
		System::ComponentModel::Container ^components;

#pragma region Windows Form Designer generated code
		void InitializeComponent(void)
		{
			this->panel1 = (gcnew System::Windows::Forms::Panel());
			this->label1 = (gcnew System::Windows::Forms::Label());
			this->btnOffline = (gcnew System::Windows::Forms::Button());
			this->btnOnline = (gcnew System::Windows::Forms::Button());
			this->panel1->SuspendLayout();
			this->SuspendLayout();
			// 
			// panel1
			// 
			this->panel1->BackColor = System::Drawing::SystemColors::ActiveCaptionText;
			this->panel1->Controls->Add(this->label1);
			this->panel1->Dock = System::Windows::Forms::DockStyle::Top;
			this->panel1->Location = System::Drawing::Point(0, 0);
			this->panel1->Name = L"panel1";
			this->panel1->Size = System::Drawing::Size(284, 50);
			this->panel1->TabIndex = 0;
			// 
			// label1
			// 
			this->label1->AutoSize = true;
			this->label1->Font = (gcnew System::Drawing::Font(L"Segoe UI", 12, System::Drawing::FontStyle::Bold));
			this->label1->ForeColor = System::Drawing::SystemColors::ButtonFace;
			this->label1->Location = System::Drawing::Point(70, 15);
			this->label1->Name = L"label1";
			this->label1->Size = System::Drawing::Size(144, 21);
			this->label1->TabIndex = 0;
			this->label1->Text = L"เลือกโหมดการดู";
			// 
			// btnOffline
			// 
			this->btnOffline->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(220)), 
				static_cast<System::Int32>(static_cast<System::Byte>(53)), static_cast<System::Int32>(static_cast<System::Byte>(69)));
			this->btnOffline->FlatAppearance->BorderSize = 0;
			this->btnOffline->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
			this->btnOffline->Font = (gcnew System::Drawing::Font(L"Segoe UI", 11, System::Drawing::FontStyle::Bold));
			this->btnOffline->ForeColor = System::Drawing::SystemColors::ButtonHighlight;
			this->btnOffline->Location = System::Drawing::Point(50, 80);
			this->btnOffline->Name = L"btnOffline";
			this->btnOffline->Size = System::Drawing::Size(180, 50);
			this->btnOffline->TabIndex = 1;
			this->btnOffline->Text = L"Offline\r\n(วิดีโอ/รูปภาพ)";
			this->btnOffline->UseVisualStyleBackColor = false;
			this->btnOffline->Click += gcnew System::EventHandler(this, &popup1::btnOffline_Click);
			// 
			// btnOnline
			// 
			this->btnOnline->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(40)), 
				static_cast<System::Int32>(static_cast<System::Byte>(167)), static_cast<System::Int32>(static_cast<System::Byte>(69)));
			this->btnOnline->FlatAppearance->BorderSize = 0;
			this->btnOnline->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
			this->btnOnline->Font = (gcnew System::Drawing::Font(L"Segoe UI", 11, System::Drawing::FontStyle::Bold));
			this->btnOnline->ForeColor = System::Drawing::SystemColors::ButtonHighlight;
			this->btnOnline->Location = System::Drawing::Point(50, 145);
			this->btnOnline->Name = L"btnOnline";
			this->btnOnline->Size = System::Drawing::Size(180, 50);
			this->btnOnline->TabIndex = 2;
			this->btnOnline->Text = L"Online\r\n(กล้องสด)";
			this->btnOnline->UseVisualStyleBackColor = false;
			this->btnOnline->Click += gcnew System::EventHandler(this, &popup1::btnOnline_Click);
			// 
			// popup1
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->BackColor = System::Drawing::Color::WhiteSmoke;
			this->ClientSize = System::Drawing::Size(284, 230);
			this->Controls->Add(this->btnOnline);
			this->Controls->Add(this->btnOffline);
			this->Controls->Add(this->panel1);
			this->FormBorderStyle = System::Windows::Forms::FormBorderStyle::FixedDialog;
			this->MaximizeBox = false;
			this->MinimizeBox = false;
			this->Name = L"popup1";
			this->StartPosition = System::Windows::Forms::FormStartPosition::CenterParent;
			this->Text = L"เลือกโหมดการใช้งาน";
			this->panel1->ResumeLayout(false);
			this->panel1->PerformLayout();
			this->ResumeLayout(false);

		}
#pragma endregion
	
	private: System::Void btnOffline_Click(System::Object^ sender, System::EventArgs^ e) {
		// เปิดหน้าฟอร์ม Offline (OfflineUploadForm)
		OfflineUploadForm^ offlineForm = gcnew OfflineUploadForm();
		this->Hide();
		offlineForm->ShowDialog();
		this->Close();
	}

	private: System::Void btnOnline_Click(System::Object^ sender, System::EventArgs^ e) {
		// เปิดหน้าฟอร์ม Online (UploadForm)
		UploadForm^ onlineForm = gcnew UploadForm();
		this->Hide();
		onlineForm->ShowDialog();
		this->Close();
	}
	};
}
