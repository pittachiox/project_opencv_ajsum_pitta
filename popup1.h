#pragma once
#include "online1.h"
#include "offline1.h"
#include <msclr/marshal_cppstd.h>

// The wrappers are now defined and mapped exclusively in main.cpp

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
		// [GPU UPDATE] Automatically fetch GPU names via nvidia-smi
		cmbGpuSelect->Items->Clear();
		try {
			System::Diagnostics::Process^ process = gcnew System::Diagnostics::Process();
			process->StartInfo->FileName = "nvidia-smi";
			process->StartInfo->Arguments = "--query-gpu=index,name --format=csv,noheader";
			process->StartInfo->UseShellExecute = false;
			process->StartInfo->RedirectStandardOutput = true;
			process->StartInfo->CreateNoWindow = true;
			process->Start();
			System::String^ output = process->StandardOutput->ReadToEnd();
			process->WaitForExit();
			
			cli::array<System::String^>^ lines = output->Split(gcnew cli::array<System::Char>{'\n', '\r'}, System::StringSplitOptions::RemoveEmptyEntries);
			for each (System::String^ line in lines) {
				cmbGpuSelect->Items->Add("GPU " + line); // Output is like "0, NVIDIA GeForce RTX 3050 Ti Laptop GPU"
			}
		} catch (...) { }
		
		// Fallback if nvidia-smi fails
		if (cmbGpuSelect->Items->Count == 0) {
			cmbGpuSelect->Items->AddRange(gcnew cli::array< System::Object^  >(4) { L"GPU 0 (Primary)", L"GPU 1 (Secondary)", L"GPU 2", L"GPU 3" });
		}
		
		cmbGpuSelect->SelectedIndex = 0; // Default to GPU 0
	}
	
	private: System::Void btnStartWeb_Click(System::Object^ sender, System::EventArgs^ e) {
		// Save Selected GPU ID globally
		g_selectedGpuId = cmbGpuSelect->SelectedIndex;

		// Hide the GUI completely and run as a service
		this->Opacity = 0;
		this->ShowInTaskbar = false;
		this->Hide();

		// Start the global MjpegServer on port 8080 (Callbacks are already bound in main.cpp)
		if (g_globalWebServer) {
			g_globalWebServer->Start();
		}

		// Initialize UploadForm silently (This is when the AI Model actually loads with the chosen GPU!)
		if (UploadForm::Instance == nullptr) {
			UploadForm^ onlineForm = gcnew UploadForm();
			onlineForm->Hide();
		}
		
		// Open the default browser to the web interface
		System::Diagnostics::Process::Start("http://localhost:8080/");
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
	private: System::Windows::Forms::Label^ labelGpu;
	private: System::Windows::Forms::ComboBox^ cmbGpuSelect;
	private: System::Windows::Forms::Button^ btnStartWeb;
	protected:

	private:
		System::ComponentModel::Container ^components;

#pragma region Windows Form Designer generated code
		void InitializeComponent(void)
		{
			this->panel1 = (gcnew System::Windows::Forms::Panel());
			this->label1 = (gcnew System::Windows::Forms::Label());
			this->labelGpu = (gcnew System::Windows::Forms::Label());
			this->cmbGpuSelect = (gcnew System::Windows::Forms::ComboBox());
			this->btnStartWeb = (gcnew System::Windows::Forms::Button());
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
			this->panel1->Size = System::Drawing::Size(300, 50);
			this->panel1->TabIndex = 0;
			// 
			// label1
			// 
			this->label1->AutoSize = true;
			this->label1->Font = (gcnew System::Drawing::Font(L"Segoe UI", 12, System::Drawing::FontStyle::Bold));
			this->label1->ForeColor = System::Drawing::SystemColors::ButtonFace;
			this->label1->Location = System::Drawing::Point(60, 15);
			this->label1->Name = L"label1";
			this->label1->Size = System::Drawing::Size(180, 21);
			this->label1->TabIndex = 0;
			this->label1->Text = L"System Configuration";
			// 
			// labelGpu
			// 
			this->labelGpu->AutoSize = true;
			this->labelGpu->Font = (gcnew System::Drawing::Font(L"Segoe UI", 10, System::Drawing::FontStyle::Bold));
			this->labelGpu->Location = System::Drawing::Point(30, 80);
			this->labelGpu->Name = L"labelGpu";
			this->labelGpu->Size = System::Drawing::Size(120, 19);
			this->labelGpu->TabIndex = 1;
			this->labelGpu->Text = L"Select GPU (ID):";
			// 
			// cmbGpuSelect
			// 
			this->cmbGpuSelect->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
			this->cmbGpuSelect->Font = (gcnew System::Drawing::Font(L"Segoe UI", 10));
			this->cmbGpuSelect->FormattingEnabled = true;
			this->cmbGpuSelect->Items->AddRange(gcnew cli::array< System::Object^  >(4) { L"GPU 0 (Primary)", L"GPU 1 (Secondary)", L"GPU 2", L"GPU 3" });
			this->cmbGpuSelect->Location = System::Drawing::Point(34, 105);
			this->cmbGpuSelect->Name = L"cmbGpuSelect";
			this->cmbGpuSelect->Size = System::Drawing::Size(230, 25);
			this->cmbGpuSelect->TabIndex = 2;
			// 
			// btnStartWeb
			// 
			this->btnStartWeb->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(40)), 
				static_cast<System::Int32>(static_cast<System::Byte>(167)), static_cast<System::Int32>(static_cast<System::Byte>(69)));
			this->btnStartWeb->FlatAppearance->BorderSize = 0;
			this->btnStartWeb->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
			this->btnStartWeb->Font = (gcnew System::Drawing::Font(L"Segoe UI", 11, System::Drawing::FontStyle::Bold));
			this->btnStartWeb->ForeColor = System::Drawing::SystemColors::ButtonHighlight;
			this->btnStartWeb->Location = System::Drawing::Point(34, 150);
			this->btnStartWeb->Name = L"btnStartWeb";
			this->btnStartWeb->Size = System::Drawing::Size(230, 45);
			this->btnStartWeb->TabIndex = 3;
			this->btnStartWeb->Text = L"🚀 Launch Web System";
			this->btnStartWeb->UseVisualStyleBackColor = false;
			this->btnStartWeb->Click += gcnew System::EventHandler(this, &popup1::btnStartWeb_Click);
			// 
			// popup1
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->BackColor = System::Drawing::Color::WhiteSmoke;
			this->ClientSize = System::Drawing::Size(300, 220);
			this->Controls->Add(this->btnStartWeb);
			this->Controls->Add(this->cmbGpuSelect);
			this->Controls->Add(this->labelGpu);
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
	
	// [Removed old Offline/Online switch buttons since we boot directly to web now]

	};
}
