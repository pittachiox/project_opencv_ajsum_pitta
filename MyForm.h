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

	private: System::Windows::Forms::MenuStrip^ menuStrip1;
	private: System::Windows::Forms::OpenFileDialog^ openFileDialog;
	private: System::Windows::Forms::SaveFileDialog^ saveFileDialog;
	private: System::Windows::Forms::ToolStripMenuItem^ uploadToolStripMenuItem;
	private: System::Windows::Forms::SplitContainer^ splitContainer1;
	private: System::Windows::Forms::Label^ label2;
	private: System::Windows::Forms::Label^ label1;
	private: System::Windows::Forms::PictureBox^ pictureBox1;
	private: System::Windows::Forms::Label^ label3;
	private: System::Windows::Forms::Label^ label4;


	private: System::Windows::Forms::Button^ btnSeeCamera;


	private: System::Windows::Forms::Button^ btnParkingSetup;
	private: System::ComponentModel::Container^ components;


	private: System::Windows::Forms::Button^ btnID3NormalZone;
	private: System::Windows::Forms::Button^ btnID2NormalZone;

	private:
		Bitmap^ bmp;

#pragma region Windows Form Designer generated code
		void InitializeComponent(void)
		{
			this->menuStrip1 = (gcnew System::Windows::Forms::MenuStrip());
			this->uploadToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			this->openFileDialog = (gcnew System::Windows::Forms::OpenFileDialog());
			this->saveFileDialog = (gcnew System::Windows::Forms::SaveFileDialog());
			this->splitContainer1 = (gcnew System::Windows::Forms::SplitContainer());
			this->label2 = (gcnew System::Windows::Forms::Label());
			this->label1 = (gcnew System::Windows::Forms::Label());
			this->pictureBox1 = (gcnew System::Windows::Forms::PictureBox());
			this->btnParkingSetup = (gcnew System::Windows::Forms::Button());
			this->btnSeeCamera = (gcnew System::Windows::Forms::Button());
			this->label4 = (gcnew System::Windows::Forms::Label());
			this->label3 = (gcnew System::Windows::Forms::Label());
			this->btnID3NormalZone = (gcnew System::Windows::Forms::Button());
			this->btnID2NormalZone = (gcnew System::Windows::Forms::Button());
			this->menuStrip1->SuspendLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->splitContainer1))->BeginInit();
			this->splitContainer1->Panel1->SuspendLayout();
			this->splitContainer1->Panel2->SuspendLayout();
			this->splitContainer1->SuspendLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox1))->BeginInit();
			this->SuspendLayout();
			// 
			// menuStrip1
			// 
			this->menuStrip1->Dock = System::Windows::Forms::DockStyle::None;
			this->menuStrip1->ImageScalingSize = System::Drawing::Size(20, 20);
			this->menuStrip1->Items->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(1) { this->uploadToolStripMenuItem });
			this->menuStrip1->Location = System::Drawing::Point(48, 73);
			this->menuStrip1->Name = L"menuStrip1";
			this->menuStrip1->Size = System::Drawing::Size(64, 24);
			this->menuStrip1->TabIndex = 0;
			this->menuStrip1->Text = L"menuStrip1";
			// 
			// uploadToolStripMenuItem
			// 
			this->uploadToolStripMenuItem->BackColor = System::Drawing::Color::PaleGreen;
			this->uploadToolStripMenuItem->Name = L"uploadToolStripMenuItem";
			this->uploadToolStripMenuItem->Size = System::Drawing::Size(56, 20);
			this->uploadToolStripMenuItem->Text = L"upload";
			this->uploadToolStripMenuItem->Click += gcnew System::EventHandler(this, &MyForm::uploadToolStripMenuItem_Click);
			// 
			// openFileDialog
			// 
			this->openFileDialog->FileName = L"openFileDialog";
			this->openFileDialog->Filter = L"Image files|*.jpg;*.png;*.jpeg;*.bmp";
			// 
			// saveFileDialog
			// 
			this->saveFileDialog->DefaultExt = L"png";
			this->saveFileDialog->Filter = L"Image files|*.jpg;*.png;*.bmp";
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
			this->splitContainer1->Panel1->Controls->Add(this->menuStrip1);
			this->splitContainer1->Panel1->Controls->Add(this->label2);
			this->splitContainer1->Panel1->Controls->Add(this->label1);
			this->splitContainer1->Panel1->Controls->Add(this->pictureBox1);
			this->splitContainer1->Panel1->Padding = System::Windows::Forms::Padding(30);
			// 
			// splitContainer1.Panel2
			// 
			this->splitContainer1->Panel2->BackColor = System::Drawing::Color::OldLace;
			this->splitContainer1->Panel2->Controls->Add(this->btnParkingSetup);
			this->splitContainer1->Panel2->Controls->Add(this->btnSeeCamera);
			this->splitContainer1->Panel2->Controls->Add(this->btnID3NormalZone);
			this->splitContainer1->Panel2->Controls->Add(this->btnID2NormalZone);
			this->splitContainer1->Panel2->Controls->Add(this->label4);
			this->splitContainer1->Panel2->Controls->Add(this->label3);
			this->splitContainer1->Panel2->ForeColor = System::Drawing::SystemColors::ActiveCaption;
			this->splitContainer1->Size = System::Drawing::Size(1443, 759);
			this->splitContainer1->SplitterDistance = 1009;
			this->splitContainer1->TabIndex = 1;
			// 
			// label2
			// 
			this->label2->AutoSize = true;
			this->label2->Font = (gcnew System::Drawing::Font(L"Segoe UI", 12.75F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->label2->ForeColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(80)), static_cast<System::Int32>(static_cast<System::Byte>(80)),
				static_cast<System::Int32>(static_cast<System::Byte>(80)));
			this->label2->Location = System::Drawing::Point(156, 39);
			this->label2->Name = L"label2";
			this->label2->Size = System::Drawing::Size(245, 23);
			this->label2->TabIndex = 1;
			this->label2->Text = L"ลากเมาส์บนภาพเพื่อกำหนดช่องจอด";
			// 
			// label1
			// 
			this->label1->AutoSize = true;
			this->label1->BackColor = System::Drawing::Color::White;
			this->label1->Font = (gcnew System::Drawing::Font(L"Segoe UI", 16.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->label1->ForeColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(45)), static_cast<System::Int32>(static_cast<System::Byte>(45)),
				static_cast<System::Int32>(static_cast<System::Byte>(48)));
			this->label1->Location = System::Drawing::Point(44, 39);
			this->label1->Name = L"label1";
			this->label1->Size = System::Drawing::Size(102, 30);
			this->label1->TabIndex = 0;
			this->label1->Text = L"camera1";
			// 
			// pictureBox1
			// 
			this->pictureBox1->BackColor = System::Drawing::Color::White;
			this->pictureBox1->Dock = System::Windows::Forms::DockStyle::Fill;
			this->pictureBox1->Location = System::Drawing::Point(30, 30);
			this->pictureBox1->Margin = System::Windows::Forms::Padding(20);
			this->pictureBox1->Name = L"pictureBox1";
			this->pictureBox1->Size = System::Drawing::Size(949, 699);
			this->pictureBox1->SizeMode = System::Windows::Forms::PictureBoxSizeMode::Zoom;
			this->pictureBox1->TabIndex = 0;
			this->pictureBox1->TabStop = false;
			// 
			// btnParkingSetup
			// 
			this->btnParkingSetup->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(138)), static_cast<System::Int32>(static_cast<System::Byte>(43)),
				static_cast<System::Int32>(static_cast<System::Byte>(226)));
			this->btnParkingSetup->FlatAppearance->BorderSize = 0;
			this->btnParkingSetup->FlatAppearance->MouseOverBackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(120)),
				static_cast<System::Int32>(static_cast<System::Byte>(30)), static_cast<System::Int32>(static_cast<System::Byte>(200)));
			this->btnParkingSetup->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
			this->btnParkingSetup->Font = (gcnew System::Drawing::Font(L"Segoe UI", 13.75F, System::Drawing::FontStyle::Bold));
			this->btnParkingSetup->ForeColor = System::Drawing::Color::White;
			this->btnParkingSetup->Location = System::Drawing::Point(109, 392);
			this->btnParkingSetup->Name = L"btnParkingSetup";
			this->btnParkingSetup->Size = System::Drawing::Size(253, 60);
			this->btnParkingSetup->TabIndex = 9;
			this->btnParkingSetup->Text = L"🅿️ Parking Setup";
			this->btnParkingSetup->UseVisualStyleBackColor = false;
			this->btnParkingSetup->Click += gcnew System::EventHandler(this, &MyForm::btnParkingSetup_Click);
			// 
			// btnSeeCamera
			// 
			this->btnSeeCamera->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(255)), static_cast<System::Int32>(static_cast<System::Byte>(193)),
				static_cast<System::Int32>(static_cast<System::Byte>(7)));
			this->btnSeeCamera->FlatAppearance->BorderSize = 0;
			this->btnSeeCamera->FlatAppearance->MouseOverBackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(255)),
				static_cast<System::Int32>(static_cast<System::Byte>(175)), static_cast<System::Int32>(static_cast<System::Byte>(0)));
			this->btnSeeCamera->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
			this->btnSeeCamera->Font = (gcnew System::Drawing::Font(L"Segoe UI", 14.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->btnSeeCamera->ForeColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(45)), static_cast<System::Int32>(static_cast<System::Byte>(45)),
				static_cast<System::Int32>(static_cast<System::Byte>(48)));
			this->btnSeeCamera->Location = System::Drawing::Point(123, 341);
			this->btnSeeCamera->Name = L"btnSeeCamera";
			this->btnSeeCamera->Size = System::Drawing::Size(225, 45);
			this->btnSeeCamera->TabIndex = 8;
			this->btnSeeCamera->Text = L"see camera";
			this->btnSeeCamera->UseVisualStyleBackColor = false;
			this->btnSeeCamera->Click += gcnew System::EventHandler(this, &MyForm::btnSeeCamera_Click);
			// 
			// label4
			// 
			this->label4->AutoSize = true;
			this->label4->BackColor = System::Drawing::Color::Transparent;
			this->label4->Font = (gcnew System::Drawing::Font(L"Segoe UI", 15.75F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->label4->ForeColor = System::Drawing::Color::Black;
			this->label4->Location = System::Drawing::Point(26, 72);
			this->label4->Name = L"label4";
			this->label4->Size = System::Drawing::Size(257, 30);
			this->label4->TabIndex = 3;
			this->label4->Text = L"รายการช่องจอด (ROI List)";
			// 
			// label3
			// 
			this->label3->Anchor = System::Windows::Forms::AnchorStyles::Top;
			this->label3->AutoSize = true;
			this->label3->Font = (gcnew System::Drawing::Font(L"Segoe UI", 16.75F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->label3->ForeColor = System::Drawing::Color::Black;
			this->label3->Location = System::Drawing::Point(133, 30);
			this->label3->Margin = System::Windows::Forms::Padding(100, 0, 3, 0);
			this->label3->Name = L"label3";
			this->label3->Size = System::Drawing::Size(215, 31);
			this->label3->TabIndex = 2;
			this->label3->Text = L"เลือกเฟรม & เครื่องมือ";
			// 
			// btnID3NormalZone
			// 
			this->btnID3NormalZone->BackColor = System::Drawing::Color::LightSteelBlue;
			this->btnID3NormalZone->FlatAppearance->BorderSize = 0;
			this->btnID3NormalZone->FlatAppearance->MouseOverBackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(0)),
				static_cast<System::Int32>(static_cast<System::Byte>(100)), static_cast<System::Int32>(static_cast<System::Byte>(200)));
			this->btnID3NormalZone->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
			this->btnID3NormalZone->Font = (gcnew System::Drawing::Font(L"Segoe UI", 13.75F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->btnID3NormalZone->ForeColor = System::Drawing::Color::White;
			this->btnID3NormalZone->Location = System::Drawing::Point(109, 174);
			this->btnID3NormalZone->Name = L"btnID3NormalZone";
			this->btnID3NormalZone->Size = System::Drawing::Size(257, 45);
			this->btnID3NormalZone->TabIndex = 5;
			this->btnID3NormalZone->Text = L"ID3 Normal Zone";
			this->btnID3NormalZone->UseVisualStyleBackColor = false;
			// 
			// btnID2NormalZone
			// 
			this->btnID2NormalZone->BackColor = System::Drawing::Color::LightSteelBlue;
			this->btnID2NormalZone->FlatAppearance->BorderSize = 0;
			this->btnID2NormalZone->FlatAppearance->MouseOverBackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(0)),
				static_cast<System::Int32>(static_cast<System::Byte>(100)), static_cast<System::Int32>(static_cast<System::Byte>(200)));
			this->btnID2NormalZone->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
			this->btnID2NormalZone->Font = (gcnew System::Drawing::Font(L"Segoe UI", 13.75F, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->btnID2NormalZone->ForeColor = System::Drawing::Color::White;
			this->btnID2NormalZone->Location = System::Drawing::Point(109, 123);
			this->btnID2NormalZone->Name = L"btnID2NormalZone";
			this->btnID2NormalZone->Size = System::Drawing::Size(258, 45);
			this->btnID2NormalZone->TabIndex = 4;
			this->btnID2NormalZone->Text = L"ID2 Normal Zone";
			this->btnID2NormalZone->UseVisualStyleBackColor = false;
			// 
			// MyForm
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(1443, 759);
			this->Controls->Add(this->splitContainer1);
			this->MainMenuStrip = this->menuStrip1;
			this->Name = L"MyForm";
			this->Text = L"My Image Viewer";
			this->menuStrip1->ResumeLayout(false);
			this->menuStrip1->PerformLayout();
			this->splitContainer1->Panel1->ResumeLayout(false);
			this->splitContainer1->Panel1->PerformLayout();
			this->splitContainer1->Panel2->ResumeLayout(false);
			this->splitContainer1->Panel2->PerformLayout();
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->splitContainer1))->EndInit();
			this->splitContainer1->ResumeLayout(false);
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox1))->EndInit();
			this->ResumeLayout(false);

		}
#pragma endregion

	private: System::Void uploadToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {
		UploadForm^ form = gcnew UploadForm();
		form->Show();
	}

	private: System::Void btnSeeCamera_Click(System::Object^ sender, System::EventArgs^ e) {
		popup1^ form = gcnew popup1();
		form->ShowDialog();
	}

	private: System::Void btnSave_Click(System::Object^ sender, System::EventArgs^ e) {
		if ((bmp != nullptr) && (saveFileDialog->ShowDialog() == System::Windows::Forms::DialogResult::OK))
		{
			bmp->Save(saveFileDialog->FileName);
		}
	}

	private: System::Void btnParkingSetup_Click(System::Object^ sender, System::EventArgs^ e) {
		ParkingSetupForm^ form = gcnew ParkingSetupForm();
		form->ShowDialog();
	}
	};
}
