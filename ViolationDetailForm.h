#pragma once
#include <msclr/marshal_cppstd.h>

namespace ConsoleApplication3 {
	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Drawing;

	public ref class ViolationDetailForm : public System::Windows::Forms::Form
	{
	public:
		ViolationDetailForm(int carId, Bitmap^ screenshot, Bitmap^ visualizationBitmap, System::String^ violationType, 
						   System::DateTime captureTime, int durationSeconds)
		{
			InitializeComponent();
			
			this->carId = carId;
			this->screenshot = screenshot;
			this->visualizationBitmap = visualizationBitmap;
			this->violationType = violationType;
			this->captureTime = captureTime;
			this->durationSeconds = durationSeconds;
			
			// Set form title
			this->Text = System::String::Format(L"Violation Details - ID {0}", carId);
			
			// Display visualization (darkened + bright car)
			if (visualizationBitmap != nullptr) {
				pictureBoxViolation->Image = visualizationBitmap;
			}
			else if (screenshot != nullptr) {
				pictureBoxViolation->Image = screenshot;
			}
			
			// Display info
			lblCarID->Text = System::String::Format(L"Car ID: {0}", carId);
			lblType->Text = System::String::Format(L"Type: {0}", violationType);
			lblTime->Text = System::String::Format(L"Time: {0:HH:mm:ss}", captureTime);
			lblDuration->Text = System::String::Format(L"Duration: {0} seconds", durationSeconds);
		}

	protected:
		~ViolationDetailForm()
		{
			if (components)
			{
				delete components;
			}
		}

	private:
		int carId;
		Bitmap^ screenshot;
		Bitmap^ visualizationBitmap;
		System::String^ violationType;
		System::DateTime captureTime;
		int durationSeconds;

		System::ComponentModel::IContainer^ components;
		System::Windows::Forms::PictureBox^ pictureBoxViolation;
		System::Windows::Forms::Label^ lblCarID;
		System::Windows::Forms::Label^ lblType;
		System::Windows::Forms::Label^ lblTime;
		System::Windows::Forms::Label^ lblDuration;
		System::Windows::Forms::Button^ btnClose;
		System::Windows::Forms::Button^ btnSave;

#pragma region Windows Form Designer generated code
		void InitializeComponent(void)
		{
			this->components = (gcnew System::ComponentModel::Container());
			this->pictureBoxViolation = (gcnew System::Windows::Forms::PictureBox());
			this->lblCarID = (gcnew System::Windows::Forms::Label());
			this->lblType = (gcnew System::Windows::Forms::Label());
			this->lblTime = (gcnew System::Windows::Forms::Label());
			this->lblDuration = (gcnew System::Windows::Forms::Label());
			this->btnClose = (gcnew System::Windows::Forms::Button());
			this->btnSave = (gcnew System::Windows::Forms::Button());
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBoxViolation))->BeginInit();
			this->SuspendLayout();

			// pictureBoxViolation
			this->pictureBoxViolation->BackColor = System::Drawing::Color::White;
			this->pictureBoxViolation->Location = System::Drawing::Point(20, 20);
			this->pictureBoxViolation->Name = L"pictureBoxViolation";
			this->pictureBoxViolation->Size = System::Drawing::Size(400, 300);
			this->pictureBoxViolation->SizeMode = System::Windows::Forms::PictureBoxSizeMode::Zoom;
			this->pictureBoxViolation->TabIndex = 0;
			this->pictureBoxViolation->TabStop = false;

			// lblCarID
			this->lblCarID->AutoSize = true;
			this->lblCarID->Font = (gcnew System::Drawing::Font(L"Segoe UI", 12, System::Drawing::FontStyle::Bold));
			this->lblCarID->Location = System::Drawing::Point(20, 330);
			this->lblCarID->Name = L"lblCarID";
			this->lblCarID->Size = System::Drawing::Size(100, 21);
			this->lblCarID->TabIndex = 1;
			this->lblCarID->Text = L"Car ID: ";

			// lblType
			this->lblType->AutoSize = true;
			this->lblType->Font = (gcnew System::Drawing::Font(L"Segoe UI", 12, System::Drawing::FontStyle::Bold));
			this->lblType->Location = System::Drawing::Point(20, 360);
			this->lblType->Name = L"lblType";
			this->lblType->Size = System::Drawing::Size(100, 21);
			this->lblType->TabIndex = 2;
			this->lblType->Text = L"Type: ";

			// lblTime
			this->lblTime->AutoSize = true;
			this->lblTime->Font = (gcnew System::Drawing::Font(L"Segoe UI", 12, System::Drawing::FontStyle::Bold));
			this->lblTime->Location = System::Drawing::Point(20, 390);
			this->lblTime->Name = L"lblTime";
			this->lblTime->Size = System::Drawing::Size(100, 21);
			this->lblTime->TabIndex = 3;
			this->lblTime->Text = L"Time: ";

			// lblDuration
			this->lblDuration->AutoSize = true;
			this->lblDuration->Font = (gcnew System::Drawing::Font(L"Segoe UI", 12, System::Drawing::FontStyle::Bold));
			this->lblDuration->Location = System::Drawing::Point(20, 420);
			this->lblDuration->Name = L"lblDuration";
			this->lblDuration->Size = System::Drawing::Size(100, 21);
			this->lblDuration->TabIndex = 4;
			this->lblDuration->Text = L"Duration: ";

			// btnSave
			this->btnSave->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(40)), 
				static_cast<System::Int32>(static_cast<System::Byte>(167)), static_cast<System::Int32>(static_cast<System::Byte>(69)));
			this->btnSave->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
			this->btnSave->Font = (gcnew System::Drawing::Font(L"Segoe UI", 11, System::Drawing::FontStyle::Bold));
			this->btnSave->ForeColor = System::Drawing::Color::White;
			this->btnSave->Location = System::Drawing::Point(240, 460);
			this->btnSave->Name = L"btnSave";
			this->btnSave->Size = System::Drawing::Size(90, 35);
			this->btnSave->TabIndex = 5;
			this->btnSave->Text = L"Save";
			this->btnSave->UseVisualStyleBackColor = false;
			this->btnSave->Click += gcnew System::EventHandler(this, &ViolationDetailForm::btnSave_Click);

			// btnClose
			this->btnClose->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(220)), 
				static_cast<System::Int32>(static_cast<System::Byte>(53)), static_cast<System::Int32>(static_cast<System::Byte>(69)));
			this->btnClose->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
			this->btnClose->Font = (gcnew System::Drawing::Font(L"Segoe UI", 11, System::Drawing::FontStyle::Bold));
			this->btnClose->ForeColor = System::Drawing::Color::White;
			this->btnClose->Location = System::Drawing::Point(340, 460);
			this->btnClose->Name = L"btnClose";
			this->btnClose->Size = System::Drawing::Size(80, 35);
			this->btnClose->TabIndex = 6;
			this->btnClose->Text = L"Close";
			this->btnClose->UseVisualStyleBackColor = false;
			this->btnClose->Click += gcnew System::EventHandler(this, &ViolationDetailForm::btnClose_Click);

			// ViolationDetailForm
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(440, 510);
			this->Controls->Add(this->btnSave);
			this->Controls->Add(this->btnClose);
			this->Controls->Add(this->lblDuration);
			this->Controls->Add(this->lblTime);
			this->Controls->Add(this->lblType);
			this->Controls->Add(this->lblCarID);
			this->Controls->Add(this->pictureBoxViolation);
			this->FormBorderStyle = System::Windows::Forms::FormBorderStyle::FixedDialog;
			this->MaximizeBox = false;
			this->MinimizeBox = false;
			this->Name = L"ViolationDetailForm";
			this->StartPosition = System::Windows::Forms::FormStartPosition::CenterParent;
			this->Text = L"Violation Details";
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBoxViolation))->EndInit();
			this->ResumeLayout(false);
			this->PerformLayout();
		}
#pragma endregion

	private: System::Void btnSave_Click(System::Object^ sender, System::EventArgs^ e)
	{
		if (visualizationBitmap == nullptr) {
			MessageBox::Show("No image to save!", "Error", MessageBoxButtons::OK, MessageBoxIcon::Error);
			return;
		}

		SaveFileDialog^ sfd = gcnew SaveFileDialog();
		sfd->Filter = "PNG Image|*.png|JPEG Image|*.jpg|Bitmap Image|*.bmp";
		sfd->DefaultExt = "png";
		sfd->FileName = System::String::Format(L"Violation_ID{0}_{1:yyyyMMdd_HHmmss}", carId, captureTime);
		
		if (sfd->ShowDialog() == System::Windows::Forms::DialogResult::OK) {
			try {
				visualizationBitmap->Save(sfd->FileName);
				MessageBox::Show(
					System::String::Format(L"Image saved successfully!\n\n{0}", sfd->FileName),
					"Success",
					MessageBoxButtons::OK,
					MessageBoxIcon::Information
				);
			}
			catch (Exception^ ex) {
				MessageBox::Show("Error saving image: " + ex->Message, "Error", MessageBoxButtons::OK, MessageBoxIcon::Error);
			}
		}
	}

	private: System::Void btnClose_Click(System::Object^ sender, System::EventArgs^ e)
	{
		this->Close();
	}
	};
}
