#include "pch.h"
#include "online1.h"
#include "popup1.h"
#include "MjpegServer.h"
#include <msclr/marshal_cppstd.h>
#include <thread>
#include <atomic>
#include <fstream>

using namespace System;
using namespace System::Windows::Forms;

// ===================== Unmanaged wrapper functions =====================
#pragma managed(push, off)
inline bool TriggerOnlineCameraHeadlessWrapperMain(int cameraId, std::string ip, std::string port, std::string path);
inline bool TriggerSaveTemplateHeadlessWrapperMain(int cameraId, std::string xmlContent);
inline cv::Mat GetCurrentFrameWrapperMain(int cameraId);

// This replaces the old WinForms displayTimer — runs at 30fps
// Pulls the latest processed frame and pushes it to the MJPEG server
static std::atomic<bool> g_streamThreadRunning(false);

void StreamingThreadFunc() {
    std::map<int, long long> lastSeqs;
    while (g_streamThreadRunning.load()) {
        for (int i = 1; i <= 4; ++i) {            
            cv::Mat outFrame;
            long long displaySeq = 0;

            GetCam(i)->GetProcessedFrameOnline(outFrame, displaySeq);

            if (!outFrame.empty() && displaySeq != lastSeqs[i]) {
                if (g_globalWebServer) {
                    g_globalWebServer->SetLatestFrame(i, outFrame);
                }
                lastSeqs[i] = displaySeq;
            } else if (outFrame.empty()) {
                cv::Mat raw;
                long long rawSeq = 0;
                GetCam(i)->GetRawFrameOnline(raw, rawSeq);
                if (!raw.empty() && g_globalWebServer) {
                    g_globalWebServer->SetLatestFrame(i, raw);
                }
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(33)); // ~30 FPS
    }
}
#pragma managed(pop)

inline cv::Mat GetProcessedFrameWrapperMain(int cameraId) {
    cv::Mat frame;
    long long seq;
    GetCam(cameraId)->GetProcessedFrameOnline(frame, seq);
    return frame.clone();
}

inline cv::Mat GetRawFrameWrapperMain(int cameraId) {
    cv::Mat frame;
    long long seq;
    GetCam(cameraId)->GetRawFrameOnline(frame, seq);
    return frame.clone();
}

inline bool TriggerSaveTemplateHeadlessWrapperMain(int cameraId, std::string xmlContent) {
    std::string templateName = "web_template_" + std::to_string(cameraId);
    size_t nameStart = xmlContent.find("<name>");
    if (nameStart != std::string::npos) {
        size_t nameEnd = xmlContent.find("</name>", nameStart);
        if (nameEnd != std::string::npos) {
            templateName = xmlContent.substr(nameStart + 6, nameEnd - nameStart - 6);
            templateName.erase(std::remove(templateName.begin(), templateName.end(), '\"'), templateName.end());
        }
    }
    
    std::string filename = "parking_templates/" + templateName + ".xml";
    DumpLog("[API] Extracted template name: " + templateName + " -> Saving to: " + filename);
    std::ofstream out(filename);
    if (!out) {
        DumpLog("[API] ERROR: Could not open " + filename + " for writing!");
        return false;
    }
    out << xmlContent;
    out.close();
    
    bool loadResult = GetCam(cameraId)->LoadParkingTemplate_Online(filename);
    DumpLog("[API] LoadParkingTemplate_Online returned: " + std::string(loadResult ? "true" : "false"));
    return loadResult;
}


inline bool TriggerOnlineCameraHeadlessWrapperMain(int cameraId, std::string ip, std::string port, std::string path) {
    try {
        DumpLog("[CONNECT] Received connect request for cam " + std::to_string(cameraId) + ": " + ip + ":" + port + path);
        if (ConsoleApplication3::UploadForm::Instance != nullptr) {
            System::String^ sysIp = msclr::interop::marshal_as<System::String^>(ip);
            System::String^ sysPort = msclr::interop::marshal_as<System::String^>(port);
            System::String^ sysPath = msclr::interop::marshal_as<System::String^>(path);
            bool result = ConsoleApplication3::UploadForm::Instance->StartCameraHeadless(cameraId, sysIp, sysPort, sysPath);
            DumpLog("[CONNECT] StartCameraHeadless returned: " + std::string(result ? "SUCCESS" : "FAILED"));
            return result;
        }
        DumpLog("[CONNECT] ERROR: UploadForm::Instance is null!");
        return false;
    } catch (System::Exception^ ex) {
        DumpLog("[CONNECT EXCEPTION MS] " + msclr::interop::marshal_as<std::string>(ex->Message));
        return false;
    } catch (const std::exception& e) {
        DumpLog(std::string("[CONNECT EXCEPTION STD] ") + e.what());
        return false;
    } catch (...) {
        DumpLog("[CONNECT EXCEPTION] Unknown exception occurred!");
        return false;
    }
}

inline void TriggerDisconnectHeadlessWrapperMain(int cameraId) {
    DumpLog("[DISCONNECT] Received disconnect request for cam " + std::to_string(cameraId) + ".");
    if (ConsoleApplication3::UploadForm::Instance != nullptr) {
        ConsoleApplication3::UploadForm::Instance->StopProcessingPublic(cameraId);
        DumpLog("[DISCONNECT] Camera stopped successfully.");
    } else {
        DumpLog("[DISCONNECT] WARNING: UploadForm::Instance is null.");
    }
}

// ===================== Managed: get local IP =====================
System::String^ GetLocalIPMain() {
    System::String^ bestIP = "127.0.0.1";
    try {
        cli::array<System::Net::NetworkInformation::NetworkInterface^>^ interfaces = System::Net::NetworkInformation::NetworkInterface::GetAllNetworkInterfaces();
        for each (System::Net::NetworkInformation::NetworkInterface^ adapter in interfaces) {
            if (adapter->OperationalStatus == System::Net::NetworkInformation::OperationalStatus::Up) {
                System::String^ desc = adapter->Description->ToLower();
                if (desc->Contains("virtual") || desc->Contains("vpn") || desc->Contains("vmware") || desc->Contains("radmin") || desc->Contains("hamachi")) continue;
                
                System::Net::NetworkInformation::IPInterfaceProperties^ properties = adapter->GetIPProperties();
                for each (System::Net::NetworkInformation::UnicastIPAddressInformation^ ip in properties->UnicastAddresses) {
                    if (ip->Address->AddressFamily == System::Net::Sockets::AddressFamily::InterNetwork) {
                        System::String^ ipStr = ip->Address->ToString();
                        if (ipStr->StartsWith("26.") || ipStr->StartsWith("169.254.")) continue;
                        bestIP = ipStr;
                        if (ipStr->StartsWith("192.168.") || ipStr->StartsWith("172.") || ipStr->StartsWith("10.")) {
                            return bestIP;
                        }
                    }
                }
            }
        }
    } catch (...) {}
    return bestIP;
}

#include <windows.h>
#include <stdio.h>

// ===================== Entry point =====================
[STAThreadAttribute]
void Main(array<String^>^ args) {
    // Force a console window to appear for easy debugging
    AllocConsole();
    FILE* dummy;
    freopen_s(&dummy, "CONOUT$", "w", stdout);
    freopen_s(&dummy, "CONOUT$", "w", stderr);

    Application::EnableVisualStyles();
    Application::SetCompatibleTextRenderingDefault(false);

    // [GPU UPDATE] Instead of initializing everything here in a headless loop, 
    // we give control back to popup1 (The Pre-Launch GPU Configuration Screen)
    // popup1 will handle launching the Web server and UploadForm after GPU selection.
    
    printf("[SERVER] Booting GPU Configuration Launcher...\n");
    fflush(stdout);

    // Prepare Web Server callbacks BEFORE launching popup1
    g_globalWebServer = new MjpegServer(8080);
    g_globalWebServer->onGetFrame = GetProcessedFrameWrapperMain;
    g_globalWebServer->onGetRawFrame = GetRawFrameWrapperMain;
    g_globalWebServer->onSaveTemplate = TriggerSaveTemplateHeadlessWrapperMain;
    g_globalWebServer->onConnectOnline = TriggerOnlineCameraHeadlessWrapperMain;
    g_globalWebServer->onDisconnect = TriggerDisconnectHeadlessWrapperMain;

    // Start the streamer thread to continuously feed frames to the MjpegServer
    g_streamThreadRunning = true;
    std::thread(StreamingThreadFunc).detach();

    ConsoleApplication3::popup1^ launcher = gcnew ConsoleApplication3::popup1();
    Application::Run(launcher);
}
