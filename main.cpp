#include "pch.h"
#include "online1.h"
#include "MjpegServer.h"
#include <msclr/marshal_cppstd.h>
#include <thread>
#include <atomic>
#include <fstream>

using namespace System;
using namespace System::Windows::Forms;

// ===================== Unmanaged wrapper functions =====================
#pragma managed(push, off)
inline bool TriggerOnlineCameraHeadlessWrapperMain(std::string ip, std::string port, std::string path);
inline bool TriggerSaveTemplateHeadlessWrapperMain(std::string xmlContent);
inline cv::Mat GetCurrentFrameWrapperMain();

// This replaces the old WinForms displayTimer — runs at 30fps
// Pulls the latest processed frame and pushes it to the MJPEG server
static std::atomic<bool> g_streamThreadRunning(false);

void StreamingThreadFunc() {
    long long lastSeq = -1;
    while (g_streamThreadRunning.load()) {
        cv::Mat outFrame;
        long long displaySeq = 0;

        // Get the processed frame (AI-annotated)
        GetProcessedFrameOnline(outFrame, displaySeq);

        if (!outFrame.empty() && displaySeq != lastSeq) {
            // Overlay parking zones on top (mirrors DrawSceneOnline)
            cv::Mat result;
            DrawSceneOnline(outFrame, displaySeq, result);
            if (!result.empty() && g_globalWebServer) {
                g_globalWebServer->SetLatestFrame(result);
            }
            lastSeq = displaySeq;
        } else if (outFrame.empty()) {
            // No processed frame yet — try showing raw camera frame
            cv::Mat raw;
            long long rawSeq = 0;
            GetRawFrameOnline(raw, rawSeq);
            if (!raw.empty() && g_globalWebServer) {
                g_globalWebServer->SetLatestFrame(raw);
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(33)); // ~30 FPS
    }
}
#pragma managed(pop)

inline cv::Mat GetProcessedFrameWrapperMain() {
    cv::Mat frame;
    long long seq;
    GetProcessedFrameOnline(frame, seq);
    return frame.clone();
}

inline cv::Mat GetRawFrameWrapperMain() {
    cv::Mat frame;
    long long seq;
    GetRawFrameOnline(frame, seq);
    return frame.clone();
}

inline bool TriggerSaveTemplateHeadlessWrapperMain(std::string xmlContent) {
    std::string templateName = "web_template";
    size_t nameStart = xmlContent.find("<name>");
    if (nameStart != std::string::npos) {
        size_t nameEnd = xmlContent.find("</name>", nameStart);
        if (nameEnd != std::string::npos) {
            templateName = xmlContent.substr(nameStart + 6, nameEnd - nameStart - 6);
            templateName.erase(std::remove(templateName.begin(), templateName.end(), '\"'), templateName.end());
        }
    }
    
    std::string filename = "parking_templates/" + templateName + ".xml";
    std::ofstream out(filename);
    if (!out) return false;
    out << xmlContent;
    out.close();
    return LoadParkingTemplate_Online(filename);
}


inline bool TriggerOnlineCameraHeadlessWrapperMain(std::string ip, std::string port, std::string path) {
    try {
        DumpLog("[CONNECT] Received connect request: " + ip + ":" + port + path);
        if (ConsoleApplication3::UploadForm::Instance != nullptr) {
            System::String^ sysIp = msclr::interop::marshal_as<System::String^>(ip);
            System::String^ sysPort = msclr::interop::marshal_as<System::String^>(port);
            System::String^ sysPath = msclr::interop::marshal_as<System::String^>(path);
            bool result = ConsoleApplication3::UploadForm::Instance->StartCameraHeadless(sysIp, sysPort, sysPath);
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

    // Start MJPEG + HTTP server
    if (!g_globalWebServer) {
        g_globalWebServer = new MjpegServer(8080);
        g_globalWebServer->Start();
    }

    // CRITICAL: point g_mjpegServer_online (used by online1.h processing worker)
    // to the same server instance so SetLatestFrame() calls actually feed our web server
    g_mjpegServer_online = g_globalWebServer;

    // Initialize UploadForm silently (loads AI model in background)
    ConsoleApplication3::UploadForm^ onlineForm = gcnew ConsoleApplication3::UploadForm();

    // Register web API callbacks
    g_globalWebServer->SetConnectOnlineCallback(&TriggerOnlineCameraHeadlessWrapperMain);
    g_globalWebServer->SetSaveTemplateCallback(&TriggerSaveTemplateHeadlessWrapperMain);
    g_globalWebServer->SetGetFrameCallback(&GetProcessedFrameWrapperMain);
    g_globalWebServer->SetGetRawFrameCallback(&GetRawFrameWrapperMain);

    printf("[SERVER] Smart Parking Web Server is running!\n");
    printf("[SERVER] Open a browser to view the interface.\n");
    fflush(stdout);

    // Open browser to home page using LAN IP
    System::String^ localIp = GetLocalIPMain();
    System::Diagnostics::Process::Start("http://" + localIp + ":8080/");

    // Headless message loop
    Application::Run();
}
