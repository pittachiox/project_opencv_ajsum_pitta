#pragma once
#define NOMINMAX
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>

#pragma comment(lib, "ws2_32.lib")

// ============================================================
//  MjpegServer
//  Routes:
//    GET /        → HTML page (browser opens this, JS connects to /events)
//    GET /stream  → MJPEG video stream
//    GET /events  → SSE log stream (for F12 console)
// ============================================================
class MjpegServer {
private:
    SOCKET serverSocket;
    std::atomic<bool> isRunning;
    std::thread serverThread;

    // MJPEG clients
    std::vector<SOCKET> clientSockets;
    std::mutex clientsMutex;

    cv::Mat latestFrame;
    std::mutex frameMutex;
    std::condition_variable frameCV;
    bool newFrameAvailable = false;

    // SSE clients - we send messages to all of them
    std::vector<SOCKET> sseSockets;
    std::mutex sseMutex;

    int port;

    // --------------------------------------------------------
    //  Parse request: returns path, also extracts Range header
    // --------------------------------------------------------
    std::string ParseRequest(SOCKET clientSocket, std::string& outRangeHeader) {
        char buf[4096] = {};
        int received = recv(clientSocket, buf, sizeof(buf) - 1, 0);
        if (received <= 0) return "";

        std::string req(buf, received);

        // First line: "GET /path HTTP/1.1"
        size_t start = req.find(' ');
        if (start == std::string::npos) return "/";
        size_t end = req.find(' ', start + 1);
        if (end == std::string::npos) return "/";
        std::string path = req.substr(start + 1, end - start - 1);

        // Extract Range header (e.g. "bytes=0-" or "bytes=1024-2047")
        outRangeHeader = "";
        size_t ri = req.find("Range: bytes=");
        if (ri != std::string::npos) {
            size_t re = req.find('\r', ri);
            if (re == std::string::npos) re = req.size();
            outRangeHeader = req.substr(ri + 13, re - ri - 13);
        }
        return path;
    }

    // --------------------------------------------------------
    //  Server accept loop
    // --------------------------------------------------------
    void ServerLoop() {
        while (isRunning) {
            fd_set readfds;
            FD_ZERO(&readfds);
            FD_SET(serverSocket, &readfds);

            timeval timeout;
            timeout.tv_sec = 0;
            timeout.tv_usec = 100000;

            int activity = select(0, &readfds, NULL, NULL, &timeout);

            if (activity > 0 && FD_ISSET(serverSocket, &readfds)) {
                SOCKET clientSocket = accept(serverSocket, NULL, NULL);
                if (clientSocket != INVALID_SOCKET) {
                    std::thread(&MjpegServer::HandleClient, this, clientSocket).detach();
                }
            }
        }
    }

    void HandleClient(SOCKET clientSocket) {
        std::string rangeHeader;
        std::string path = ParseRequest(clientSocket, rangeHeader);

        if (path == "/events") {
            HandleSSE(clientSocket);
        } else if (path == "/stream") {
            HandleMjpeg(clientSocket);
        } else if (path.substr(0, 12) == "/violations/") {
            std::string filename = path.substr(12);
            std::string filepath = "C:\\logpic\\" + filename;
            ServeFile(clientSocket, filepath, "image/jpeg", rangeHeader);
        } else if (path.substr(0, 10) == "/locvideo/") {
            std::string relpath = path.substr(10);
            std::replace(relpath.begin(), relpath.end(), '/', '\\');
            std::string filepath = "C:\\locvideo\\" + relpath;
            ServeFile(clientSocket, filepath, "video/webm", rangeHeader);
        } else {
            HandlePage(clientSocket);
        }
    }

    // --------------------------------------------------------
    //  Serve a file from disk — supports HTTP Range requests (206)
    //  Required for video: Chrome seeks to moov atom via Range
    // --------------------------------------------------------
    void ServeFile(SOCKET clientSocket, const std::string& filepath,
                   const std::string& contentType, const std::string& rangeHeader) {
        FILE* f = nullptr;
        if (fopen_s(&f, filepath.c_str(), "rb") != 0 || !f) {
            std::string err = "HTTP/1.1 404 Not Found\r\nContent-Length: 9\r\nConnection: close\r\n\r\nNot Found";
            send(clientSocket, err.c_str(), (int)err.size(), 0);
            closesocket(clientSocket);
            return;
        }
        fseek(f, 0, SEEK_END);
        long fileSize = ftell(f);

        long rangeStart = 0, rangeEnd = fileSize - 1;
        bool isRangeRequest = !rangeHeader.empty();

        if (isRangeRequest) {
            size_t dash = rangeHeader.find('-');
            if (dash != std::string::npos) {
                std::string sStr = rangeHeader.substr(0, dash);
                std::string eStr = rangeHeader.substr(dash + 1);
                if (!sStr.empty()) rangeStart = std::stol(sStr);
                if (!eStr.empty()) rangeEnd   = std::stol(eStr);
            }
            if (rangeEnd >= fileSize) rangeEnd = fileSize - 1;
        }

        long sendSize = rangeEnd - rangeStart + 1;
        fseek(f, rangeStart, SEEK_SET);

        // Build and send HTTP header
        std::string header;
        if (isRangeRequest) {
            header =
                "HTTP/1.1 206 Partial Content\r\n"
                "Content-Type: " + contentType + "\r\n"
                "Content-Range: bytes " + std::to_string(rangeStart) + "-" +
                    std::to_string(rangeEnd) + "/" + std::to_string(fileSize) + "\r\n"
                "Content-Length: " + std::to_string(sendSize) + "\r\n"
                "Accept-Ranges: bytes\r\n"
                "Access-Control-Allow-Origin: *\r\n"
                "Connection: close\r\n\r\n";
        } else {
            header =
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: " + contentType + "\r\n"
                "Content-Length: " + std::to_string(sendSize) + "\r\n"
                "Accept-Ranges: bytes\r\n"
                "Access-Control-Allow-Origin: *\r\n"
                "Connection: close\r\n\r\n";
        }
        send(clientSocket, header.c_str(), (int)header.size(), 0);

        // *** Stream file in 64KB chunks — single send() can't handle large files ***
        const int CHUNK = 65536;
        std::vector<char> chunk(CHUNK);
        long remaining = sendSize;
        while (remaining > 0) {
            int toRead = (int)(std::min((long)CHUNK, remaining));
            int bytesRead = (int)fread(chunk.data(), 1, toRead, f);
            if (bytesRead <= 0) break;
            const char* ptr = chunk.data();
            int left = bytesRead;
            while (left > 0) {
                int sent = send(clientSocket, ptr, left, 0);
                if (sent <= 0) goto done; // connection closed
                ptr  += sent;
                left -= sent;
            }
            remaining -= bytesRead;
        }
        done:
        fclose(f);
        closesocket(clientSocket);
    }

    // --------------------------------------------------------
    //  Route: / — HTML page
    // --------------------------------------------------------
    void HandlePage(SOCKET clientSocket) {
        // HTML page that:
        //  1. Shows MJPEG video
        //  2. Connects to /events (SSE) and console.log() every message
        std::string html = R"html(<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Parking Monitor</title>
<style>
  body { margin:0; background:#111; color:#0f0; font-family:monospace; }
  h2   { margin:8px; color:#0f0; }
  img  { display:block; max-width:100%; border:2px solid #0f0; }
  #stats { padding:8px; font-size:14px; }
</style>
</head>
<body>
<h2>&#128247; Parking Monitor — Live</h2>
<img src="/stream" alt="Camera stream">
<div id="stats">Connecting to log stream...</div>
<script>
(function(){
  var statsEl = document.getElementById('stats');
  var es = new EventSource('/events');

  es.onmessage = function(e){
    try {
      var d = JSON.parse(e.data);
      // Print to F12 console
      console.log('[PARKING]', JSON.stringify(d));
      // Update on-page stats too
      statsEl.textContent =
        'Empty: ' + d.available_slots +
        ' | Occupied: ' + d.occupied_slots +
        ' | Violations: ' + d.violation_slots +
        ' | ' + d.timestamp;
    } catch(err) {
      console.warn('[PARKING] parse error', err);
    }
  };

  es.onerror = function(){
    console.warn('[PARKING] SSE connection lost, retrying...');
    statsEl.textContent = 'Connection lost — retrying...';
  };
})();
</script>
</body>
</html>)html";

        std::string response =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/html; charset=utf-8\r\n"
            "Content-Length: " + std::to_string(html.size()) + "\r\n"
            "Connection: close\r\n\r\n" + html;

        send(clientSocket, response.c_str(), (int)response.size(), 0);
        closesocket(clientSocket);
    }

    // --------------------------------------------------------
    //  Route: /stream — MJPEG video
    // --------------------------------------------------------
    void HandleMjpeg(SOCKET clientSocket) {
        {
            std::lock_guard<std::mutex> lock(clientsMutex);
            clientSockets.push_back(clientSocket);
        }

        std::string httpHeader =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: multipart/x-mixed-replace; boundary=mjpegstream\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Connection: close\r\n\r\n";
        send(clientSocket, httpHeader.c_str(), (int)httpHeader.length(), 0);

        std::vector<uchar> buffer;
        std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 70};

        while (isRunning) {
            cv::Mat frameToSend;
            {
                std::unique_lock<std::mutex> lock(frameMutex);
                if (!frameCV.wait_for(lock, std::chrono::milliseconds(500),
                        [this] { return newFrameAvailable || !isRunning; })) {
                    int error = 0; socklen_t len = sizeof(error);
                    if (getsockopt(clientSocket, SOL_SOCKET, SO_ERROR, (char*)&error, &len) != 0 || error != 0)
                        break;
                    continue;
                }
                if (!isRunning) break;
                if (latestFrame.empty()) { newFrameAvailable = false; continue; }
                frameToSend = latestFrame.clone();
                newFrameAvailable = false;
            }

            if (!frameToSend.empty()) {
                cv::imencode(".jpg", frameToSend, buffer, params);

                std::string frameHeader =
                    "--mjpegstream\r\n"
                    "Content-Type: image/jpeg\r\n"
                    "Content-Length: " + std::to_string(buffer.size()) + "\r\n\r\n";

                if (send(clientSocket, frameHeader.c_str(), (int)frameHeader.length(), 0) == SOCKET_ERROR) break;
                if (send(clientSocket, (const char*)buffer.data(), (int)buffer.size(), 0) == SOCKET_ERROR) break;
                if (send(clientSocket, "\r\n", 2, 0) == SOCKET_ERROR) break;
            }
        }

        {
            std::lock_guard<std::mutex> lock(clientsMutex);
            auto it = std::find(clientSockets.begin(), clientSockets.end(), clientSocket);
            if (it != clientSockets.end()) clientSockets.erase(it);
        }
        closesocket(clientSocket);
    }

    // --------------------------------------------------------
    //  Route: /events — SSE log stream
    // --------------------------------------------------------
    void HandleSSE(SOCKET clientSocket) {
        {
            std::lock_guard<std::mutex> lock(sseMutex);
            sseSockets.push_back(clientSocket);
        }

        // SSE headers — keep connection alive
        std::string header =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Connection: keep-alive\r\n\r\n";
        send(clientSocket, header.c_str(), (int)header.length(), 0);

        // Send an initial comment to confirm the stream is live
        std::string hello = ": connected\n\n";
        send(clientSocket, hello.c_str(), (int)hello.length(), 0);

        // Keep thread alive — PushLogEvent() sends data; we just wait here
        while (isRunning) {
            // Heartbeat every 15s to keep the connection alive through proxies
            Sleep(15000);
            if (!isRunning) break;
            std::string hb = ": heartbeat\n\n";
            if (send(clientSocket, hb.c_str(), (int)hb.length(), 0) == SOCKET_ERROR) break;
        }

        {
            std::lock_guard<std::mutex> lock(sseMutex);
            auto it = std::find(sseSockets.begin(), sseSockets.end(), clientSocket);
            if (it != sseSockets.end()) sseSockets.erase(it);
        }
        closesocket(clientSocket);
    }

public:
    MjpegServer(int listenPort = 8080) : isRunning(false), serverSocket(INVALID_SOCKET), port(listenPort) {}

    ~MjpegServer() { Stop(); }

    bool Start() {
        if (isRunning) return true;

        WSADATA wsaData;
        if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) return false;

        serverSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (serverSocket == INVALID_SOCKET) { WSACleanup(); return false; }

        char opt = 1;
        setsockopt(serverSocket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        sockaddr_in serverAddr;
        serverAddr.sin_family = AF_INET;
        serverAddr.sin_addr.s_addr = INADDR_ANY;
        serverAddr.sin_port = htons(port);

        if (bind(serverSocket, (SOCKADDR*)&serverAddr, sizeof(serverAddr)) == SOCKET_ERROR) {
            closesocket(serverSocket); WSACleanup(); return false;
        }
        if (listen(serverSocket, SOMAXCONN) == SOCKET_ERROR) {
            closesocket(serverSocket); WSACleanup(); return false;
        }

        isRunning = true;
        serverThread = std::thread(&MjpegServer::ServerLoop, this);
        return true;
    }

    void Stop() {
        if (!isRunning) return;
        isRunning = false;
        frameCV.notify_all();

        if (serverSocket != INVALID_SOCKET) {
            closesocket(serverSocket);
            serverSocket = INVALID_SOCKET;
        }
        if (serverThread.joinable()) serverThread.join();

        {
            std::lock_guard<std::mutex> lock(clientsMutex);
            for (SOCKET s : clientSockets) closesocket(s);
            clientSockets.clear();
        }
        {
            std::lock_guard<std::mutex> lock(sseMutex);
            for (SOCKET s : sseSockets) closesocket(s);
            sseSockets.clear();
        }
        WSACleanup();
    }

    // --------------------------------------------------------
    //  Called by CameraReaderLoop — push new MJPEG frame
    // --------------------------------------------------------
    void SetLatestFrame(const cv::Mat& frame) {
        if (!isRunning || clientSockets.empty()) return;
        {
            std::lock_guard<std::mutex> lock(frameMutex);
            latestFrame = frame.clone();
            newFrameAvailable = true;
        }
        frameCV.notify_all();
    }

    // --------------------------------------------------------
    //  Called by timer1_Tick — broadcast JSON to all SSE clients
    // --------------------------------------------------------
    void PushLogEvent(const std::string& jsonPayload) {
        if (!isRunning) return;

        // SSE format: "data: <payload>\n\n"
        std::string msg = "data: " + jsonPayload + "\n\n";

        std::lock_guard<std::mutex> lock(sseMutex);
        std::vector<SOCKET> dead;
        for (SOCKET s : sseSockets) {
            if (send(s, msg.c_str(), (int)msg.size(), 0) == SOCKET_ERROR) {
                dead.push_back(s);
            }
        }
        for (SOCKET s : dead) {
            closesocket(s);
            sseSockets.erase(std::find(sseSockets.begin(), sseSockets.end(), s));
        }
    }

    int GetClientCount() {
        std::lock_guard<std::mutex> lock(clientsMutex);
        return (int)clientSockets.size();
    }

    int GetPort() const { return port; }
};
