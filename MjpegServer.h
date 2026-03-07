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
#include <functional>
#include <fstream>
#include <mutex>
#include <algorithm>
#include <cctype>
static std::mutex g_logMutex;
inline void DumpLog(const std::string& msg) {
    std::lock_guard<std::mutex> lock(g_logMutex);
    std::cout << msg << std::endl;
}

class MjpegServer;
__declspec(selectany) MjpegServer* g_globalWebServer = nullptr;
using ConnectOnlineCallback = std::function<bool(std::string, std::string, std::string)>;

#pragma comment(lib, "ws2_32.lib")

class MjpegServer {
private:
    SOCKET serverSocket;
    std::atomic<bool> isRunning;
    std::thread serverThread;
    
    std::vector<SOCKET> clientSockets;
    std::mutex clientsMutex;
    
    cv::Mat latestFrame;
    std::mutex frameMutex;
    std::condition_variable frameCV;
    bool newFrameAvailable = false;
    
    int port;

    std::string latestStatsJson = "{\"empty\":0,\"normal\":0,\"carEmpty\":0,\"carNormal\":0,\"motoEmpty\":0,\"motoNormal\":0,\"violation\":0,\"logs\":[]}";
    std::mutex statsMutex;

    using SaveTemplateCallback = std::function<bool(std::string)>;
    using GetFrameCallback = std::function<cv::Mat()>;
    using DisconnectCallback = std::function<void()>;
public:
    ConnectOnlineCallback onConnectOnline;
    SaveTemplateCallback onSaveTemplate;
    GetFrameCallback onGetFrame;
    GetFrameCallback onGetRawFrame;
    DisconnectCallback onDisconnect;
    
    void SetConnectOnlineCallback(ConnectOnlineCallback cb) { onConnectOnline = cb; }
    void SetSaveTemplateCallback(SaveTemplateCallback cb) { onSaveTemplate = cb; }
    void SetGetFrameCallback(GetFrameCallback cb) { onGetFrame = cb; }
    void SetGetRawFrameCallback(GetFrameCallback cb) { onGetRawFrame = cb; }
    void SetDisconnectCallback(DisconnectCallback cb) { onDisconnect = cb; }
private:

    void ServerLoop() {
        while (isRunning) {
            fd_set readfds;
            FD_ZERO(&readfds);
            FD_SET(serverSocket, &readfds);
            
            timeval timeout;
            timeout.tv_sec = 0;
            timeout.tv_usec = 100000; // 100ms timeout
            
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
        char buffer[4096];
        int bytesRead = recv(clientSocket, buffer, sizeof(buffer) - 1, 0);
        if (bytesRead <= 0) {
            closesocket(clientSocket);
            return;
        }
        buffer[bytesRead] = '\0';
        std::string request(buffer);
        
        // Only try to read a full body for POST requests — skip for GET/streaming
        bool isPost = (request.size() >= 4 && request[0] == 'P' && request[1] == 'O');
        if (isPost) {
            // Find Content-Length in headers (case-insensitive search on just the header portion)
            size_t headersEnd = request.find("\r\n\r\n");
            if (headersEnd != std::string::npos) {
                std::string headers = request.substr(0, headersEnd);
                // Lowercase only the small headers block
                std::string headersLower = headers;
                std::transform(headersLower.begin(), headersLower.end(), headersLower.begin(), ::tolower);
                
                size_t contentLengthPos = headersLower.find("content-length: ");
                if (contentLengthPos != std::string::npos) {
                    size_t clEnd = headersLower.find("\r\n", contentLengthPos);
                    if (clEnd != std::string::npos) {
                        int expectedLength = std::stoi(headersLower.substr(contentLengthPos + 16, clEnd - (contentLengthPos + 16)));
                        int currentBodyLength = (int)request.length() - (int)(headersEnd + 4);
                        int remainingBytes = expectedLength - currentBodyLength;
                        while (remainingBytes > 0) {
                            int chunkRead = recv(clientSocket, buffer, std::min((int)sizeof(buffer) - 1, remainingBytes), 0);
                            if (chunkRead <= 0) break;
                            buffer[chunkRead] = '\0';
                            request += buffer;
                            remainingBytes -= chunkRead;
                        }
                    }
                }
            }
        }
        
        size_t firstSpace = request.find(' ');
        size_t secondSpace = request.find(' ', firstSpace + 1);
        if (firstSpace == std::string::npos || secondSpace == std::string::npos) {
            closesocket(clientSocket);
            return;
        }
        
        std::string method = request.substr(0, firstSpace);
        std::string path = request.substr(firstSpace + 1, secondSpace - firstSpace - 1);
        // Strip query string (e.g. ?t=123456) before route matching
        size_t qpos = path.find('?');
        if (qpos != std::string::npos) path = path.substr(0, qpos);

        if (path == "/video") {
            ServeMjpegStream(clientSocket);
        } else if (path == "/api/stats") {
            ServeStats(clientSocket);
        } else if (path == "/api/current_frame") {
            ServeCurrentFrame(clientSocket);
        } else if (path == "/api/raw_frame") {
            ServeRawFrame(clientSocket);
        } else if (path == "/api/save_template" && request.find("POST") == 0) {
            ServeSaveTemplate(clientSocket, request);
        } else if (path == "/api/connect_online" && request.find("POST") == 0) {
            ServeConnectOnline(clientSocket, request);
        } else if (path == "/api/disconnect" && request.find("POST") == 0) {
            ServeDisconnect(clientSocket);
        } else if (path == "/api/anomaly_events") {
            ServeAnomalyEvents(clientSocket, request);
        } else if (path == "/api/parking_areas") {
            ServeParkingAreas(clientSocket, request);
        } else if (path.find("/locvideo/") == 0) {
            ServeFileDirectly(clientSocket, "C:" + path, request);
        } else if (path.find("/smart_parking_violations/") == 0) {
            ServeFileDirectly(clientSocket, "C:" + path, request);
        } else if (path.find("/violations/") == 0) {
            std::string realPath = path;
            realPath.replace(0, 12, "/smart_parking_violations/");
            ServeFileDirectly(clientSocket, "C:" + realPath, request);
        } else if (path == "/setup_online") {
            ServeHtml(clientSocket, "setup_online.html");
        } else if (path == "/setup_parking") {
            ServeHtml(clientSocket, "setup_parking.html");
        } else if (path == "/camera") {
            ServeHtml(clientSocket, "camera.html");
        } else if (path == "/dashboard") {
            ServeHtml(clientSocket, "index.html");
        } else if (path == "/" || path == "/index.html" || path == "/home") {
            ServeHtml(clientSocket, "home.html");
        } else {
            std::string notFound = "HTTP/1.1 404 Not Found\r\nContent-Length: 9\r\n\r\nNot Found";
            send(clientSocket, notFound.c_str(), notFound.length(), 0);
            closesocket(clientSocket);
        }
    }

    // ==========================================
    // [PHASE 3] NEW ENDPOINTS FOR JSON API
    // ==========================================
    void ServeAnomalyEvents(SOCKET clientSocket, const std::string& request) {
        // Default to today
        SYSTEMTIME st;
        GetLocalTime(&st);
        char defaultDate[32];
        sprintf_s(defaultDate, sizeof(defaultDate), "%04d%02d%02d", st.wYear, st.wMonth, st.wDay);
        std::string dateTarget = defaultDate;

        // Extract ?date=YYYY-MM-DD to YYYYMMDD
        size_t queryPos = request.find("date=");
        if (queryPos != std::string::npos) {
            size_t spacePos = request.find(' ', queryPos);
            size_t ampPos = request.find('&', queryPos);
            size_t endPos = (ampPos != std::string::npos && ampPos < spacePos) ? ampPos : spacePos;
            
            std::string dateStr = request.substr(queryPos + 5, endPos - (queryPos + 5));
            dateTarget = "";
            for(char c : dateStr) if (c != '-') dateTarget += c;
        }

        int limit = -1;
        size_t limitPos = request.find("limit=");
        if (limitPos != std::string::npos) {
            size_t spacePos = request.find(' ', limitPos);
            size_t ampPos = request.find('&', limitPos);
            size_t endPos = (ampPos != std::string::npos && ampPos < spacePos) ? ampPos : spacePos;
            
            try {
                limit = std::stoi(request.substr(limitPos + 6, endPos - (limitPos + 6)));
            } catch(...) {}
        }

        std::string dirPath = "C:\\loc_json\\anomaly_events\\" + dateTarget;
        ServeJsonDirectoryAsArray(clientSocket, dirPath, limit);
    }

    void ServeParkingAreas(SOCKET clientSocket, const std::string& request) {
        // Default to today
        SYSTEMTIME st;
        GetLocalTime(&st);
        char defaultDate[32];
        sprintf_s(defaultDate, sizeof(defaultDate), "%04d%02d%02d", st.wYear, st.wMonth, st.wDay);
        std::string dateTarget = defaultDate;

        // Extract ?date=YYYY-MM-DD
        size_t queryPos = request.find("date=");
        if (queryPos != std::string::npos) {
            size_t spacePos = request.find(' ', queryPos);
            size_t ampPos = request.find('&', queryPos);
            size_t endPos = (ampPos != std::string::npos && ampPos < spacePos) ? ampPos : spacePos;
            
            std::string dateStr = request.substr(queryPos + 5, endPos - (queryPos + 5));
            dateTarget = "";
            for(char c : dateStr) if (c != '-') dateTarget += c;
        }

        int limit = -1;
        size_t limitPos = request.find("limit=");
        if (limitPos != std::string::npos) {
            size_t spacePos = request.find(' ', limitPos);
            size_t ampPos = request.find('&', limitPos);
            size_t endPos = (ampPos != std::string::npos && ampPos < spacePos) ? ampPos : spacePos;
            
            try {
                limit = std::stoi(request.substr(limitPos + 6, endPos - (limitPos + 6)));
            } catch(...) {}
        }

        std::string dirPath = "C:\\loc_json\\parking_areas\\" + dateTarget;
        ServeJsonDirectoryAsArray(clientSocket, dirPath, limit);
    }

    void ServeJsonDirectoryAsArray(SOCKET clientSocket, const std::string& dirPath, int limit = -1) {
        std::vector<std::string> fileNames;
        WIN32_FIND_DATAA findData;
        HANDLE hFind = FindFirstFileA((dirPath + "\\*.json").c_str(), &findData);
        
        if (hFind != INVALID_HANDLE_VALUE) {
            do {
                if (!(findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                    fileNames.push_back(findData.cFileName);
                }
            } while (FindNextFileA(hFind, &findData));
            FindClose(hFind);
        }
        
        // Sort descending (newest epoch first)
        std::sort(fileNames.begin(), fileNames.end(), std::greater<std::string>());
        
        std::string jsonContent = "[";
        bool first = true;
        int count = 0;
        
        for (const std::string& fileName : fileNames) {
            if (limit > 0 && count >= limit) break;
            
            std::string filePath = dirPath + "\\" + fileName;
            std::ifstream inFile(filePath);
            if (inFile.is_open()) {
                std::stringstream buffer;
                buffer << inFile.rdbuf();
                
                if (!first) jsonContent += ",";
                jsonContent += buffer.str();
                first = false;
                
                inFile.close();
                count++;
            }
        }
        
        jsonContent += "]";

        std::string header = "HTTP/1.1 200 OK\r\n"
                             "Content-Type: application/json; charset=utf-8\r\n"
                             "Access-Control-Allow-Origin: *\r\n"
                             "Connection: close\r\n"
                             "Content-Length: " + std::to_string(jsonContent.size()) + "\r\n\r\n";
        
        send(clientSocket, header.c_str(), header.length(), 0);
        send(clientSocket, jsonContent.c_str(), jsonContent.length(), 0);
        closesocket(clientSocket);
    }

    void ServeFileDirectly(SOCKET clientSocket, const std::string& fsPath, const std::string& httpRequest) {
        // Strip out the query param safely (e.g. ?time=45)
        std::string cleanPath = fsPath;
        size_t queryPos = cleanPath.find("?");
        if (queryPos != std::string::npos) {
            cleanPath = cleanPath.substr(0, queryPos);
        }
        // Force replace / with \\ for windows paths just in case
        std::replace(cleanPath.begin(), cleanPath.end(), '/', '\\');

        std::ifstream file(cleanPath, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            std::string notFound = "HTTP/1.1 404 Not Found\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: 9\r\n\r\nNot Found";
            send(clientSocket, notFound.c_str(), notFound.length(), 0);
            closesocket(clientSocket);
            return;
        }

        std::streamsize fileSize = file.tellg();
        file.seekg(0, std::ios::beg);

        std::string contentType = "application/octet-stream";
        if (cleanPath.find(".jpg") != std::string::npos || cleanPath.find(".jpeg") != std::string::npos) contentType = "image/jpeg";
        else if (cleanPath.find(".webm") != std::string::npos) contentType = "video/webm";
        else if (cleanPath.find(".mp4") != std::string::npos) contentType = "video/mp4";

        long long start = 0;
        long long end = fileSize - 1;
        bool isRange = false;

        size_t rangePos = httpRequest.find("Range: bytes=");
        if (rangePos != std::string::npos) {
            isRange = true;
            size_t startPos = rangePos + 13;
            size_t dashPos = httpRequest.find("-", startPos);
            size_t endLinePos = httpRequest.find("\r", startPos);
            if (endLinePos == std::string::npos) endLinePos = httpRequest.find("\n", startPos);
            
            if (dashPos != std::string::npos && dashPos < endLinePos) {
                std::string startStr = httpRequest.substr(startPos, dashPos - startPos);
                std::string endStr = httpRequest.substr(dashPos + 1, endLinePos - dashPos - 1);
                try { if (!startStr.empty()) start = std::stoll(startStr); } catch (...) {}
                try { if (!endStr.empty() && endStr != " ") end = std::stoll(endStr); } catch (...) {}
            }
        }
        
        if (end >= fileSize) end = fileSize - 1;
        if (start < 0) start = 0;
        long long contentLength = end - start + 1;

        std::string responseHeader;
        if (isRange) {
            responseHeader = "HTTP/1.1 206 Partial Content\r\n"
                             "Content-Type: " + contentType + "\r\n"
                             "Accept-Ranges: bytes\r\n"
                             "Content-Range: bytes " + std::to_string(start) + "-" + std::to_string(end) + "/" + std::to_string(fileSize) + "\r\n"
                             "Access-Control-Allow-Origin: *\r\n"
                             "Connection: close\r\n"
                             "Content-Length: " + std::to_string(contentLength) + "\r\n\r\n";
        } else {
            responseHeader = "HTTP/1.1 200 OK\r\n"
                             "Content-Type: " + contentType + "\r\n"
                             "Accept-Ranges: bytes\r\n"
                             "Access-Control-Allow-Origin: *\r\n"
                             "Connection: close\r\n"
                             "Content-Length: " + std::to_string(contentLength) + "\r\n\r\n";
        }

        send(clientSocket, responseHeader.c_str(), responseHeader.length(), 0);

        file.seekg(start, std::ios::beg);
        char buffer[8192];
        long long bytesToRead = contentLength;
        while (bytesToRead > 0 && file.read(buffer, (std::streamsize)std::min((long long)sizeof(buffer), bytesToRead))) {
            long long readBytes = file.gcount();
            send(clientSocket, buffer, (int)readBytes, 0);
            bytesToRead -= readBytes;
        }

        if (bytesToRead > 0 && file.gcount() > 0) {
            send(clientSocket, buffer, (int)file.gcount(), 0);
        }

        file.close();
        closesocket(clientSocket);
    }

    void ServeStats(SOCKET clientSocket) {
        std::string json;
        {
            std::lock_guard<std::mutex> lock(statsMutex);
            json = latestStatsJson;
        }
        std::string header = "HTTP/1.1 200 OK\r\n"
                             "Content-Type: application/json; charset=utf-8\r\n"
                             "Access-Control-Allow-Origin: *\r\n"
                             "Connection: close\r\n"
                             "Content-Length: " + std::to_string(json.length()) + "\r\n\r\n";
        send(clientSocket, header.c_str(), header.length(), 0);
        send(clientSocket, json.c_str(), json.length(), 0);
        closesocket(clientSocket);
    }

    void ServeDisconnect(SOCKET clientSocket) {
        DumpLog("[HTTP] Received /api/disconnect");
        if (onDisconnect) {
            DisconnectCallback cb = onDisconnect;
            std::thread([cb]() { cb(); }).detach();
        }
        std::string jsonResponse = "{\"status\":\"disconnected\"}";
        std::string response =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: application/json\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Content-Length: " + std::to_string(jsonResponse.length()) + "\r\n"
            "Connection: close\r\n\r\n" + jsonResponse;
        send(clientSocket, response.c_str(), response.length(), 0);
        closesocket(clientSocket);
    }

    void ServeConnectOnline(SOCKET clientSocket, const std::string& request) {
        std::string ip = "192.168.1.100", port = "8080", reqPath = "/video";
        
        size_t bodyStart = request.find("\r\n\r\n");
        if (bodyStart != std::string::npos) {
            std::string body = request.substr(bodyStart + 4);
            auto extString = [](const std::string& b, const std::string& key) {
                size_t p = b.find("\"" + key + "\":");
                if (p != std::string::npos) {
                    size_t s = b.find("\"", p + key.length() + 3);
                    if (s != std::string::npos) {
                        size_t e = b.find("\"", s + 1);
                        if (e != std::string::npos) return b.substr(s + 1, e - s - 1);
                    }
                }
                return std::string("");
            };
            std::string parsedIp = extString(body, "ip");
            if (!parsedIp.empty()) ip = parsedIp;
            std::string parsedPort = extString(body, "port");
            if (!parsedPort.empty()) port = parsedPort;
            std::string parsedPath = extString(body, "path");
            if (!parsedPath.empty()) reqPath = parsedPath;
        }

        // Fire the connection in a background thread — don't block the HTTP response
        DumpLog("[HTTP] Received /api/connect_online for " + ip + ":" + port + reqPath);

        if (onConnectOnline) {
            std::string capturedIp = ip, capturedPort = port, capturedPath = reqPath;
            ConnectOnlineCallback cb = onConnectOnline;
            std::thread([cb, capturedIp, capturedPort, capturedPath]() {
                cb(capturedIp, capturedPort, capturedPath);
            }).detach();
        }

        // Return 202 Accepted immediately
        std::string jsonResponse = "{\"status\":\"connecting\"}";
        std::string response = 
            "HTTP/1.1 202 Accepted\r\n"
            "Content-Type: application/json\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Content-Length: " + std::to_string(jsonResponse.length()) + "\r\n"
            "Connection: close\r\n\r\n" + jsonResponse;

        send(clientSocket, response.c_str(), response.length(), 0);
        closesocket(clientSocket);
    }

    void ServeCurrentFrame(SOCKET clientSocket) {
        if (!onGetFrame) {
            std::string r = "HTTP/1.1 503 Service Unavailable\r\n\r\n";
            send(clientSocket, r.c_str(), r.length(), 0);
            closesocket(clientSocket);
            return;
        }
        cv::Mat frame = onGetFrame();
        bool isEmpty = frame.empty();
        if (isEmpty) {
            frame = cv::Mat(480, 640, CV_8UC3, cv::Scalar(50, 50, 50));
            cv::putText(frame, "Waiting for AI Processing...", cv::Point(80, 240), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
        }

        std::string statusLine = isEmpty ? "HTTP/1.1 503 Service Unavailable\r\n" : "HTTP/1.1 200 OK\r\n";
        std::vector<uchar> buf;
        std::vector<int> params = { cv::IMWRITE_JPEG_QUALITY, 80 };
        cv::imencode(".jpg", frame, buf, params);
        std::string response = statusLine +
                               "Content-Type: image/jpeg\r\n"
                               "Content-Length: " + std::to_string(buf.size()) + "\r\n"
                               "Cache-Control: no-cache\r\n"
                               "Connection: close\r\n\r\n";
        send(clientSocket, response.c_str(), response.length(), 0);
        send(clientSocket, (const char*)buf.data(), buf.size(), 0);
        closesocket(clientSocket);
    }

    void ServeRawFrame(SOCKET clientSocket) {
        if (!onGetRawFrame) {
            std::string r = "HTTP/1.1 503 Service Unavailable\r\n\r\n";
            send(clientSocket, r.c_str(), r.length(), 0);
            closesocket(clientSocket);
            return;
        }
        cv::Mat frame = onGetRawFrame();
        bool isEmpty = frame.empty();
        if (isEmpty) {
            frame = cv::Mat(480, 640, CV_8UC3, cv::Scalar(50, 50, 50));
            cv::putText(frame, "Waiting for stream...", cv::Point(150, 240), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
        }

        std::string statusLine = isEmpty ? "HTTP/1.1 503 Service Unavailable\r\n" : "HTTP/1.1 200 OK\r\n";
        std::vector<uchar> buf;
        std::vector<int> params = { cv::IMWRITE_JPEG_QUALITY, 80 };
        cv::imencode(".jpg", frame, buf, params);
        std::string response = statusLine +
                               "Content-Type: image/jpeg\r\n"
                               "Content-Length: " + std::to_string(buf.size()) + "\r\n"
                               "Cache-Control: no-cache\r\n"
                               "Connection: close\r\n\r\n";
        send(clientSocket, response.c_str(), response.length(), 0);
        send(clientSocket, (const char*)buf.data(), buf.size(), 0);
        closesocket(clientSocket);
    }

    void ServeSaveTemplate(SOCKET clientSocket, const std::string& request) {
        size_t bodyPos = request.find("\r\n\r\n");
        if (bodyPos == std::string::npos) {
            DumpLog("[API] /save_template -> Failed to find body delimiter.");
            closesocket(clientSocket);
            return;
        }
        std::string body = request.substr(bodyPos + 4);
        DumpLog("[API] /save_template -> Extracted XML body of size " + std::to_string(body.length()) + " bytes.");

        bool success = false;
        if (onSaveTemplate) {
            success = onSaveTemplate(body);
            DumpLog("[API] /save_template -> onSaveTemplate returned " + std::string(success ? "true" : "false"));
        }

        std::string jsonResponse = "{\"status\":\"" + std::string(success ? "success" : "error") + "\"}";
        std::string response = 
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: application/json\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Content-Length: " + std::to_string(jsonResponse.length()) + "\r\n"
            "Connection: close\r\n\r\n" + jsonResponse;

        send(clientSocket, response.c_str(), response.length(), 0);
        closesocket(clientSocket);
    }

    void ServeHtml(SOCKET clientSocket, const std::string& filename) {
        FILE* f = nullptr;
        fopen_s(&f, filename.c_str(), "rb");
        if (!f) {
            std::string msg = "HTTP/1.1 404 Not Found\r\n\r\nFile not found. Please create index.html in the app directory.";
            send(clientSocket, msg.c_str(), msg.length(), 0);
            closesocket(clientSocket);
            return;
        }
        fseek(f, 0, SEEK_END);
        long fsize = ftell(f);
        fseek(f, 0, SEEK_SET);

        std::vector<char> content(fsize);
        if (fsize > 0) {
            fread(content.data(), 1, fsize, f);
        }
        fclose(f);

        std::string header = "HTTP/1.1 200 OK\r\n"
                             "Content-Type: text/html; charset=utf-8\r\n"
                             "Connection: close\r\n"
                             "Content-Length: " + std::to_string(fsize) + "\r\n\r\n";
        send(clientSocket, header.c_str(), header.length(), 0);
        if (fsize > 0) {
            send(clientSocket, content.data(), fsize, 0);
        }
        closesocket(clientSocket);
    }

    void ServeMjpegStream(SOCKET clientSocket) {
        // Add to client list
        {
            std::lock_guard<std::mutex> lock(clientsMutex);
            clientSockets.push_back(clientSocket);
        }
        
        // HTTP Stream Header
        std::string httpHeader = "HTTP/1.1 200 OK\r\n"
                                 "Content-Type: multipart/x-mixed-replace; boundary=mjpegstream\r\n"
                                 "Connection: keep-alive\r\n"
                                 "Cache-Control: no-cache\r\n"
                                 "Pragma: no-cache\r\n\r\n";
        send(clientSocket, httpHeader.c_str(), httpHeader.length(), 0);

        std::vector<uchar> buffer;
        std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 70}; 

        while (isRunning) {
            cv::Mat frameToSend;
            {
                std::unique_lock<std::mutex> lock(frameMutex);
                if (!frameCV.wait_for(lock, std::chrono::milliseconds(500), [this] { return newFrameAvailable || !isRunning; })) {
                    // Timeout, check connection
                    int error = 0;
                    socklen_t len = sizeof(error);
                    if (getsockopt(clientSocket, SOL_SOCKET, SO_ERROR, (char*)&error, &len) != 0 || error != 0) {
                        break; // Connection is dead
                    }
                    continue; // Timeout, no new frame, loop again
                }
                
                if (!isRunning) break;
                
                if (latestFrame.empty()) {
                    newFrameAvailable = false;
                    continue;
                }
                
                // Copy frame to avoid holding lock during encoding
                frameToSend = latestFrame.clone();
                newFrameAvailable = false;
            }

            if (!frameToSend.empty()) {
                cv::imencode(".jpg", frameToSend, buffer, params);
                
                std::string frameHeader = "--mjpegstream\r\n"
                                          "Content-Type: image/jpeg\r\n"
                                          "Content-Length: " + std::to_string(buffer.size()) + "\r\n\r\n";
                
                // Send Header
                int bytesSent = send(clientSocket, frameHeader.c_str(), frameHeader.length(), 0);
                if (bytesSent == SOCKET_ERROR) break;
                
                // Send Image Data
                bytesSent = send(clientSocket, (const char*)buffer.data(), buffer.size(), 0);
                if (bytesSent == SOCKET_ERROR) break;
                
                // Send Footer
                std::string frameFooter = "\r\n";
                bytesSent = send(clientSocket, frameFooter.c_str(), frameFooter.length(), 0);
                if (bytesSent == SOCKET_ERROR) break;
            }
        }

        // Remove from client list
        {
            std::lock_guard<std::mutex> lock(clientsMutex);
            auto it = std::find(clientSockets.begin(), clientSockets.end(), clientSocket);
            if (it != clientSockets.end()) {
                clientSockets.erase(it);
            }
        }
        closesocket(clientSocket);
    }

public:
    MjpegServer(int listenPort = 8080) : isRunning(false), serverSocket(INVALID_SOCKET), port(listenPort) {}

    ~MjpegServer() {
        Stop();
    }

    bool Start() {
        if (isRunning) return true;

        WSADATA wsaData;
        if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
            return false;
        }

        serverSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (serverSocket == INVALID_SOCKET) {
            WSACleanup();
            return false;
        }

        // Allow port reuse
        char opt = 1;
        setsockopt(serverSocket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        sockaddr_in serverAddr;
        serverAddr.sin_family = AF_INET;
        serverAddr.sin_addr.s_addr = INADDR_ANY;
        serverAddr.sin_port = htons(port);

        if (bind(serverSocket, (SOCKADDR*)&serverAddr, sizeof(serverAddr)) == SOCKET_ERROR) {
            closesocket(serverSocket);
            WSACleanup();
            return false;
        }

        if (listen(serverSocket, SOMAXCONN) == SOCKET_ERROR) {
            closesocket(serverSocket);
            WSACleanup();
            return false;
        }

        isRunning = true;
        serverThread = std::thread(&MjpegServer::ServerLoop, this);
        return true;
    }

    void Stop() {
        if (!isRunning) return;
        isRunning = false;
        
        // Notify any waiting threads to wake up and exit
        frameCV.notify_all();

        if (serverSocket != INVALID_SOCKET) {
            closesocket(serverSocket);
            serverSocket = INVALID_SOCKET;
        }

        if (serverThread.joinable()) {
            serverThread.join();
        }

        // Close all client sockets
        {
            std::lock_guard<std::mutex> lock(clientsMutex);
            for (SOCKET sock : clientSockets) {
                closesocket(sock);
            }
            clientSockets.clear();
        }

        WSACleanup();
    }

    void SetLatestFrame(const cv::Mat& frame) {
        if (!isRunning) return;
        
        {
            std::lock_guard<std::mutex> lock(frameMutex);
            latestFrame = frame.clone();
            newFrameAvailable = true;
        }
        frameCV.notify_all();

		// (Removed debug print here to save resources)
    }

    void SetStats(const std::string& json) {
        std::lock_guard<std::mutex> lock(statsMutex);
        latestStatsJson = json;
    }
    
    int GetClientCount() {
        std::lock_guard<std::mutex> lock(clientsMutex);
        return clientSockets.size();
    }
    
    int GetPort() const {
        return port;
    }
};
