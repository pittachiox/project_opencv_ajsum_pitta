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
        // Add to client list
        {
            std::lock_guard<std::mutex> lock(clientsMutex);
            clientSockets.push_back(clientSocket);
        }
        
        // HTTP Stream Header
        std::string httpHeader = "HTTP/1.1 200 OK\r\n"
                                 "Content-Type: multipart/x-mixed-replace; boundary=mjpegstream\r\n"
                                 "Connection: close\r\n\r\n";
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
        if (!isRunning || clientSockets.empty()) return; // Don't encode if no clients
        
        {
            std::lock_guard<std::mutex> lock(frameMutex);
            latestFrame = frame.clone();
            newFrameAvailable = true;
        }
        frameCV.notify_all();
    }
    
    int GetClientCount() {
        std::lock_guard<std::mutex> lock(clientsMutex);
        return clientSockets.size();
    }
    
    int GetPort() const {
        return port;
    }
};
