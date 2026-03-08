#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <fstream>

// Parking slot status
enum class SlotStatus {
    EMPTY,          // ???? (?????)
    OCCUPIED_GOOD,  // ????? >60% (?????)
    OCCUPIED_OK,    // ???????? 45-60% (??????)
    OCCUPIED_BAD,   // ?????? <45% (?????)
    ILLEGAL         // ????????? (?????)
};

// Parking slot structure
struct ParkingSlot {
    int id;
    std::vector<cv::Point> polygon;  // ??????????????????????????
    SlotStatus status;
    int occupiedByTrackId;           // Track ID ???????????????
    float occupancyPercent;          // % ??????????????????
    std::string type;                // "Car" or "Motorcycle"
    int tempOccupiedBy;              // Transient: occupied by track ID in current frame
    int tempClassId;                 // Transient: class ID occupying in current frame
    int framesOccupied;              // Stabilization: frames continuously occupied
    int framesEmpty;                 // Stabilization: frames continuously empty
    
    ParkingSlot() : id(-1), type("Car"), status(SlotStatus::EMPTY), occupiedByTrackId(-1), occupancyPercent(0.0f), 
                    tempOccupiedBy(-1), tempClassId(-1), framesOccupied(0), framesEmpty(0) {}
    
    ParkingSlot(int _id, const std::vector<cv::Point>& _poly, const std::string& _type = "Car") 
        : id(_id), polygon(_poly), type(_type), status(SlotStatus::EMPTY), occupiedByTrackId(-1), occupancyPercent(0.0f),
          tempOccupiedBy(-1), tempClassId(-1), framesOccupied(0), framesEmpty(0) {}
    
    // Get bounding box of polygon
    cv::Rect getBoundingBox() const {
        if (polygon.empty()) return cv::Rect();
        return cv::boundingRect(polygon);
    }
    
    // Get center point
    cv::Point getCenter() const {
        if (polygon.empty()) return cv::Point(0, 0);
        cv::Moments m = cv::moments(polygon);
        return cv::Point(static_cast<int>(m.m10 / m.m00), static_cast<int>(m.m01 / m.m00));
    }
    
    // Get top-right point of bounding box
    cv::Point getTopRight() const {
        cv::Rect bbox = getBoundingBox();
        return cv::Point(bbox.x + bbox.width, bbox.y);
    }
    
    // Calculate polygon area
    double getArea() const {
        if (polygon.size() < 3) return 0.0;
        return cv::contourArea(polygon);
    }
};

// Template structure for saving/loading
struct ParkingTemplate {
    std::string name;
    std::string description;
    std::vector<ParkingSlot> slots;
    cv::Size imageSize;  // ?????????????????? template
    
    // Save to file
    bool saveToFile(const std::string& filename) const {
        try {
            // [FIXED] Validate imageSize before saving
            if (imageSize.width <= 0 || imageSize.height <= 0) {
                return false; // Invalid image size
            }
            
            cv::FileStorage fs(filename, cv::FileStorage::WRITE);
            if (!fs.isOpened()) {
                return false; // Cannot open file for writing
            }
            
            fs << "name" << name;
            fs << "description" << description;
            fs << "imageWidth" << imageSize.width;
            fs << "imageHeight" << imageSize.height;
            fs << "slotCount" << (int)slots.size();
            
            for (size_t i = 0; i < slots.size(); i++) {
                std::string prefix = "slot_" + std::to_string(i);
                fs << (prefix + "_id") << slots[i].id;
                fs << (prefix + "_type") << slots[i].type;
                fs << (prefix + "_points") << slots[i].polygon;
            }
            
            fs.release();
            return true;
        }
        catch (...) {
            return false; // Handle any exceptions
        }
    }
    
    // Load from file
    bool loadFromFile(const std::string& filename) {
        try {
            cv::FileStorage fs(filename, cv::FileStorage::READ);
            if (!fs.isOpened()) {
                OutputDebugStringA(("[ERROR] loadFromFile: Failed to open FileStorage for " + filename + "\n").c_str());
                return false;
            }
            
            fs["name"] >> name;
            fs["description"] >> description;
            fs["imageWidth"] >> imageSize.width;
            fs["imageHeight"] >> imageSize.height;
            
            int slotCount = 0;
            fs["slotCount"] >> slotCount;
            
            OutputDebugStringA(("[INFO] loadFromFile: Parsed generic info. slotCount: " + std::to_string(slotCount) + "\n").c_str());
            
            slots.clear();
            for (int i = 0; i < slotCount; i++) {
                std::string prefix = "slot_" + std::to_string(i);
                ParkingSlot slot;
                
                if (fs[prefix + "_id"].empty()) {
                    OutputDebugStringA(("[ERROR] loadFromFile: Missing " + prefix + "_id node\n").c_str());
                    return false;
                }
                fs[prefix + "_id"] >> slot.id;
                
                cv::FileNode typeNode = fs[prefix + "_type"];
                if (typeNode.empty()) slot.type = "Car"; // Backward compatibility
                else typeNode >> slot.type;
                
                if (fs[prefix + "_points"].empty()) {
                    OutputDebugStringA(("[ERROR] loadFromFile: Missing " + prefix + "_points node\n").c_str());
                    return false;
                }
                fs[prefix + "_points"] >> slot.polygon;
                slots.push_back(slot);
            }
            
            fs.release();
            OutputDebugStringA(("[INFO] loadFromFile: Successfully loaded template with " + std::to_string(slots.size()) + " slots\n").c_str());
            return true;
        }
        catch (const cv::Exception& e) {
            OutputDebugStringA(("[ERROR] cv::Exception in loadFromFile: " + std::string(e.what()) + "\n").c_str());
            return false;
        }
        catch (const std::exception& e) {
            OutputDebugStringA(("[ERROR] std::exception in loadFromFile: " + std::string(e.what()) + "\n").c_str());
            return false;
        }
        catch (...) {
            OutputDebugStringA("[ERROR] Unknown exception in loadFromFile\n");
            return false;
        }
    }
};

// Parking Manager
class ParkingManager {
private:
    std::vector<ParkingSlot> slots;
    cv::Mat templateFrame;  // First frame for template creation
    
    // [OPTIMIZED] Check if a car's center is inside the slot
    bool isCarInSlot(const cv::Rect& carBbox, const ParkingSlot& slot) const {
        if (slot.polygon.size() < 3) return false;
        cv::Point center(carBbox.x + carBbox.width / 2, carBbox.y + carBbox.height / 2);
        return cv::pointPolygonTest(slot.polygon, center, false) >= 0;
    }

public:
    // Set template frame (first frame of video)
    void setTemplateFrame(const cv::Mat& frame) {
        templateFrame = frame.clone();
    }
    
    // Add parking slot
    void addSlot(const std::vector<cv::Point>& polygon, const std::string& type = "Car") {
        int newId = (int)slots.size() + 1;
        slots.push_back(ParkingSlot(newId, polygon, type));
    }
    
    // Clear all slots
    void clearSlots() {
        slots.clear();
    }
    
    // Get slots
    std::vector<ParkingSlot>& getSlots() {
        return slots;
    }
    
    // Save template
    bool saveTemplate(const std::string& filename, const std::string& name, const std::string& desc) {
        // [FIXED] Ensure templateFrame is not empty before saving
        if (templateFrame.empty()) {
            return false; // Cannot save if templateFrame is empty
        }
        
        ParkingTemplate templ;
        templ.name = name;
        templ.description = desc;
        templ.slots = slots;
        templ.imageSize = templateFrame.size();
        
        // [FIXED] Double-check imageSize is valid
        if (templ.imageSize.width <= 0 || templ.imageSize.height <= 0) {
            return false; // Invalid image dimensions
        }
        
        return templ.saveToFile(filename);
    }
    
    // Load template
    bool loadTemplate(const std::string& filename) {
        ParkingTemplate templ;
        if (!templ.loadFromFile(filename)) return false;
        slots = templ.slots;
        return true;
    }
    
    // Update slot status based on tracked objects
    void updateSlotStatus(const std::vector<TrackedObject>& trackedObjects) {
        // First reset transient info for this specific frame
        for (auto& slot : slots) {
            slot.occupancyPercent = 0.0f; // Reset transient
        }
        
        // Check each tracked object
        for (const auto& obj : trackedObjects) {
            // Skip if not vehicle (car, truck, bus, motorcycle)
            if (obj.classId != 2 && obj.classId != 3 && obj.classId != 5 && obj.classId != 7) {
                continue;
            }
            
            bool foundSlot = false;
            int bestSlotIdx = -1;
            
            // Find best matching slot (first slot that contains the center)
            for (size_t i = 0; i < slots.size(); i++) {
                if (isCarInSlot(obj.bbox, slots[i])) {
                    bestSlotIdx = static_cast<int>(i);
                    break;
                }
            }
            
            // Note occupancy in current frame
            if (bestSlotIdx >= 0) {
                slots[bestSlotIdx].occupancyPercent = 100.0f;
                slots[bestSlotIdx].tempOccupiedBy = obj.id;
                slots[bestSlotIdx].tempClassId = obj.classId;
            }
        }
        
        // Stabilize statuses across frames (3 frame delay)
        for (auto& slot : slots) {
            if (slot.occupancyPercent > 0.0f) {
                slot.framesOccupied++;
                slot.framesEmpty = 0;
            } else {
                slot.framesEmpty++;
                slot.framesOccupied = 0;
            }
            
            if (slot.status == SlotStatus::EMPTY && slot.framesOccupied >= 3) {
                slot.occupiedByTrackId = slot.tempOccupiedBy;
                
                // Check for Wrong Vehicle Type violation
                bool isCarObj = (slot.tempClassId == 2 || slot.tempClassId == 5 || slot.tempClassId == 7);
                bool isMotoObj = (slot.tempClassId == 3);
                bool isCarSlot = (slot.type == "Car");
                bool isMotoSlot = (slot.type == "Motorcycle");
                
                if ((isCarObj && isMotoSlot) || (isMotoObj && isCarSlot)) {
                    slot.status = SlotStatus::ILLEGAL;
                } else {
                    slot.status = SlotStatus::OCCUPIED_GOOD;
                }
            } 
            else if (slot.status != SlotStatus::EMPTY && slot.framesEmpty >= 3) {
                slot.status = SlotStatus::EMPTY;
                slot.occupiedByTrackId = -1;
            }
            else if (slot.status != SlotStatus::EMPTY && slot.occupancyPercent > 0.0f) {
                // Instantly update who is occupying if type changed without fully emptying
                slot.occupiedByTrackId = slot.tempOccupiedBy;
            }
        }
    }
    
    // Draw slots on image
    cv::Mat drawSlots(const cv::Mat& frame) const {
        cv::Mat result = frame.clone();
        
        for (const auto& slot : slots) {
            cv::Scalar color;
            std::string statusText;
            
            switch (slot.status) {
                case SlotStatus::EMPTY:
                    color = cv::Scalar(255, 255, 255);  // White overlay base, we use dot color
                    statusText = "Empty";
                    break;
                case SlotStatus::OCCUPIED_GOOD:
                case SlotStatus::OCCUPIED_OK:
                case SlotStatus::OCCUPIED_BAD:
                    color = (slot.type == "Car") ? cv::Scalar(255, 144, 30) : cv::Scalar(0, 165, 255); // Blue for Car, Orange for Moto
                    statusText = "Occupied";
                    break;
                case SlotStatus::ILLEGAL:
                    color = cv::Scalar(0, 0, 255);      // Red
                    statusText = "Wrong Type";
                    break;
            }
            
            // Draw polygon
            std::vector<std::vector<cv::Point>> contours = { slot.polygon };
            cv::drawContours(result, contours, 0, color, 2);
            
            // Fill with semi-transparent color
            cv::Mat overlay = result.clone();
            cv::drawContours(overlay, contours, 0, color, -1);
            cv::addWeighted(overlay, 0.3, result, 0.7, 0, result);

            // Draw a clear, non-intrusive colored dot in the center to indicate vehicle type
            cv::Point center = slot.getCenter();
            
            // Define colors for the center dot (Blue for Car, Orange for Moto)
            cv::Scalar dotColor = (slot.type == "Car") ? cv::Scalar(255, 144, 30) : cv::Scalar(0, 165, 255); 
            
            // Draw a subtle dark border for the dot, then the dot itself
            int radius = 8;
            cv::circle(result, center, radius + 1, cv::Scalar(0, 0, 0), cv::FILLED);
            cv::circle(result, center, radius, dotColor, cv::FILLED);
            
            // Draw slot ID and status
            std::string label = "S" + std::to_string(slot.id) + " (" + slot.type.substr(0,1) + ")";
            
            cv::putText(result, label, cv::Point(center.x - 30, center.y - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 2);
            cv::putText(result, label, cv::Point(center.x - 30, center.y - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
            
            cv::putText(result, statusText, cv::Point(center.x - 30, center.y + 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(0, 0, 0), 2);
            cv::putText(result, statusText, cv::Point(center.x - 30, center.y + 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(255, 255, 255), 1);
        }
        
        // Draw statistics
        int emptyCount = 0, occupiedCount = 0, illegalCount = 0;
        for (const auto& slot : slots) {
            if (slot.status == SlotStatus::EMPTY) emptyCount++;
            else if (slot.status != SlotStatus::ILLEGAL) occupiedCount++;
        }
        
        std::string stats = "Total: " + std::to_string(slots.size()) + 
                           " | Empty: " + std::to_string(emptyCount) +
                           " | Occupied: " + std::to_string(occupiedCount);
        
        cv::rectangle(result, cv::Point(5, 5), cv::Point(400, 50), cv::Scalar(0, 0, 0), -1);
        cv::putText(result, stats, cv::Point(10, 30),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        
        return result;
    }
    
    // Get template frame
    cv::Mat getTemplateFrame() const {
        return templateFrame;
    }
};
