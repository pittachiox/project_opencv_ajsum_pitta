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
    
    ParkingSlot() : id(-1), status(SlotStatus::EMPTY), occupiedByTrackId(-1), occupancyPercent(0.0f) {}
    
    ParkingSlot(int _id, const std::vector<cv::Point>& _poly) 
        : id(_id), polygon(_poly), status(SlotStatus::EMPTY), occupiedByTrackId(-1), occupancyPercent(0.0f) {}
    
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
        cv::FileStorage fs(filename, cv::FileStorage::WRITE);
        if (!fs.isOpened()) return false;
        
        fs << "name" << name;
        fs << "description" << description;
        fs << "imageWidth" << imageSize.width;
        fs << "imageHeight" << imageSize.height;
        fs << "slotCount" << (int)slots.size();
        
        for (size_t i = 0; i < slots.size(); i++) {
            std::string prefix = "slot_" + std::to_string(i);
            fs << (prefix + "_id") << slots[i].id;
            fs << (prefix + "_points") << slots[i].polygon;
        }
        
        fs.release();
        return true;
    }
    
    // Load from file
    bool loadFromFile(const std::string& filename) {
        cv::FileStorage fs(filename, cv::FileStorage::READ);
        if (!fs.isOpened()) return false;
        
        fs["name"] >> name;
        fs["description"] >> description;
        fs["imageWidth"] >> imageSize.width;
        fs["imageHeight"] >> imageSize.height;
        
        int slotCount;
        fs["slotCount"] >> slotCount;
        
        slots.clear();
        for (int i = 0; i < slotCount; i++) {
            std::string prefix = "slot_" + std::to_string(i);
            ParkingSlot slot;
            fs[prefix + "_id"] >> slot.id;
            fs[prefix + "_points"] >> slot.polygon;
            slots.push_back(slot);
        }
        
        fs.release();
        return true;
    }
};

// Parking Manager
class ParkingManager {
private:
    std::vector<ParkingSlot> slots;
    cv::Mat templateFrame;  // First frame for template creation
    
    // Calculate intersection area between bbox and polygon
    float calculateIntersectionRatio(const cv::Rect& bbox, const std::vector<cv::Point>& polygon) {
        if (polygon.size() < 3 || bbox.area() == 0) return 0.0f;
        
        // Create masks
        cv::Mat bboxMask = cv::Mat::zeros(templateFrame.size(), CV_8UC1);
        cv::Mat polyMask = cv::Mat::zeros(templateFrame.size(), CV_8UC1);
        
        // Draw bbox
        cv::rectangle(bboxMask, bbox, cv::Scalar(255), -1);
        
        // Draw polygon
        std::vector<std::vector<cv::Point>> contours = { polygon };
        cv::drawContours(polyMask, contours, 0, cv::Scalar(255), -1);
        
        // Calculate intersection
        cv::Mat intersection;
        cv::bitwise_and(bboxMask, polyMask, intersection);
        
        int intersectionArea = cv::countNonZero(intersection);
        int bboxArea = bbox.area();
        
        return bboxArea > 0 ? (float)intersectionArea / bboxArea * 100.0f : 0.0f;
    }

public:
    ParkingManager() {}
    
    // [NEW] ?????????????????????????????
    bool isCarInSlot(const cv::Rect& carBbox, const ParkingSlot& slot) const {
        float ratio = const_cast<ParkingManager*>(this)->calculateIntersectionRatio(carBbox, slot.polygon);
        return ratio > 5.0f; // ???????????????? 5% = ??????????
    }
    
    // Set template frame (first frame of video)
    void setTemplateFrame(const cv::Mat& frame) {
        templateFrame = frame.clone();
    }
    
    // Add parking slot
    void addSlot(const std::vector<cv::Point>& polygon) {
        int newId = slots.size() + 1;
        slots.push_back(ParkingSlot(newId, polygon));
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
        ParkingTemplate templ;
        templ.name = name;
        templ.description = desc;
        templ.slots = slots;
        templ.imageSize = templateFrame.size();
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
        // Reset all slots to empty
        for (auto& slot : slots) {
            slot.status = SlotStatus::EMPTY;
            slot.occupiedByTrackId = -1;
            slot.occupancyPercent = 0.0f;
        }
        
        // Check each tracked object
        for (const auto& obj : trackedObjects) {
            // Skip if not vehicle (car, truck, bus, motorcycle)
            if (obj.classId != 2 && obj.classId != 3 && obj.classId != 5 && obj.classId != 7) {
                continue;
            }
            
            bool foundSlot = false;
            int bestSlotIdx = -1;
            float bestRatio = 0.0f;
            
            // Find best matching slot
            for (size_t i = 0; i < slots.size(); i++) {
                float ratio = calculateIntersectionRatio(obj.bbox, slots[i].polygon);
                
                if (ratio > bestRatio) {
                    bestRatio = ratio;
                    bestSlotIdx = i;
                }
            }
            
            // Update slot status based on ratio
            if (bestSlotIdx >= 0 && bestRatio > 5.0f) { // At least 5% overlap
                foundSlot = true;
                slots[bestSlotIdx].occupiedByTrackId = obj.id;
                slots[bestSlotIdx].occupancyPercent = bestRatio;
                
                if (bestRatio >= 60.0f) {
                    slots[bestSlotIdx].status = SlotStatus::OCCUPIED_GOOD;  // ?????
                } else if (bestRatio >= 45.0f) {
                    slots[bestSlotIdx].status = SlotStatus::OCCUPIED_OK;    // ??????
                } else {
                    slots[bestSlotIdx].status = SlotStatus::OCCUPIED_BAD;   // ?????
                }
            }
            
            // If vehicle not in any slot (illegal parking)
            if (!foundSlot || bestRatio < 5.0f) {
                // Create virtual "illegal" slot for this vehicle
                ParkingSlot illegalSlot;
                illegalSlot.id = -1;
                illegalSlot.status = SlotStatus::ILLEGAL;
                illegalSlot.occupiedByTrackId = obj.id;
                illegalSlot.occupancyPercent = 0.0f;
                
                // Use bbox as polygon
                illegalSlot.polygon = {
                    cv::Point(obj.bbox.x, obj.bbox.y),
                    cv::Point(obj.bbox.x + obj.bbox.width, obj.bbox.y),
                    cv::Point(obj.bbox.x + obj.bbox.width, obj.bbox.y + obj.bbox.height),
                    cv::Point(obj.bbox.x, obj.bbox.y + obj.bbox.height)
                };
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
                    color = cv::Scalar(255, 255, 255);  // ???
                    statusText = "Empty";
                    break;
                case SlotStatus::OCCUPIED_GOOD:
                    color = cv::Scalar(255, 200, 0);    // ???
                    statusText = "OK " + std::to_string((int)slot.occupancyPercent) + "%";
                    break;
                case SlotStatus::OCCUPIED_OK:
                    color = cv::Scalar(255, 0, 255);    // ????
                    statusText = "Fair " + std::to_string((int)slot.occupancyPercent) + "%";
                    break;
                case SlotStatus::OCCUPIED_BAD:
                    color = cv::Scalar(0, 0, 255);      // ???
                    statusText = "Bad " + std::to_string((int)slot.occupancyPercent) + "%";
                    break;
                case SlotStatus::ILLEGAL:
                    color = cv::Scalar(0, 0, 255);      // ???
                    statusText = "ILLEGAL!";
                    break;
            }
            
            // Draw polygon
            std::vector<std::vector<cv::Point>> contours = { slot.polygon };
            cv::drawContours(result, contours, 0, color, 2);
            
            // Fill with semi-transparent color
            cv::Mat overlay = result.clone();
            cv::drawContours(overlay, contours, 0, color, -1);
            cv::addWeighted(overlay, 0.3, result, 0.7, 0, result);
            
            // Draw slot ID and status
            cv::Point center = slot.getCenter();
            std::string label = "Slot " + std::to_string(slot.id);
            
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
