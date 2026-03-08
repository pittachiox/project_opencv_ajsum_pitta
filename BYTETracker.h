#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
#include <algorithm>

// Improved ByteTrack implementation with better tracking persistence
struct TrackedObject {
    int id;
    cv::Rect bbox;
    int classId;
    float confidence;
    int framesLost;
    bool isActive;
    
    // Motion prediction
    cv::Point2f velocity;
    cv::Rect predictedBbox;
    
    // [NEW] ?????????????????????????????
    int framesStill;         // ??????????????????????
    cv::Point lastCenter;    // ???????????????????????
    
    TrackedObject() : id(-1), framesLost(0), isActive(true), confidence(0.0f), classId(-1), 
                      velocity(0, 0), predictedBbox(), framesStill(0), lastCenter(0, 0) {}
    
    TrackedObject(int _id, cv::Rect _bbox, int _classId, float _conf) 
        : id(_id), bbox(_bbox), classId(_classId), confidence(_conf), framesLost(0), isActive(true),
          velocity(0, 0), predictedBbox(_bbox), framesStill(0), 
          lastCenter(_bbox.x + _bbox.width/2, _bbox.y + _bbox.height/2) {}
    
    // Predict next position based on velocity
    void predict() {
        predictedBbox.x = bbox.x + static_cast<int>(velocity.x);
        predictedBbox.y = bbox.y + static_cast<int>(velocity.y);
        predictedBbox.width = bbox.width;
        predictedBbox.height = bbox.height;
    }
    
    // Update velocity based on movement
    void updateVelocity(const cv::Rect& newBbox) {
        float alpha = 0.7f; // Smoothing factor
        cv::Point2f newVelocity(
            newBbox.x + newBbox.width/2.0f - (bbox.x + bbox.width/2.0f),
            newBbox.y + newBbox.height/2.0f - (bbox.y + bbox.height/2.0f)
        );
        velocity.x = alpha * velocity.x + (1 - alpha) * newVelocity.x;
        velocity.y = alpha * velocity.y + (1 - alpha) * newVelocity.y;
        
        // [NEW] ?????? framesStill
        cv::Point currentCenter(newBbox.x + newBbox.width/2, newBbox.y + newBbox.height/2);
        float distance = (float)cv::norm(currentCenter - lastCenter);
        
        // ????????????????????? 5 pixels = ??????????
        if (distance < 5.0f) {
            framesStill++;
        } else {
            framesStill = 0;
        }
        
        lastCenter = currentCenter;
    }
};

class BYTETracker {
private:
    int nextId;
    std::map<int, TrackedObject> trackedObjects;
    int maxFramesLost;
    float iouThreshold;
    float lowIouThreshold; // For lost tracks
    
    // Calculate Intersection over Union (IoU)
    float calculateIoU(const cv::Rect& box1, const cv::Rect& box2) {
        int x1 = (std::max)(box1.x, box2.x);
        int y1 = (std::max)(box1.y, box2.y);
        int x2 = (std::min)(box1.x + box1.width, box2.x + box2.width);
        int y2 = (std::min)(box1.y + box1.height, box2.y + box2.height);
        
        int intersection = (std::max)(0, x2 - x1) * (std::max)(0, y2 - y1);
        int union_area = box1.area() + box2.area() - intersection;
        
        return union_area > 0 ? static_cast<float>(intersection) / union_area : 0.0f;
    }

public:
    BYTETracker(int maxLost = 90, float iouThresh = 0.25f) 
        : nextId(1), maxFramesLost(maxLost), iouThreshold(iouThresh), lowIouThreshold(0.15f) {}
    
    // Update tracker with new detections
    std::vector<TrackedObject> update(const std::vector<cv::Rect>& bboxes, 
                                       const std::vector<int>& classIds,
                                       const std::vector<float>& confidences) {
        // Predict next positions for all existing tracks
        for (auto& pair : trackedObjects) {
            pair.second.predict();
            pair.second.isActive = false;
        }
        
        // Match detections with existing tracks
        std::vector<bool> matched(bboxes.size(), false);
        
        // First pass: High confidence matches with active tracks
        for (size_t i = 0; i < bboxes.size(); i++) {
            int bestMatch = -1;
            float bestIoU = iouThreshold;
            
            for (auto& pair : trackedObjects) {
                if (pair.second.classId != classIds[i]) continue;
                if (pair.second.framesLost > 5) continue; // Skip long-lost tracks in first pass
                
                // Try matching with both current and predicted bbox
                float iou1 = calculateIoU(bboxes[i], pair.second.bbox);
                float iou2 = calculateIoU(bboxes[i], pair.second.predictedBbox);
                float iou = (std::max)(iou1, iou2);
                
                if (iou > bestIoU) {
                    bestIoU = iou;
                    bestMatch = pair.first;
                }
            }
            
            if (bestMatch != -1) {
                // Update existing track
                trackedObjects[bestMatch].updateVelocity(bboxes[i]);
                trackedObjects[bestMatch].bbox = bboxes[i];
                trackedObjects[bestMatch].confidence = confidences[i];
                trackedObjects[bestMatch].framesLost = 0;
                trackedObjects[bestMatch].isActive = true;
                matched[i] = true;
            }
        }
        
        // Second pass: Match remaining detections with lost tracks (more lenient)
        for (size_t i = 0; i < bboxes.size(); i++) {
            if (matched[i]) continue;
            
            int bestMatch = -1;
            float bestIoU = lowIouThreshold; // Lower threshold for lost tracks
            
            for (auto& pair : trackedObjects) {
                if (pair.second.classId != classIds[i]) continue;
                if (pair.second.isActive) continue; // Already matched
                
                // Use predicted position for lost tracks
                float iou = calculateIoU(bboxes[i], pair.second.predictedBbox);
                
                if (iou > bestIoU) {
                    bestIoU = iou;
                    bestMatch = pair.first;
                }
            }
            
            if (bestMatch != -1) {
                // Recover lost track
                trackedObjects[bestMatch].updateVelocity(bboxes[i]);
                trackedObjects[bestMatch].bbox = bboxes[i];
                trackedObjects[bestMatch].confidence = confidences[i];
                trackedObjects[bestMatch].framesLost = 0;
                trackedObjects[bestMatch].isActive = true;
                matched[i] = true;
            }
        }
        
        // Third pass: Create new tracks for ALL unmatched detections (no confidence filter)
        for (size_t i = 0; i < bboxes.size(); i++) {
            if (!matched[i]) {
                TrackedObject newTrack(nextId++, bboxes[i], classIds[i], confidences[i]);
                trackedObjects[newTrack.id] = newTrack;
            }
        }
        
        // Update and remove lost tracks
        auto it = trackedObjects.begin();
        while (it != trackedObjects.end()) {
            if (!it->second.isActive) {
                it->second.framesLost++;
                if (it->second.framesLost > maxFramesLost) {
                    it = trackedObjects.erase(it);
                    continue;
                }
            }
            ++it;
        }
        
        // Return all tracks (including lost ones for visualization)
        std::vector<TrackedObject> result;
        for (const auto& pair : trackedObjects) {
            if (pair.second.framesLost < 10) { // Only show tracks lost < 10 frames
                result.push_back(pair.second);
            }
        }
        return result;
    }
    
    // Reset tracker
    void reset() {
        trackedObjects.clear();
        nextId = 1;
    }
    
    // Get number of active tracks
    int getTrackCount() const {
        return static_cast<int>(trackedObjects.size());
    }
};
