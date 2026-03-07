#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <onnxruntime_cxx_api.h>

__declspec(selectany) int g_selectedGpuId = 0;

class OnnxYoloInference {
public:
    OnnxYoloInference() : env_(ORT_LOGGING_LEVEL_WARNING, "YOLOInference") {}
    
    ~OnnxYoloInference() {
        session_.reset();
    }
    
    bool loadModel(const std::string& modelPath, bool useGPU = true, int gpuDeviceId = 0) {
        try {
            Ort::SessionOptions session_options;
            // [GPU EXTREME OPTIMIZATION] Limit CPU threads to force GPU reliance
            session_options.SetIntraOpNumThreads(1);
            session_options.SetInterOpNumThreads(1);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            
            if (useGPU) {
                try {
                    // Try CUDA first
                    OrtCUDAProviderOptions cuda_options;
                    cuda_options.device_id = gpuDeviceId; // [NEW] User selectable GPU
                    
                    // [GPU EXTREME OPTIMIZATION] Force heavy use of cuDNN
                    cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
                    cuda_options.gpu_mem_limit = SIZE_MAX; // Take all available VRAM if needed
                    cuda_options.arena_extend_strategy = 0; // 0 = kNextPowerOfTwo for better memory reuse
                    cuda_options.do_copy_in_default_stream = 1;
                    cuda_options.has_user_compute_stream = 0;
                    
                    session_options.AppendExecutionProvider_CUDA(cuda_options);
                    OutputDebugStringA("[ONNX] CUDA provider added successfully with Extreme Optimization\n");
                }
                catch (...) {
                    OutputDebugStringA("[ONNX] CUDA provider failed, using CPU\n");
                }
            }
            
#ifdef _WIN32
            std::wstring wideModelPath(modelPath.begin(), modelPath.end());
            session_ = std::make_unique<Ort::Session>(env_, wideModelPath.c_str(), session_options);
#else
            session_ = std::make_unique<Ort::Session>(env_, modelPath.c_str(), session_options);
#endif
            
            // Get input/output info
            Ort::AllocatorWithDefaultOptions allocator;
            
            input_name_ = session_->GetInputNameAllocated(0, allocator).get();
            output_name_ = session_->GetOutputNameAllocated(0, allocator).get();
            
            auto input_type_info = session_->GetInputTypeInfo(0);
            auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
            input_dims_ = input_tensor_info.GetShape();
            
            auto output_type_info = session_->GetOutputTypeInfo(0);
            auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
            output_dims_ = output_tensor_info.GetShape();
            
            char msg[512];
            sprintf_s(msg, "[ONNX] Model loaded: %s\n[ONNX] Input: [%lld,%lld,%lld,%lld], Output: [%lld,%lld,%lld]\n",
                modelPath.c_str(),
                input_dims_[0], input_dims_[1], input_dims_[2], input_dims_[3],
                output_dims_[0], output_dims_[1], output_dims_[2]);
            OutputDebugStringA(msg);
            
            return true;
        }
        catch (const Ort::Exception& e) {
            char msg[512];
            sprintf_s(msg, "[ONNX ERROR] %s\n", e.what());
            OutputDebugStringA(msg);
            return false;
        }
    }
    
    bool forward(const cv::Mat& blob, std::vector<cv::Mat>& outputs) {
        if (!session_) return false;
        
        try {
            // Create input tensor
            std::vector<int64_t> input_shape = { 1, 3, 640, 640 };
            size_t input_tensor_size = 1 * 3 * 640 * 640;
            
            // blob is already in CHW format from cv::dnn::blobFromImage
            std::vector<float> input_tensor_values(input_tensor_size);
            memcpy(input_tensor_values.data(), blob.data, input_tensor_size * sizeof(float));
            
            Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info,
                input_tensor_values.data(),
                input_tensor_size,
                input_shape.data(),
                input_shape.size()
            );
            
            // Run inference
            const char* input_names[] = { input_name_.c_str() };
            const char* output_names[] = { output_name_.c_str() };
            
            auto output_tensors = session_->Run(
                Ort::RunOptions{ nullptr },
                input_names,
                &input_tensor,
                1,
                output_names,
                1
            );
            
            // Convert output to cv::Mat
            float* output_data = output_tensors[0].GetTensorMutableData<float>();
            auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
            
            // Create cv::Mat with same structure as OpenCV DNN output
            // output_shape: [1, num_detections, num_features]
            int dims[] = { (int)output_shape[0], (int)output_shape[1], (int)output_shape[2] };
            cv::Mat output_mat(3, dims, CV_32F, output_data);
            
            outputs.clear();
            outputs.push_back(output_mat.clone()); // Clone to own the data
            
            return true;
        }
        catch (const Ort::Exception& e) {
            char msg[512];
            sprintf_s(msg, "[ONNX ERROR] Forward failed: %s\n", e.what());
            OutputDebugStringA(msg);
            return false;
        }
    }
    
private:
    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    std::string input_name_;
    std::string output_name_;
    std::vector<int64_t> input_dims_;
    std::vector<int64_t> output_dims_;
};
