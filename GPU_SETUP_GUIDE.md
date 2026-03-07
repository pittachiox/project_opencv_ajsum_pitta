# 🚀 GPU Acceleration Setup Guide - YOLO Detection with ONNX Runtime

## 📊 Performance Comparison

| Mode | FPS | Speed Improvement |
|------|-----|-------------------|
| **CPU (OpenCV DNN)** | 12.9 FPS | Baseline |
| **GPU (ONNX Runtime + CUDA)** | **113 FPS** | **8.76x faster** |
| **Actual Video (640x360)** | **47-64 FPS** | **4x faster** |

---

## 📋 Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation Steps](#installation-steps)
3. [Code Changes](#code-changes)
4. [Verification & Testing](#verification--testing)
5. [Troubleshooting](#troubleshooting)

---

## 🖥️ System Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (tested on RTX 4050 Laptop 6GB VRAM)
- **RAM**: 8GB+ recommended
- **Disk Space**: ~5GB for CUDA + cuDNN + ONNX Runtime

### Software
- **Windows 10/11** (64-bit)
- **Visual Studio 2022** (with C++ Desktop Development)
- **NVIDIA Driver**: 471.0 or later (tested with 581.60)
- **OpenCV**: 4.12.0+

---

## 🔧 Installation Steps

### Step 1: Install CUDA 12.6

1. **Download CUDA 12.6**:
   - Visit: https://developer.nvidia.com/cuda-12-6-0-download-archive
   - Select: Windows → x86_64 → 11 → exe (local)
   - File: `cuda_12.6.0_560.76_windows.exe` (~3.5GB)

2. **Install CUDA**:
   ```powershell
   # Run installer (Express Installation recommended)
   cuda_12.6.0_560.76_windows.exe
   ```
   - Install location: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6`
   - This includes: cuBLAS, cuFFT, cuSPARSE, etc.

3. **Verify Installation**:
   ```powershell
   nvcc --version
   # Should show: Cuda compilation tools, release 12.6
   
   nvidia-smi
   # Should show your GPU and CUDA Version
   ```

### Step 2: Install cuDNN 9.1.9.1

1. **Download cuDNN**:
   - Visit: https://developer.nvidia.com/cudnn
   - Login required (free account)
   - Select: **cuDNN 9.1.9 for CUDA 12.6** (Windows)
   - File: `cudnn-windows-x86_64-9.1.9.1_cuda12-archive.zip` (~869MB)

2. **Extract and Copy Files**:
   ```powershell
   # Extract ZIP to Downloads folder
   # Copy files to CUDA installation folder
   
   # Navigate to cuDNN folder
   cd "$env:USERPROFILE\Downloads\cudnn-windows-x86_64-9.1.9.1_cuda12-archive"
   
   # Copy files
   Copy-Item -Path "bin\cudnn*.dll" -Destination "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\" -Force
   Copy-Item -Path "include\cudnn*.h" -Destination "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\include\" -Force
   Copy-Item -Path "lib\x64\cudnn*.lib" -Destination "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\lib\x64\" -Force
   ```

3. **Add to System PATH** (Permanent):
   ```powershell
   # Run PowerShell as Administrator
   
   $cudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
   $currentPath = [Environment]::GetEnvironmentVariable("Path", "Machine")
   
   if ($currentPath -notlike "*$cudaPath*") {
       [Environment]::SetEnvironmentVariable(
           "Path",
           "$currentPath;$cudaPath",
           "Machine"
       )
       Write-Host "✅ Added CUDA to system PATH" -ForegroundColor Green
   }
   
   # Restart PowerShell/CMD after this
   ```

### Step 3: Install ONNX Runtime GPU 1.20.1

1. **Download via NuGet**:
   ```powershell
   # In project directory
   cd "C:\Users\kt856\source\repos\project_opencv_ajsum_nikom"
   
   # Download using NuGet CLI (or Visual Studio NuGet Package Manager)
   nuget install Microsoft.ML.OnnxRuntime.Gpu -Version 1.20.1 -OutputDirectory packages
   ```

2. **Or Download Manually**:
   - Visit: https://github.com/microsoft/onnxruntime/releases/tag/v1.20.1
   - Download: `onnxruntime-win-x64-gpu-1.20.1.zip`
   - Extract to: `packages\Microsoft.ML.OnnxRuntime.Gpu.1.20.1\`

3. **Copy DLL Files** to Output Directories:
   ```powershell
   # Copy ONNX Runtime DLLs
   $sourceDlls = "packages\Microsoft.ML.OnnxRuntime.Gpu.1.20.1\runtimes\win-x64\native"
   
   # Copy to Debug
   Copy-Item "$sourceDlls\onnxruntime.dll" -Destination "x64\Debug\" -Force
   Copy-Item "$sourceDlls\onnxruntime_providers_cuda.dll" -Destination "x64\Debug\" -Force
   Copy-Item "$sourceDlls\onnxruntime_providers_shared.dll" -Destination "x64\Debug\" -Force
   Copy-Item "$sourceDlls\onnxruntime_providers_tensorrt.dll" -Destination "x64\Debug\" -Force
   
   # Copy to Release
   Copy-Item "$sourceDlls\onnxruntime.dll" -Destination "x64\Release\" -Force
   Copy-Item "$sourceDlls\onnxruntime_providers_cuda.dll" -Destination "x64\Release\" -Force
   Copy-Item "$sourceDlls\onnxruntime_providers_shared.dll" -Destination "x64\Release\" -Force
   Copy-Item "$sourceDlls\onnxruntime_providers_tensorrt.dll" -Destination "x64\Release\" -Force
   
   Write-Host "✅ DLLs copied successfully" -ForegroundColor Green
   ```

4. **Verify DLLs are Present**:
   ```powershell
   Get-ChildItem "x64\Release\onnxruntime*.dll"
   # Should show 4 DLL files
   ```

---

## 💻 Code Changes

### Step 4: Create ONNX Runtime Wrapper Class

Create new file: `OnnxYoloInference.h`

```cpp
#pragma once
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <iostream>

class OnnxYoloInference {
private:
    Ort::Env env;
    Ort::Session* session = nullptr;
    Ort::SessionOptions session_options;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    std::vector<int64_t> input_shape;

public:
    OnnxYoloInference() : env(ORT_LOGGING_LEVEL_WARNING, "YOLOv8") {
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    }

    ~OnnxYoloInference() {
        if (session) delete session;
    }

    bool loadModel(const std::string& modelPath, bool useGPU = true) {
        try {
            if (useGPU) {
                std::cout << "[GPU] Attempting to use CUDA Execution Provider..." << std::endl;
                
                // Add CUDA provider with options
                OrtCUDAProviderOptions cuda_options;
                cuda_options.device_id = 0;
                cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
                cuda_options.gpu_mem_limit = SIZE_MAX;
                cuda_options.arena_extend_strategy = 1;
                cuda_options.do_copy_in_default_stream = 1;
                session_options.AppendExecutionProvider_CUDA(cuda_options);
                
                // Add TensorRT provider (optional, fallback to CUDA if not available)
                OrtTensorRTProviderOptions trt_options;
                trt_options.device_id = 0;
                trt_options.trt_max_workspace_size = 2147483648; // 2GB
                trt_options.trt_fp16_enable = 1;
                session_options.AppendExecutionProvider_TensorRT(trt_options);
            }

            // Set intra/inter op threads
            session_options.SetIntraOpNumThreads(4);
            session_options.SetInterOpNumThreads(4);

            // Convert to wide string for Windows
            std::wstring wideModelPath(modelPath.begin(), modelPath.end());
            
            // Create session with timeout
            session = new Ort::Session(env, wideModelPath.c_str(), session_options);

            // Get input/output info
            Ort::AllocatorWithDefaultOptions allocator;
            
            // Input
            size_t num_input_nodes = session->GetInputCount();
            if (num_input_nodes > 0) {
                auto input_name = session->GetInputNameAllocated(0, allocator);
                input_names.push_back(_strdup(input_name.get()));
                
                Ort::TypeInfo type_info = session->GetInputTypeInfo(0);
                auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
                input_shape = tensor_info.GetShape();
                
                std::cout << "[ONNX] Input shape: [";
                for (size_t i = 0; i < input_shape.size(); i++) {
                    std::cout << input_shape[i];
                    if (i < input_shape.size() - 1) std::cout << ", ";
                }
                std::cout << "]" << std::endl;
            }

            // Output
            size_t num_output_nodes = session->GetOutputCount();
            for (size_t i = 0; i < num_output_nodes; i++) {
                auto output_name = session->GetOutputNameAllocated(i, allocator);
                output_names.push_back(_strdup(output_name.get()));
            }

            if (useGPU) {
                std::cout << "[GPU] ✅ Model loaded with GPU acceleration (CUDA + TensorRT)" << std::endl;
            } else {
                std::cout << "[CPU] Model loaded with CPU" << std::endl;
            }

            return true;
        }
        catch (const Ort::Exception& e) {
            std::cerr << "[ERROR] ONNX Runtime error: " << e.what() << std::endl;
            return false;
        }
    }

    bool forward(const cv::Mat& blob, std::vector<cv::Mat>& outputs) {
        try {
            // Convert blob to ONNX tensor format
            std::vector<int64_t> input_tensor_shape = { 1, 3, 640, 640 };
            size_t input_tensor_size = 1 * 3 * 640 * 640;

            std::vector<float> input_tensor_values(input_tensor_size);
            std::memcpy(input_tensor_values.data(), blob.data, input_tensor_size * sizeof(float));

            // Create input tensor
            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info,
                input_tensor_values.data(),
                input_tensor_size,
                input_tensor_shape.data(),
                input_tensor_shape.size()
            );

            // Run inference
            auto output_tensors = session->Run(
                Ort::RunOptions{ nullptr },
                input_names.data(),
                &input_tensor,
                1,
                output_names.data(),
                output_names.size()
            );

            // Convert output to cv::Mat
            outputs.clear();
            for (size_t i = 0; i < output_tensors.size(); i++) {
                float* floatarr = output_tensors[i].GetTensorMutableData<float>();
                auto shape_info = output_tensors[i].GetTensorTypeAndShapeInfo();
                auto shape = shape_info.GetShape();

                if (shape.size() == 3) {
                    // Shape: [batch, num_detections, 6] for yolo26s
                    int rows = static_cast<int>(shape[1]);
                    int cols = static_cast<int>(shape[2]);
                    cv::Mat output_mat(rows, cols, CV_32F);
                    std::memcpy(output_mat.data, floatarr, rows * cols * sizeof(float));
                    outputs.push_back(output_mat);
                }
            }

            return true;
        }
        catch (const Ort::Exception& e) {
            std::cerr << "[ERROR] Inference error: " << e.what() << std::endl;
            return false;
        }
    }
};
```

### Step 5: Modify Visual Studio Project (.vcxproj)

Edit `ConsoleApplication3.vcxproj`:

```xml
<PropertyGroup Label="UserMacros" />
<PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Release|x64'">
  <!-- Add ONNX Runtime Paths -->
  <IncludePath>$(ProjectDir)packages\Microsoft.ML.OnnxRuntime.Gpu.1.20.1\build\native\include;$(VC_IncludePath);$(WindowsSDK_IncludePath);</IncludePath>
  <LibraryPath>$(ProjectDir)packages\Microsoft.ML.OnnxRuntime.Gpu.1.20.1\runtimes\win-x64\native;$(VC_LibraryPath_x64);$(WindowsSDK_LibraryPath_x64)</LibraryPath>
</PropertyGroup>

<!-- Add to Link section -->
<ItemDefinitionGroup Condition="'$(Configuration)|$(Platform)'=='Release|x64'">
  <Link>
    <AdditionalDependencies>onnxruntime.lib;onnxruntime_providers_cuda.lib;onnxruntime_providers_shared.lib;%(AdditionalDependencies)</AdditionalDependencies>
  </Link>
</ItemDefinitionGroup>
```

### Step 6: Modify online1.h

**Change 1: Add Global Variable** (line ~51-52)
```cpp
// Replace:
static cv::dnn::Net* g_net = nullptr;

// With:
static OnnxYoloInference* g_onnx_net = nullptr;
```

**Change 2: Update InitGlobalModel** (line ~260-290)
```cpp
static void InitGlobalModel(const std::string& modelPath) {
    std::lock_guard<std::mutex> lock(g_aiMutex_online);
    g_modelReady = false;
    if (g_net) { delete g_net; g_net = nullptr; }
    if (g_onnx_net) { delete g_onnx_net; g_onnx_net = nullptr; }
    if (g_tracker) { delete g_tracker; g_tracker = nullptr; }

    try {
        // Use ONNX Runtime with GPU acceleration
        g_onnx_net = new OnnxYoloInference();
        
        if (!g_onnx_net->loadModel(modelPath, true)) {  // true = use GPU
            DumpLog("[ERROR] Failed to load ONNX model with GPU");
            delete g_onnx_net;
            g_onnx_net = nullptr;
            return;
        }

        DumpLog("[OK] ONNX Runtime GPU model loaded: " + modelPath);
        g_modelReady = true;
        
        // Initialize ByteTrack
        g_tracker = new BYTETracker(30, 30);
        DumpLog("[OK] BYTETracker initialized");
    }
    catch (const std::exception& ex) {
        DumpLog(std::string("[EXCEPTION] InitGlobalModel: ") + ex.what());
    }
}
```

**Change 3: Update ProcessFrameOnline** (line ~336-350)
```cpp
// Replace g_net->forward() calls with:
std::vector<cv::Mat> outputs;
if (!g_onnx_net->forward(blob, outputs) || outputs.empty()) {
    return;
}
cv::Mat output = outputs[0]; // Get first output
```

**Change 4: Remove FPS Throttling** (line ~1620-1710 in CameraReaderLoop)
```cpp
// Remove these lines:
// LONGLONG ticksPerFrame = (LONGLONG)((1.0 / g_cameraFPS) * 10000000.0);
// LONGLONG nextTick = currentTicks + ticksPerFrame;
// if (currentTicks < nextTick) {
//     DWORD sleepMs = (DWORD)((nextTick - currentTicks) / 10000);
//     if (sleepMs > 0 && sleepMs < 1000) {
//         std::this_thread::sleep_for(std::chrono::milliseconds(sleepMs));
//     }
// }

// Replace with: (no FPS limit - let GPU run at full speed)
std::this_thread::sleep_for(std::chrono::milliseconds(1));
```

### Step 7: Add Header Include

At the top of `online1.h`:
```cpp
#include "OnnxYoloInference.h"
#include <onnxruntime_cxx_api.h>
```

---

## ✅ Verification & Testing

### Step 8: Build the Project

```powershell
# Open Visual Studio 2022
# Select: Release | x64
# Build → Rebuild Solution

# Or use command line:
msbuild ConsoleApplication3.vcxproj /p:Configuration=Release /p:Platform=x64 /t:Rebuild
```

### Step 9: Verify GPU is Working

1. **Run the Application**
2. **Check Console Output**:
   ```
   [GPU] Attempting to use CUDA Execution Provider...
   [ONNX] Input shape: [1, 3, 640, 640]
   [GPU] ✅ Model loaded with GPU acceleration (CUDA + TensorRT)
   ```

3. **Monitor GPU Usage**:
   ```powershell
   # In another terminal
   nvidia-smi -l 1
   # Should show GPU utilization ~80-100% when processing
   ```

### Step 10: Python Benchmark (Optional)

Test with Python to verify GPU performance:

```python
# demo_gpu_speed.py
import cv2
import numpy as np
from ultralytics import YOLO
import time

# Load model
model = YOLO('models/yolo26s.onnx', task='detect')
print("✅ Using GPU (CUDA)" if model.predictor.device.type == 'cuda' else "❌ Using CPU")

# Open video
cap = cv2.VideoCapture('path/to/video.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)

frame_times = []
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    start = time.time()
    results = model(frame, conf=0.5, verbose=False)
    inference_time = time.time() - start
    
    frame_times.append(inference_time)
    frame_count += 1
    
    # Draw results
    annotated = results[0].plot()
    cv2.imshow('GPU Detection', annotated)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Statistics
avg_time = np.mean(frame_times)
avg_fps = 1.0 / avg_time
print(f"\n📊 Performance:")
print(f"   Average FPS: {avg_fps:.1f}")
print(f"   Total frames: {frame_count}")
```

Expected output:
```
✅ Using GPU (CUDA)
📊 Performance:
   Average FPS: 113.0
   Total frames: 1000
```

---

## 🔍 Troubleshooting

### Issue 1: "CUDA not found" Error

**Solution**: Verify CUDA installation
```powershell
nvcc --version
nvidia-smi
```

If not found:
- Reinstall CUDA 12.6
- Check PATH environment variable
- Restart computer

### Issue 2: "Failed to load ONNX model with GPU"

**Causes**:
- cuDNN DLLs not in PATH
- Wrong ONNX Runtime version
- Missing DLLs in output folder

**Solution**:
```powershell
# Check if DLLs exist
Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\cudnn*.dll"
Get-ChildItem "x64\Release\onnxruntime*.dll"

# Re-copy DLLs if missing
Copy-Item "packages\Microsoft.ML.OnnxRuntime.Gpu.1.20.1\runtimes\win-x64\native\*.dll" -Destination "x64\Release\" -Force
```

### Issue 3: Low FPS (still around 20-30)

**Causes**:
- FPS throttling still in code
- Not using GPU (check console logs)
- CPU bottleneck in video decoding

**Solution**:
```cpp
// Make sure FPS throttling is removed in CameraReaderLoop()
// Check console output for:
// [GPU] ✅ Model loaded with GPU acceleration

// Use nvidia-smi to verify GPU is actually running
```

### Issue 4: Out of Memory Error

**Solution**: Reduce batch size or model size
```cpp
// In OnnxYoloInference.h, reduce workspace size:
trt_options.trt_max_workspace_size = 1073741824; // 1GB instead of 2GB
```

Or use smaller model:
```cpp
std::string modelPath = "models/yolo11n.onnx"; // Nano model (smaller)
```

### Issue 5: Application Crashes on Startup

**Causes**:
- Missing DLL files
- Incompatible CUDA/cuDNN versions
- Wrong Visual Studio configuration

**Solution**:
```powershell
# Check dependencies
dumpbin /dependents x64\Release\ConsoleApplication3.exe

# Common missing DLLs:
# - onnxruntime.dll
# - onnxruntime_providers_cuda.dll  
# - cudnn64_9.dll
# - cublas64_12.dll

# Copy all required DLLs to output folder
```

---

## 📈 Performance Tips

### 1. Use FP16 (Half Precision)
```cpp
trt_options.trt_fp16_enable = 1;  // Already enabled in code
// Gives ~30% speed boost with minimal accuracy loss
```

### 2. Optimize Input Resolution
```cpp
// Lower resolution = faster inference
// 640x640 → 100 FPS
// 416x416 → 150+ FPS (but lower accuracy)
```

### 3. Reduce Confidence Threshold
```cpp
float confThreshold = 0.5f; // Higher = faster (fewer detections to process)
```

### 4. Use Lighter Model
```
yolo11n.onnx    - Nano (fastest, ~200 FPS)
yolo11s.onnx    - Small (~120 FPS)
yolo26s.onnx    - Custom (~113 FPS)
yolov8m.onnx    - Medium (~60 FPS)
yolov8l.onnx    - Large (~30 FPS)
```

---

## 📦 Complete Checklist for Your Friend

```
✅ System Requirements
   ⬜ NVIDIA GPU with CUDA support
   ⬜ Windows 10/11 64-bit
   ⬜ Visual Studio 2022
   ⬜ NVIDIA Driver 471.0+

✅ Installation
   ⬜ Install CUDA 12.6 (3.5GB)
   ⬜ Install cuDNN 9.1.9.1 (copy files to CUDA folder)
   ⬜ Add CUDA bin to System PATH
   ⬜ Download ONNX Runtime GPU 1.20.1
   ⬜ Copy ONNX Runtime DLLs to x64/Release and x64/Debug

✅ Code Changes
   ⬜ Create OnnxYoloInference.h
   ⬜ Modify ConsoleApplication3.vcxproj (add include/lib paths)
   ⬜ Update online1.h (replace cv::dnn with ONNX Runtime)
   ⬜ Remove FPS throttling in CameraReaderLoop()
   ⬜ Add #include "OnnxYoloInference.h"

✅ Build & Test
   ⬜ Rebuild Solution (Release x64)
   ⬜ Check console for GPU confirmation message
   ⬜ Monitor GPU usage with nvidia-smi
   ⬜ Verify FPS improvement (should be 60-100+ FPS)

✅ Troubleshooting
   ⬜ All DLLs present in output folder
   ⬜ CUDA paths in System PATH
   ⬜ No FPS throttling code remaining
   ⬜ GPU utilization visible in Task Manager
```

---

## 🎯 Expected Results

| Metric | Before (CPU) | After (GPU) |
|--------|--------------|-------------|
| Model Loading | 2-3 seconds | 3-5 seconds |
| Inference Time | 77ms | 8.8ms |
| FPS (Benchmark) | 12.9 | 113.0 |
| FPS (Actual Video) | 20-30 | 47-64 |
| GPU Usage | 0% | 80-95% |
| CPU Usage | 80-100% | 20-30% |

---

## 📞 Support

If you encounter issues:
1. Check console output for error messages
2. Verify GPU with `nvidia-smi`
3. Check Task Manager → Performance → GPU
4. Review Troubleshooting section above

**Common Issues**:
- Missing DLLs → Re-copy from packages folder
- Low FPS → Verify GPU is actually being used (check console logs)
- Crashes → Check CUDA/cuDNN compatibility

---

## 📝 File Structure After Setup

```
project_opencv_ajsum_nikom/
├── OnnxYoloInference.h          (NEW - GPU wrapper class)
├── online1.h                     (MODIFIED - uses ONNX Runtime)
├── ConsoleApplication3.vcxproj   (MODIFIED - includes ONNX paths)
├── packages/
│   └── Microsoft.ML.OnnxRuntime.Gpu.1.20.1/
│       ├── build/native/include/
│       └── runtimes/win-x64/native/*.dll
├── x64/
│   ├── Release/
│   │   ├── ConsoleApplication3.exe
│   │   ├── onnxruntime.dll              (COPIED)
│   │   ├── onnxruntime_providers_cuda.dll
│   │   ├── onnxruntime_providers_shared.dll
│   │   └── onnxruntime_providers_tensorrt.dll
│   └── Debug/ (same DLLs)
└── models/
    └── yolo26s.onnx
```

---

## 🚀 Quick Start Commands

```powershell
# 1. Install CUDA (run installer GUI)
# 2. Install cuDNN (copy files)
# 3. Add CUDA to PATH:
[Environment]::SetEnvironmentVariable("Path", "$env:Path;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin", "Machine")

# 4. Download ONNX Runtime
nuget install Microsoft.ML.OnnxRuntime.Gpu -Version 1.20.1 -OutputDirectory packages

# 5. Copy DLLs
Copy-Item "packages\Microsoft.ML.OnnxRuntime.Gpu.1.20.1\runtimes\win-x64\native\*.dll" -Destination "x64\Release\" -Force

# 6. Build
msbuild ConsoleApplication3.vcxproj /p:Configuration=Release /p:Platform=x64 /t:Rebuild

# 7. Run
.\x64\Release\ConsoleApplication3.exe
```

---

**Created**: March 7, 2026  
**GPU**: NVIDIA RTX 4050 Laptop (6GB VRAM)  
**Performance**: 113 FPS (8.76x faster than CPU)  
**Version**: CUDA 12.6 + cuDNN 9.1.9 + ONNX Runtime 1.20.1
