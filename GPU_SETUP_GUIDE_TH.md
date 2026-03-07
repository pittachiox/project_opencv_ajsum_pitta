# 🚀 คู่มือการติดตั้ง GPU สำหรับ YOLO Detection (ฉบับภาษาไทย)

## 📊 เปรียบเทียบความเร็ว

| โหมด | FPS | เร็วขึ้นกว่าเดิม |
|------|-----|-----------------|
| **CPU (OpenCV DNN)** | 12.9 FPS | จุดเริ่มต้น |
| **GPU (ONNX Runtime + CUDA)** | **113 FPS** | **เร็วขึ้น 8.76 เท่า** |
| **วิดีโอจริง (640x360)** | **47-64 FPS** | **เร็วขึ้น 4 เท่า** |

---

## 📋 สารบัญ
1. [ความต้องการของระบบ](#ความต้องการของระบบ)
2. [ขั้นตอนการติดตั้ง](#ขั้นตอนการติดตั้ง)
3. [การแก้ไขโค้ด](#การแก้ไขโค้ด)
4. [การทดสอบ](#การทดสอบ)
5. [แก้ไขปัญหา](#แก้ไขปัญหา)

---

## 🖥️ ความต้องการของระบบ

### ฮาร์ดแวร์
- **การ์ดจอ**: NVIDIA GPU ที่รองรับ CUDA (ทดสอบกับ RTX 4050 Laptop 6GB VRAM)
- **แรม**: แนะนำ 8GB ขึ้นไป
- **พื้นที่ดิสก์**: ประมาณ 5GB สำหรับ CUDA + cuDNN + ONNX Runtime

### ซอฟต์แวร์
- **Windows 10/11** (64-bit)
- **Visual Studio 2022** (พร้อม C++ Desktop Development)
- **NVIDIA Driver**: 471.0 ขึ้นไป (ทดสอบกับ 581.60)
- **OpenCV**: 4.12.0+

---

## 🔧 ขั้นตอนการติดตั้ง

### ขั้นตอนที่ 1: ติดตั้ง CUDA 12.6

1. **ดาวน์โหลด CUDA 12.6**:
   - เว็บไซต์: https://developer.nvidia.com/cuda-12-6-0-download-archive
   - เลือก: Windows → x86_64 → 11 → exe (local)
   - ไฟล์: `cuda_12.6.0_560.76_windows.exe` (~3.5GB)

2. **ติดตั้ง CUDA**:
   - เปิดไฟล์ติดตั้ง แนะนำให้เลือก "Express Installation"
   - จะติดตั้งไปที่: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6`
   - รวมไลบรารี: cuBLAS, cuFFT, cuSPARSE, etc.

3. **ตรวจสอบการติดตั้ง**:
   ```powershell
   nvcc --version
   # ควรแสดง: Cuda compilation tools, release 12.6
   
   nvidia-smi
   # ควรแสดงข้อมูลการ์ดจอและเวอร์ชัน CUDA
   ```

### ขั้นตอนที่ 2: ติดตั้ง cuDNN 9.1.9.1

1. **ดาวน์โหลด cuDNN**:
   - เว็บไซต์: https://developer.nvidia.com/cudnn
   - ต้อง login (สมัครฟรี)
   - เลือก: **cuDNN 9.1.9 for CUDA 12.6** (Windows)
   - ไฟล์: `cudnn-windows-x86_64-9.1.9.1_cuda12-archive.zip` (~869MB)

2. **แตกไฟล์และคัดลอก**:
   ```powershell
   # แตกไฟล์ ZIP ที่ Downloads
   # คัดลอกไฟล์ไปยังโฟลเดอร์ CUDA
   
   # เข้าไปในโฟลเดอร์ cuDNN
   cd "$env:USERPROFILE\Downloads\cudnn-windows-x86_64-9.1.9.1_cuda12-archive"
   
   # คัดลอกไฟล์
   Copy-Item -Path "bin\cudnn*.dll" -Destination "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\" -Force
   Copy-Item -Path "include\cudnn*.h" -Destination "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\include\" -Force
   Copy-Item -Path "lib\x64\cudnn*.lib" -Destination "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\lib\x64\" -Force
   ```

3. **เพิ่มเข้า System PATH (ถาวร)**:
   ```powershell
   # เปิด PowerShell ในโหมด Administrator
   
   $cudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
   $currentPath = [Environment]::GetEnvironmentVariable("Path", "Machine")
   
   if ($currentPath -notlike "*$cudaPath*") {
       [Environment]::SetEnvironmentVariable(
           "Path",
           "$currentPath;$cudaPath",
           "Machine"
       )
       Write-Host "✅ เพิ่ม CUDA เข้า PATH แล้ว" -ForegroundColor Green
   }
   
   # รีสตาร์ท PowerShell/CMD หลังจากนี้
   ```

### ขั้นตอนที่ 3: ติดตั้ง ONNX Runtime GPU 1.20.1

1. **ดาวน์โหลดผ่าน NuGet**:
   ```powershell
   # ในโฟลเดอร์โปรเจ็กต์
   cd "C:\Users\kt856\source\repos\project_opencv_ajsum_nikom"
   
   # ดาวน์โหลดด้วย NuGet CLI (หรือใช้ Visual Studio NuGet Package Manager)
   nuget install Microsoft.ML.OnnxRuntime.Gpu -Version 1.20.1 -OutputDirectory packages
   ```

2. **หรือดาวน์โหลดด้วยตนเอง**:
   - เว็บไซต์: https://github.com/microsoft/onnxruntime/releases/tag/v1.20.1
   - ดาวน์โหลด: `onnxruntime-win-x64-gpu-1.20.1.zip`
   - แตกไฟล์ไปที่: `packages\Microsoft.ML.OnnxRuntime.Gpu.1.20.1\`

3. **คัดลอกไฟล์ DLL** ไปยังโฟลเดอร์ Output:
   ```powershell
   # คัดลอก ONNX Runtime DLLs
   $sourceDlls = "packages\Microsoft.ML.OnnxRuntime.Gpu.1.20.1\runtimes\win-x64\native"
   
   # คัดลอกไปที่ Debug
   Copy-Item "$sourceDlls\onnxruntime.dll" -Destination "x64\Debug\" -Force
   Copy-Item "$sourceDlls\onnxruntime_providers_cuda.dll" -Destination "x64\Debug\" -Force
   Copy-Item "$sourceDlls\onnxruntime_providers_shared.dll" -Destination "x64\Debug\" -Force
   Copy-Item "$sourceDlls\onnxruntime_providers_tensorrt.dll" -Destination "x64\Debug\" -Force
   
   # คัดลอกไปที่ Release
   Copy-Item "$sourceDlls\onnxruntime.dll" -Destination "x64\Release\" -Force
   Copy-Item "$sourceDlls\onnxruntime_providers_cuda.dll" -Destination "x64\Release\" -Force
   Copy-Item "$sourceDlls\onnxruntime_providers_shared.dll" -Destination "x64\Release\" -Force
   Copy-Item "$sourceDlls\onnxruntime_providers_tensorrt.dll" -Destination "x64\Release\" -Force
   
   Write-Host "✅ คัดลอก DLLs เรียบร้อย" -ForegroundColor Green
   ```

4. **ตรวจสอบว่า DLLs อยู่ครบ**:
   ```powershell
   Get-ChildItem "x64\Release\onnxruntime*.dll"
   # ควรแสดง 4 ไฟล์ DLL
   ```

---

## 💻 การแก้ไขโค้ด

### ขั้นตอนที่ 4: สร้างไฟล์ Wrapper Class สำหรับ ONNX Runtime

สร้างไฟล์ใหม่: `OnnxYoloInference.h` (ดูโค้ดเต็มใน GPU_SETUP_GUIDE.md ฉบับภาษาอังกฤษ)

**สิ่งที่ไฟล์นี้ทำ**:
- จัดการ ONNX Runtime Session
- เปิดใช้งาน CUDA และ TensorRT providers
- แปลง OpenCV Mat เป็น ONNX tensor
- ทำ inference บน GPU
- แปลง output กลับเป็น OpenCV Mat

### ขั้นตอนที่ 5: แก้ไข Visual Studio Project (.vcxproj)

แก้ไข `ConsoleApplication3.vcxproj` เพื่อเพิ่ม ONNX Runtime paths:

**เพิ่มใน PropertyGroup**:
```xml
<IncludePath>$(ProjectDir)packages\Microsoft.ML.OnnxRuntime.Gpu.1.20.1\build\native\include;$(VC_IncludePath);$(WindowsSDK_IncludePath);</IncludePath>
<LibraryPath>$(ProjectDir)packages\Microsoft.ML.OnnxRuntime.Gpu.1.20.1\runtimes\win-x64\native;$(VC_LibraryPath_x64);$(WindowsSDK_LibraryPath_x64)</LibraryPath>
```

**เพิ่มใน Link section**:
```xml
<AdditionalDependencies>onnxruntime.lib;onnxruntime_providers_cuda.lib;onnxruntime_providers_shared.lib;%(AdditionalDependencies)</AdditionalDependencies>
```

### ขั้นตอนที่ 6: แก้ไข online1.h

**การเปลี่ยนแปลงที่ 1: เพิ่ม Header Include** (ตอนต้นไฟล์)
```cpp
#include "OnnxYoloInference.h"
#include <onnxruntime_cxx_api.h>
```

**การเปลี่ยนแปลงที่ 2: แก้ Global Variable** (บรรทัดประมาณ 51-52)
```cpp
// เปลี่ยนจาก:
static cv::dnn::Net* g_net = nullptr;

// เป็น:
static OnnxYoloInference* g_onnx_net = nullptr;
```

**การเปลี่ยนแปลงที่ 3: แก้ InitGlobalModel** (บรรทัด ~260-290)
```cpp
static void InitGlobalModel(const std::string& modelPath) {
    std::lock_guard<std::mutex> lock(g_aiMutex_online);
    g_modelReady = false;
    
    // ลบ network เก่า
    if (g_net) { delete g_net; g_net = nullptr; }
    if (g_onnx_net) { delete g_onnx_net; g_onnx_net = nullptr; }
    if (g_tracker) { delete g_tracker; g_tracker = nullptr; }

    try {
        // ใช้ ONNX Runtime พร้อม GPU acceleration
        g_onnx_net = new OnnxYoloInference();
        
        if (!g_onnx_net->loadModel(modelPath, true)) {  // true = ใช้ GPU
            DumpLog("[ERROR] โหลด ONNX model ด้วย GPU ไม่สำเร็จ");
            delete g_onnx_net;
            g_onnx_net = nullptr;
            return;
        }

        DumpLog("[OK] โหลด ONNX Runtime GPU model สำเร็จ: " + modelPath);
        g_modelReady = true;
        
        // Initialize ByteTrack
        g_tracker = new BYTETracker(30, 30);
        DumpLog("[OK] BYTETracker เริ่มทำงานแล้ว");
    }
    catch (const std::exception& ex) {
        DumpLog(std::string("[EXCEPTION] InitGlobalModel: ") + ex.what());
    }
}
```

**การเปลี่ยนแปลงที่ 4: แก้ ProcessFrameOnline** (บรรทัด ~336-350)
```cpp
// เปลี่ยนจาก g_net->forward() เป็น:
std::vector<cv::Mat> outputs;
if (!g_onnx_net->forward(blob, outputs) || outputs.empty()) {
    return;
}
cv::Mat output = outputs[0]; // รับ output แรก
```

**การเปลี่ยนแปลงที่ 5: ลบการจำกัด FPS** (บรรทัด ~1620-1710 ใน CameraReaderLoop)
```cpp
// ลบบรรทัดเหล่านี้:
// LONGLONG ticksPerFrame = (LONGLONG)((1.0 / g_cameraFPS) * 10000000.0);
// LONGLONG nextTick = currentTicks + ticksPerFrame;
// if (currentTicks < nextTick) {
//     DWORD sleepMs = (DWORD)((nextTick - currentTicks) / 10000);
//     if (sleepMs > 0 && sleepMs < 1000) {
//         std::this_thread::sleep_for(std::chrono::milliseconds(sleepMs));
//     }
// }

// แทนที่ด้วย: (ไม่จำกัด FPS - ให้ GPU ทำงานเต็มความเร็ว)
std::this_thread::sleep_for(std::chrono::milliseconds(1));
```

---

## ✅ การทดสอบ

### ขั้นตอนที่ 7: Build โปรเจ็กต์

```powershell
# วิธีที่ 1: ใช้ Visual Studio 2022
# เลือก: Release | x64
# Build → Rebuild Solution

# วิธีที่ 2: ใช้ command line
msbuild ConsoleApplication3.vcxproj /p:Configuration=Release /p:Platform=x64 /t:Rebuild
```

### ขั้นตอนที่ 8: ตรวจสอบว่า GPU ทำงาน

1. **รันโปรแกรม**
2. **ดูที่ Console Output** ควรเห็น:
   ```
   [GPU] Attempting to use CUDA Execution Provider...
   [ONNX] Input shape: [1, 3, 640, 640]
   [GPU] ✅ Model loaded with GPU acceleration (CUDA + TensorRT)
   ```

3. **ตรวจสอบการใช้งาน GPU**:
   ```powershell
   # เปิด terminal อีกหน้าต่าง
   nvidia-smi -l 1
   # ควรเห็น GPU utilization ประมาณ 80-100% ตอนประมวลผล
   ```

4. **ดูใน Task Manager**:
   - กด `Ctrl + Shift + Esc`
   - ไปที่แท็บ "Performance"
   - เลือก "GPU"
   - ควรเห็น "CUDA" เพิ่มขึ้นเวลาโปรแกรมทำงาน

### ขั้นตอนที่ 9: ทดสอบด้วย Python (ถ้าต้องการ)

```python
# demo_gpu_speed.py
import cv2
import numpy as np
from ultralytics import YOLO
import time

# โหลด model
model = YOLO('models/yolo26s.onnx', task='detect')
device_type = model.predictor.device.type
print(f"{'✅ Using GPU (CUDA)' if device_type == 'cuda' else '❌ Using CPU'}")

# เปิดวิดีโอ
cap = cv2.VideoCapture('path/to/video.mp4')

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
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

# แสดงสถิติ
avg_time = np.mean(frame_times)
avg_fps = 1.0 / avg_time
print(f"\n📊 ผลการทดสอบ:")
print(f"   FPS เฉลี่ย: {avg_fps:.1f}")
print(f"   จำนวนเฟรมทั้งหมด: {frame_count}")
```

ผลลัพธ์ที่คาดหวัง:
```
✅ Using GPU (CUDA)
📊 ผลการทดสอบ:
   FPS เฉลี่ย: 113.0
   จำนวนเฟรมทั้งหมด: 1000
```

---

## 🔍 แก้ไขปัญหา

### ปัญหาที่ 1: ขึ้น Error "CUDA not found"

**วิธีแก้**: ตรวจสอบการติดตั้ง CUDA
```powershell
nvcc --version
nvidia-smi
```

ถ้ายังไม่พบ:
- ติดตั้ง CUDA 12.6 ใหม่
- เช็ค PATH environment variable
- รีสตาร์ทเครื่อง

### ปัญหาที่ 2: "Failed to load ONNX model with GPU"

**สาเหตุที่เป็นไปได้**:
- cuDNN DLLs ไม่อยู่ใน PATH
- เวอร์ชัน ONNX Runtime ผิด
- DLLs หายในโฟลเดอร์ output

**วิธีแก้**:
```powershell
# เช็คว่า DLLs มีหรือไม่
Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\cudnn*.dll"
Get-ChildItem "x64\Release\onnxruntime*.dll"

# คัดลอก DLLs ใหม่ถ้าหาย
Copy-Item "packages\Microsoft.ML.OnnxRuntime.Gpu.1.20.1\runtimes\win-x64\native\*.dll" -Destination "x64\Release\" -Force
```

### ปัญหาที่ 3: FPS ยังต่ำอยู่ (ประมาณ 20-30)

**สาเหตุที่เป็นไปได้**:
- ยังมีโค้ดจำกัด FPS อยู่
- ไม่ได้ใช้ GPU (ดูที่ console logs)
- CPU bottleneck ในการ decode วิดีโอ

**วิธีแก้**:
```cpp
// ตรวจสอบว่าลบโค้ดจำกัด FPS ใน CameraReaderLoop() แล้วหรือยัง
// เช็ค console output ควรมี:
// [GPU] ✅ Model loaded with GPU acceleration

// ใช้ nvidia-smi ตรวจสอบว่า GPU กำลังทำงานจริงๆ
```

### ปัญหาที่ 4: Out of Memory Error

**วิธีแก้**: ลด batch size หรือใช้โมเดลเล็กลง
```cpp
// ใน OnnxYoloInference.h ลด workspace size:
trt_options.trt_max_workspace_size = 1073741824; // 1GB แทนที่จะเป็น 2GB
```

หรือใช้โมเดลที่เล็กกว่า:
```cpp
std::string modelPath = "models/yolo11n.onnx"; // Nano model (เล็กกว่า)
```

### ปัญหาที่ 5: โปรแกรม Crash ตอนเริ่ม

**สาเหตุที่เป็นไปได้**:
- DLL files หาย
- เวอร์ชัน CUDA/cuDNN ไม่เข้ากัน
- Visual Studio configuration ผิด

**วิธีแก้**:
```powershell
# ตรวจสอบ dependencies
dumpbin /dependents x64\Release\ConsoleApplication3.exe

# DLL ที่มักหาย:
# - onnxruntime.dll
# - onnxruntime_providers_cuda.dll  
# - cudnn64_9.dll
# - cublas64_12.dll

# คัดลอก DLL ที่จำเป็นทั้งหมดไปยังโฟลเดอร์ output
```

---

## 📈 เทคนิคเพิ่มความเร็ว

### 1. ใช้ FP16 (Half Precision)
```cpp
trt_options.trt_fp16_enable = 1;  // เปิดใช้อยู่แล้วในโค้ด
// ให้ความเร็วเพิ่ม ~30% โดยความแม่นยำลดน้อยมาก
```

### 2. ลดความละเอียด Input
```cpp
// ความละเอียดต่ำ = inference เร็วขึ้น
// 640x640 → 100 FPS
// 416x416 → 150+ FPS (แต่ความแม่นยำลดลง)
```

### 3. เพิ่ม Confidence Threshold
```cpp
float confThreshold = 0.5f; // ค่าสูง = เร็วขึ้น (detect น้อยลง)
```

### 4. ใช้โมเดลเล็กลง
```
yolo11n.onnx    - Nano (เร็วที่สุด, ~200 FPS)
yolo11s.onnx    - Small (~120 FPS)
yolo26s.onnx    - Custom (~113 FPS)
yolov8m.onnx    - Medium (~60 FPS)
yolov8l.onnx    - Large (~30 FPS)
```

---

## 📦 Checklist สำหรับเพื่อน

```
✅ ความต้องการระบบ
   ⬜ การ์ดจอ NVIDIA ที่รองรับ CUDA
   ⬜ Windows 10/11 64-bit
   ⬜ Visual Studio 2022
   ⬜ NVIDIA Driver 471.0+

✅ การติดตั้ง
   ⬜ ติดตั้ง CUDA 12.6 (3.5GB)
   ⬜ ติดตั้ง cuDNN 9.1.9.1 (คัดลอกไฟล์ไปโฟลเดอร์ CUDA)
   ⬜ เพิ่ม CUDA bin เข้า System PATH
   ⬜ ดาวน์โหลด ONNX Runtime GPU 1.20.1
   ⬜ คัดลอก ONNX Runtime DLLs ไป x64/Release และ x64/Debug

✅ การแก้ไขโค้ด
   ⬜ สร้าง OnnxYoloInference.h
   ⬜ แก้ ConsoleApplication3.vcxproj (เพิ่ม include/lib paths)
   ⬜ แก้ online1.h (เปลี่ยนจาก cv::dnn เป็น ONNX Runtime)
   ⬜ ลบโค้ดจำกัด FPS ใน CameraReaderLoop()
   ⬜ เพิ่ม #include "OnnxYoloInference.h"

✅ Build & ทดสอบ
   ⬜ Rebuild Solution (Release x64)
   ⬜ เช็ค console ว่ามีข้อความยืนยัน GPU
   ⬜ ดูการใช้งาน GPU ด้วย nvidia-smi
   ⬜ ตรวจสอบ FPS เพิ่มขึ้น (ควรได้ 60-100+ FPS)

✅ แก้ไขปัญหา
   ⬜ DLLs ครบในโฟลเดอร์ output
   ⬜ CUDA paths ใน System PATH
   ⬜ ไม่มีโค้ดจำกัด FPS เหลืออยู่
   ⬜ เห็นการใช้งาน GPU ใน Task Manager
```

---

## 🎯 ผลลัพธ์ที่คาดหวัง

| ตัวชี้วัด | ก่อน (CPU) | หลัง (GPU) |
|----------|------------|------------|
| โหลด Model | 2-3 วินาที | 3-5 วินาที |
| Inference Time | 77ms | 8.8ms |
| FPS (Benchmark) | 12.9 | 113.0 |
| FPS (วิดีโอจริง) | 20-30 | 47-64 |
| GPU Usage | 0% | 80-95% |
| CPU Usage | 80-100% | 20-30% |

---

## 🚀 คำสั่งด่วน (Quick Start)

```powershell
# 1. ติดตั้ง CUDA (รันไฟล์ติดตั้ง)
# 2. ติดตั้ง cuDNN (คัดลอกไฟล์)
# 3. เพิ่ม CUDA เข้า PATH:
[Environment]::SetEnvironmentVariable("Path", "$env:Path;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin", "Machine")

# 4. ดาวน์โหลด ONNX Runtime
nuget install Microsoft.ML.OnnxRuntime.Gpu -Version 1.20.1 -OutputDirectory packages

# 5. คัดลอก DLLs
Copy-Item "packages\Microsoft.ML.OnnxRuntime.Gpu.1.20.1\runtimes\win-x64\native\*.dll" -Destination "x64\Release\" -Force

# 6. Build
msbuild ConsoleApplication3.vcxproj /p:Configuration=Release /p:Platform=x64 /t:Rebuild

# 7. รัน
.\x64\Release\ConsoleApplication3.exe
```

---

## 💡 เคล็ดลับสำคัญ

1. **ต้องคัดลอก DLLs ทุกครั้งที่ Clean Solution**
2. **ตรวจสอบ console output เสมอ** เพื่อดูว่า GPU ทำงานหรือไม่
3. **ใช้ nvidia-smi** เพื่อดูการใช้งาน GPU แบบ real-time
4. **ลบโค้ดจำกัด FPS** เพื่อให้ GPU ทำงานเต็มที่
5. **เปิด Task Manager → Performance → GPU** เพื่อดูกราฟการใช้งาน

---

## 📁 โครงสร้างไฟล์หลังติดตั้ง

```
project_opencv_ajsum_nikom/
├── OnnxYoloInference.h          (ใหม่ - GPU wrapper class)
├── online1.h                     (แก้ไข - ใช้ ONNX Runtime)
├── ConsoleApplication3.vcxproj   (แก้ไข - มี ONNX paths)
├── packages/
│   └── Microsoft.ML.OnnxRuntime.Gpu.1.20.1/
│       ├── build/native/include/
│       └── runtimes/win-x64/native/*.dll
├── x64/
│   ├── Release/
│   │   ├── ConsoleApplication3.exe
│   │   ├── onnxruntime.dll              (คัดลอกแล้ว)
│   │   ├── onnxruntime_providers_cuda.dll
│   │   ├── onnxruntime_providers_shared.dll
│   │   └── onnxruntime_providers_tensorrt.dll
│   └── Debug/ (DLLs เหมือนกัน)
└── models/
    └── yolo26s.onnx
```

---

## 📞 ติดต่อสอบถาม

หากพบปัญหา:
1. ดูข้อความ error ใน console
2. ตรวจสอบ GPU ด้วย `nvidia-smi`
3. เช็ค Task Manager → Performance → GPU
4. ดูส่วน "แก้ไขปัญหา" ด้านบน

**ปัญหาที่พบบ่อย**:
- DLL หาย → คัดลอกจากโฟลเดอร์ packages อีกครั้ง
- FPS ต่ำ → ตรวจสอบว่าใช้ GPU จริงๆ (ดูที่ console logs)
- Crash → เช็คความเข้ากันได้ของ CUDA/cuDNN

---

**สร้างเมื่อ**: 7 มีนาคม 2026  
**การ์ดจอ**: NVIDIA RTX 4050 Laptop (6GB VRAM)  
**ผลการทดสอบ**: 113 FPS (เร็วขึ้น 8.76 เท่า)  
**เวอร์ชัน**: CUDA 12.6 + cuDNN 9.1.9 + ONNX Runtime 1.20.1

---

## 🎓 คำอธิบายเพิ่มเติม

### ทำไมต้องใช้ GPU?
- **CPU**: ประมวลผลทีละขั้นตอน (serial processing)
- **GPU**: ประมวลผลหลายพันขั้นตอนพร้อมกัน (parallel processing)
- **YOLO**: ต้องคำนวณหลายล้านครั้งต่อเฟรม → เหมาะกับ GPU

### ความแตกต่างระหว่าง CUDA, cuDNN, และ ONNX Runtime
- **CUDA**: ภาษาและไลบรารีพื้นฐานสำหรับการเขียนโปรแกรมบน GPU
- **cuDNN**: ไลบรารีเฉพาะสำหรับ Deep Learning (เร็วกว่า CUDA ธรรมดา)
- **ONNX Runtime**: ตัวรัน AI models ที่รองรับหลาย framework (PyTorch, TensorFlow, etc.)

### ทำไมต้องคัดลอก DLLs?
- โปรแกรม C++ ต้องการ DLL files ในโฟลเดอร์เดียวกับ .exe
- ONNX Runtime DLLs ไม่ได้อยู่ใน system PATH โดยตรง
- การคัดลอกทำให้โปรแกรมหา DLLs เจอทันที

### การทำงานของ TensorRT
- TensorRT จะ optimize โมเดลโดยอัตโนมัติ
- ครั้งแรกจะใช้เวลานานหน่อย (building engine)
- ครั้งต่อไปจะเร็วมาก (ใช้ cached engine)

---

**หมายเหตุ**: เอกสารนี้เขียนขึ้นเพื่อให้เพื่อนของคุณสามารถติดตั้งและใช้งาน GPU acceleration ได้เอง โดยไม่ต้องมีความรู้ด้าน GPU มาก่อน ถ้ามีคำถามเพิ่มเติมสามารถศึกษาจากเอกสาร NVIDIA CUDA และ ONNX Runtime documentation ได้
