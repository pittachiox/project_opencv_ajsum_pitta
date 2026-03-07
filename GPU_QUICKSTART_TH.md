# 🚀 คู่มือติดตั้ง ONNX Runtime GPU ฉบับรวบรัด (Quick Start Guide)

ไฟล์นี้จะรวมเฉพาะคำสั่งและขั้นตอนที่จำเป็น **สำหรับการ Compile และ Run โปรแกรม** ด้วย ONNX Runtime GPU (ไม่ต้องคอนฟิกเยอะ)

---

## 🛠 1. สิ่งที่ต้องติดตั้งก่อน (Prerequisites)
ก่อนเริ่มคอมไพล์ เครื่องต้องมี 3 อย่างนี้ติดตั้งไว้แล้ว:
1. **NVIDIA GPU Driver** (อัปเดตล่าสุด)
2. **CUDA Toolkit 12.6** 
3. **cuDNN 9.1.9**

*(ถ้ายังไม่มีให้โหลดและติดตั้งแบบปกติ)*

---

## 📦 2. ดาวน์โหลด ONNX Runtime GPU

1. ดาวน์โหลดไฟล์ ZIP: [onnxruntime-win-x64-gpu-1.20.1.zip](https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-win-x64-gpu-1.20.1.zip)
2. นำไปแตกไฟล์ และ **นำโฟลเดอร์มาวางไว้ที่ root ของโปรเจกต์** ให้โครงสร้างเป็นแบบนี้:
   ```text
   C:\Users\HP\source\repos\final\
   └── onnxruntime-win-x64-gpu-1.20.1\
       ├── include\
       ├── lib\
       └── ...
   ```
*(หมายเหตุ: ระวังการแตกไฟล์แล้วได้โฟลเดอร์ซ้อนกันแบบ `onnxruntime...\onnxruntime...\lib` ต้องย้ายออกมาให้เหลือชั้นเดียว)*

---

## 🚀 3. คัดลอก DLLs ไปยัง Output Folder

ไฟล์ DLL จำเป็นต้องอยู่โฟลเดอร์เดียวกับไฟล์ `.exe` ตอนรันโปรแกรม เปิด PowerShell และรันคำสั่ง:

```powershell
cd "C:\Users\HP\source\repos\final"

# สำหรับรันโหมด Debug
Copy-Item "onnxruntime-win-x64-gpu-1.20.1\lib\*.dll" -Destination "x64\Debug\" -Force

# สำหรับรันโหมด Release
Copy-Item "onnxruntime-win-x64-gpu-1.20.1\lib\*.dll" -Destination "x64\Release\" -Force
```

---

## 🔨 4. Compile (Build Solution)

ใช้คำสั่ง `MSBuild` โค้ดด้านล่างเพื่อ Build โปรแกรมผ่าน PowerShell (คำสั่งนี้ใช้กับไดร์ฟ D ถ้าลง VS ไว้ที่ไดร์ฟอื่นให้แก้ Path):

**คำสั่งสำหรับ Build โหมด Debug:**
```powershell
& "D:\VS2022\MSBuild\Current\Bin\MSBuild.exe" "C:\Users\HP\source\repos\final\ConsoleApplication3.vcxproj" /p:Configuration=Debug /p:Platform=x64 /t:Rebuild /m
```

**คำสั่งสำหรับ Build โหมด Release (แนะนำสำหรับใช้งานจริง - เร็วกว่ามาก):**
```powershell
& "D:\VS2022\MSBuild\Current\Bin\MSBuild.exe" "C:\Users\HP\source\repos\final\ConsoleApplication3.vcxproj" /p:Configuration=Release /p:Platform=x64 /t:Rebuild /m
```

---

## ▶️ 5. Run โปรแกรม

ทำการรันไฟล์ `.exe` ที่ Build เสร็จแล้ว:

**รันโหมด Debug:**
```powershell
cd "C:\Users\HP\source\repos\final\x64\Debug"
.\ConsoleApplication3.exe
```

**รันโหมด Release:**
```powershell
cd "C:\Users\HP\source\repos\final\x64\Release"
.\ConsoleApplication3.exe
```

🎉 **เสร็จเรียบร้อย! โปรแกรมจะทำงานโดยใช้การ์ดจอ (GPU) ในการรัน AI โมเดลแล้ว**
