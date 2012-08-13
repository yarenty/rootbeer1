@echo off
@call "C:\Program Files (x86)\Microsoft Visual Studio 10.0\VC\vcvarsall.bat" amd64
cl /I"%JAVA_HOME%\include" /I"%JAVA_HOME%\include\win32" /I"%CUDA_INC_PATH%" CudaRuntime2.c FastMemory.c Handles.c Cuda2DeviceMemory.c /link "%CUDA_LIB_PATH%\cuda.lib" /DLL /OUT:cudaruntime.dll /MACHINE:X64
move /-y .\cudaruntime.* ..\src\edu\syr\pcpratts\rootbeer\runtime2\native\
