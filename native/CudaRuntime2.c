#include "edu_syr_pcpratts_rootbeer_runtime2_cuda_CudaRuntime2.h"

#include <assert.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK_STATUS(env,msg,status) \
if (CUDA_SUCCESS != status) {\
  throw_cuda_errror_exception(env, msg, status);\
  return;\
}

#define CHECK_STATUS_RTN(env,msg,status,rtn) \
if (CUDA_SUCCESS != status) {\
  throw_cuda_errror_exception(env, msg, status);\
  return rtn;\
}

#define GPUCARD_CLASS_NAME "edu/syr/pcpratts/rootbeer/runtime/GpuCard"
#define GPUCARD_CLASS_NAME_SIG "Ledu/syr/pcpratts/rootbeer/runtime/GpuCard;"
#define CUDA_ERROR_EXCEPTION_CLASS_NAME "edu/syr/pcpratts/rootbeer/runtime2/cuda/CudaErrorException"

static CUmodule cuModule;
static CUfunction cuFunction;

static jobject thisRef;
static jclass thisRefClass;

static jlong heapEndPtr;

static CUtexref    cache;

static jlong classMemSize;
static int textureMemSize;

static size_t gc_space_size = 1024;


/**
 * Returns the current GPU Device
 */
jobject getCurrentGPUDevice(JNIEnv *env){
    
    jfieldID fid = (*env)->GetFieldID(env, thisRefClass, "currentGpuCard", GPUCARD_CLASS_NAME_SIG);
    jobject currentGpuCard = (*env)->GetObjectField(env, thisRef, fid);
        
    return currentGpuCard;
}

/**
 * Sets the current GPU Device
 */
void setCurrentGPUDevice(JNIEnv *env, jobject gpuDevice){
    
    jfieldID fid = (*env)->GetFieldID(env, thisRefClass, "currentGpuCard", GPUCARD_CLASS_NAME_SIG);
    (*env)->SetObjectField(env, thisRef, fid, gpuDevice);
    
    return;
}


jint getIntField(JNIEnv *env, jobject obj, const char * name){
    
    jclass c = (*env)->GetObjectClass(env, obj);
    jfieldID fid = (*env)->GetFieldID(env, c, name, "I");
    jint value = (*env)->GetIntField(env, obj, fid);
    
    return value;
}

void setLongField(JNIEnv *env, jobject obj, const char * name, jlong value){
    
    jclass c = (*env)->GetObjectClass(env, obj);
    jfieldID fid = (*env)->GetFieldID(env, c, name, "J");
    (*env)->SetLongField(env, obj, fid, value);
    
    return;
}

jlong getLongField(JNIEnv *env, jobject obj, const char * name){
    
    jclass c = (*env)->GetObjectClass(env, obj);
    jfieldID fid = (*env)->GetFieldID(env, c, name, "J");
    jlong value = (*env)->GetLongField(env, obj, fid);
    
    return value;
}


/**
* Throws a runtimeexception called CudaMemoryException
* allocd - number of bytes tried to allocate
* id - variable the memory assignment was for
*/
void throw_cuda_errror_exception(JNIEnv *env, const char *message, int error) {
  char msg[1024];
  jclass exp;
  jfieldID fid;
  int a = 0;
  int b = 0;
  char name[1024];

  if(error == CUDA_SUCCESS){
    return;
  }

  // Get the current GPU Device and ID
  jobject currentGPUDevice = getCurrentGPUDevice(env);
  jint currentGPUDeviceID = getIntField(env, currentGPUDevice, "cardID");
    
  CUdevice cuDevice;
  int status = cuDeviceGet(&cuDevice, currentGPUDeviceID);
  CHECK_STATUS(env,"error in cuDeviceGet",status)

  exp = (*env)->FindClass(env,CUDA_ERROR_EXCEPTION_CLASS_NAME);

  // we truncate the message to 900 characters to stop any buffer overflow
  switch(error){
    case CUDA_ERROR_OUT_OF_MEMORY:
      sprintf(msg, "CUDA_ERROR_OUT_OF_MEMORY: %.900s",message);
      break;
    case CUDA_ERROR_NO_BINARY_FOR_GPU:
      cuDeviceGetName(name,1024,cuDevice);
      cuDeviceComputeCapability(&a, &b, cuDevice);
      sprintf(msg, "No binary for gpu. Selected %s (%d.%d). 2.0 compatibility required.", name, a, b);
      break;
    default:
      sprintf(msg, "ERROR STATUS:%i : %.900s", error, message);
  }

  fid = (*env)->GetFieldID(env, exp, "cudaError_enum", "I");
  (*env)->SetLongField(env, exp, fid, (jint)error);

  (*env)->ThrowNew(env,exp,msg);
  
  return;
}


/*
 * Class:     edu_syr_pcpratts_rootbeer_runtime2_cuda_CudaRuntime2
 * Method:    setupGpuCards
 * Signature: (II[J)Ljava/util/List;
 */
JNIEXPORT jobject JNICALL Java_edu_syr_pcpratts_rootbeer_runtime2_cuda_CudaRuntime2_setupGpuCards
(JNIEnv * env, jobject this_obj, jint _max_blocks_per_proc, jint _max_threads_per_block, jlongArray _reserve_mem_list) {

    int num_devices = 0;
    // Temp variables
    int i, a=0, b=0, status;
    long j;
    // Best GPU
    int max_multiprocessors = -1;
    jobject bestGpuCardObject;
    
    // GPU Property variables
    char name[1024];
    int computeCapabilityA=0, computeCapabilityB=0;
    size_t free_mem, total_mem;
    //int free_mem_Mbytes=0, total_mem_Mbytes=0;
    int max_registers_per_block=0;
    int warp_size=0;
    int max_pitch=0;
    int max_threads_per_block=0;
    int max_shared_memory_per_block=0;
    float clock_rate=0.0;
    float memory_clock_rate=0.0;
    float total_constant_memory=0.0;
    int integrated=0;
    int max_threads_per_multiprocessor=0;
    int multiprocessor_count=0; //numMultiProcessors
    int max_block_dim_x=0;
    int max_block_dim_y=0;
    int max_block_dim_z=0;
    int max_grid_dim_x=0;  //maxGridDim
    int max_grid_dim_y=0;
    int max_grid_dim_z=0;
    
    jlong reserve_mem=0;
    size_t to_space_size;
    jlong numBlocks=0;
    
    // Set static this references
    thisRef = this_obj;
    thisRefClass = (*env)->GetObjectClass(env, this_obj);
    
    // ArrayList Class and Constructor and Add Method
    jclass arrayListClass = (*env)->FindClass(env, "java/util/ArrayList");
    jmethodID arrayListCons =  (*env)->GetMethodID(env, arrayListClass,
                                                   "<init>", "()V");
    jmethodID arrayListAdd = (*env)->GetMethodID(env, arrayListClass,
                                                 "add", "(Ljava/lang/Object;)Z");
    // Create new ArrayList<Object>
    jobject gpuCardList = (*env)->NewObject(env, arrayListClass, arrayListCons);
    
    // GpuCard Class and Constructor
    jclass gpuCardClass = (*env)->FindClass(env,GPUCARD_CLASS_NAME);
    jmethodID gpuCardCons = (*env)->GetMethodID(env, gpuCardClass,
                                                "<init>", "(ILjava/lang/String;IIIIIIIIIFFFIIIIIIIIIJJJ)V");
    if (gpuCardCons == NULL) return NULL;
    
    // Initializes the driver API
    status = cuInit(0);
    CHECK_STATUS(env,"error in cuInit",status)
 
    // Get number of CUDA devices
    cuDeviceGetCount(&num_devices);
    CHECK_STATUS_RTN(env,"error in cuDeviceGetCount",status, 0);
    
    if(num_devices == 0)
        throw_cuda_errror_exception(env,"0 Cuda Devices were found",0);
    
    for (i = 0; i < num_devices; ++i)
    {
        CUdevice cuDevice;
        status = cuDeviceGet(&cuDevice, i);
        CHECK_STATUS(env,"error in cuDeviceGet",status)
        
        CUcontext cuContext;
        status = cuCtxCreate(&cuContext, CU_CTX_MAP_HOST, cuDevice);
        CHECK_STATUS(env,"error in cuCtxCreate",status)
        
        if(cuDeviceComputeCapability(&a, &b, cuDevice) == CUDA_SUCCESS){
            computeCapabilityA = a;
            computeCapabilityB = b;
        }
        
        cuDeviceGetName(name,1024,cuDevice);
        
        cuMemGetInfo(&free_mem, &total_mem);
        /*
         == CUDA_SUCCESS){
            free_mem_Mbytes = free_mem/1024/1024;
            total_mem_Mbytes = total_mem/1024/1024;
        }*/
        
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,cuDevice) == CUDA_SUCCESS)
            max_registers_per_block = a;
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_WARP_SIZE,cuDevice) == CUDA_SUCCESS)
            warp_size = a;
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_MAX_PITCH,cuDevice) == CUDA_SUCCESS)
            max_pitch = a;
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,cuDevice) == CUDA_SUCCESS)
            max_threads_per_block = a;
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,cuDevice) == CUDA_SUCCESS)
            max_shared_memory_per_block = a/1024.0;
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_CLOCK_RATE,cuDevice) == CUDA_SUCCESS)
            clock_rate = a/1000000.0;
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,cuDevice) == CUDA_SUCCESS)
            memory_clock_rate = a/1000000.0;
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY,cuDevice) == CUDA_SUCCESS)
            total_constant_memory = a/1024.0/1024.0;
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_INTEGRATED,cuDevice) == CUDA_SUCCESS)
            integrated = a;
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,cuDevice) == CUDA_SUCCESS)
            max_threads_per_multiprocessor = a;
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,cuDevice) == CUDA_SUCCESS)
            multiprocessor_count = a;
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,cuDevice) == CUDA_SUCCESS)
            max_block_dim_x = a;
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,cuDevice) == CUDA_SUCCESS)
            max_block_dim_y = a;
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,cuDevice) == CUDA_SUCCESS)
            max_block_dim_z = a;
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,cuDevice) == CUDA_SUCCESS)
            max_grid_dim_x = a;
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,cuDevice) == CUDA_SUCCESS)
            max_grid_dim_y = a;
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z,cuDevice) == CUDA_SUCCESS)
            max_grid_dim_z = a;
        
        
        // Calculate to_space_size
        to_space_size = free_mem;
        //space for 100 types in the scene
        classMemSize = sizeof(jint)*100;
        numBlocks = multiprocessor_count * _max_threads_per_block * _max_blocks_per_proc;
        
        to_space_size -= (numBlocks * sizeof(jlong));
        to_space_size -= (numBlocks * sizeof(jlong));
        to_space_size -= gc_space_size;
        to_space_size -= classMemSize;
        //leave 10MB for module
        to_space_size -= 10L*1024L*1024L;
        
        
        // Get reserve_mem for current GPU
        reserve_mem = 0;
        if (_reserve_mem_list != NULL) {
            jlong* reserve_memory_list = (*env)->GetLongArrayElements(env, _reserve_mem_list, NULL);
            if (reserve_memory_list == NULL) return NULL;
            
            reserve_mem = reserve_memory_list[i];
        }
        
        // Try to find reserve_mem if no reserve_mem was configured or invalid
        if ((_reserve_mem_list == NULL) || (reserve_mem == 0)) {
            
            void * toSpace;
            CUdeviceptr gpuToSpace;
            CUdeviceptr gpuClassMemory;
            void * handlesMemory;
            CUdeviceptr gpuHandlesMemory;
            void * exceptionsMemory;
            CUdeviceptr gpuExceptionsMemory;
            CUdeviceptr gcInfoSpace;
            CUdeviceptr gpuHeapEndPtr;
            CUdeviceptr gpuBufferSize;
            
            for(j = 1024L*1024L; j < to_space_size; j += 100L*1024L*1024L){
                size_t temp_size = to_space_size - j;
                
                printf("attempting allocation with temp_size: %lu to_space_size: %lu i: %ld\n", temp_size, to_space_size, j);
                
                status = cuMemHostAlloc(&toSpace, temp_size, 0);
                if(status != CUDA_SUCCESS) {
                    cuMemFreeHost(toSpace);
                    continue;
                }
                status = cuMemAlloc(&gpuToSpace, temp_size);
                if(status != CUDA_SUCCESS) {
                    cuMemFreeHost(toSpace);
                    cuMemFree(gpuToSpace);
                    continue;
                }
                status = cuMemAlloc(&gpuClassMemory, classMemSize);
                if(status != CUDA_SUCCESS) {
                    cuMemFreeHost(toSpace);
                    cuMemFree(gpuToSpace);
                    cuMemFree(gpuClassMemory);
                    continue;
                }
                status = cuMemHostAlloc(&handlesMemory, numBlocks * sizeof(jlong), 0);
                if(status != CUDA_SUCCESS) {
                    cuMemFreeHost(toSpace);
                    cuMemFree(gpuToSpace);
                    cuMemFree(gpuClassMemory);
                    cuMemFreeHost(handlesMemory);
                    continue;
                }
                status = cuMemAlloc(&gpuHandlesMemory, numBlocks * sizeof(jlong));
                if(status != CUDA_SUCCESS) {
                    cuMemFreeHost(toSpace);
                    cuMemFree(gpuToSpace);
                    cuMemFree(gpuClassMemory);
                    cuMemFreeHost(handlesMemory);
                    cuMemFree(gpuHandlesMemory);
                    continue;
                }
                status = cuMemHostAlloc(&exceptionsMemory, numBlocks * sizeof(jlong), 0);
                if(status != CUDA_SUCCESS) {
                    cuMemFreeHost(toSpace);
                    cuMemFree(gpuToSpace);
                    cuMemFree(gpuClassMemory);
                    cuMemFreeHost(handlesMemory);
                    cuMemFree(gpuHandlesMemory);
                    cuMemFreeHost(exceptionsMemory);
                    continue;
                }
                status = cuMemAlloc(&gpuExceptionsMemory, numBlocks * sizeof(jlong));
                if(status != CUDA_SUCCESS) {
                    cuMemFreeHost(toSpace);
                    cuMemFree(gpuToSpace);
                    cuMemFree(gpuClassMemory);
                    cuMemFreeHost(handlesMemory);
                    cuMemFree(gpuHandlesMemory);
                    cuMemFreeHost(exceptionsMemory);
                    cuMemFree(gpuExceptionsMemory);
                    continue;
                }
                status = cuMemAlloc(&gcInfoSpace, gc_space_size);
                if(status != CUDA_SUCCESS) {
                    cuMemFreeHost(toSpace);
                    cuMemFree(gpuToSpace);
                    cuMemFree(gpuClassMemory);
                    cuMemFreeHost(handlesMemory);
                    cuMemFree(gpuHandlesMemory);
                    cuMemFreeHost(exceptionsMemory);
                    cuMemFree(gpuExceptionsMemory);
                    cuMemFree(gcInfoSpace);
                    continue;
                }
                status = cuMemAlloc(&gpuHeapEndPtr, 8);
                if(status != CUDA_SUCCESS) {
                    cuMemFreeHost(toSpace);
                    cuMemFree(gpuToSpace);
                    cuMemFree(gpuClassMemory);
                    cuMemFreeHost(handlesMemory);
                    cuMemFree(gpuHandlesMemory);
                    cuMemFreeHost(exceptionsMemory);
                    cuMemFree(gpuExceptionsMemory);
                    cuMemFree(gcInfoSpace);
                    cuMemFree(gpuHeapEndPtr);
                    continue;
                }
                status = cuMemAlloc(&gpuBufferSize, 8);
                if(status != CUDA_SUCCESS) {
                    cuMemFreeHost(toSpace);
                    cuMemFree(gpuToSpace);
                    cuMemFree(gpuClassMemory);
                    cuMemFreeHost(handlesMemory);
                    cuMemFree(gpuHandlesMemory);
                    cuMemFreeHost(exceptionsMemory);
                    cuMemFree(gpuExceptionsMemory);
                    cuMemFree(gcInfoSpace);
                    cuMemFree(gpuHeapEndPtr);
                    cuMemFree(gpuBufferSize);
                    continue;
                }
                
                cuMemFreeHost(toSpace);
                cuMemFree(gpuToSpace);
                cuMemFree(gpuClassMemory);
                cuMemFreeHost(handlesMemory);
                cuMemFree(gpuHandlesMemory);
                cuMemFreeHost(exceptionsMemory);
                cuMemFree(gpuExceptionsMemory);
                cuMemFree(gcInfoSpace);
                cuMemFree(gpuHeapEndPtr);
                cuMemFree(gpuBufferSize);
                
                to_space_size = temp_size;
                reserve_mem = j;
                break; //exit to_space_size and reserve_mem was found
            }
            if (reserve_mem==0)
                throw_cuda_errror_exception(env, "unable to find enough space using CUDA", 0); 
        }
        
        jobject gpuCardObject = (*env)->NewObject(env, gpuCardClass,
                                                  gpuCardCons,
                                                  i,
                                                  (*env)->NewStringUTF(env,name),
                                                  computeCapabilityA,
                                                  computeCapabilityB,
                                                  total_mem,
                                                  free_mem,
                                                  max_registers_per_block,
                                                  warp_size,
                                                  max_pitch,
                                                  max_threads_per_block,
                                                  max_shared_memory_per_block,
                                                  clock_rate,
                                                  memory_clock_rate,
                                                  total_constant_memory,
                                                  integrated,
                                                  max_threads_per_multiprocessor,
                                                  multiprocessor_count,
                                                  max_block_dim_x,
                                                  max_block_dim_y,
                                                  max_block_dim_z,
                                                  max_grid_dim_x,
                                                  max_grid_dim_y,
                                                  max_grid_dim_z,
                                                  to_space_size,
                                                  numBlocks,
                                                  reserve_mem);
        if (gpuCardObject == NULL) return NULL;
        
        // Find best GPU and save gpu_id
        if(multiprocessor_count > max_multiprocessors) {
            max_multiprocessors = multiprocessor_count;
            bestGpuCardObject = gpuCardObject;
        }
        
        jboolean jbool = (*env)->CallBooleanMethod(env, gpuCardList,
                                                   arrayListAdd, gpuCardObject);
        
        cuCtxDestroy(cuContext);
    }
	
    // Set current currentGpuCard to best one
    setCurrentGPUDevice(env,bestGpuCardObject);
    
	return gpuCardList;
}

/*
 * Class:     edu_syr_pcpratts_rootbeer_runtime2_cuda_CudaRuntime2
 * Method:    initGpuCard
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_edu_syr_pcpratts_rootbeer_runtime2_cuda_CudaRuntime2_initGpuCard
(JNIEnv * env, jobject this_obj) {

    int status;
    int deviceCount = 0;
    //textureMemSize = 1;
    
    status = cuDeviceGetCount(&deviceCount);
    CHECK_STATUS(env,"error in cuDeviceGetCount",status)

    // Get the current GPU Device and ID
    jobject currentGPUDevice = getCurrentGPUDevice(env);
    jint currentGPUDeviceID = getIntField(env, currentGPUDevice, "cardID");    
    
    CUdevice cuDevice;
    status = cuDeviceGet(&cuDevice, currentGPUDeviceID);
    CHECK_STATUS(env,"error in cuDeviceGet",status)
    
    CUcontext cuContext;
    status = cuCtxCreate(&cuContext, CU_CTX_MAP_HOST, cuDevice);
    CHECK_STATUS(env,"error in cuCtxCreate",status)
    
    //size_t f_mem = getLongField(env, currentGPUDevice, "freeMemory");
    //size_t t_mem = getLongField(env, currentGPUDevice, "totalMemory");
    jlong numBlocks = getLongField(env, currentGPUDevice, "numBlocks");
    size_t to_space_size = getLongField(env, currentGPUDevice, "toSpaceSize");

    //space for 100 types in the scene
    classMemSize = sizeof(jint)*100;
    
    void * toSpace;
    status = cuMemHostAlloc(&toSpace, to_space_size, 0);
    CHECK_STATUS(env,"toSpace memory allocation failed",status)
    setLongField(env, currentGPUDevice, "toSpaceAddr", (jlong)toSpace);
    
    CUdeviceptr gpuToSpace;
    status = cuMemAlloc(&gpuToSpace, to_space_size);
    CHECK_STATUS(env,"gpuToSpace memory allocation failed",status)
    setLongField(env, currentGPUDevice, "gpuToSpaceAddr", (jlong)gpuToSpace);
    
    CUdeviceptr gpuClassMemory;
    status = cuMemAlloc(&gpuClassMemory, classMemSize);
    CHECK_STATUS(env,"gpuClassMemory memory allocation failed",status)
    setLongField(env, currentGPUDevice, "gpuClassAddr", (jlong)gpuClassMemory);
    
    /*
     status = cuMemHostAlloc(&textureMemory, textureMemSize, 0);
     if (CUDA_SUCCESS != status)
     {
     printf("error in cuMemHostAlloc textureMemory %d\n", status);
     }
     
     status = cuMemAlloc(&gpuTexture, textureMemSize);
     if (CUDA_SUCCESS != status)
     {
     printf("error in cuMemAlloc gpuTexture %d\n", status);
     }
     */
    
    void * handlesMemory;
    status = cuMemHostAlloc(&handlesMemory, numBlocks * sizeof(jlong), CU_MEMHOSTALLOC_WRITECOMBINED);
    CHECK_STATUS(env,"handlesMemory memory allocation failed",status)
    setLongField(env, currentGPUDevice, "handlesAddr", (jlong)handlesMemory);

    CUdeviceptr gpuHandlesMemory;
    status = cuMemAlloc(&gpuHandlesMemory, numBlocks * sizeof(jlong));
    CHECK_STATUS(env,"gpuHandlesMemory memory allocation failed",status)
    setLongField(env, currentGPUDevice, "gpuHandlesAddr", (jlong)gpuHandlesMemory);
    
    void * exceptionsMemory;
    status = cuMemHostAlloc(&exceptionsMemory, numBlocks * sizeof(jlong), 0);
    CHECK_STATUS(env,"exceptionsMemory memory allocation failed",status)
    setLongField(env, currentGPUDevice, "exceptionsHandlesAddr", (jlong)exceptionsMemory);
    
    CUdeviceptr gpuExceptionsMemory;
    status = cuMemAlloc(&gpuExceptionsMemory, numBlocks * sizeof(jlong));
    CHECK_STATUS(env,"gpuExceptionsMemory memory allocation failed",status)
    setLongField(env, currentGPUDevice, "gpuExceptionsHandlesAddr", (jlong)gpuExceptionsMemory);
    
    CUdeviceptr gcInfoSpace;
    status = cuMemAlloc(&gcInfoSpace, gc_space_size);
    CHECK_STATUS(env,"gcInfoSpace memory allocation failed",status)
    setLongField(env, currentGPUDevice, "gcInfoSpace", (jlong)gcInfoSpace);
    
    CUdeviceptr gpuHeapEndPtr;
    status = cuMemAlloc(&gpuHeapEndPtr, 8);
    CHECK_STATUS(env,"gpuHeapEndPtr memory allocation failed",status)
    setLongField(env, currentGPUDevice, "gpuHeapEndPtr", (jlong)gpuHeapEndPtr);
    
    CUdeviceptr gpuBufferSize;
    status = cuMemAlloc(&gpuBufferSize, 8);
    CHECK_STATUS(env,"gpuBufferSize memory allocation failed",status)
    setLongField(env, currentGPUDevice, "gpuBufferSize", (jlong)gpuBufferSize);
    
    cuCtxDestroy(cuContext);
    return;
}

/*
 * Class:     edu_syr_pcpratts_rootbeer_runtime2_cuda_CudaRuntime2
 * Method:    freeCurrentGpuCard
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_edu_syr_pcpratts_rootbeer_runtime2_cuda_CudaRuntime2_freeCurrentGpuCard
(JNIEnv * env, jobject this_obj) {

    int deviceCount = 0;
    int status = cuDeviceGetCount(&deviceCount);
    CHECK_STATUS(env,"error in cuDeviceGetCount",status)
    
    // Get the current GPU Device and ID
    jobject currentGPUDevice = getCurrentGPUDevice(env);
    jint currentGPUDeviceID = getIntField(env, currentGPUDevice, "cardID");
    
    CUdevice cuDevice;
    status = cuDeviceGet(&cuDevice, currentGPUDeviceID);
    CHECK_STATUS(env,"error in cuDeviceGet",status)
    
    CUcontext cuContext;
    status = cuCtxCreate(&cuContext, CU_CTX_MAP_HOST, cuDevice);
    CHECK_STATUS(env,"error in cuCtxCreate",status)
    
    void * toSpace = (void *) getLongField(env, currentGPUDevice, "toSpaceAddr");
    cuMemFreeHost(toSpace);
    
    CUdeviceptr gpuToSpace = getLongField(env, currentGPUDevice, "gpuToSpaceAddr");
    cuMemFree(gpuToSpace);
    
    CUdeviceptr gpuClassMemory = getLongField(env, currentGPUDevice, "gpuClassAddr");
    cuMemFree(gpuClassMemory);
    
    void * handlesMemory = (void *)getLongField(env, currentGPUDevice, "handlesAddr");
    cuMemFreeHost(handlesMemory);
    
    CUdeviceptr gpuHandlesMemory = getLongField(env, currentGPUDevice, "gpuHandlesAddr");
    cuMemFree(gpuHandlesMemory);
    
    void * exceptionsMemory = (void *)getLongField(env, currentGPUDevice, "exceptionsHandlesAddr");
    cuMemFreeHost(exceptionsMemory);
    
    CUdeviceptr gpuExceptionsMemory = getLongField(env, currentGPUDevice, "gpuExceptionsHandlesAddr");
    cuMemFree(gpuExceptionsMemory);
    
    CUdeviceptr gcInfoSpace = getLongField(env, currentGPUDevice, "gcInfoSpace");
    cuMemFree(gcInfoSpace);
    
    CUdeviceptr gpuHeapEndPtr = getLongField(env, currentGPUDevice, "gpuHeapEndPtr");
    cuMemFree(gpuHeapEndPtr);
    
    CUdeviceptr gpuBufferSize = getLongField(env, currentGPUDevice, "gpuBufferSize");
    cuMemFree(gpuBufferSize);
    
    cuCtxDestroy(cuContext);
     
    return;
}
/*
 * Class:     edu_syr_pcpratts_rootbeer_runtime2_cuda_CudaRuntime2
 * Method:    printDeviceInfo
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_edu_syr_pcpratts_rootbeer_runtime2_cuda_CudaRuntime2_printDeviceInfo
  (JNIEnv *env, jclass cls)
{
    int i, a=0, b=0, status;
    int num_devices = 0;
    char str[1024];
    size_t free_mem, total_mem;
 
    status = cuInit(0);
    CHECK_STATUS(env,"error in cuInit",status)
    
    cuDeviceGetCount(&num_devices);
    printf("%d cuda gpus found\n", num_devices);
 
    for (i = 0; i < num_devices; ++i)
    {
        CUdevice dev;
        status = cuDeviceGet(&dev, i);
        CHECK_STATUS(env,"error in cuDeviceGet",status)

        CUcontext cuContext;
        status = cuCtxCreate(&cuContext, CU_CTX_MAP_HOST, dev);
        CHECK_STATUS(env,"error in cuCtxCreate",status)
                
        printf("\nGPU:%d\n", i);
        
        if(cuDeviceComputeCapability(&a, &b, dev) == CUDA_SUCCESS)
            printf("Version:                       %i.%i\n", a, b);
        
        if(cuDeviceGetName(str,1024,dev) == CUDA_SUCCESS)
            printf("Name:                          %s\n", str);
        
        if(cuMemGetInfo(&free_mem, &total_mem) == CUDA_SUCCESS){
          #if (defined linux || defined __APPLE_CC__)
            printf("Total global memory:           %zu/%zu (Free/Total) MBytes\n", free_mem/1024/1024, total_mem/1024/1024);
          #else
            printf("Total global memory:           %Iu/%Iu (Free/Total) MBytes\n", free_mem/1024/1024, total_mem/1024/1024);
          #endif
        }
        
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,dev) == CUDA_SUCCESS)
            printf("Total registers per block:     %i\n", a);
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_WARP_SIZE,dev) == CUDA_SUCCESS)
            printf("Warp size:                     %i\n", a);
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_MAX_PITCH,dev) == CUDA_SUCCESS)
            printf("Maximum memory pitch:          %i\n", a);
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,dev) == CUDA_SUCCESS)
            printf("Maximum threads per block:     %i\n", a);        
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,dev) == CUDA_SUCCESS)
            printf("Total shared memory per block  %.2f KB\n", a/1024.0);
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_CLOCK_RATE,dev) == CUDA_SUCCESS)
            printf("Clock rate:                    %.2f MHz\n",  a/1000000.0);
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,dev) == CUDA_SUCCESS)
            printf("Memory Clock rate:             %.2f\n",  a/1000000.0);
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY,dev) == CUDA_SUCCESS)
            printf("Total constant memory:         %.2f MB\n",  a/1024.0/1024.0);
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_INTEGRATED,dev) == CUDA_SUCCESS)
            printf("Integrated:                    %i\n",  a);
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,dev) == CUDA_SUCCESS)
            printf("Max threads per multiprocessor:%i\n",  a);    
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,dev) == CUDA_SUCCESS)
            printf("Number of multiprocessors:     %i\n",  a);    
      
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,dev) == CUDA_SUCCESS)
            printf("Maximum dimension x of block:  %i\n", a);
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,dev) == CUDA_SUCCESS)
            printf("Maximum dimension y of block:  %i\n", a);
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,dev) == CUDA_SUCCESS)
            printf("Maximum dimension z of block:  %i\n", a);
        
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,dev) == CUDA_SUCCESS)
            printf("Maximum dimension x of grid:   %i\n", a);
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,dev) == CUDA_SUCCESS)
            printf("Maximum dimension y of grid:   %i\n", a);
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z,dev) == CUDA_SUCCESS)
            printf("Maximum dimension z of grid:   %i\n", a);
			
        cuCtxDestroy(cuContext);
    } 
	
	return;
}


void * readCubinFile(const char * filename){

  int i;
  jlong size;
  char * ret;
  int block_size;
  int num_blocks;
  int leftover;
  char * dest;
  
  FILE * file = fopen(filename, "r");
  fseek(file, 0, SEEK_END);
  size = ftell(file);
  fseek(file, 0, SEEK_SET);

  ret = (char *) malloc(size);
  block_size = 4096;
  num_blocks = (int) (size / block_size);
  leftover = (int) (size % block_size);

  dest = ret;
  for(i = 0; i < num_blocks; ++i){
    fread(dest, 1, block_size, file);
    dest += block_size;
  }
  if(leftover != 0){
    fread(dest, 1, leftover, file);
  }

  fclose(file);
  return (void *) ret;
}

void * readCubinFileFromBuffers(JNIEnv *env, jobject buffers, jint size, jint total_size){
  int i, j;
  int dest_offset = 0;
  int len;
  char * data;
  char * ret = (char *) malloc(total_size);

  jclass cls = (*env)->GetObjectClass(env, buffers);
  jmethodID mid = (*env)->GetMethodID(env, cls, "get", "(I)Ljava/lang/Object;");
  for(i = 0; i < size; ++i){
    jobject buffer = (*env)->CallObjectMethod(env, buffers, mid, i);
    jbyteArray * arr = (jbyteArray*) &buffer;
    len = (*env)->GetArrayLength(env, *arr);
    data = (*env)->GetByteArrayElements(env, *arr, NULL);
    memcpy((void *) (ret + dest_offset), (void *) data, len);
    dest_offset += len;
    (*env)->ReleaseByteArrayElements(env, *arr, data, 0);
  }

  return (void *) ret;
}

/*
 * Class:     edu_syr_pcpratts_rootbeer_runtime2_cuda_CudaRuntime2
 * Method:    writeClassTypeRef
 * Signature: ([I)V
 */
JNIEXPORT void JNICALL Java_edu_syr_pcpratts_rootbeer_runtime2_cuda_CudaRuntime2_writeClassTypeRef
  (JNIEnv *env, jobject this_ref, jintArray jarray)
{
  int i;
  jint * native_array = (*env)->GetIntArrayElements(env, jarray, 0);
  
  // Get the current GPU Device and gpuClassMemory
  jobject currentGPUDevice = getCurrentGPUDevice(env);
  CUdeviceptr gpuClassMemory = getLongField(env, currentGPUDevice, "gpuClassAddr");
  
  cuMemcpyHtoD(gpuClassMemory, native_array, classMemSize);
    
  (*env)->ReleaseIntArrayElements(env, jarray, native_array, 0);
  
  return;
}

/*
 * Class:     edu_syr_pcpratts_rootbeer_runtime2_cuda_CudaRuntime2
 * Method:    loadFunction
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_edu_syr_pcpratts_rootbeer_runtime2_cuda_CudaRuntime2_loadFunction
  (JNIEnv *env, jobject this_obj, jlong heap_end_ptr, jobject buffers, jint size, jint total_size, jint num_blocks){

  void * fatcubin;
  int offset;
  CUresult status;
  char * native_filename;
  heapEndPtr = heap_end_ptr;
  
  // Get the current GPU Device and ID
  jobject currentGPUDevice = getCurrentGPUDevice(env);
  jint currentGPUDeviceID = getIntField(env, currentGPUDevice, "cardID");
      
  CUdevice cuDevice;
  status = cuDeviceGet(&cuDevice, currentGPUDeviceID);
  CHECK_STATUS(env,"error in cuDeviceGet",status)
      
  CUcontext cuContext;
  status = cuCtxCreate(&cuContext, CU_CTX_MAP_HOST, cuDevice);
  CHECK_STATUS(env,"error in cuCtxCreate",status)

  cuCtxPushCurrent(cuContext);
      
  fatcubin = readCubinFileFromBuffers(env, buffers, size, total_size);
  status = cuModuleLoadFatBinary(&cuModule, fatcubin);
  CHECK_STATUS(env, "error in cuModuleLoad", status);
  free(fatcubin);

  status = cuModuleGetFunction(&cuFunction, cuModule, "_Z5entryPcS_PiPxS1_S0_S0_i"); 
  CHECK_STATUS(env,"error in cuModuleGetFunction",status)

  status = cuFuncSetCacheConfig(cuFunction, CU_FUNC_CACHE_PREFER_L1);
  CHECK_STATUS(env,"error in cuFuncSetCacheConfig",status)

  status = cuParamSetSize(cuFunction, (7 * sizeof(CUdeviceptr) + sizeof(int))); 
  CHECK_STATUS(env,"error in cuParamSetSize",status)

  offset = 0;
  CUdeviceptr gcInfoSpace = getLongField(env, currentGPUDevice, "gcInfoSpace");
  status = cuParamSetv(cuFunction, offset, (void *) &gcInfoSpace, sizeof(CUdeviceptr)); 
  CHECK_STATUS(env,"error in cuParamSetv gcInfoSpace",status)
  offset += sizeof(CUdeviceptr);

  CUdeviceptr gpuToSpace = getLongField(env, currentGPUDevice, "gpuToSpaceAddr");
  status = cuParamSetv(cuFunction, offset, (void *) &gpuToSpace, sizeof(CUdeviceptr)); 
  CHECK_STATUS(env,"error in cuParamSetv gpuToSpace",status)
  offset += sizeof(CUdeviceptr);

  CUdeviceptr gpuHandlesMemory = getLongField(env, currentGPUDevice, "gpuHandlesAddr");
  status = cuParamSetv(cuFunction, offset, (void *) &gpuHandlesMemory, sizeof(CUdeviceptr)); 
  CHECK_STATUS(env,"error in cuParamSetv gpuHandlesMemory %",status)
  offset += sizeof(CUdeviceptr);

  CUdeviceptr gpuHeapEndPtr = getLongField(env, currentGPUDevice, "gpuHeapEndPtr");
  status = cuParamSetv(cuFunction, offset, (void *) &gpuHeapEndPtr, sizeof(CUdeviceptr)); 
  CHECK_STATUS(env,"error in cuParamSetv gpuHeapEndPtr",status)
  offset += sizeof(CUdeviceptr);

  CUdeviceptr gpuBufferSize = getLongField(env, currentGPUDevice, "gpuBufferSize");
  status = cuParamSetv(cuFunction, offset, (void *) &gpuBufferSize, sizeof(CUdeviceptr));
  CHECK_STATUS(env,"error in cuParamSetv gpuBufferSize",status)
  offset += sizeof(CUdeviceptr); 

  CUdeviceptr gpuExceptionsMemory = getLongField(env, currentGPUDevice, "gpuExceptionsHandlesAddr");
  status = cuParamSetv(cuFunction, offset, (void *) &gpuExceptionsMemory, sizeof(CUdeviceptr)); 
  CHECK_STATUS(env,"error in cuParamSetv gpuExceptionsMemory",status)
  offset += sizeof(CUdeviceptr);

  CUdeviceptr gpuClassMemory = getLongField(env, currentGPUDevice, "gpuClassAddr");
  status = cuParamSetv(cuFunction, offset, (void *) &gpuClassMemory, sizeof(CUdeviceptr)); 
  CHECK_STATUS(env,"error in cuParamSetv gpuClassMemory",status)
  offset += sizeof(CUdeviceptr);

  jlong numBlocks = getLongField(env, currentGPUDevice, "numBlocks");
  status = cuParamSeti(cuFunction, offset, num_blocks); 
  CHECK_STATUS(env,"error in cuParamSetv num_blocks",status)
  offset += sizeof(int);
      
  cuCtxPopCurrent(&cuContext);
  cuCtxDestroy(cuContext);
      
  return;
}

/*
 * Class:     edu_syr_pcpratts_rootbeer_runtime2_cuda_CudaRuntime2
 * Method:    runBlocks
 * Signature: (I)V
 */
JNIEXPORT jint JNICALL Java_edu_syr_pcpratts_rootbeer_runtime2_cuda_CudaRuntime2_runBlocks
  (JNIEnv *env, jobject this_obj, jint num_blocks, jint block_shape, jint grid_shape){

  CUresult status;
  jlong * infoSpace = (jlong *) malloc(gc_space_size);
  infoSpace[1] = heapEndPtr;
      
  // Get the current GPU Device and ID
  jobject currentGPUDevice = getCurrentGPUDevice(env);
  jint currentGPUDeviceID = getIntField(env, currentGPUDevice, "cardID");
      
  CUdevice cuDevice;
  status = cuDeviceGet(&cuDevice, currentGPUDeviceID);
  CHECK_STATUS(env,"error in cuDeviceGet",status)
      
  CUcontext cuContext;
  status = cuCtxCreate(&cuContext, CU_CTX_MAP_HOST, cuDevice);
  CHECK_STATUS(env,"error in cuCtxCreate",status)

  void * toSpace = (void *) getLongField(env, currentGPUDevice, "toSpaceAddr");
  CUdeviceptr gpuToSpace = getLongField(env, currentGPUDevice, "gpuToSpaceAddr");
  void * handlesMemory = (void *)getLongField(env, currentGPUDevice, "handlesAddr");
  CUdeviceptr gpuHandlesMemory = getLongField(env, currentGPUDevice, "gpuHandlesAddr");
  void * exceptionsMemory = (void *)getLongField(env, currentGPUDevice, "exceptionsHandlesAddr");
  CUdeviceptr gpuExceptionsMemory = getLongField(env, currentGPUDevice, "gpuExceptionsHandlesAddr");
  CUdeviceptr gcInfoSpace = getLongField(env, currentGPUDevice, "gcInfoSpace");
  CUdeviceptr gpuHeapEndPtr = getLongField(env, currentGPUDevice, "gpuHeapEndPtr");
  CUdeviceptr gpuBufferSize = getLongField(env, currentGPUDevice, "gpuBufferSize");
  jlong bufferSize = getLongField(env, currentGPUDevice, "toSpaceSize");
      
  cuCtxPushCurrent(cuContext);

  cuMemcpyHtoD(gcInfoSpace, infoSpace, gc_space_size);
  cuMemcpyHtoD(gpuToSpace, toSpace, heapEndPtr);
  //cuMemcpyHtoD(gpuTexture, textureMemory, textureMemSize);
  cuMemcpyHtoD(gpuHandlesMemory, handlesMemory, num_blocks * sizeof(jlong));
  cuMemcpyHtoD(gpuHeapEndPtr, &heapEndPtr, sizeof(jlong));
  cuMemcpyHtoD(gpuBufferSize, &bufferSize, sizeof(jlong));
  
/*
  status = cuModuleGetTexRef(&cache, cuModule, "m_Cache");  
  if (CUDA_SUCCESS != status) 
  {
    printf("error in cuModuleGetTexRef %d\n", status);
  }

  status = cuTexRefSetAddress(0, cache, gpuTexture, textureMemSize);
  if (CUDA_SUCCESS != status) 
  {
    printf("error in cuTextRefSetAddress %d\n", status);
  }
*/

  status = cuFuncSetBlockShape(cuFunction, block_shape, 1, 1);
  if(status != CUDA_SUCCESS){
    free(infoSpace);
    cuCtxPopCurrent(&cuContext);
  }
  CHECK_STATUS_RTN(env,"error in cuFuncSetBlockShape",status, (jint)status);

  status = cuLaunchGrid(cuFunction, grid_shape, 1);
  if(status != CUDA_SUCCESS){
    free(infoSpace);
    cuCtxPopCurrent(&cuContext);
  }
  CHECK_STATUS_RTN(env,"error in cuLaunchGrid",status, (jint)status)

  status = cuCtxSynchronize();  
  if(status != CUDA_SUCCESS){
    free(infoSpace);
    cuCtxPopCurrent(&cuContext);
  }
  CHECK_STATUS_RTN(env,"error in cuCtxSynchronize",status, (jint)status)
  
  cuMemcpyDtoH(infoSpace, gcInfoSpace, gc_space_size);
  heapEndPtr = infoSpace[1];
  cuMemcpyDtoH(toSpace, gpuToSpace, heapEndPtr);
  cuMemcpyDtoH(exceptionsMemory, gpuExceptionsMemory, num_blocks * sizeof(jlong));
  free(infoSpace);
      
  cuCtxPopCurrent(&cuContext);
  cuCtxDestroy(cuContext);
      
  return 0;
}

/*
 * Class:     edu_syr_pcpratts_rootbeer_runtime2_cuda_CudaRuntime2
 * Method:    unload
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_edu_syr_pcpratts_rootbeer_runtime2_cuda_CudaRuntime2_unload
  (JNIEnv *env, jobject this_obj){

  cuModuleUnload(cuModule);
  cuFunction = (CUfunction) 0;  
 
  return;
}


