#include "edu_syr_pcpratts_rootbeer_runtime2_cuda_CudaRuntime2.h"

#include <assert.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK_STATUS(env,this_obj,msg,status) \
if (CUDA_SUCCESS != status) {\
  throw_cuda_error_exception(env,this_obj,msg,status);\
  return;\
}

#define CHECK_STATUS_RTN(env,this_obj,msg,status,rtn) \
if (CUDA_SUCCESS != status) {\
  throw_cuda_error_exception(env,this_obj,msg,status);\
  return rtn;\
}

#define GPUCARD_CLASS_NAME "edu/syr/pcpratts/rootbeer/runtime/GpuCard"
#define GPUCARD_CLASS_NAME_SIG "Ledu/syr/pcpratts/rootbeer/runtime/GpuCard;"
#define CUDA_ERROR_EXCEPTION_CLASS_NAME "edu/syr/pcpratts/rootbeer/runtime2/cuda/CudaErrorException"


// Global Variables
static jlong heapEndPtr;

static size_t gc_space_size = 1024;
static jlong classMemSize = sizeof(jint)*100; //space for 100 types in the scene

//static int textureMemSize;

static CUmodule cuModule;
static CUfunction cuFunction;

static CUtexref    cache;


/**
 * JNI IntField Setter
 */
jint getIntField(JNIEnv *env, jobject obj, const char * name){
    
    jclass c = (*env)->GetObjectClass(env, obj);
    jfieldID fid = (*env)->GetFieldID(env, c, name, "I");
    jint value = (*env)->GetIntField(env, obj, fid);
    return value;
}

/**
 * JNI LongField Setter
 */
void setLongField(JNIEnv *env, jobject obj, const char * name, jlong value){
    
    jclass c = (*env)->GetObjectClass(env, obj);
    jfieldID fid = (*env)->GetFieldID(env, c, name, "J");
    (*env)->SetLongField(env, obj, fid, value);
    return;
}

/**
 * JNI LongField Getter
 */
jlong getLongField(JNIEnv *env, jobject obj, const char * name){
    
    jclass c = (*env)->GetObjectClass(env, obj);
    jfieldID fid = (*env)->GetFieldID(env, c, name, "J");
    jlong value = (*env)->GetLongField(env, obj, fid);
    return value;
}

/**
 * JNI ObjectField Setter
 */
void setObjectField(JNIEnv *env, jobject obj, const char * name, jobject value, const char * valueClass){
    
    jclass c = (*env)->GetObjectClass(env, obj);
    jfieldID fid = (*env)->GetFieldID(env, c, name, valueClass);
    (*env)->SetObjectField(env, obj, fid, value);
    return;
}

/**
 * JNI ObjectField Getter
 */
jobject getObjectField(JNIEnv *env, jobject obj, const char * name, const char * valueClass){
    
    jclass c = (*env)->GetObjectClass(env, obj);
    jfieldID fid = (*env)->GetFieldID(env, c, name, valueClass);
    jobject value = (*env)->GetObjectField(env, obj, fid);
    return value;
}

/**
 * Returns the current GPU Device
 */
jobject getCurrentGPUDevice(JNIEnv *env, jobject this_obj){
    return getObjectField(env, this_obj, "currentGpuCard", GPUCARD_CLASS_NAME_SIG);
}

/**
 * Sets the current GPU Device
 */
void setCurrentGPUDevice(JNIEnv *env, jobject this_obj, jobject gpuDevice){
    setObjectField(env, this_obj, "currentGpuCard", gpuDevice, GPUCARD_CLASS_NAME_SIG);
}


/**
 * Throws a edu/syr/pcpratts/rootbeer/runtime2/cuda/CudaErrorException
 * without checking error code and GPU detailed information
 */
void throw_error_exception(JNIEnv *env, const char *message, int error) {
    char msg[1024];
    jclass exp;
    jfieldID fid;
    
    if(error == CUDA_SUCCESS){
        return;
    }
    exp = (*env)->FindClass(env,CUDA_ERROR_EXCEPTION_CLASS_NAME);
    
    sprintf(msg, "ERROR STATUS:%i : %.900s", error, message);
    
    fid = (*env)->GetFieldID(env, exp, "cudaError_enum", "I");
    (*env)->SetLongField(env, exp, fid, (jint)error);
    
    (*env)->ThrowNew(env,exp,msg);
    return;
}

/**
 * Throws a edu/syr/pcpratts/rootbeer/runtime2/cuda/CudaErrorException
 * WITH checking error code and GPU detailed information
 */
void throw_cuda_error_exception(JNIEnv *env, jobject this_obj, const char *message, int error) {
  char msg[1024];
  jclass exp;
  jfieldID fid;
  int a = 0;
  int b = 0;
  char name[1024];
  jobject currentGPUDevice;
  jint currentGPUDeviceID;
  CUdevice cuDevice;
  int status;

  if(error == CUDA_SUCCESS){
    return;
  }

  // Get the current GPU Device and ID
  currentGPUDevice = getCurrentGPUDevice(env,this_obj);
  currentGPUDeviceID = getIntField(env, currentGPUDevice, "cardID");
    
  cuDevice;
  status = cuDeviceGet(&cuDevice, currentGPUDeviceID);
  if (CUDA_SUCCESS != status) {
    throw_error_exception(env, "error in cuDeviceGet", status);
    return;
  }
    
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
    // Global temp variables
    int i, a=0, b=0, status;
    long j;
    // For searching the best GPU Device
    int max_multiprocessors = -1;
    jobject bestGpuCard = NULL;

    // GPU Property variables
    char name[1024];
    int computeCapabilityA=0, computeCapabilityB=0;
    size_t free_mem, total_mem;
    //int free_mem_Mbytes=0, total_mem_Mbytes=0;
    int max_registers_per_block=0;
    int warp_size=0;
    int max_pitch=0;
    int max_threads_per_block=0;
    long max_shared_memory_per_block=0;
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
    
	jclass arrayListClass;
	jmethodID arrayListCons;
	jmethodID arrayListAdd;
	jobject gpuCardList;
	jclass gpuCardClass;
	jmethodID gpuCardCons;
	CUdevice cuDevice;
	CUcontext cuContext;
	jlong* reserve_memory_list;
	
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
	size_t temp_size;
	jobject gpuCardObject;
	jboolean jbool;

    // ArrayList Class and Constructor and Add Method
    arrayListClass = (*env)->FindClass(env, "java/util/ArrayList");
    arrayListCons =  (*env)->GetMethodID(env, arrayListClass,
                                                   "<init>", "()V");
    arrayListAdd = (*env)->GetMethodID(env, arrayListClass,
                                                 "add", "(Ljava/lang/Object;)Z");
    //jmethodID arrayListGet= (*env)->GetMethodID(env, arrayListClass,
    //                                             "get", "(I)Ljava/lang/Object;");
    
    // Create new ArrayList<Object>
    gpuCardList = (*env)->NewObject(env, arrayListClass, arrayListCons);
    
    // GpuCard Class and Constructor
    gpuCardClass = (*env)->FindClass(env,GPUCARD_CLASS_NAME);
    gpuCardCons = (*env)->GetMethodID(env, gpuCardClass,
                                                "<init>", "(ILjava/lang/String;IIJJIIIIJFFFIIIIIIIIIJJJ)V");
    if (gpuCardCons == NULL) return NULL;
    
    // Initializes the driver API
    status = cuInit(0);
    CHECK_STATUS_RTN(env,this_obj,"error in cuInit",status, 0);
 
    // Get number of CUDA devices
    cuDeviceGetCount(&num_devices);
    CHECK_STATUS_RTN(env,this_obj,"error in cuDeviceGetCount",status, 0);
    
    if(num_devices == 0)
        throw_cuda_error_exception(env,this_obj,"0 Cuda Devices were found",0);
    
    for (i = 0; i < num_devices; ++i)
    {
        status = cuDeviceGet(&cuDevice, i);
        CHECK_STATUS_RTN(env,this_obj,"error in cuDeviceGet",status, 0)
        
        status = cuCtxCreate(&cuContext, CU_CTX_MAP_HOST, cuDevice);
        CHECK_STATUS_RTN(env,this_obj,"error in cuCtxCreate",status, 0)
        
        if(cuDeviceComputeCapability(&a, &b, cuDevice) == CUDA_SUCCESS){
            computeCapabilityA = a;
            computeCapabilityB = b;
        }
        
        cuDeviceGetName(name,1024,cuDevice);
        
        cuMemGetInfo(&free_mem, &total_mem);
        /*
        free_mem_Mbytes = free_mem/1024/1024;
        total_mem_Mbytes = total_mem/1024/1024;
        */
        
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,cuDevice) == CUDA_SUCCESS)
            max_registers_per_block = a;
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_WARP_SIZE,cuDevice) == CUDA_SUCCESS)
            warp_size = a;
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_MAX_PITCH,cuDevice) == CUDA_SUCCESS)
            max_pitch = a;
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,cuDevice) == CUDA_SUCCESS)
            max_threads_per_block = a;
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,cuDevice) == CUDA_SUCCESS)
            max_shared_memory_per_block = a; // /1024.0;
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_CLOCK_RATE,cuDevice) == CUDA_SUCCESS)
            clock_rate = a/1000000.0;
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,cuDevice) == CUDA_SUCCESS)
            memory_clock_rate = a/1000000.0;
        if(cuDeviceGetAttribute(&a, CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY,cuDevice) == CUDA_SUCCESS)
            total_constant_memory = a; // /1024.0/1024.0;
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
            reserve_memory_list = (*env)->GetLongArrayElements(env, _reserve_mem_list, NULL);
            if (reserve_memory_list == NULL) return NULL;
            
            reserve_mem = reserve_memory_list[i];
            to_space_size -= reserve_mem;
        }
        
        // Try to find reserve_mem if no reserve_mem was configured or invalid
        if ((_reserve_mem_list == NULL) || (reserve_mem == 0)) {
            
            for(j = 1024L*1024L; j < to_space_size; j += 100L*1024L*1024L){
                temp_size = to_space_size - j;
                
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
                throw_cuda_error_exception(env, this_obj, "unable to find enough space using CUDA", 0);
        }
        
        gpuCardObject = (*env)->NewObject(env, gpuCardClass,
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
        
        // Find best GPU Device
        if(multiprocessor_count > max_multiprocessors) {
            max_multiprocessors = multiprocessor_count;
            bestGpuCard = gpuCardObject;
        }
        
        jbool = (*env)->CallBooleanMethod(env, gpuCardList,
                                               arrayListAdd, gpuCardObject);
        
        cuCtxDestroy(cuContext);
    }
    
    // Set best GPU Device to currentGpuCard
    setCurrentGPUDevice(env, this_obj, bestGpuCard);
    
	return gpuCardList;
}


/*
 * Class:     edu_syr_pcpratts_rootbeer_runtime2_cuda_CudaRuntime2
 * Method:    initCurrentGpuCard
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_edu_syr_pcpratts_rootbeer_runtime2_cuda_CudaRuntime2_initCurrentGpuCard
(JNIEnv * env, jobject this_obj) {

    int status;
	jobject currentGPUDevice;
	jint currentGPUDeviceID;
	CUdevice cuDevice;
    CUcontext cuContext;
	jlong numBlocks;
	jlong to_space_size;
    void * toSpace;
    void * handlesMemory;
    CUdeviceptr gpuHandlesMemory;
    void * exceptionsMemory;
    CUdeviceptr gpuExceptionsMemory;
    CUdeviceptr gcInfoSpace;
    CUdeviceptr gpuHeapEndPtr;
    CUdeviceptr gpuBufferSize;
    CUdeviceptr gpuToSpace;
    CUdeviceptr gpuClassMemory;
    
    // Get the current GPU Device and ID
    currentGPUDevice = getCurrentGPUDevice(env, this_obj);
    currentGPUDeviceID = getIntField(env, currentGPUDevice, "cardID");    
    // DEBUG Info
    //printf("STARTED: initCurrentGpuCard with currentGPUDeviceID: %d\n",currentGPUDeviceID);
    
    status = cuDeviceGet(&cuDevice, currentGPUDeviceID);
    CHECK_STATUS(env,this_obj,"error in cuDeviceGet",status)
    
    status = cuCtxCreate(&cuContext, CU_CTX_MAP_HOST, cuDevice);
    CHECK_STATUS(env,this_obj,"error in cuCtxCreate",status)
    
    // Context Management
    // Save cuContext in currentGPUDevice
    setLongField(env, currentGPUDevice, "cudaContext", (jlong)cuContext);
    
    //size_t f_mem = getLongField(env, currentGPUDevice, "freeMemory");
    //size_t t_mem = getLongField(env, currentGPUDevice, "totalMemory");
    numBlocks = getLongField(env, currentGPUDevice, "numBlocks");
    to_space_size = getLongField(env, currentGPUDevice, "toSpaceSize");
    
    status = cuMemHostAlloc(&toSpace, to_space_size, 0);
    CHECK_STATUS(env,this_obj,"toSpace memory allocation failed",status)
    setLongField(env, currentGPUDevice, "toSpaceAddr", (jlong)toSpace);
    
    status = cuMemAlloc(&gpuToSpace, to_space_size);
    CHECK_STATUS(env,this_obj,"gpuToSpace memory allocation failed",status)
    setLongField(env, currentGPUDevice, "gpuToSpaceAddr", (jlong)gpuToSpace);
    
    status = cuMemAlloc(&gpuClassMemory, classMemSize);
    CHECK_STATUS(env,this_obj,"gpuClassMemory memory allocation failed",status)
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
    
    status = cuMemHostAlloc(&handlesMemory, numBlocks * sizeof(jlong), CU_MEMHOSTALLOC_WRITECOMBINED);
    CHECK_STATUS(env,this_obj,"handlesMemory memory allocation failed",status)
    setLongField(env, currentGPUDevice, "handlesAddr", (jlong)handlesMemory);
    
    status = cuMemAlloc(&gpuHandlesMemory, numBlocks * sizeof(jlong));
    CHECK_STATUS(env,this_obj,"gpuHandlesMemory memory allocation failed",status)
    setLongField(env, currentGPUDevice, "gpuHandlesAddr", (jlong)gpuHandlesMemory);
    
    status = cuMemHostAlloc(&exceptionsMemory, numBlocks * sizeof(jlong), 0);
    CHECK_STATUS(env,this_obj,"exceptionsMemory memory allocation failed",status)
    setLongField(env, currentGPUDevice, "exceptionsHandlesAddr", (jlong)exceptionsMemory);
    
    status = cuMemAlloc(&gpuExceptionsMemory, numBlocks * sizeof(jlong));
    CHECK_STATUS(env,this_obj,"gpuExceptionsMemory memory allocation failed",status)
    setLongField(env, currentGPUDevice, "gpuExceptionsHandlesAddr", (jlong)gpuExceptionsMemory);
    
    status = cuMemAlloc(&gcInfoSpace, gc_space_size);
    CHECK_STATUS(env,this_obj,"gcInfoSpace memory allocation failed",status)
    setLongField(env, currentGPUDevice, "gcInfoSpace", (jlong)gcInfoSpace);
    
    status = cuMemAlloc(&gpuHeapEndPtr, 8);
    CHECK_STATUS(env,this_obj,"gpuHeapEndPtr memory allocation failed",status)
    setLongField(env, currentGPUDevice, "gpuHeapEndPtr", (jlong)gpuHeapEndPtr);
    
    status = cuMemAlloc(&gpuBufferSize, 8);
    CHECK_STATUS(env,this_obj,"gpuBufferSize memory allocation failed",status)
    setLongField(env, currentGPUDevice, "gpuBufferSize", (jlong)gpuBufferSize);
    
    return;
}


/*
 * Class:     edu_syr_pcpratts_rootbeer_runtime2_cuda_CudaRuntime2
 * Method:    freeCurrentGpuCard
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_edu_syr_pcpratts_rootbeer_runtime2_cuda_CudaRuntime2_freeCurrentGpuCard
(JNIEnv * env, jobject this_obj) {
    
    // Get the current GPU Device and ID
    jobject currentGPUDevice = getCurrentGPUDevice(env,this_obj);
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
	long cuContextPointer;
	CUcontext cuContext;
    
    // DEBUG INFO
    //jint currentGPUDeviceID = getIntField(env, currentGPUDevice, "cardID");
    //printf("STARTED: freeCurrentGpuCard with currentGPUDeviceID: %d\n",currentGPUDeviceID);
    
    toSpace = (void *) getLongField(env, currentGPUDevice, "toSpaceAddr");
    cuMemFreeHost(toSpace);
    setLongField(env, currentGPUDevice, "toSpaceAddr", 0);
    
    gpuToSpace = getLongField(env, currentGPUDevice, "gpuToSpaceAddr");
    cuMemFree(gpuToSpace);
    setLongField(env, currentGPUDevice, "gpuToSpaceAddr", 0);
    
    gpuClassMemory = getLongField(env, currentGPUDevice, "gpuClassAddr");
    cuMemFree(gpuClassMemory);
    setLongField(env, currentGPUDevice, "gpuClassAddr", 0);
    
    handlesMemory = (void *)getLongField(env, currentGPUDevice, "handlesAddr");
    cuMemFreeHost(handlesMemory);
    setLongField(env, currentGPUDevice, "handlesAddr", 0);
    
    gpuHandlesMemory = getLongField(env, currentGPUDevice, "gpuHandlesAddr");
    cuMemFree(gpuHandlesMemory);
    setLongField(env, currentGPUDevice, "gpuHandlesAddr", 0);
    
    exceptionsMemory = (void *)getLongField(env, currentGPUDevice, "exceptionsHandlesAddr");
    cuMemFreeHost(exceptionsMemory);
    setLongField(env, currentGPUDevice, "exceptionsHandlesAddr", 0);
    
    gpuExceptionsMemory = getLongField(env, currentGPUDevice, "gpuExceptionsHandlesAddr");
    cuMemFree(gpuExceptionsMemory);
    setLongField(env, currentGPUDevice, "gpuExceptionsHandlesAddr", 0);
    
    gcInfoSpace = getLongField(env, currentGPUDevice, "gcInfoSpace");
    cuMemFree(gcInfoSpace);
    setLongField(env, currentGPUDevice, "gcInfoSpace", 0);
    
    gpuHeapEndPtr = getLongField(env, currentGPUDevice, "gpuHeapEndPtr");
    cuMemFree(gpuHeapEndPtr);
    setLongField(env, currentGPUDevice, "gpuHeapEndPtr", 0);
    
    gpuBufferSize = getLongField(env, currentGPUDevice, "gpuBufferSize");
    cuMemFree(gpuBufferSize);
    setLongField(env, currentGPUDevice, "gpuBufferSize", 0);
    
    // Get cuContext from currentGPUDevice
    cuContextPointer = getLongField(env, currentGPUDevice, "cudaContext");
    if (cuContextPointer == 0)
      throw_error_exception(env, "cudaContext was not set in currentGPUDevice!", CUDA_ERROR_INVALID_VALUE);
    
    cuContext = (CUcontext)cuContextPointer;
    
    cuCtxDestroy(cuContext);
    setLongField(env, currentGPUDevice, "cudaContext", 0);
    
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
  (JNIEnv *env, jobject this_obj, jintArray jarray)
{
    
  // Get the current GPU Device
  jobject currentGPUDevice = getCurrentGPUDevice(env,this_obj);

  // DEBUG INFO
  //jint currentGPUDeviceID = getIntField(env, currentGPUDevice, "cardID");
  //printf("STARTED: writeClassTypeRef with currentGPUDeviceID: %d\n",currentGPUDeviceID);
    
  CUdeviceptr gpuClassMemory = getLongField(env, currentGPUDevice, "gpuClassAddr");
  //printf("STARTED: writeClassTypeRef with gpuClassMemory: %ld\n", gpuClassMemory);

  jint * native_array = (*env)->GetIntArrayElements(env, jarray, 0);
  cuMemcpyHtoD(gpuClassMemory, native_array, classMemSize);
  (*env)->ReleaseIntArrayElements(env, jarray, native_array, 0);
  
  return;
}


/*
 * Class:     edu_syr_pcpratts_rootbeer_runtime2_cuda_CudaRuntime2
 * Method:    loadFunction
 * Signature: (JLjava/lang/Object;III)V
 */
JNIEXPORT void JNICALL Java_edu_syr_pcpratts_rootbeer_runtime2_cuda_CudaRuntime2_loadFunction
  (JNIEnv *env, jobject this_obj, jlong heap_end_ptr, jobject buffers, jint size, jint total_size, jint num_blocks){

  void * fatcubin;
  int offset;
  CUresult status;
  char * native_filename;
  jobject currentGPUDevice;
  long cuContextPointer;
  CUcontext cuContext;
  CUdeviceptr gcInfoSpace;
  CUdeviceptr gpuToSpace;
  CUdeviceptr gpuHandlesMemory;
  CUdeviceptr gpuHeapEndPtr;
  CUdeviceptr gpuBufferSize;
  CUdeviceptr gpuExceptionsMemory;
  CUdeviceptr gpuClassMemory;
  jlong numBlocks;
	  
  heapEndPtr = heap_end_ptr;
  
  // Get the current GPU Device
  currentGPUDevice = getCurrentGPUDevice(env,this_obj);
      
  // DEBUG INFO
  //jint currentGPUDeviceID = getIntField(env, currentGPUDevice, "cardID");
  //printf("STARTED: loadFunction with currentGPUDeviceID: %d\n",currentGPUDeviceID);
       
  // Get cuContext from currentGPUDevice
  cuContextPointer = getLongField(env, currentGPUDevice, "cudaContext");
  if (cuContextPointer == 0)
    throw_error_exception(env, "cudaContext was not set in currentGPUDevice!", CUDA_ERROR_INVALID_VALUE);
      
  cuContext = (CUcontext)cuContextPointer;
      
  cuCtxPushCurrent(cuContext);
      
  fatcubin = readCubinFileFromBuffers(env, buffers, size, total_size);
  status = cuModuleLoadFatBinary(&cuModule, fatcubin);
  CHECK_STATUS(env,this_obj,"error in cuModuleLoad", status);
  free(fatcubin);

  status = cuModuleGetFunction(&cuFunction, cuModule, "_Z5entryPcS_PiPxS1_S0_S0_i"); 
  CHECK_STATUS(env,this_obj,"error in cuModuleGetFunction",status)

  status = cuFuncSetCacheConfig(cuFunction, CU_FUNC_CACHE_PREFER_L1);
  CHECK_STATUS(env,this_obj,"error in cuFuncSetCacheConfig",status)

  status = cuParamSetSize(cuFunction, (7 * sizeof(CUdeviceptr) + sizeof(int))); 
  CHECK_STATUS(env,this_obj,"error in cuParamSetSize",status)

  offset = 0;
  gcInfoSpace = getLongField(env, currentGPUDevice, "gcInfoSpace");
  status = cuParamSetv(cuFunction, offset, (void *) &gcInfoSpace, sizeof(CUdeviceptr)); 
  CHECK_STATUS(env,this_obj,"error in cuParamSetv gcInfoSpace",status)
  offset += sizeof(CUdeviceptr);

  gpuToSpace = getLongField(env, currentGPUDevice, "gpuToSpaceAddr");
  status = cuParamSetv(cuFunction, offset, (void *) &gpuToSpace, sizeof(CUdeviceptr)); 
  CHECK_STATUS(env,this_obj,"error in cuParamSetv gpuToSpace",status)
  offset += sizeof(CUdeviceptr);

  gpuHandlesMemory = getLongField(env, currentGPUDevice, "gpuHandlesAddr");
  status = cuParamSetv(cuFunction, offset, (void *) &gpuHandlesMemory, sizeof(CUdeviceptr)); 
  CHECK_STATUS(env,this_obj,"error in cuParamSetv gpuHandlesMemory %",status)
  offset += sizeof(CUdeviceptr);

  gpuHeapEndPtr = getLongField(env, currentGPUDevice, "gpuHeapEndPtr");
  status = cuParamSetv(cuFunction, offset, (void *) &gpuHeapEndPtr, sizeof(CUdeviceptr)); 
  CHECK_STATUS(env,this_obj,"error in cuParamSetv gpuHeapEndPtr",status)
  offset += sizeof(CUdeviceptr);

  gpuBufferSize = getLongField(env, currentGPUDevice, "gpuBufferSize");
  status = cuParamSetv(cuFunction, offset, (void *) &gpuBufferSize, sizeof(CUdeviceptr));
  CHECK_STATUS(env,this_obj,"error in cuParamSetv gpuBufferSize",status)
  offset += sizeof(CUdeviceptr); 

  gpuExceptionsMemory = getLongField(env, currentGPUDevice, "gpuExceptionsHandlesAddr");
  status = cuParamSetv(cuFunction, offset, (void *) &gpuExceptionsMemory, sizeof(CUdeviceptr)); 
  CHECK_STATUS(env,this_obj,"error in cuParamSetv gpuExceptionsMemory",status)
  offset += sizeof(CUdeviceptr);

  gpuClassMemory = getLongField(env, currentGPUDevice, "gpuClassAddr");
  status = cuParamSetv(cuFunction, offset, (void *) &gpuClassMemory, sizeof(CUdeviceptr)); 
  CHECK_STATUS(env,this_obj,"error in cuParamSetv gpuClassMemory",status)
  offset += sizeof(CUdeviceptr);

  numBlocks = getLongField(env, currentGPUDevice, "numBlocks");
  status = cuParamSeti(cuFunction, offset, num_blocks); 
  CHECK_STATUS(env,this_obj,"error in cuParamSetv num_blocks",status)
  offset += sizeof(int);
      
  cuCtxPopCurrent(&cuContext);
      
  return;
}


/*
 * Class:     edu_syr_pcpratts_rootbeer_runtime2_cuda_CudaRuntime2
 * Method:    runBlocks
 * Signature: (III)I
 */
JNIEXPORT jint JNICALL Java_edu_syr_pcpratts_rootbeer_runtime2_cuda_CudaRuntime2_runBlocks
  (JNIEnv *env, jobject this_obj, jint num_blocks, jint block_shape, jint grid_shape){

  CUresult status;
  jlong * infoSpace;
  jobject currentGPUDevice;
  long cuContextPointer;
  CUcontext cuContext;
  void * toSpace;
  CUdeviceptr gpuToSpace;
  void * handlesMemory;
  CUdeviceptr gpuHandlesMemory;
  void * exceptionsMemory;
  CUdeviceptr gpuExceptionsMemory;
  CUdeviceptr gcInfoSpace;
  CUdeviceptr gpuHeapEndPtr;
  CUdeviceptr gpuBufferSize;
  jlong bufferSize;
  
  infoSpace = (jlong *) malloc(gc_space_size);

  infoSpace[1] = heapEndPtr;
      
  // Get the current GPU Device
  currentGPUDevice = getCurrentGPUDevice(env,this_obj);
      
  // DEBUG INFO
  //jint currentGPUDeviceID = getIntField(env, currentGPUDevice, "cardID");
  //printf("STARTED: runBlocks with currentGPUDeviceID: %d\n",currentGPUDeviceID);

  // Get cuContext from currentGPUDevice
  cuContextPointer = getLongField(env, currentGPUDevice, "cudaContext");
  if (cuContextPointer == 0)
    throw_error_exception(env, "cudaContext was not set in currentGPUDevice!", CUDA_ERROR_INVALID_VALUE);
      
  cuContext = (CUcontext)cuContextPointer;
      
  cuCtxPushCurrent(cuContext);
      
  // Init variables
  toSpace = (void *) getLongField(env, currentGPUDevice, "toSpaceAddr");
  gpuToSpace = getLongField(env, currentGPUDevice, "gpuToSpaceAddr");
  handlesMemory = (void *)getLongField(env, currentGPUDevice, "handlesAddr");
  gpuHandlesMemory = getLongField(env, currentGPUDevice, "gpuHandlesAddr");
  exceptionsMemory = (void *)getLongField(env, currentGPUDevice, "exceptionsHandlesAddr");
  gpuExceptionsMemory = getLongField(env, currentGPUDevice, "gpuExceptionsHandlesAddr");
  gcInfoSpace = getLongField(env, currentGPUDevice, "gcInfoSpace");
  gpuHeapEndPtr = getLongField(env, currentGPUDevice, "gpuHeapEndPtr");
  gpuBufferSize = getLongField(env, currentGPUDevice, "gpuBufferSize");
  bufferSize = getLongField(env, currentGPUDevice, "toSpaceSize");

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
  CHECK_STATUS_RTN(env,this_obj,"error in cuFuncSetBlockShape",status, (jint)status);

  status = cuLaunchGrid(cuFunction, grid_shape, 1);
  if(status != CUDA_SUCCESS){
    free(infoSpace);
    cuCtxPopCurrent(&cuContext);
  }
  CHECK_STATUS_RTN(env,this_obj,"error in cuLaunchGrid",status, (jint)status)

  status = cuCtxSynchronize();  
  if(status != CUDA_SUCCESS){
    free(infoSpace);
    cuCtxPopCurrent(&cuContext);
  }
  CHECK_STATUS_RTN(env,this_obj,"error in cuCtxSynchronize",status, (jint)status)
  
  cuMemcpyDtoH(infoSpace, gcInfoSpace, gc_space_size);
  heapEndPtr = infoSpace[1];
  cuMemcpyDtoH(toSpace, gpuToSpace, heapEndPtr);
  cuMemcpyDtoH(exceptionsMemory, gpuExceptionsMemory, num_blocks * sizeof(jlong));
  free(infoSpace);
      
  cuCtxPopCurrent(&cuContext);
      
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
    if (CUDA_SUCCESS != status) {
        throw_error_exception(env, "error in cuInit", status);
        return;
    }
    
    cuDeviceGetCount(&num_devices);
    printf("%d cuda gpus found\n", num_devices);
    
    for (i = 0; i < num_devices; ++i)
    {
        CUdevice dev;
        CUcontext cuContext;
		
        status = cuDeviceGet(&dev, i);
        if (CUDA_SUCCESS != status) {
            throw_error_exception(env, "error in cuDeviceGet", status);
            return;
        }
        
        status = cuCtxCreate(&cuContext, CU_CTX_MAP_HOST, dev);
        if (CUDA_SUCCESS != status) {
            throw_error_exception(env, "error in cuCtxCreate", status);
            return;
        }
        
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


