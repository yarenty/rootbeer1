#include "CUDARuntime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>

JNIEXPORT jobject JNICALL Java_org_trifort_rootbeer_runtime_OpenCLRuntime_loadGpuDevices
  (JNIEnv * env, jobject this_ref)
{
  int i;
  int status;
  
  jclass array_list_class;
  jmethodID array_list_init;
  jmethodID array_list_add;
  jobject ret;

  jclass gpu_device_class;
  jmethodID gpu_device_init;
  jobject gpu_device;

  cl_int num_devices;

  array_list_class = (*env)->FindClass(env, "java/util/ArrayList");
  array_list_init = (*env)->GetMethodID(env, array_list_class, "<init>", "()V");
  array_list_add = (*env)->GetMethodID(env, array_list_class, "add", "(Ljava/lang/Object;)Z");

  ret = (*env)->NewObject(env, array_list_class, array_list_init);

  gpu_device_class = (*env)->FindClass(env, "org/trifort/rootbeer/runtime/GpuDevice");
  gpu_device_init = (*env)->GetStaticMethodID(env, gpu_device_class, "newOpenCLDevice", 
    "()Lorg/trifort/rootbeer/runtime/GpuDevice;");

  num_devices = clGetDeviceIDs(0, CL_DEVICE_TYPE_ALL, 0, NULL, NULL);
  printf("num opencl devices: %d\n", num_devices);

  /*  
  for(i = 0; i < num_devices; ++i){
    status = cuDeviceGet(&device, i);
    if(status != CUDA_SUCCESS){
      continue;
    }

    gpu_device = (*env)->CallObjectMethod(env, gpu_device_class, gpu_device_init);
    (*env)->CallBooleanMethod(env, ret, array_list_add, gpu_device);
  }
  */

  return ret;
}
