
__device__ int
org_trifort_classConstant(int type_num){
  int * temp = (int *) m_Local[2];   
  return temp[type_num];
}

__device__  char *
org_trifort_gc_deref(int handle){

  char * data_arr = (char * ) m_Local[0];
  long long lhandle = handle;
  lhandle = lhandle << 4;
  return &data_arr[lhandle];
}

__device__ int
org_trifort_gc_malloc(int size){
  unsigned long long space_size = m_Local[1];
  int ret = org_trifort_gc_malloc_no_fail(size);
  unsigned long long long_ret = ret << 4;
  unsigned long long end = long_ret + size + 8L;
  if(end >= space_size){
    return -1;
  }
  return ret;
}

//TODO: don't pass gc_info everywhere because free pointer is __device__
__device__ int * global_free_pointer; 

__device__ int
org_trifort_gc_malloc_no_fail(int size){
  if(size % 16 != 0){
    size += (16 - (size %16));
  }
  size >>= 4;

  int ret;
  ret = atomicAdd(global_free_pointer, size);
  return ret;
}

__device__  void
org_trifort_gc_init(int * free_pointer, char * to_space, size_t space_size, int * java_lang_class_refs){
  
  if(threadIdx.x == 0){
    m_Local[0] = (size_t) to_space;
    m_Local[1] = (size_t) space_size;
    m_Local[2] = (size_t) java_lang_class_refs;
    
    global_free_pointer = free_pointer;
  }
}

__device__
long long java_lang_System_nanoTime(int * exception){
  return (long long) clock64();
}

__global__ void entry(char * gc_info, char * to_space, int * handles, 
  int * free_pointer, long long * space_size, int * exceptions,
  int * java_lang_class_refs, int num_blocks, int using_kernel_templates){

  org_trifort_gc_init(free_pointer, to_space, *space_size, java_lang_class_refs);
  __syncthreads();
      
  int loop_control = blockIdx.x * blockDim.x + threadIdx.x;
  if(loop_control >= num_blocks){
    return;
  } else {
    int handle;
    if(using_kernel_templates){
      handle = handles[0];
    } else {
      handle = handles[loop_control];
    }
    int exception = 0; 
    %%invoke_run%%(handle, &exception);  
    if(%%using_exceptions%%){
      exceptions[loop_control] = exception;
    }

    int * result_free_pointer = (int *) (gc_info + TO_SPACE_FREE_POINTER_OFFSET);
    *result_free_pointer = *global_free_pointer;
  }
}
