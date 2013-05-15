
DWORD run(void * data)
{
  int index;
  int curr_thread_idxx;
  int curr_block_idxx;
  long long lhandle;
  int exception;
  int handle;    

  while(1){
    lock_thread_id();
    index = thread_id;
    ++thread_id;
    unlock_thread_id();
    
    if(index >= global_num_threads){
      break;
    }

    curr_block_idxx = index / global_block_shape;
    curr_thread_idxx = index % global_block_shape;
    
    TlsSetValue(threadIdKey, (void *) index);
    TlsSetValue(threadIdxxKey, (void *) curr_thread_idxx);
    TlsSetValue(blockIdxxKey, (void *) curr_block_idxx);
    TlsSetValue(blockDimxKey, (void *) global_block_shape);

    lhandle = global_handles[index];
    lhandle = lhandle >> 4;
    handle = (int) lhandle;
    exception = 0;
    %%invoke_run%%(global_gc_info, handle, &exception);
    global_exceptions[index] = exception;
  }

  return 0;
}

void entry(char * gc_info_space,
           long long * to_space,
           long long * handles,
           long long * to_space_free_ptr,
           long long * exceptions,
           int * java_lang_class_refs,
           long long space_size,
           int num_threads,
           int block_shape,
           int thread_shape){
  int i;
  int rc;
  int num_cores;
  char * gc_info;
  HANDLE * threads;

  gc_info = edu_syr_pcpratts_gc_init(gc_info_space, to_space,
    *to_space_free_ptr, space_size);
  global_num_threads = num_threads;
  global_block_shape = block_shape;
  global_thread_shape = thread_shape;
  thread_id = 0;
  global_gc_info = gc_info;
  global_handles = handles;
  global_exceptions = exceptions;
  global_class_refs = java_lang_class_refs;

  InitializeCriticalSection(&thread_id_mutex);
  InitializeCriticalSection(&atom_add_mutex);
  threadIdKey = TlsAlloc();
  threadIdxxKey = TlsAlloc();
  blockIdxxKey = TlsAlloc();

  num_cores = 4;
  threads = (HANDLE *) malloc(sizeof(HANDLE)*num_cores);

  for(i = 0; i < num_cores; ++i){
    threads[i] = CreateThread(NULL, 0, &run, NULL, 0, NULL);
  }

  for(i = 0; i < num_cores; ++i){
    WaitForSingleObject(threads[i], INFINITE);
  }

  free(threads);
  fflush(stdout);
}