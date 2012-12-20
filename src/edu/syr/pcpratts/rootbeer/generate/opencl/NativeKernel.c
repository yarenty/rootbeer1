
char * global_gc_info;
int * global_handles;
int thread_id;
int global_num_threads;
int * global_exceptions;
int * global_class_refs;

void synchthreads()
{
}

char *
edu_syr_pcpratts_gc_deref(char * gc_info, int handle){
  long long lhandle;
  long long * to_space;
  long space_size;
  long long array;
  long long offset;
  long long address;
  char * data_arr;

  lhandle = handle;
  lhandle = lhandle << 4;
  to_space = edu_syr_pcpratts_gc_get_to_space_address(gc_info);
  space_size = edu_syr_pcpratts_getlong(gc_info, 16);
  array = lhandle / space_size;
  offset = lhandle % space_size;

  address = to_space[array];
  data_arr = (char *) address;
  return &data_arr[offset];
}

int
edu_syr_pcpratts_gc_malloc(char * gc_info, long long size){
  long long * addr;
  long long space_size;
  long long ret;
  int mod;
  long long start_array;
  long long end_array;

  addr = (long long *) (gc_info + TO_SPACE_FREE_POINTER_OFFSET);
  space_size = edu_syr_pcpratts_getlong(gc_info, 16);
  size += 8;
  while(true){
    ret = atom_add(addr, (long) size);
    mod = ret % 8;
    if(mod != 0)
      ret += (8 - mod);

    start_array = ret / space_size;
    end_array = (ret + size) / space_size;

    if(start_array != end_array){
      continue;
    }

    ret = ret >> 4;
    return (int) ret;
  }
}

int
edu_syr_pcpratts_classConstant(int type_num){
  return global_class_refs[type_num];
}

char *
edu_syr_pcpratts_gc_init(char * gc_info_space,
                         long long * to_space,
                         long long to_space_free_ptr,
                         long long space_size){

  edu_syr_pcpratts_setlong(gc_info_space, 0, (long long) to_space);
  edu_syr_pcpratts_setlong(gc_info_space, 8, to_space_free_ptr);
  edu_syr_pcpratts_setlong(gc_info_space, 16, space_size);
    
  return (char *) gc_info_space;
}

static void * run(void * data){
  int index;
  long long lhandle;
  int exception;
  int handle;    

  while(1){
    lock_thread_id();
    index = thread_id;
    ++thread_id;
    unlock_thread_id();
    
    if(index >= global_num_threads)
      break;

#if (defined linux || defined __APPLE_CC__)
    pthread_setspecific(threadIdKey, (void *) index);
#else
    TlsSetValue(threadIdKey, (void *) index);
#endif

    lhandle = global_handles[index];
    lhandle = lhandle >> 4;
    handle = (int) lhandle;
    exception = 0;
    %%invoke_run%%(global_gc_info, handle, &exception);
    global_exceptions[index] = exception;
  }
  return NULL;
}

#if (defined linux || defined __APPLE_CC__)  
void entry(char * gc_info_space,
           long long * to_space,
           long long * handles,
           long long * to_space_free_ptr,
           long long * exceptions,
           int * java_lang_class_refs,
           long long space_size,
           int num_threads){
#else
__declspec(dllexport)
void entry(char * gc_info_space,
           long long * to_space,
           long long * handles,
           long long * to_space_free_ptr,
           long long * exceptions,
           int * java_lang_class_refs,
           long long space_size,
           int num_threads){
#endif
  int i;
  int rc;
  int num_cores;

#if (defined linux || defined __APPLE_CC__)
  pthread_t ** threads;
  pthread_t * thread;
#else
  HANDLE * threads;
#endif

  char * gc_info = edu_syr_pcpratts_gc_init(gc_info_space, to_space,
    *to_space_free_ptr, space_size);
  global_num_threads = num_threads;
  thread_id = 0;
  global_gc_info = gc_info;
  global_handles = handles;
  global_exceptions = exceptions;
  global_class_refs = java_lang_class_refs;
  
#if (defined linux || defined __APPLE_CC__)

  pthread_mutex_init(&thread_id_mutex, NULL);
  pthread_mutex_init(&atom_add_mutex, NULL);
  pthread_key_create(&threadIdKey, NULL);
   
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  num_cores = 4;
  threads = (pthread_t **) malloc(sizeof(pthread_t *)*num_cores);

  for(i = 0; i < num_cores; ++i){
    thread = (pthread_t *) malloc(sizeof(pthread_t));
    pthread_create(thread, &attr, &run, NULL);
    threads[i] = thread;
  }

  for(i = 0; i < num_cores; ++i){
    thread = threads[i];
    rc = pthread_join(*thread, NULL);
    if (rc) {
      printf("ERROR; return code from pthread_join() is %d\n", rc);
      exit(-1);
    }
  } 

  free(threads);

#else

  InitializeCriticalSection(&thread_id_mutex);
  InitializeCriticalSection(&atom_add_mutex);
  threadIdKey = TlsAlloc();

  num_cores = 4;
  threads = (HANDLE *) malloc(sizeof(HANDLE)*num_cores);

  for(i = 0; i < num_cores; ++i){
    threads[i] = CreateThread(NULL, 0, &run, NULL, 0, NULL);
  }

  for(i = 0; i < num_cores; ++i){
    WaitForSingleObject(&threads[i], INFINITE);
  }

  free(threads);

#endif

  fflush(stdout);

}