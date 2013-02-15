
static void * run(void * data){
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
    
    pthread_setspecific(blockIdxxKey, (void *) curr_block_idxx);
    pthread_setspecific(threadIdxxKey, (void *) curr_thread_idxx);
    pthread_setspecific(threadIdKey, (void *) index);

    lhandle = global_handles[index];
    lhandle = lhandle >> 4;
    handle = (int) lhandle;
    exception = 0;
    %%invoke_run%%(global_gc_info, handle, &exception);
    global_exceptions[index] = exception;
  }

  return NULL;
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
  pthread_t ** threads;
  pthread_t * thread;

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

  pthread_mutex_init(&thread_id_mutex, NULL);
  pthread_mutex_init(&atom_add_mutex, NULL);
  pthread_key_create(&threadIdKey, NULL);
  pthread_key_create(&threadIdxxKey, NULL);
  pthread_key_create(&blockIdxxKey, NULL);
  
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
  fflush(stdout);
}