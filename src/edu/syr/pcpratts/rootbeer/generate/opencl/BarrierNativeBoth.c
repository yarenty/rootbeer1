
void edu_syr_pcpratts_barrier(){
  volatile int count;

  barrier_mutex_lock();
  global_barrier_count2 = 0;
  global_barrier_count1++;
  printf("global_barrier_count1: %d thread_id: %d\n", global_barrier_count1, getThreadId());
  while(1){
    edu_syr_pcpratts_sleep(2);
    if(0){
      break;
    }
  }
  barrier_mutex_unlock();

  thread_gate_mutex_lock();
  global_thread_gate_count++;
  printf("global_thread_gate_count: %d\n", global_thread_gate_count);
  thread_gate_mutex_unlock();

  while(1){
    barrier_mutex_lock();
    count = global_barrier_count1;
    barrier_mutex_unlock();

    if(count == global_thread_count){
      break;
    } else {
      edu_syr_pcpratts_sleep(2);
    }
  }

  barrier_mutex_lock();
  global_barrier_count3 = 0;
  global_barrier_count2++;
  barrier_mutex_unlock();

  while(1){
    barrier_mutex_lock();
    count = global_barrier_count2;
    barrier_mutex_unlock();

    if(count == global_thread_count){
      break;
    } else {
      edu_syr_pcpratts_sleep(2);
    }
  }

  barrier_mutex_lock();
  global_barrier_count1 = 0;
  global_barrier_count3++;
  barrier_mutex_unlock();

  thread_gate_mutex_lock();
  global_thread_gate_count = global_num_cores;
  thread_gate_mutex_unlock();

  while(1){
    barrier_mutex_lock();
    count = global_barrier_count3;
    barrier_mutex_unlock();

    if(count == global_thread_count){
      break;
    } else {
      edu_syr_pcpratts_sleep(2);
    }
  }
}
