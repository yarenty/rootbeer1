#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#if (defined linux || defined __APPLE_CC__)

  #include <sys/time.h>
  #include <pthread.h>

  pthread_key_t threadIdKey = 0;
  pthread_mutex_t atom_add_mutex;
  pthread_mutex_t thread_id_mutex;
  pthread_attr_t attr;

  void lock_atom_add(){
    pthread_mutex_lock(&atom_add_mutex);
  }

  void unlock_atom_add(){
    pthread_mutex_unlock(&atom_add_mutex);
  }

  void lock_thread_id(){
    pthread_mutex_lock(&thread_id_mutex);
  }

  void unlock_thread_id(){
    pthread_mutex_unlock(&thread_id_mutex);
  }

  int getThreadId(){
    return (int) pthread_getspecific(threadIdKey);
  }

  long long java_lang_System_nanoTime(char * gc_info, int * exception){
    struct timeval tm;
    gettimeofday(&tm, 0);
    return tm.tv_sec * 1000000 + tm.tv_usec;
  }
#else

  #include <Windows.h>

  DWORD threadIdKey;
  CRITICAL_SECTION atom_add_mutex;
  CRITICAL_SECTION thread_id_mutex;

  void lock_atom_add(){
    EnterCriticalSection(&atom_add_mutex);
  }

  void unlock_atom_add(){
    LeaveCriticalSection(&atom_add_mutex);
  }

  void lock_thread_id(){
    EnterCriticalSection(&thread_id_mutex);
  }

  void unlock_thread_id(){
    LeaveCriticalSection(&thread_id_mutex);
  }

  int getThreadId(){
    return (int) TlsGetValue(threadIdKey);
  }

  long long java_lang_System_nanoTime(char * gc_info, int * exception){
    SYSTEMTIME system_time;
    GetSystemTime(&system_time);
    return system_time.wMilliseconds;
  }
#endif

long long atom_add(long long * addr, long long value){
  long long ret;
  lock_atom_add();
  ret = *addr;
  *addr += value;
  unlock_atom_add();
  return ret;
}

unsigned long long atomicAdd(unsigned long long * addr, long long value){
  long long ret;
  lock_atom_add();
  ret = *addr;
  *addr += value;
  unlock_atom_add();
  return ret;
}

int atomicCAS(int * addr, int compare, int set){
  int ret;
  lock_atom_add();
  ret = *addr;
  if(ret == compare)
    *addr = set;
  unlock_atom_add();
  return ret;
}

int atomicExch(int * addr, int value){
  int ret;
  lock_atom_add();
  ret = *addr;
  *addr = value;
  unlock_atom_add();
  return ret;
}

int getThreadId();
void synchthreads();

void __threadfence(){ }

long long m_Local[3];
int * m_Cache;