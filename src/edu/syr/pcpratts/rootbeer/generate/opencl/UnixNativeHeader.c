#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <sys/time.h>
#include <pthread.h>

pthread_key_t threadIdKey = 0;
pthread_key_t threadIdxxKey = 0;
pthread_key_t blockIdxxKey = 0;
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

int getThreadIdxx(){
  return (int) pthread_getspecific(threadIdxxKey);
}

int getBlockIdxx(){
  return (int) pthread_getspecific(blockIdxxKey);
}

long long java_lang_System_nanoTime(char * gc_info, int * exception){
  struct timeval tm;
  gettimeofday(&tm, 0);
  return tm.tv_sec * 1000000 + tm.tv_usec;
}
