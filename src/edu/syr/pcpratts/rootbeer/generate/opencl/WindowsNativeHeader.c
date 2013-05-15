#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <Windows.h>

DWORD threadIdKey;
DWORD threadIdxxKey;
DWORD blockIdxxKey;
DWORD blockDimxKey;
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

int getThreadIdxx(){
  return (int) TlsGetValue(threadIdxxKey);
}

int getBlockIdxx(){
  return (int) TlsGetValue(blockIdxxKey);
}

int getBlockDimx(){
  return (int) TlsGetValue(blockDimxKey);
}

long long java_lang_System_nanoTime(char * gc_info, int * exception){
  SYSTEMTIME system_time;
  GetSystemTime(&system_time);
  return system_time.wMilliseconds;
}