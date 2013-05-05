
#define NAN 0x7ff8000000000000L
#define INFINITY 0x7ff0000000000000L

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
int getThreadIdxx();
int getBlockIdxx();
int getBlockDimx();

void edu_syr_pcpratts_syncthreads();

void __threadfence(){ }

long long m_Local[3];
int * m_Cache;
long long m_shared[40*1024];