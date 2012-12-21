
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