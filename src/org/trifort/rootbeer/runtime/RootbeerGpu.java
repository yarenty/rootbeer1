/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.runtime;

import java.util.Map;
import java.util.TreeMap;

public class RootbeerGpu {

  private static boolean m_isOnGpu;
  private static short[] m_sharedMem;
  private static int m_blockIdxx;
  private static int m_threadIdxx;
  private static int m_threadId;
  private static long m_gridDimx;
  private static int m_blockDimx;
  
  private static Map<Integer, Object> m_sharedArrayMap;
  
  static {
    m_isOnGpu = false;
    m_sharedMem = new short[48*1024];
    m_sharedArrayMap = new TreeMap<Integer, Object>();
  }
  
  public static boolean isOnGpu(){
    return m_isOnGpu;
  }
  
  public static void setIsOnGpu(boolean value){
    m_isOnGpu = value;
  }

  /**
   * @return blockIdx.x * blockDim.x + threadIdx.x;
   */
  public static int getThreadId() {
    return m_threadId;
  }
 
  public static int getThreadIdxx() {
    return m_threadIdxx;
  }

  public static int getBlockIdxx() {
    return m_blockIdxx;
  }
  
  public static int getBlockDimx(){
    return m_blockDimx;
  }
  
  public static long getGridDimx(){
    return m_gridDimx;
  }
  
  public static void setThreadId(int thread_id){
    m_threadId = thread_id;
  }

  public static void setThreadIdxx(int thread_idxx){
    m_threadIdxx = thread_idxx;
  }
  
  public static void setBlockIdxx(int block_idxx){
    m_blockIdxx = block_idxx;
  }
  
  public static void setBlockDimx(int block_dimx){
    m_blockDimx = block_dimx;
  }
  
  public static void setGridDimx(long grid_dimx){
    m_gridDimx = grid_dimx;
  }
  
  public static void syncthreads(){ 
  }

  public static void threadfence(){ 
  }
  
  public static void threadfenceBlock(){ 
  }
  
  public static void threadfenceSystem(){
  }
  
  public static long getRef(Object obj) {
    return 0;
  }

  public static Object getSharedObject(int index){
    return null;
  }
  
  public static void setSharedObject(int index, Object value){
  }
  
  public static byte getSharedByte(int index){
    return (byte) m_sharedMem[index];
  }
  
  public static void setSharedByte(int index, byte value){
    m_sharedMem[index] = value;
  }
  
  public static char getSharedChar(int index){
    char ret = 0;
    ret |= m_sharedMem[index];
    ret |= (m_sharedMem[index + 1] << 8);
    return ret;
  }
  
  public static void setSharedChar(int index, char value){ 
    m_sharedMem[index] = (short) (value & 0xff);
    m_sharedMem[index + 1] = (short) ((value >> 8) & 0xff);
  }
  
  public static boolean getSharedBoolean(int index){
    if(m_sharedMem[index] == 1){
      return true;
    } else {
      return false;
    }
  }
  
  public static void setSharedBoolean(int index, boolean value){
    byte value_byte;
    if(value == true){
      value_byte = 1;
    } else {
      value_byte = 0; 
    }
    m_sharedMem[index] = value_byte;
  }
  
  public static short getSharedShort(int index){
    short ret = 0;
    ret |= m_sharedMem[index];
    ret |= (m_sharedMem[index + 1] << 8);
    return ret;
  }
  
  public static void setSharedShort(int index, short value){
    m_sharedMem[index] = (short) (value & 0xff);
    m_sharedMem[index + 1] = (short) ((value >> 8) & 0xff);
  }
  
  public static int getSharedInteger(int index){
    int ret = 0;
    ret |= m_sharedMem[index];
    ret |= (m_sharedMem[index + 1] <<  8);
    ret |= (m_sharedMem[index + 2] << 16);
    ret |= (m_sharedMem[index + 3] << 24);
    return ret;  
  }
  
  public static void setSharedInteger(int index, int value){
    m_sharedMem[index] = (short) (value & 0xff);
    m_sharedMem[index + 1] = (short) ((value >> 8)  & 0xff);
    m_sharedMem[index + 2] = (short) ((value >> 16) & 0xff);
    m_sharedMem[index + 3] = (short) ((value >> 24) & 0xff);
  }
  
  public static long getSharedLong(int index){
    long ret = 0;
    ret |=  ((long) m_sharedMem[index]);
    ret |= (((long) m_sharedMem[index + 1]) <<  8);
    ret |= (((long) m_sharedMem[index + 2]) << 16);
    ret |= (((long) m_sharedMem[index + 3]) << 24);
    ret |= (((long) m_sharedMem[index + 4]) << 32);
    ret |= (((long) m_sharedMem[index + 5]) << 40);
    ret |= (((long) m_sharedMem[index + 6]) << 48);
    ret |= (((long) m_sharedMem[index + 7]) << 56);
    return ret;    
  }
  
  public static void setSharedLong(int index, long value){
    m_sharedMem[index] = (short) (value & 0xff);
    m_sharedMem[index + 1] = (short) ((value >> 8)  & 0xff);
    m_sharedMem[index + 2] = (short) ((value >> 16) & 0xff);
    m_sharedMem[index + 3] = (short) ((value >> 24) & 0xff);
    m_sharedMem[index + 4] = (short) ((value >> 32) & 0xff);
    m_sharedMem[index + 5] = (short) ((value >> 40) & 0xff);
    m_sharedMem[index + 6] = (short) ((value >> 48) & 0xff);
    m_sharedMem[index + 7] = (short) ((value >> 56) & 0xff);
  }
  
  public static float getSharedFloat(int index){
    int value_int = getSharedInteger(index);
    return Float.intBitsToFloat(value_int);
  }
  
  public static void setSharedFloat(int index, float value){ 
    int value_int = Float.floatToIntBits(value);
    setSharedInteger(index, value_int);
  }
  
  public static double getSharedDouble(int index){
    long value_long = getSharedLong(index);
    return Double.longBitsToDouble(value_long);
  }
  
  public static void setSharedDouble(int index, double value){
    long value_long = Double.doubleToLongBits(value);
    setSharedLong(index, value_long);
  }
  
  public static double sin(double value){
    return 0;
  }  
  
  public static int atomicAddGlobal(int[] array, int index, int addValue){
    synchronized(array){
      int ret = array[index];
      array[index] += addValue;
      return ret;
    }
  }
  
  public static long atomicAddGlobal(long[] array, int index, long addValue){
    synchronized(array){
      long ret = array[index];
      array[index] += addValue;
      return ret;
    }
  }

  public static float atomicAddGlobal(float[] array, int index, float addValue){
    synchronized(array){
      float ret = array[index];
      array[index] += addValue;
      return ret;
    }
  }
  public static int atomicSubGlobal(int[] array, int index, int subValue){
    synchronized(array){
      int ret = array[index];
      array[index] -= subValue;
      return ret;
    }
  }
  
  public static int atomicExchGlobal(int[] array, int index, int value){
    synchronized(array){
      int ret = array[index];
      array[index] = value;
      return ret;
    }
  }
  
  public static long atomicExchGlobal(long[] array, int index, long value){
    synchronized(array){
      long ret = array[index];
      array[index] = value;
      return ret;
    }
  }
  
  public static float atomicExchGlobal(float[] array, int index, float value){
    synchronized(array){
      float ret = array[index];
      array[index] = value;
      return ret;
    }
  }
  
  public static int atomicMinGlobal(int[] array, int index, int value){
    synchronized(array){
      int old = array[index];
      if(value < old){
        array[index] = value;
      }
      return old;
    }
  }
  
  public static int atomicMaxGlobal(int[] array, int index, int value){
    synchronized(array){
      int old = array[index];
      if(value > old){
        array[index] = value;
      }
      return old;
    }
  }
  
  public static int atomicCASGlobal(int[] array, int index, int compare, int value){
    synchronized(array){
      int old = array[index];
      if(old == compare){
        array[index] = value;
      }
      return old;
    }
  }

  public static int atomicAndGlobal(int[] array, int index, int value){
    synchronized(array){
      int old = array[index];
      array[index] = old & value;
      return old;
    }
  }
  
  public static int atomicOrGlobal(int[] array, int index, int value){
    synchronized(array){
      int old = array[index];
      array[index] = old | value;
      return old;
    }
  }

  public static int atomicXorGlobal(int[] array, int index, int value){
    synchronized(array){
      int old = array[index];
      array[index] = old ^ value;
      return old;
    }
  }
  
  /*
  //TODO: working on this
  public static int[] createSharedIntArray(int index, int length){
    int[] ret = new int[length];
    m_sharedArrayMap.put(index, ret);
    return ret;
  }
  
  public static int[] getSharedIntArray(int index){
    if(m_sharedArrayMap.containsKey(index)){
      return (int[]) m_sharedArrayMap.get(index);
    } else {
      throw new IllegalArgumentException();
    }
  }
  */
}
