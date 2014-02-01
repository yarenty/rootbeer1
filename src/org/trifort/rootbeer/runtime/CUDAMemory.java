package org.trifort.rootbeer.runtime;

import java.util.concurrent.atomic.AtomicLong;

public class CUDAMemory implements GpuMemory {

  private long m_size;
  private long m_reserve;
  private long m_address;
    
  public CUDAMemory(long size){
    m_size = size;
    m_reserve = 1024;
  }
  
  private long currPointer(){
    //return m_CurrMemPointer.m_Pointer;
    return 0;
  }  
  
  @Override
  public byte readByte() {
    byte ret = doReadByte(currPointer(), m_address);
    incrementAddress(1);
    return ret;
  }

  @Override
  public boolean readBoolean() {
    boolean ret = doReadBoolean(currPointer(), m_address);
    incrementAddress(1);
    return ret;
  }

  @Override
  public short readShort() {
    short ret = doReadShort(currPointer(), m_address);
    incrementAddress(2);
    return ret;
  }

  @Override
  public int readInt() {
    int ret = doReadInt(currPointer(), m_address);
    incrementAddress(4);
    return ret;
  }

  @Override
  public float readFloat() {
    float ret = doReadFloat(currPointer(), m_address);
    incrementAddress(4);
    return ret;
  }

  @Override
  public double readDouble() {
    double ret = doReadDouble(currPointer(), m_address);
    incrementAddress(8);
    return ret;
  }

  @Override
  public long readLong() {
    long ret = doReadLong(currPointer(), m_address);
    incrementAddress(8);
    return ret;
  }
  
  @Override
  public long readRef() {
    long ret = readInt();
    ret = ret << 4;
    return ret;
  }


  @Override
  public void writeByte(byte value) {
    doWriteByte(currPointer(), value, m_address);
    incrementAddress(1);
  }

  @Override
  public void writeBoolean(boolean value) {
    doWriteBoolean(currPointer(), value, m_address);
    incrementAddress(1);
  }

  @Override
  public void writeShort(short value) {
    doWriteShort(currPointer(), value, m_address);
    incrementAddress(2);
  }

  @Override
  public void writeInt(int value) {
    doWriteInt(currPointer(), value, m_address);
    incrementAddress(4);
  }

  @Override
  public void writeRef(long value) {
    value = value >> 4;
    writeInt((int) value);
  }
  
  @Override
  public void writeFloat(float value) {
    doWriteFloat(currPointer(), value, m_address);
    incrementAddress(4);
  }

  @Override
  public void writeDouble(double value) {
    doWriteDouble(currPointer(), value, m_address);
    incrementAddress(8);
  }

  @Override
  public void writeLong(long value) {
    doWriteLong(currPointer(), value, m_address);
    incrementAddress(8);
  }
  
  @Override
  public void readArray(byte[] array){
    doReadByteArray(array, m_address+currPointer(), 0, array.length);  
  }
  
  @Override
  public void readArray(boolean[] array){
    doReadBooleanArray(array, m_address+currPointer(), 0, array.length);  
  }
    
  @Override
  public void readArray(short[] array){
    doReadShortArray(array, m_address+currPointer(), 0, array.length);  
  }
      
  @Override
  public void readArray(int[] array){
    doReadIntArray(array, m_address+currPointer(), 0, array.length);  
  }
    
  @Override
  public void readArray(float[] array){
    doReadFloatArray(array, m_address+currPointer(), 0, array.length);  
  }
  
  @Override
  public void readArray(double[] array){
    doReadDoubleArray(array, m_address+currPointer(), 0, array.length);  
  }
    
  @Override
  public void readArray(long[] array){
    doReadLongArray(array, m_address+currPointer(), 0, array.length);  
  }
    
  @Override
  public void writeArray(byte[] array){
    doWriteByteArray(array, m_address+currPointer(), 0, array.length);
  }
    
  @Override
  public void writeArray(boolean[] array){
    doWriteBooleanArray(array, m_address+currPointer(), 0, array.length);
  }
    
  @Override
  public void writeArray(short[] array){
    doWriteShortArray(array, m_address+currPointer(), 0, array.length);
  }
    
  @Override
  public void writeArray(int[] array){
    doWriteIntArray(array, m_address+currPointer(), 0, array.length);
  }
    
  @Override
  public void writeArray(float[] array){
    doWriteFloatArray(array, m_address+currPointer(), 0, array.length);
  }
  
  @Override
  public void writeArray(double[] array){
    doWriteDoubleArray(array, m_address+currPointer(), 0, array.length);
  }
    
  @Override
  public void writeArray(long[] array){
    doWriteLongArray(array, m_address+currPointer(), 0, array.length);
  }
  
  public void incrementAddress(int offset) {
    //m_CurrMemPointer.incrementAddress(offset);
  }
    
  public native void doReadByteArray(byte[] array, long addr, int start, int len);
  public native void doReadBooleanArray(boolean[] array, long addr, int start, int len);
  public native void doReadShortArray(short[] array, long addr, int start, int len);
  public native void doReadIntArray(int[] array, long addr, int start, int len);
  public native void doReadFloatArray(float[] array, long addr, int start, int len);
  public native void doReadDoubleArray(double[] array, long addr, int start, int len);
  public native void doReadLongArray(long[] array, long addr, int start, int len);
  
  public native void doWriteByteArray(byte[] array, long addr, int start, int len);
  public native void doWriteBooleanArray(boolean[] array, long addr, int start, int len);
  public native void doWriteShortArray(short[] array, long addr, int start, int len);
  public native void doWriteIntArray(int[] array, long addr, int start, int len);
  public native void doWriteFloatArray(float[] array, long addr, int start, int len);
  public native void doWriteDoubleArray(double[] array, long addr, int start, int len);
  public native void doWriteLongArray(long[] array, long addr, int start, int len);
  
  public native byte doReadByte(long ptr, long cpu_base);
  public native boolean doReadBoolean(long ptr, long cpu_base);
  public native short doReadShort(long ptr, long cpu_base);
  public native int doReadInt(long ptr, long cpu_base);
  public native float doReadFloat(long ptr, long cpu_base);
  public native double doReadDouble(long ptr, long cpu_base);
  public native long doReadLong(long ptr, long cpu_base);
  public native void doWriteByte(long ptr, byte value, long cpu_base);
  public native void doWriteBoolean(long ptr, boolean value, long cpu_base);
  public native void doWriteShort(long ptr, short value, long cpu_base);
  public native void doWriteInt(long ptr, int value, long cpu_base);
  public native void doWriteFloat(long ptr, float value, long cpu_base);
  public native void doWriteDouble(long ptr, double value, long cpu_base);
  public native void doWriteLong(long ptr, long value, long cpu_base);

  @Override
  public void reset() {
    // TODO Auto-generated method stub
    
  }

  @Override
  public void setAddress(long pos) {
    // TODO Auto-generated method stub
    
  }

  @Override
  public long getPointer() {
    // TODO Auto-generated method stub
    return 0;
  }
}
