package org.trifort.rootbeer.runtime;

public interface GpuMemory {

  public void reset();
  public void setAddress(long pos);
  public long getPointer();
  
  public byte readByte();
  public boolean readBoolean();
  public short readShort();
  public int readInt();
  public float readFloat();
  public double readDouble();
  public long readLong();
  public long readRef();
  public void readArray(byte[] array);
  public void readArray(boolean[] array);
  public void readArray(short[] array);
  public void readArray(int[] array);
  public void readArray(float[] array);
  public void readArray(double[] array);
  public void readArray(long[] array);
  
  public void writeByte(byte value);
  public void writeBoolean(boolean value);
  public void writeShort(short value);
  public void writeInt(int value);
  public void writeFloat(float value);
  public void writeDouble(double value);
  public void writeLong(long value);
  public void writeRef(long value);
  public void writeArray(byte[] array);
  public void writeArray(boolean[] array);
  public void writeArray(short[] array);
  public void writeArray(int[] array);
  public void writeArray(float[] array);
  public void writeArray(double[] array);
  public void writeArray(long[] array);
}
