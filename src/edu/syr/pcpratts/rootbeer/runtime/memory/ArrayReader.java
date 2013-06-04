/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.runtime.memory;

public class ArrayReader {

  private byte[] m_array;
  private int m_index;
  
  public ArrayReader(byte[] array){
    m_array = array;
    m_index = 0;
  }
  
  public byte readByte(){
    byte ret = m_array[m_index];
    ++m_index;
    return ret;
  }

  public boolean readBoolean(){
    byte b = readByte();
    if(b != 0)
      return true;
    return false;
  }
  
  public int readInt(){
    int value = (int) (((int) m_array[m_index+3] << 24) & 0xff000000);
    value |= (int) (((int) m_array[m_index+2] << 16) & 0x00ff0000);
    value |= (int) (((int) m_array[m_index+1] << 8) & 0x0000ff00);
    value |= (int) ((int) m_array[m_index] & 0x000000ff);

    m_index += 4;
    return value;
  }
  
  public short readShort(){
    short value = (short) (((short) m_array[m_index+1] << 8) & 0x0000ff00);
    value |=  (short) (((short) m_array[m_index]) & 0x000000ff);

    m_index += 2;
    return value;
  }

  public float readFloat(){
    int intValue = readInt();
    return Float.intBitsToFloat(intValue);
  }

  public double readDouble(){
    long longValue = readLong();
    return Double.longBitsToDouble(longValue);
  }

  public long readLong(){
    long value = (((long) m_array[m_index+7] << 56) & 0xff00000000000000L);
    value |= (((long) m_array[m_index+6] << 48)     & 0x00ff000000000000L);
    value |= (((long) m_array[m_index+5] << 40)     & 0x0000ff0000000000L);
    value |= (((long) m_array[m_index+4] << 32)     & 0x000000ff00000000L);
    value |= (((long) m_array[m_index+3] << 24)     & 0x00000000ff000000L);
    value |= (((long) m_array[m_index+2] << 16)     & 0x0000000000ff0000L);
    value |= (((long) m_array[m_index+1] << 8)      & 0x000000000000ff00L);
    value |= ((long) m_array[m_index]               & 0x00000000000000ffL);

    m_index += 8;
    return value;
  }
  
  public int getIndex(){
    return m_index;
  }
  
  public int size(){
    return m_array.length;
  }
}
