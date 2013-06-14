/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.runtimegpu;

import edu.syr.pcpratts.rootbeer.runtime.Sentinal;

public class GpuException {

  public int m_arrayLength;
  public int m_arrayIndex;
  public int m_array;
  
  public StringBuilder m_sb;
  
  public static GpuException exception(){
    GpuException ret = new GpuException();
    ret.m_sb = new StringBuilder();
    return ret;
  }

  public static GpuException arrayOutOfBounds(int index, int array, int length){
    GpuException ret = new GpuException();
    ret.m_arrayLength = length;
    ret.m_arrayIndex = index;
    ret.m_array = array;
    ret.m_sb = new StringBuilder();
    return ret;
  }
  
  public void addStackTrace(String str){
    m_sb.append(str);
  }
          
  public void throwArrayOutOfBounds(){
    throw new ArrayIndexOutOfBoundsException("array_index: "+m_arrayIndex+" array_length: "+m_arrayLength+" array: "+m_array+"\n"+m_sb.toString());
  }
}
