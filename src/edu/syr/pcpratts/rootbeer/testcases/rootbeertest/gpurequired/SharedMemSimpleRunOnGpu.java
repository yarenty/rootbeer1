/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.testcases.rootbeertest.gpurequired;

import edu.syr.pcpratts.rootbeer.runtime.Kernel;
import edu.syr.pcpratts.rootbeer.runtime.RootbeerGpu;

public class SharedMemSimpleRunOnGpu implements Kernel {

  private boolean m_boolean;
  private byte m_byte;
  private char m_char;
  private short m_short;
  private int m_integer;
  private long m_long;
  private float m_float;
  private double m_double;
  
  public void gpuMethod() {
    RootbeerGpu.setSharedBoolean(0, true);
    RootbeerGpu.setSharedByte(1, (byte) 2);
    RootbeerGpu.setSharedChar(2, (char) 3);
    RootbeerGpu.setSharedShort(4, (short) 4);
    RootbeerGpu.setSharedInteger(6, 5);
    RootbeerGpu.setSharedLong(10, 6);
    RootbeerGpu.setSharedFloat(18, 7.1f);
    RootbeerGpu.setSharedDouble(22, 8.2);
    
    RootbeerGpu.synchthreads();
    
    m_boolean = RootbeerGpu.getSharedBoolean(0);
    m_byte = RootbeerGpu.getSharedByte(1);
    m_char = RootbeerGpu.getSharedChar(2);
    m_short = RootbeerGpu.getSharedShort(4);
    m_integer = RootbeerGpu.getSharedInteger(6);
    m_long = RootbeerGpu.getSharedLong(10);
    m_float = RootbeerGpu.getSharedFloat(18);
    m_double = RootbeerGpu.getSharedDouble(22);
  }

  public boolean compare(SharedMemSimpleRunOnGpu rhs) {
    if(m_boolean != rhs.m_boolean){
      System.out.println("m_boolean");
      return false;
    }
    if(m_byte != rhs.m_byte){
      System.out.println("m_byte");
      return false;
    }
    if(m_char != rhs.m_char){
      System.out.println("m_char");
      return false;
    } 
    if(m_short != rhs.m_short){
      System.out.println("m_short");
      return false;
    }
    if(m_integer != rhs.m_integer){
      System.out.println("m_integer");
      return false;
    }
    if(m_long != rhs.m_long){
      System.out.println("m_long");
      return false;
    }
    if(m_float != rhs.m_float){
      System.out.append("m_float");
      return false;
    }
    if(m_double != rhs.m_double){
      System.out.println("m_double");
      return false;
    }
    return true;
  }
}
