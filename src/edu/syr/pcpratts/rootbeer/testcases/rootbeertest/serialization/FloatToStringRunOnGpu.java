/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.testcases.rootbeertest.serialization;

import edu.syr.pcpratts.rootbeer.runtime.Kernel;

public class FloatToStringRunOnGpu implements Kernel {

  private String m_toString;
  private float m_value;
  
  public FloatToStringRunOnGpu(float value){
    m_toString = "str";
    m_value = value;
  }
  
  public void gpuMethod() {
    m_toString = "" + m_value * m_value;
  }

  public boolean compare(FloatToStringRunOnGpu rhs) {
    if(rhs.m_toString == null){
      System.out.println("rhs.m_toString == null");
      return false;
    }
    String lhs_str = m_toString;
    lhs_str = pad(lhs_str);
    if(rhs.m_toString.equals(lhs_str) == false){
      System.out.println("m_toString");
      System.out.println("  lhs: "+m_toString);
      System.out.println("  rhs: "+rhs.m_toString);
      return false;
    }
    return true;
  }

  private String pad(String lhs_str) {
    while(needPad(lhs_str)){
      lhs_str += "0";
    }
    return lhs_str;
  }

  private boolean needPad(String lhs_str) {
    String[] tokens = lhs_str.split("\\.");
    String token1 = tokens[1];
    if(token1.length() < 7){
      return true;
    }
    return false;
  }
}
