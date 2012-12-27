/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.testcases.rootbeertest.serialization;

import edu.syr.pcpratts.rootbeer.runtime.Kernel;

public class StringConstantKernel implements Kernel {

  private String m_string;
  
  public void gpuMethod() {
    m_string = "hello world";
  }

  boolean compare(StringConstantKernel rhs) {
    if(m_string.equals(rhs.m_string)){
      System.out.println("rhs.m_string: "+rhs.m_string);
      return false;
    }
    return true;
  }

}
