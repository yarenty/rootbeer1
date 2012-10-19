/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.testcases.rootbeertest.gpurequired;

import edu.syr.pcpratts.rootbeer.runtime.Kernel;
import edu.syr.pcpratts.rootbeer.runtime.RootbeerGpu;

public class DotClassRunOnGpu implements Kernel {

  private String m_name;
  private String m_name2;
  private long m_ref;
  
  public void gpuMethod() {
    m_name = DotClassRunOnGpu.class.getName();
    m_name2 = int[][].class.getName();
  }
  
  public boolean compare(DotClassRunOnGpu rhs){    
    if(m_name.equals(rhs.m_name) == false){
      System.out.println("m_name: "+m_name);
      System.out.println("rhs.m_name: "+rhs.m_name);
      return false;
    }
    if(m_name2.equals(rhs.m_name2) == false){
      System.out.println("m_name2: "+m_name2);
      System.out.println("rhs.m_name2: "+rhs.m_name2);
      return false;
    }
    return true;
  }
}
