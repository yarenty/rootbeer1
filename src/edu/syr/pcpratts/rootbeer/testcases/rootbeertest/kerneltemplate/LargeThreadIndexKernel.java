/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.testcases.rootbeertest.kerneltemplate;

import edu.syr.pcpratts.rootbeer.runtime.Kernel;
import edu.syr.pcpratts.rootbeer.runtime.RootbeerGpu;

public class LargeThreadIndexKernel implements Kernel {

  private int[] m_result;
  
  public LargeThreadIndexKernel(){
    m_result = new int[17*1024];
  }
  
  public void gpuMethod() {
    int thread_id = RootbeerGpu.getThreadId();
    m_result[thread_id] = thread_id;
  }

  public boolean compare(LargeThreadIndexKernel rhs) {
    if(m_result.length != rhs.m_result.length){
      return false;
    }
    for(int i = 0; i < m_result.length; ++i){
      if(m_result[i] != i){
        System.out.println("lhs mismatch: ");
        System.out.println("  m_result["+i+"]: "+m_result[i]);
        return false;
      }
      if(rhs.m_result[i] != i){
        System.out.println("rhs mismatch: ");
        System.out.println("  rhs.m_result["+i+"]: "+rhs.m_result[i]);
        return false;
      }
    }
    return true;
  }
  
}
