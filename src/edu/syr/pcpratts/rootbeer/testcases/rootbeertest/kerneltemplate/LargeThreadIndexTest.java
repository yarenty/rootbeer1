/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.testcases.rootbeertest.kerneltemplate;

import edu.syr.pcpratts.rootbeer.runtime.Kernel;
import edu.syr.pcpratts.rootbeer.runtime.ThreadConfig;
import edu.syr.pcpratts.rootbeer.test.TestKernelTemplate;

public class LargeThreadIndexTest implements TestKernelTemplate {

  public Kernel create() {
    return new LargeThreadIndexKernel();
  }

  public ThreadConfig getThreadConfig() {
    return new ThreadConfig(17, 1024);
  }

  public boolean compare(Kernel original, Kernel from_heap) {
    LargeThreadIndexKernel lhs = (LargeThreadIndexKernel) original;
    LargeThreadIndexKernel rhs = (LargeThreadIndexKernel) from_heap;
    return lhs.compare(rhs);
  }
  
}
