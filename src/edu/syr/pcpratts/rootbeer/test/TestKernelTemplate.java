/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.test;

import edu.syr.pcpratts.rootbeer.runtime.Kernel;
import edu.syr.pcpratts.rootbeer.runtime.ThreadConfig;
import java.util.List;

public interface TestKernelTemplate {

  Kernel create();
  ThreadConfig getThreadConfig();
  boolean compare(Kernel original, Kernel from_heap);
  
}
