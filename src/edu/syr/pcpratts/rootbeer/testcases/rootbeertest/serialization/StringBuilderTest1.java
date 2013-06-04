/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.testcases.rootbeertest.serialization;

import edu.syr.pcpratts.rootbeer.runtime.Kernel;
import edu.syr.pcpratts.rootbeer.test.TestSerialization;
import java.util.ArrayList;
import java.util.List;

public class StringBuilderTest1 implements TestSerialization {

  public List<Kernel> create() {
    List<Kernel> ret = new ArrayList<Kernel>();
    for(int i = 0; i < 5; ++i){
      ret.add(new StringBuilderRunOnGpu1());
    }
    return ret;
  }

  public boolean compare(Kernel original, Kernel from_heap) {
    StringBuilderRunOnGpu1 lhs = (StringBuilderRunOnGpu1) original;
    StringBuilderRunOnGpu1 rhs = (StringBuilderRunOnGpu1) from_heap;
    return lhs.compare(rhs);
  }
  
}
