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

public class FloatToStringTest implements TestSerialization {

  public List<Kernel> create() {
    List<Kernel> ret = new ArrayList<Kernel>();
    for(int i = 0; i < 5; ++i){
      ret.add(new FloatToStringRunOnGpu(i + 0.125f));
    }
    return ret;
  }

  public boolean compare(Kernel original, Kernel from_heap) {
    FloatToStringRunOnGpu lhs = (FloatToStringRunOnGpu) original;
    FloatToStringRunOnGpu rhs = (FloatToStringRunOnGpu) from_heap;
    return lhs.compare(rhs);
  }
}
